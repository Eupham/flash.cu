#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h> // For __half, if FP16/BF16 support is added later
#include <cmath>       // For expf, logf, fabsf, sqrtf
#include <algorithm>   // For std::max, std::min
#include <limits>      // For std::numeric_limits

// Default tile sizes for Q and K/V sequence dimensions, and max head dimension for shared memory.
// These constants define the tiling strategy and can be tuned for performance on different hardware.
// T_r: Tile size along the query sequence length dimension.
// T_c: Tile size along the key/value sequence length dimension.
// HEAD_DIM_MAX: Maximum head dimension supported by kernel versions with fixed shared memory.
//               Kernels are templated on HEAD_DIM, and this acts as an upper bound for dispatch.
// Optimized tile sizes similar to Triton implementations
// These can be overwritten at compile time with -DT_r_DEFAULT=... etc.
#ifndef T_r_DEFAULT
#define T_r_DEFAULT 64
#endif
#ifndef T_c_DEFAULT
#define T_c_DEFAULT 64
#endif
#ifndef HEAD_DIM_MAX
#define HEAD_DIM_MAX 128
#endif

constexpr int T_r_DEFAULT_VAL = T_r_DEFAULT;
constexpr int T_c_DEFAULT_VAL = T_c_DEFAULT;
constexpr int HEAD_DIM_MAX_VAL = HEAD_DIM_MAX;

// Forward declaration for the backward pass CUDA dispatcher function.
// The actual definition resides in flash_attn_bwd.cu.
// This is necessary for the Pybind11 module definition at the end of this file,
// which binds both forward and backward functions to the Python module.
void flash_attention_backward_cuda(
    const torch::Tensor& Q, const torch::Tensor& K, const torch::Tensor& V,
    const torch::Tensor& O, const torch::Tensor& dO, const torch::Tensor& L,
    torch::Tensor& dQ, torch::Tensor& dK, torch::Tensor& dV,
    bool is_causal, float sm_scale
);

/**
 * @brief CUDA kernel for FlashAttention forward pass - Triton-style implementation.
 * 
 * Based on Alex Dremov's blog post: each job loads a single Q tile, 
 * iterates over all tiles in K and V, and accumulates the result.
 * This follows the exact algorithm from the blog with proper online softmax.
 */
template <int BLOCK_M, int BLOCK_N, int HEAD_DIM>
__global__ void flash_attention_forward_kernel(
    const float* Q, const float* K, const float* V, 
    const int N, const int d,
    const int Tc, const int Tr, 
    const float softmax_scale,
    float* l, float* m, float* O,
    bool is_causal
) {
    // Each block processes one Q tile (BLOCK_M rows)
    int batch_head_idx = blockIdx.x * gridDim.y + blockIdx.y;
    int q_block_idx = blockIdx.z;  // Which Q tile this block processes
    int tid = threadIdx.x;
    
    // Calculate offsets for this batch/head
    int qkv_offset = batch_head_idx * N * d;
    int lm_offset = batch_head_idx * N;
    int q_offset = q_block_idx * BLOCK_M;

    // Shared memory layout
    extern __shared__ float sram[];
    float* q_tile = sram;                                    // [BLOCK_M, HEAD_DIM]
    float* k_tile = sram + BLOCK_M * HEAD_DIM;               // [BLOCK_N, HEAD_DIM] 
    float* v_tile = k_tile + BLOCK_N * HEAD_DIM;             // [BLOCK_N, HEAD_DIM]
    float* s_tile = v_tile + BLOCK_N * HEAD_DIM;             // [BLOCK_M, BLOCK_N]

    // Load Q tile once (this block's responsibility)
    for (int i = tid; i < BLOCK_M * HEAD_DIM; i += blockDim.x) {
        int row = i / HEAD_DIM;
        int col = i % HEAD_DIM;
        int global_row = q_offset + row;
        
        if (global_row < N && col < d) {
            q_tile[row * HEAD_DIM + col] = Q[qkv_offset + global_row * d + col];
        } else {
            q_tile[row * HEAD_DIM + col] = 0.0f;
        }
    }
    __syncthreads();

    // Initialize running statistics per thread (each thread handles one Q row)
    float m_i = -INFINITY;  // running max
    float l_i = 0.0f;       // running softmax denominator  
    float acc[HEAD_DIM];    // accumulator for output
    
    // Initialize accumulator to zero
    if (tid < BLOCK_M) {
        #pragma unroll
        for (int d_idx = 0; d_idx < HEAD_DIM; d_idx++) {
            acc[d_idx] = 0.0f;
        }
    }

    // Iterate over all K/V tiles (following Triton approach)
    for (int kv_tile_idx = 0; kv_tile_idx < Tc; kv_tile_idx++) {
        int kv_offset = kv_tile_idx * BLOCK_N;
        
        // Load K and V tiles cooperatively
        for (int i = tid; i < BLOCK_N * HEAD_DIM; i += blockDim.x) {
            int row = i / HEAD_DIM;
            int col = i % HEAD_DIM;
            int global_row = kv_offset + row;
            
            if (global_row < N && col < d) {
                k_tile[row * HEAD_DIM + col] = K[qkv_offset + global_row * d + col];
                v_tile[row * HEAD_DIM + col] = V[qkv_offset + global_row * d + col];
            } else {
                k_tile[row * HEAD_DIM + col] = 0.0f;
                v_tile[row * HEAD_DIM + col] = 0.0f;
            }
        }
        __syncthreads();

        // Each thread processes one Q row
        if (tid < BLOCK_M) {
            int q_idx = q_offset + tid;
            if (q_idx >= N) continue;

            // Compute QK^T for this Q row
            float m_ij = -INFINITY;  // max of current tile
            for (int k_idx = 0; k_idx < BLOCK_N; k_idx++) {
                float qk_val = 0.0f;
                
                // Dot product: Q[tid] @ K[k_idx]
                #pragma unroll
                for (int d_idx = 0; d_idx < HEAD_DIM; d_idx++) {
                    qk_val += q_tile[tid * HEAD_DIM + d_idx] * k_tile[k_idx * HEAD_DIM + d_idx];
                }
                
                qk_val *= softmax_scale;
                
                // Apply causal mask
                if (is_causal) {
                    int global_k_idx = kv_offset + k_idx;
                    if (global_k_idx > q_idx) {
                        qk_val = -INFINITY;
                    }
                }
                
                s_tile[tid * BLOCK_N + k_idx] = qk_val;
                m_ij = fmaxf(m_ij, qk_val);
            }

            // Online softmax update (following the blog's formulas)
            float m_new = fmaxf(m_i, m_ij);
            float alpha = __expf(m_i - m_new);
            
            // Compute probabilities and sum for current tile
            float l_ij = 0.0f;
            for (int k_idx = 0; k_idx < BLOCK_N; k_idx++) {
                float p_val = __expf(s_tile[tid * BLOCK_N + k_idx] - m_new);
                s_tile[tid * BLOCK_N + k_idx] = p_val;
                l_ij += p_val;
            }
            
            // Update denominator: l(x) = e^(m(x1) - m(x)) * l(x1) + l(x2)
            float l_new = alpha * l_i + l_ij;

            // Update accumulator: acc = alpha * acc + P @ V
            #pragma unroll
            for (int d_idx = 0; d_idx < HEAD_DIM; d_idx++) {
                // Scale previous accumulator
                acc[d_idx] *= alpha;
                
                // Add P @ V for current tile
                float pv = 0.0f;
                for (int k_idx = 0; k_idx < BLOCK_N; k_idx++) {
                    pv += s_tile[tid * BLOCK_N + k_idx] * v_tile[k_idx * HEAD_DIM + d_idx];
                }
                acc[d_idx] += pv;
            }

            // Update running statistics
            m_i = m_new;
            l_i = l_new;
        }
        __syncthreads();
    }

    // Final normalization and write output
    if (tid < BLOCK_M) {
        int q_idx = q_offset + tid;
        if (q_idx < N) {
            // Normalize by softmax denominator
            #pragma unroll
            for (int d_idx = 0; d_idx < HEAD_DIM; d_idx++) {
                O[qkv_offset + q_idx * d + d_idx] = acc[d_idx] / l_i;
            }
            
            // Store running statistics
            m[lm_offset + q_idx] = m_i;
            l[lm_offset + q_idx] = l_i;
        }
    }
}


/**
 * @brief C++ dispatcher for the FlashAttention forward CUDA kernel.
 * 
 * This function handles tensor validation, determines kernel launch parameters,
 * and calls the appropriate templated version of `flash_attention_forward_kernel`
 * based on the head dimension.
 * 
 * @param Q Input Query tensor.
 * @param K Input Key tensor.
 * @param V Input Value tensor.
 * @param O Output tensor, will be populated by the kernel.
 * @param L Output Logsumexp tensor, will be populated by the kernel.
 * @param is_causal Boolean flag for causal masking.
 * @param sm_scale Scaling factor for attention scores.
 */
void flash_attention_forward_cuda(
    const torch::Tensor& Q, 
    const torch::Tensor& K, 
    const torch::Tensor& V, 
    torch::Tensor& O,       
    torch::Tensor& L,       
    torch::Tensor& M,       // Add M tensor for row max values
    bool is_causal,
    float sm_scale
) {
    // --- Input Tensor Validation ---
    // Check if tensors are on CUDA device.
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");
    TORCH_CHECK(O.is_cuda(), "O must be a CUDA tensor");
    TORCH_CHECK(L.is_cuda(), "L must be a CUDA tensor");
    TORCH_CHECK(M.is_cuda(), "M must be a CUDA tensor");

    // Check tensor dimensions.
    TORCH_CHECK(Q.dim() == 4, "Q must be 4D");
    TORCH_CHECK(K.dim() == 4, "K must be 4D");
    TORCH_CHECK(V.dim() == 4, "V must be 4D");
    TORCH_CHECK(O.dim() == 4, "O must be 4D");
    TORCH_CHECK(L.dim() == 3, "L must be 3D");
    TORCH_CHECK(M.dim() == 3, "M must be 3D");
    
    // Check tensor data types. Currently, only Float32 is supported by this kernel.
    TORCH_CHECK(Q.dtype() == K.dtype() && Q.dtype() == V.dtype(), "All input tensors Q, K, V must have the same dtype");
    TORCH_CHECK(Q.dtype() == O.dtype(), "Input Q and Output O tensors must have the same dtype");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Currently only Float32 is supported for Q, K, V, O"); 
    TORCH_CHECK(L.dtype() == torch::kFloat32, "L tensor must be Float32");
    TORCH_CHECK(M.dtype() == torch::kFloat32, "M tensor must be Float32");

    // --- Shape Compatibility and Parameter Extraction ---
    const int batch_size = Q.size(0);
    const int num_heads = Q.size(1);
    const int seq_len_q = Q.size(2);
    const int head_dim = Q.size(3);
    const int seq_len_kv = K.size(2);

    TORCH_CHECK(K.size(0) == batch_size && K.size(1) == num_heads && K.size(3) == head_dim, "K shape mismatch with Q");
    TORCH_CHECK(V.size(0) == batch_size && V.size(1) == num_heads && V.size(2) == seq_len_kv && V.size(3) == head_dim, "V shape mismatch with K");
    TORCH_CHECK(O.size(0) == batch_size && O.size(1) == num_heads && O.size(2) == seq_len_q && O.size(3) == head_dim, "O shape mismatch with Q");
    TORCH_CHECK(L.size(0) == batch_size && L.size(1) == num_heads && L.size(2) == seq_len_q, "L shape mismatch");
    TORCH_CHECK(M.size(0) == batch_size && M.size(1) == num_heads && M.size(2) == seq_len_q, "M shape mismatch");

    TORCH_CHECK(head_dim <= HEAD_DIM_MAX_VAL, "Head dimension exceeds compiled maximum HEAD_DIM_MAX.");
    
    // --- Kernel Launch Configuration (Triton-style: one block per Q tile) ---
    const int BLOCK_M = 32;  // Smaller tile sizes to fit in shared memory
    const int BLOCK_N = 32;
    
    // Calculate number of blocks needed
    const int Tc = (seq_len_kv + BLOCK_N - 1) / BLOCK_N;  // Number of K/V tiles
    const int Tr = (seq_len_q + BLOCK_M - 1) / BLOCK_M;   // Number of Q tiles
    
    // Grid configuration: (batch_size, num_heads, num_Q_tiles)
    // Each block processes one Q tile, iterates through all K/V tiles
    dim3 grid_dim(batch_size, num_heads, Tr);
    dim3 block_dim(BLOCK_M);  // One thread per Q row in the tile
    
    // Calculate shared memory size needed
    // q_tile: BLOCK_M * head_dim, k_tile: BLOCK_N * head_dim, v_tile: BLOCK_N * head_dim
    // s_tile: BLOCK_M * BLOCK_N (for attention scores)
    const int sram_size = ((BLOCK_M + 2 * BLOCK_N) * head_dim + BLOCK_M * BLOCK_N) * sizeof(float);
    
    // Ensure we don't exceed shared memory limits
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    TORCH_CHECK(sram_size <= max_sram_size, 
                "Shared memory requirement (", sram_size, " bytes) exceeds device limit (", max_sram_size, " bytes)");
    
    // Get raw pointers
    const float* Q_ptr = Q.data_ptr<float>();
    const float* K_ptr = K.data_ptr<float>();
    const float* V_ptr = V.data_ptr<float>();
    float* O_ptr = O.data_ptr<float>();
    float* L_ptr = L.data_ptr<float>();
    float* M_ptr = M.data_ptr<float>();
    
    // --- Dispatch to Templated Kernel based on Head Dimension ---
    if (head_dim <= 32) {
         flash_attention_forward_kernel<32, 32, 32><<<grid_dim, block_dim, sram_size>>>(
            Q_ptr, K_ptr, V_ptr, seq_len_q, head_dim, Tc, Tr, sm_scale, L_ptr, M_ptr, O_ptr, is_causal);
    } else if (head_dim <= 64) {
         flash_attention_forward_kernel<32, 32, 64><<<grid_dim, block_dim, sram_size>>>(
            Q_ptr, K_ptr, V_ptr, seq_len_q, head_dim, Tc, Tr, sm_scale, L_ptr, M_ptr, O_ptr, is_causal);
    } else if (head_dim <= 128) {
         flash_attention_forward_kernel<32, 32, 128><<<grid_dim, block_dim, sram_size>>>(
            Q_ptr, K_ptr, V_ptr, seq_len_q, head_dim, Tc, Tr, sm_scale, L_ptr, M_ptr, O_ptr, is_causal);
    } else {
        AT_ERROR("Unsupported head_dimension: ", head_dim, ". Max supported by this build is ", HEAD_DIM_MAX_VAL);
    }

    // Check for any CUDA errors during kernel launch.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed in flash_attention_forward_cuda: ", cudaGetErrorString(err));
    }
}

// PYBIND11_MODULE: Defines the Python module structure for the CUDA extension.
// TORCH_EXTENSION_NAME is typically defined by the build system (e.g., setuptools based on 'name' in CUDAExtension).
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUDA implementations for FlashAttention components"; // Optional module docstring
    m.def("forward", &flash_attention_forward_cuda, "FlashAttention forward pass (CUDA)");
    m.def("backward", &flash_attention_backward_cuda, "FlashAttention backward pass (CUDA)");
}
