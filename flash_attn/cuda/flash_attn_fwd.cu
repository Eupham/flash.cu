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
// Conservative shared memory defaults to avoid CUDA build errors on most GPUs.
// These can be overwritten at compile time with -DT_r_DEFAULT=... etc.
#ifndef T_r_DEFAULT
#define T_r_DEFAULT 32
#endif
#ifndef T_c_DEFAULT
#define T_c_DEFAULT 32
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
 * @brief CUDA kernel for the forward pass of FlashAttention.
 * 
 * Based on the reference FlashAttention implementation pattern.
 * This kernel follows the proper tiling strategy where K/V tiles are in the outer loop
 * and Q processing is in the inner operations.
 * 
 * Template parameters:
 * @param Bc Block size for K/V sequence dimension (Bc = tile size for keys/values)
 * @param Br Block size for Q sequence dimension (Br = tile size for queries, but we process row by row)
 * @param HEAD_DIM Head dimension, used to size shared memory arrays.
 */
template <int Bc, int Br, int HEAD_DIM>
__global__ void flash_attention_forward_kernel(
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> Q_acc,
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> K_acc,
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> V_acc,
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> O_acc,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> L_acc,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> M_acc,
    bool is_causal,
    float softmax_scale
) {
    // Block and thread indices
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tx = threadIdx.x;  // Thread within block (0 to Bc-1)
    
    // Tensor dimensions
    const int N = Q_acc.size(2);  // Sequence length
    const int d = Q_acc.size(3);  // Head dimension
    const int actual_head_dim = min(d, HEAD_DIM);
    
    // Calculate number of tiles
    const int Tc = (N + Bc - 1) / Bc;  // Number of K/V tiles
    const int Tr = (N + Br - 1) / Br;  // Number of Q tiles
    
    // Shared memory allocation - same pattern as reference
    extern __shared__ float sram[];
    float* Qi = sram;                           // Size: Bc * d
    float* Kj = sram + Bc * HEAD_DIM;           // Size: Bc * d  
    float* Vj = sram + 2 * Bc * HEAD_DIM;      // Size: Bc * d
    float* S = sram + 3 * Bc * HEAD_DIM;       // Size: Bc * Bc (attention scores)
    
    // Only process if thread is within the valid range
    if (tx >= Bc) return;
    
    // Outer loop over K/V tiles (j index)
    for (int j = 0; j < Tc; j++) {
        
        // Load Kj, Vj to SRAM
        for (int x = 0; x < actual_head_dim; x++) {
            int kv_idx = j * Bc + tx;
            if (kv_idx < N) {
                Kj[tx * HEAD_DIM + x] = K_acc[batch_idx][head_idx][kv_idx][x];
                Vj[tx * HEAD_DIM + x] = V_acc[batch_idx][head_idx][kv_idx][x];
            } else {
                Kj[tx * HEAD_DIM + x] = 0.0f;
                Vj[tx * HEAD_DIM + x] = 0.0f;
            }
        }
        // Pad remaining dimensions
        for (int x = actual_head_dim; x < HEAD_DIM; x++) {
            Kj[tx * HEAD_DIM + x] = 0.0f;
            Vj[tx * HEAD_DIM + x] = 0.0f;
        }
        __syncthreads();
        
        // Inner loop over Q tiles (i index)
        for (int i = 0; i < Tr; i++) {
            
            // Load Qi to SRAM, load previous m and l values
            for (int x = 0; x < actual_head_dim; x++) {
                int q_idx = i * Br + tx;
                if (q_idx < N) {
                    Qi[tx * HEAD_DIM + x] = Q_acc[batch_idx][head_idx][q_idx][x];
                } else {
                    Qi[tx * HEAD_DIM + x] = 0.0f;
                }
            }
            // Pad remaining dimensions
            for (int x = actual_head_dim; x < HEAD_DIM; x++) {
                Qi[tx * HEAD_DIM + x] = 0.0f;
            }
            
            int q_idx = i * Br + tx;
            float row_m_prev = (q_idx < N) ? M_acc[batch_idx][head_idx][q_idx] : -INFINITY;
            float row_l_prev = (q_idx < N) ? L_acc[batch_idx][head_idx][q_idx] : 0.0f;
            
            // Compute S = QK^T, find row max
            float row_m = -INFINITY;
            for (int y = 0; y < Bc; y++) {
                float sum = 0.0f;
                for (int x = 0; x < actual_head_dim; x++) {
                    sum += Qi[tx * HEAD_DIM + x] * Kj[y * HEAD_DIM + x];
                }
                sum *= softmax_scale;
                
                // Apply causal mask if needed
                if (is_causal) {
                    int k_idx = j * Bc + y;
                    if (k_idx > q_idx) {
                        sum = -INFINITY;
                    }
                }
                
                S[tx * Bc + y] = sum;
                row_m = fmaxf(row_m, sum);
            }
            
            // Compute P = exp(S - row_m), find row sum
            float row_l = 0.0f;
            for (int y = 0; y < Bc; y++) {
                S[tx * Bc + y] = expf(S[tx * Bc + y] - row_m);
                row_l += S[tx * Bc + y];
            }
            
            // Update m and l using online softmax
            float row_m_new = fmaxf(row_m_prev, row_m);
            float row_l_new = expf(row_m_prev - row_m_new) * row_l_prev + expf(row_m - row_m_new) * row_l;
            
            // Update output O
            if (q_idx < N) {
                for (int x = 0; x < actual_head_dim; x++) {
                    float pv = 0.0f;  // P * V
                    for (int y = 0; y < Bc; y++) {
                        pv += S[tx * Bc + y] * Vj[y * HEAD_DIM + x];
                    }
                    
                    float prev_o = O_acc[batch_idx][head_idx][q_idx][x];
                    O_acc[batch_idx][head_idx][q_idx][x] = (1.0f / row_l_new) * 
                        (row_l_prev * expf(row_m_prev - row_m_new) * prev_o + expf(row_m - row_m_new) * pv);
                }
                
                // Update m and l in HBM
                M_acc[batch_idx][head_idx][q_idx] = row_m_new;
                L_acc[batch_idx][head_idx][q_idx] = row_l_new;
            }
        }
        __syncthreads();
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
    
    // --- Kernel Launch Configuration (based on reference) ---
    const int Bc = T_c_DEFAULT_VAL;  // Use Bc for block size like reference
    const int Br = T_r_DEFAULT_VAL;  // Use Br for block size like reference
    
    dim3 grid_dim(batch_size, num_heads);   // batch_size x num_heads
    dim3 block_dim(Bc);                     // Bc threads per block

    // Calculate shared memory size needed
    const int sram_size = (3 * Bc * head_dim * sizeof(float)) + (Bc * Br * sizeof(float));
    
    // Get packed tensor accessors for efficient element access in CUDA.
    auto Q_acc = Q.packed_accessor32<float,4,torch::RestrictPtrTraits>();
    auto K_acc = K.packed_accessor32<float,4,torch::RestrictPtrTraits>();
    auto V_acc = V.packed_accessor32<float,4,torch::RestrictPtrTraits>();
    auto O_acc = O.packed_accessor32<float,4,torch::RestrictPtrTraits>();
    auto L_acc = L.packed_accessor32<float,3,torch::RestrictPtrTraits>();
    auto M_acc = M.packed_accessor32<float,3,torch::RestrictPtrTraits>();
    
    // --- Dispatch to Templated Kernel based on Head Dimension ---
    // This allows using shared memory arrays sized at compile time via templates.
    if (head_dim <= 32) {
         flash_attention_forward_kernel<32, 32, 32><<<grid_dim, block_dim, sram_size>>>(
            Q_acc, K_acc, V_acc, O_acc, L_acc, M_acc, is_causal, sm_scale);
    } else if (head_dim <= 64) {
         flash_attention_forward_kernel<32, 32, 64><<<grid_dim, block_dim, sram_size>>>(
            Q_acc, K_acc, V_acc, O_acc, L_acc, M_acc, is_causal, sm_scale);
    } else if (head_dim <= 128) { // Corresponds to HEAD_DIM_MAX
         flash_attention_forward_kernel<32, 32, 128><<<grid_dim, block_dim, sram_size>>>(
            Q_acc, K_acc, V_acc, O_acc, L_acc, M_acc, is_causal, sm_scale);
    } else {
        // This case should be caught by the TORCH_CHECK for head_dim vs HEAD_DIM_MAX.
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
