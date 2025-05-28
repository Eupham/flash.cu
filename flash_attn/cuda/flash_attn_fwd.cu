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
 * Exact implementation following the reference pattern for optimal performance.
 * This kernel uses raw pointers and simple indexing for maximum efficiency.
 */
template <int Bc, int Br, int HEAD_DIM>
__global__ void flash_attention_forward_kernel(
    const float* Q, const float* K, const float* V, 
    const int N, const int d,
    const int Tc, const int Tr, 
    const float softmax_scale,
    float* l, float* m, float* O,
    bool is_causal
) {
    int tx = threadIdx.x;
    int bx = blockIdx.x; 
    int by = blockIdx.y;  // batch and head index

    // Offset into Q,K,V,O,l,m - different for each batch and head
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);  // gridDim.y = nh
    int lm_offset = (bx * gridDim.y * N) + (by * N);  // offset for l and m

    // Define SRAM for Q,K,V,S
    extern __shared__ float sram[];
    int tile_size = Bc * d;  // size of Qi, Kj, Vj
    float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[tile_size * 2];
    float* S = &sram[tile_size * 3];

    for (int j = 0; j < Tc; j++) {

        // Load Kj, Vj to SRAM
        for (int x = 0; x < d; x++) {
            int kv_idx = j * Bc + tx;
            if (kv_idx < N) {
                Kj[(tx * d) + x] = K[qkv_offset + (kv_idx * d) + x];
                Vj[(tx * d) + x] = V[qkv_offset + (kv_idx * d) + x];
            } else {
                Kj[(tx * d) + x] = 0.0f;
                Vj[(tx * d) + x] = 0.0f;
            }
        }
        __syncthreads();  // such that the inner loop can use the correct Kj, Vj

        for (int i = 0; i < Tr; i++)  {

            // Load Qi to SRAM, l and m to registers
            for (int x = 0; x < d; x++) {
                int q_idx = i * Br + tx;
                if (q_idx < N) {
                    Qi[(tx * d) + x] = Q[qkv_offset + (q_idx * d) + x];
                } else {
                    Qi[(tx * d) + x] = 0.0f;
                }
            }
            
            int q_idx = i * Br + tx;
            float row_m_prev = (q_idx < N) ? m[lm_offset + q_idx] : -INFINITY;
            float row_l_prev = (q_idx < N) ? l[lm_offset + q_idx] : 0.0f;

            // S = QK^T, row_m = rowmax(S)
            float row_m = -INFINITY;
            for (int y = 0; y < Bc; y++) {
                float sum = 0;
                for (int x = 0; x < d; x++) {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                }
                sum *= softmax_scale;
                
                // Apply causal mask if needed
                if (is_causal) {
                    int k_idx = j * Bc + y;
                    if (k_idx > q_idx) {
                        sum = -INFINITY;
                    }
                }
                
                S[(Bc * tx) + y] = sum;

                if (sum > row_m)
                    row_m = sum;
            }

            // P = exp(S - row_m), row_l = rowsum(P)
            float row_l = 0;
            for (int y = 0; y < Bc; y++) {
                S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m);
                row_l += S[(Bc * tx) + y];
            }

            // Compute new m and l
            float row_m_new = fmaxf(row_m_prev, row_m);
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

            // Write O, l, m to HBM
            if (q_idx < N) {
                for (int x = 0; x < d; x++) {
                    float pv = 0;  // Pij * Vj
                    for (int y = 0; y < Bc; y++) {
                        pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                    }
                    float prev_o = O[qkv_offset + (q_idx * d) + x];
                    O[qkv_offset + (q_idx * d) + x] = (1.0f / row_l_new) * 
                        ((row_l_prev * __expf(row_m_prev - row_m_new) * prev_o) + 
                         (__expf(row_m - row_m_new) * pv));
                }
                m[lm_offset + q_idx] = row_m_new;
                l[lm_offset + q_idx] = row_l_new;
            }
        }
        __syncthreads();  // otherwise, thread can use the wrong Kj, Vj in inner loop
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
    
    // Calculate Tc and Tr like in the reference
    const int Tc = (seq_len_q + Bc - 1) / Bc;
    const int Tr = (seq_len_q + Br - 1) / Br;
    
    dim3 grid_dim(batch_size, num_heads);   // batch_size x num_heads
    dim3 block_dim(Bc);                     // Bc threads per block

    // Calculate shared memory size needed
    const int sram_size = (3 * Bc * head_dim * sizeof(float)) + (Bc * Br * sizeof(float));
    
    // Get raw pointers like the reference
    const float* Q_ptr = Q.data_ptr<float>();
    const float* K_ptr = K.data_ptr<float>();
    const float* V_ptr = V.data_ptr<float>();
    float* O_ptr = O.data_ptr<float>();
    float* L_ptr = L.data_ptr<float>();
    float* M_ptr = M.data_ptr<float>();
    
    // --- Dispatch to Templated Kernel based on Head Dimension ---
    // This allows using shared memory arrays sized at compile time via templates.
    if (head_dim <= 32) {
         flash_attention_forward_kernel<32, 32, 32><<<grid_dim, block_dim, sram_size>>>(
            Q_ptr, K_ptr, V_ptr, seq_len_q, head_dim, Tc, Tr, sm_scale, L_ptr, M_ptr, O_ptr, is_causal);
    } else if (head_dim <= 64) {
         flash_attention_forward_kernel<32, 32, 64><<<grid_dim, block_dim, sram_size>>>(
            Q_ptr, K_ptr, V_ptr, seq_len_q, head_dim, Tc, Tr, sm_scale, L_ptr, M_ptr, O_ptr, is_causal);
    } else if (head_dim <= 128) { // Corresponds to HEAD_DIM_MAX
         flash_attention_forward_kernel<32, 32, 128><<<grid_dim, block_dim, sram_size>>>(
            Q_ptr, K_ptr, V_ptr, seq_len_q, head_dim, Tc, Tr, sm_scale, L_ptr, M_ptr, O_ptr, is_causal);
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
