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
#define T_c_DEFAULT 16
#endif
#ifndef HEAD_DIM_MAX
#define HEAD_DIM_MAX 64
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
 * This kernel computes scaled dot-product attention using tiling and online softmax
 * to reduce HBM memory reads/writes.
 * 
 * Template parameters:
 * @param T_r Tile size for the query sequence dimension (rows of Q processed by a block).
 * @param T_c Tile size for the key/value sequence dimension (columns of K/V processed per tile).
 * @param HEAD_DIM Head dimension, used to size shared memory arrays.
 * 
 * Tensor arguments (accessed via PackedTensorAccessor):
 * @param Q_acc Query tensor (Batch, NumHeads, SeqLen_Q, HeadDim).
 * @param K_acc Key tensor (Batch, NumHeads, SeqLen_KV, HeadDim).
 * @param V_acc Value tensor (Batch, NumHeads, SeqLen_KV, HeadDim).
 * @param O_acc Output tensor (Batch, NumHeads, SeqLen_Q, HeadDim).
 * @param L_acc Logsumexp tensor (Batch, NumHeads, SeqLen_Q) for storing m_i + log(l_i), useful for backward pass.
 * 
 * Other arguments:
 * @param is_causal Boolean flag to enable/disable causal masking.
 * @param sm_scale Scaling factor for dot products (typically 1/sqrt(head_dim)).
 */
template <int T_r, int T_c, int HEAD_DIM>
__global__ void flash_attention_forward_kernel(
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> Q_acc,
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> K_acc,
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> V_acc,
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> O_acc,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> L_acc,
    bool is_causal,
    float sm_scale
) {
    // --- Block and Dimension Setup ---
    // Determine batch, head, and starting query row for this thread block.
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int q_block_start_row = blockIdx.x * T_r; // Each block processes T_r rows of Q.

    // Get actual dimensions from input tensors.
    const int actual_head_dim = Q_acc.size(3); 
    const int seq_len_q = Q_acc.size(2);
    const int seq_len_kv = K_acc.size(2);

    // Ensure the kernel isn't run with a head dimension larger than it was compiled for.
    if (actual_head_dim > HEAD_DIM) {
        // This situation should ideally be prevented by the C++ dispatcher.
        return; 
    }

    // --- Shared Memory Allocation ---
    // Shared memory tiles for Q, K, V, and intermediate scores S_ij.
    // Sized using compile-time template parameters for efficiency.
    __shared__ float q_tile[T_r][HEAD_DIM]; // Stores one Q-vector per row if T_r > 1, or part of it.
    __shared__ float k_tile[T_c][HEAD_DIM]; // Stores a tile of K vectors.
    __shared__ float v_tile[T_c][HEAD_DIM]; // Stores a tile of V vectors.
    __shared__ float s_tile[T_r][T_c];      // Stores QK^T scores for the current q_tile and k_tile.

    // --- Initialize Output Accumulators and Online Softmax Statistics ---
    // Each thread handles one query row, storing state in registers
    const int local_q_idx = threadIdx.y; // Thread's query row within the block
    const int q_abs_idx = q_block_start_row + local_q_idx; // Absolute query index
    
    // Per-thread online softmax statistics and output accumulator
    float m_i = -std::numeric_limits<float>::infinity();
    float l_i = 0.0f;
    float o_i[HEAD_DIM] = {0}; // Initialize to zero
    
    // Early exit for threads beyond valid query range
    bool valid_q = (q_abs_idx < seq_len_q);
    
    // --- Load Q-block into Shared Memory (All threads cooperate) ---
    // Load T_r query vectors into shared memory in parallel
    for (int q_row = threadIdx.y; q_row < T_r; q_row += blockDim.y) {
        int q_abs_row = q_block_start_row + q_row;
        for (int h_col = threadIdx.x; h_col < actual_head_dim; h_col += blockDim.x) {
            if (q_abs_row < seq_len_q) {
                q_tile[q_row][h_col] = Q_acc[batch_idx][head_idx][q_abs_row][h_col];
            } else {
                q_tile[q_row][h_col] = 0.0f;
            }
        }
    }
    __syncthreads(); // Ensure all Q vectors are loaded

    // --- Outer Loop: Iterate over Key/Value Blocks (Tiles of K and V) ---
    // Process K and V in tiles of size T_c to manage memory movement.
    for (int kv_block_col_start = 0; kv_block_col_start < seq_len_kv; kv_block_col_start += T_c) {
        
        // --- Load K_j_tile and V_j_tile into Shared Memory ---
        // All threads cooperate to load T_c vectors of K and V into k_tile and v_tile.
        for (int tc_row = threadIdx.y; tc_row < T_c; tc_row += blockDim.y) { 
            for (int h_col = threadIdx.x; h_col < actual_head_dim; h_col += blockDim.x) { 
                int k_abs_row_idx = kv_block_col_start + tc_row; // Absolute index in K/V sequence.
                if (k_abs_row_idx < seq_len_kv) { // Boundary check for K/V sequence length.
                    k_tile[tc_row][h_col] = K_acc[batch_idx][head_idx][k_abs_row_idx][h_col];
                    v_tile[tc_row][h_col] = V_acc[batch_idx][head_idx][k_abs_row_idx][h_col];
                } else { // Pad with zeros if past actual K/V sequence length.
                    k_tile[tc_row][h_col] = 0.0f;
                    v_tile[tc_row][h_col] = 0.0f;
                }
            }
        }
        __syncthreads(); // Ensure K_tile and V_tile are fully loaded before use.

        // --- Compute Scores S_ij = (Q_i @ K_j_tile.T) * sm_scale ---
        // Optimized matrix multiplication with better memory access patterns
        if (valid_q) {
            // Compute all scores for this thread's query row in parallel
            for (int k_col = threadIdx.x; k_col < T_c; k_col += blockDim.x) {
                float dot_product = 0.0f;
                
                // Vectorized dot product computation
                #pragma unroll 4
                for (int d = 0; d < actual_head_dim; ++d) {
                    dot_product += q_tile[local_q_idx][d] * k_tile[k_col][d];
                }
                
                float score = dot_product * sm_scale;
                
                // Apply causal masking if enabled
                if (is_causal) {
                    int k_abs_idx = kv_block_col_start + k_col;
                    if (k_abs_idx > q_abs_idx) {
                        score = -std::numeric_limits<float>::infinity();
                    }
                }
                
                s_tile[local_q_idx][k_col] = score;
            }
        }
        __syncthreads(); // Ensure all S_ij scores are computed.

        // --- Online Softmax: Update Statistics and Output Accumulator ---
        if (valid_q) {
            // 1. Find maximum score in the current S_ij tile row (for this thread's Q_i).
            float block_max_s = -std::numeric_limits<float>::infinity();
            for (int k_col = 0; k_col < T_c; ++k_col) {
                if (s_tile[local_q_idx][k_col] > block_max_s) {
                    block_max_s = s_tile[local_q_idx][k_col];
                }
            }

            // 2. Update global maximum m_i
            float new_m_i = fmaxf(m_i, block_max_s);

            // 3. Compute scaling factors for previous and current contributions
            float exp_diff_old = expf(m_i - new_m_i);
            float exp_diff_new = expf(block_max_s - new_m_i);

            // 4. Compute sum of exp(s_ij - new_m_i) for current tile
            float tile_sum = 0.0f;
            for (int k_col = 0; k_col < T_c; ++k_col) {
                tile_sum += expf(s_tile[local_q_idx][k_col] - new_m_i);
            }

            // 5. Update l_i (running sum for normalizer)
            float new_l_i = exp_diff_old * l_i + tile_sum;

            // 6. Update output accumulator o_i
            // Scale previous contributions
            for (int h_col = 0; h_col < actual_head_dim; ++h_col) {
                o_i[h_col] *= exp_diff_old;
            }

            // Add new contributions: sum over k_col of P_ij * V_j
            for (int k_col = 0; k_col < T_c; ++k_col) {
                float p_ij = expf(s_tile[local_q_idx][k_col] - new_m_i);
                for (int h_col = 0; h_col < actual_head_dim; ++h_col) {
                    o_i[h_col] += p_ij * v_tile[k_col][h_col];
                }
            }

            // 7. Update statistics
            m_i = new_m_i;
            l_i = new_l_i;
        }
        __syncthreads(); // Ensure all threads finish before next KV tile
    } // End of KV tiles loop

    // --- Final Output and Statistics ---
    if (valid_q) {
        // Normalize final output
        float inv_l_i = 1.0f / l_i;
        for (int h_col = 0; h_col < actual_head_dim; ++h_col) {
            o_i[h_col] *= inv_l_i;
        }

        // Write output to global memory
        for (int h_col = threadIdx.x; h_col < actual_head_dim; h_col += blockDim.x) {
            O_acc[batch_idx][head_idx][q_abs_idx][h_col] = o_i[h_col];
        }

        // Write logsumexp (for backward pass)
        if (threadIdx.x == 0) {
            L_acc[batch_idx][head_idx][q_abs_idx] = m_i + logf(l_i);
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

    // Check tensor dimensions.
    TORCH_CHECK(Q.dim() == 4, "Q must be 4D");
    TORCH_CHECK(K.dim() == 4, "K must be 4D");
    TORCH_CHECK(V.dim() == 4, "V must be 4D");
    TORCH_CHECK(O.dim() == 4, "O must be 4D");
    TORCH_CHECK(L.dim() == 3, "L must be 3D");
    
    // Check tensor data types. Currently, only Float32 is supported by this kernel.
    TORCH_CHECK(Q.dtype() == K.dtype() && Q.dtype() == V.dtype(), "All input tensors Q, K, V must have the same dtype");
    TORCH_CHECK(Q.dtype() == O.dtype(), "Input Q and Output O tensors must have the same dtype");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Currently only Float32 is supported for Q, K, V, O"); 
    TORCH_CHECK(L.dtype() == torch::kFloat32, "L tensor must be Float32");


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

    TORCH_CHECK(head_dim <= HEAD_DIM_MAX_VAL, "Head dimension exceeds compiled maximum HEAD_DIM_MAX.");
    
    // --- Kernel Launch Configuration ---
    // Optimized thread block configuration for better performance
    // Use power-of-2 dimensions for better warp utilization
    dim3 threads_per_block;
    if (head_dim <= 32) threads_per_block = dim3(16, 8, 1); // 128 threads: 16x8 for better memory coalescing
    else if (head_dim <= 64) threads_per_block = dim3(32, 8, 1); // 256 threads: 32x8 for head_dim=64
    else threads_per_block = dim3(32, 8, 1); // 256 threads: maintain same config for larger dims

    // Define grid dimensions. Each block processes T_r_DEFAULT rows of Q.
    dim3 num_blocks((seq_len_q + T_r_DEFAULT_VAL - 1) / T_r_DEFAULT_VAL, num_heads, batch_size);

    // Get packed tensor accessors for efficient element access in CUDA.
    auto Q_acc = Q.packed_accessor32<float,4,torch::RestrictPtrTraits>();
    auto K_acc = K.packed_accessor32<float,4,torch::RestrictPtrTraits>();
    auto V_acc = V.packed_accessor32<float,4,torch::RestrictPtrTraits>();
    auto O_acc = O.packed_accessor32<float,4,torch::RestrictPtrTraits>();
    auto L_acc = L.packed_accessor32<float,3,torch::RestrictPtrTraits>();
    
    // --- Dispatch to Templated Kernel based on Head Dimension ---
    // This allows using shared memory arrays sized at compile time via templates.
    if (head_dim <= 32) {
         flash_attention_forward_kernel<T_r_DEFAULT_VAL, T_c_DEFAULT_VAL, 32><<<num_blocks, threads_per_block>>>(
            Q_acc, K_acc, V_acc, O_acc, L_acc, is_causal, sm_scale);
    } else if (head_dim <= 64) {
         flash_attention_forward_kernel<T_r_DEFAULT_VAL, T_c_DEFAULT_VAL, 64><<<num_blocks, threads_per_block>>>(
            Q_acc, K_acc, V_acc, O_acc, L_acc, is_causal, sm_scale);
    } else if (head_dim <= 128) { // Corresponds to HEAD_DIM_MAX
         flash_attention_forward_kernel<T_r_DEFAULT_VAL, T_c_DEFAULT_VAL, 128><<<num_blocks, threads_per_block>>>(
            Q_acc, K_acc, V_acc, O_acc, L_acc, is_causal, sm_scale);
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
