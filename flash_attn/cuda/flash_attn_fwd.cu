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
constexpr int T_r_DEFAULT = 64; 
constexpr int T_c_DEFAULT = 64; 
constexpr int HEAD_DIM_MAX = 128; 

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

    // --- Iterate over T_r Query Rows Processed by this Block ---
    // Each iteration of this loop processes one query vector Q_i from the block of T_r queries.
    // The online softmax statistics (m_i, l_i, o_i) are maintained per query Q_i.
    for (int q_tile_row_idx = 0; q_tile_row_idx < T_r; ++q_tile_row_idx) {
        const int q_abs_idx = q_block_start_row + q_tile_row_idx; // Absolute index in Q sequence.
        if (q_abs_idx >= seq_len_q) continue; // Boundary check: stop if past actual Q sequence length.

        // --- Initialize Online Softmax Statistics and Output Accumulator ---
        // m_i: current maximum score for Q_i (running max).
        // l_i: current sum of exp(score - m_i) for Q_i (running sum for normalizer).
        // o_i: accumulator for the output vector for Q_i (running sum of P_ij * V_j).
        float m_i = -std::numeric_limits<float>::infinity();
        float l_i = 0.0f;
        float o_i[HEAD_DIM]; // Temporary register array for the current Q_i's output vector.
        for (int h_col = 0; h_col < actual_head_dim; ++h_col) {
            o_i[h_col] = 0.0f;
        }

        // --- Load Current Query Vector (Q_i) into Shared Memory ---
        // Threads in the block cooperate to load Q_i into q_tile[q_tile_row_idx].
        // This simple load assumes q_tile[q_tile_row_idx] is for the current q_abs_idx.
        // A more complex strategy might load all T_r Q-vectors into q_tile at once
        // if threads within the block were to process multiple Q_i simultaneously.
        // Current model: one Q_i processed by the block through all K/V tiles.
        for (int h_col = threadIdx.x; h_col < actual_head_dim; h_col += blockDim.x) {
            q_tile[q_tile_row_idx][h_col] = Q_acc[batch_idx][head_idx][q_abs_idx][h_col];
        }
        // A __syncthreads() might be implied or needed here if all threads in the block
        // immediately depend on this specific Q_i being fully loaded for S_ij computation.
        // Given the s_tile computation structure, it's implicitly handled by the __syncthreads()
        // after S_ij computation.

        // --- Outer Loop: Iterate over Key/Value Blocks (Tiles of K and V) ---
        // Process K and V in tiles of size T_c to manage memory movement.
        for (int kv_block_col_start = 0; kv_block_col_start < seq_len_kv; kv_block_col_start += T_c) {
            
            // --- Load K_j_tile and V_j_tile into Shared Memory ---
            // Threads cooperate to load T_c vectors of K and V into k_tile and v_tile.
            // threadIdx.y iterates over rows of the K/V tile (up to T_c).
            // threadIdx.x iterates over elements of the head dimension for each K/V vector.
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
            // Q_i is q_tile[q_tile_row_idx]. K_j_tile is k_tile.
            // Scores are stored in s_tile[q_tile_row_idx][...].
            // Parallelism: threadIdx.y=0 computes scores for its assigned k_tile_col_idx.
            // This is a simplification; typically, more threads would participate if T_c > blockDim.x.
            if (threadIdx.y == 0) { 
                for (int k_tile_col_idx = threadIdx.x; k_tile_col_idx < T_c; k_tile_col_idx += blockDim.x) {
                    float sum_qk = 0.0f; // Dot product accumulator.
                    for (int d = 0; d < actual_head_dim; ++d) {
                        sum_qk += q_tile[q_tile_row_idx][d] * k_tile[k_tile_col_idx][d];
                    }
                    float current_s_val = sum_qk * sm_scale; // Apply scaling factor.

                    // Apply causal masking if enabled.
                    if (is_causal) {
                        int k_abs_idx = kv_block_col_start + k_tile_col_idx; // Absolute key index.
                        if (k_abs_idx > q_abs_idx) { // If key is "after" query.
                            current_s_val = -std::numeric_limits<float>::infinity();
                        }
                    }
                    s_tile[q_tile_row_idx][k_tile_col_idx] = current_s_val;
                }
            }
            __syncthreads(); // Ensure all S_ij scores for this Q_i and K_tile are computed.

            // --- Online Softmax: Update Statistics and Output Accumulator ---
            // 1. Find maximum score in the current S_ij tile row (for Q_i).
            float block_max_s = -std::numeric_limits<float>::infinity();
            // This reduction is simplified to be done by thread (0,0).
            // In practice, a parallel reduction (e.g., using warp shuffles or shared memory) is more efficient.
            if (threadIdx.x == 0 && threadIdx.y == 0) { 
                for (int k_col = 0; k_col < T_c; ++k_col) {
                    if (s_tile[q_tile_row_idx][k_col] > block_max_s) {
                        block_max_s = s_tile[q_tile_row_idx][k_col];
                    }
                }
            }
            // Broadcast block_max_s to all threads in the block.
            __shared__ float shared_block_max_s_val;
            if(threadIdx.x == 0 && threadIdx.y == 0) shared_block_max_s_val = block_max_s;
            __syncthreads(); // Synchronize to make shared_block_max_s_val visible.
            block_max_s = shared_block_max_s_val; // All threads now have the correct block max.

            // 2. Update overall m_i (running max for Q_i).
            float m_i_old = m_i;
            m_i = std::max(m_i, block_max_s);

            // 3. Rescale previous l_i and o_i based on new m_i.
            // This ensures numerical stability if m_i changes significantly.
            float p_scale_factor = expf(m_i_old - m_i); 
            for (int h_col = 0; h_col < actual_head_dim; ++h_col) { // These are register ops.
                o_i[h_col] *= p_scale_factor;
            }
            l_i *= p_scale_factor;

            // 4. Compute sum of new probabilities P_ij_new = exp(S_ij - m_i) for the current K_tile.
            float block_sum_p_new = 0.0f;
            // Simplified sum by thread (0,0). Parallel reduction is more efficient.
            if (threadIdx.x == 0 && threadIdx.y == 0) { 
                for (int k_col = 0; k_col < T_c; ++k_col) {
                    float s_val = s_tile[q_tile_row_idx][k_col];
                    if (s_val > -std::numeric_limits<float>::infinity()){ // Only consider non-masked scores.
                        block_sum_p_new += expf(s_val - m_i);
                    }
                }
            }
            // Broadcast block_sum_p_new.
            __shared__ float shared_block_sum_p_new_val;
            if(threadIdx.x == 0 && threadIdx.y == 0) shared_block_sum_p_new_val = block_sum_p_new;
            __syncthreads(); // Synchronize.
            block_sum_p_new = shared_block_sum_p_new_val; // All threads get the sum.

            // 5. Update overall l_i (running sum for normalizer for Q_i).
            l_i += block_sum_p_new;

            // 6. Accumulate V_j contributions to o_i, weighted by P_ij_new.
            // o_i += P_ij_new * V_j
            // Simplified accumulation by thread (0,0). Parallelism is key for performance here.
            // E.g., each thread (tx,ty) computes a partial sum for o_i elements it's responsible for.
            if (threadIdx.x == 0 && threadIdx.y == 0) {
                for (int k_col = 0; k_col < T_c; ++k_col) { // Iterate over columns of K_tile / V_tile
                    float s_val = s_tile[q_tile_row_idx][k_col];
                    if (s_val > -std::numeric_limits<float>::infinity()) { // If not masked
                        float p_val_new = expf(s_val - m_i); // Probability for this K_j
                        for (int h = 0; h < actual_head_dim; ++h) { // Iterate over head dimension
                            o_i[h] += p_val_new * v_tile[k_col][h];
                        }
                    }
                }
            }
            // This sync ensures that if thread (0,0) did the o_i update, other threads wait.
            // Also ensures K/V tiles in shared memory are processed before being overwritten in the next iteration.
            __syncthreads(); 
        } // End of loop over K/V blocks (kv_block_col_start)

        // --- Finalize Output for Q_i ---
        // Normalize the accumulated output o_i by the final l_i.
        if (l_i > 1e-8f) { // Avoid division by zero or very small l_i.
            float inv_l_i = 1.0f / l_i;
            // Parallel write to global memory O_acc.
            // Each thread writes a portion of the o_i vector.
            for (int h_col = threadIdx.x; h_col < actual_head_dim; h_col += blockDim.x) {
                O_acc[batch_idx][head_idx][q_abs_idx][h_col] = o_i[h_col] * inv_l_i;
            }
        } else { // Handle cases where l_i is zero (e.g., all scores were -inf).
             for (int h_col = threadIdx.x; h_col < actual_head_dim; h_col += blockDim.x) {
                O_acc[batch_idx][head_idx][q_abs_idx][h_col] = 0.0f; // Output zeros.
            }
        }
        
        // Store logsumexp L_i = m_i + log(l_i) for the backward pass.
        // Only one thread needs to write this for the current Q_i.
        if (threadIdx.x == 0 && threadIdx.y == 0) { 
            if (l_i > 1e-8f) { // Use same threshold for consistency.
                 L_acc[batch_idx][head_idx][q_abs_idx] = m_i + logf(l_i);
            } else {
                 L_acc[batch_idx][head_idx][q_abs_idx] = -std::numeric_limits<float>::infinity();
            }
        }
        // Synchronize to ensure O_acc and L_acc writes are complete before the next Q_i iteration (if any in T_r block)
        // or before the block finishes. This might be overly cautious if q_tile_row_idx loop implies sequential processing
        // by the "logical thread (0,0)" for critical sections, but safer with explicit parallel loops.
        __syncthreads(); 

    } // End of loop over q_tile_row_idx (rows of Q assigned to this block)
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

    TORCH_CHECK(head_dim <= HEAD_DIM_MAX, "Head dimension exceeds compiled maximum HEAD_DIM_MAX.");
    
    // --- Kernel Launch Configuration ---
    // Define thread block dimensions. These can be tuned.
    // blockDim.x: Threads along one dimension (e.g., head_dim elements or K_tile columns).
    // blockDim.y: Threads along another dimension (e.g., Q_tile rows or K_tile rows).
    dim3 threads_per_block;
    if (head_dim <= 32) threads_per_block = dim3(32, 4, 1); // Example: 128 threads
    else if (head_dim <= 64) threads_per_block = dim3(64, 4, 1); // Example: 256 threads
    else threads_per_block = dim3(128, 2, 1); // Example: 256 threads
    // Total threads per block = threads_per_block.x * threads_per_block.y * threads_per_block.z
    // This configuration should be chosen carefully based on kernel's parallelization strategy.

    // Define grid dimensions. Each block processes T_r_DEFAULT rows of Q.
    dim3 num_blocks((seq_len_q + T_r_DEFAULT - 1) / T_r_DEFAULT, num_heads, batch_size);

    // Get packed tensor accessors for efficient element access in CUDA.
    auto Q_acc = Q.packed_accessor32<float,4,torch::RestrictPtrTraits>();
    auto K_acc = K.packed_accessor32<float,4,torch::RestrictPtrTraits>();
    auto V_acc = V.packed_accessor32<float,4,torch::RestrictPtrTraits>();
    auto O_acc = O.packed_accessor32<float,4,torch::RestrictPtrTraits>();
    auto L_acc = L.packed_accessor32<float,3,torch::RestrictPtrTraits>();
    
    // --- Dispatch to Templated Kernel based on Head Dimension ---
    // This allows using shared memory arrays sized at compile time via templates.
    if (head_dim <= 32) {
         flash_attention_forward_kernel<T_r_DEFAULT, T_c_DEFAULT, 32><<<num_blocks, threads_per_block>>>(
            Q_acc, K_acc, V_acc, O_acc, L_acc, is_causal, sm_scale);
    } else if (head_dim <= 64) {
         flash_attention_forward_kernel<T_r_DEFAULT, T_c_DEFAULT, 64><<<num_blocks, threads_per_block>>>(
            Q_acc, K_acc, V_acc, O_acc, L_acc, is_causal, sm_scale);
    } else if (head_dim <= 128) { // Corresponds to HEAD_DIM_MAX
         flash_attention_forward_kernel<T_r_DEFAULT, T_c_DEFAULT, 128><<<num_blocks, threads_per_block>>>(
            Q_acc, K_acc, V_acc, O_acc, L_acc, is_causal, sm_scale);
    } else {
        // This case should be caught by the TORCH_CHECK for head_dim vs HEAD_DIM_MAX.
        AT_ERROR("Unsupported head_dimension: ", head_dim, ". Max supported by this build is ", HEAD_DIM_MAX);
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
