#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h> // For __half, if FP16/BF16 support is added later
#include <cmath>       // For expf, logf, fabsf
#include <algorithm>   // For std::max, std::min
#include <limits>      // For std::numeric_limits

// Default tile sizes for Q and K/V sequence dimensions, and max head dimension for shared memory.
// These constants define the tiling strategy for the backward pass.
// T_r_BWD_DEFAULT: Tile size along the query sequence length dimension (rows of Q processed by a block).
// T_c_BWD_DEFAULT: Tile size along the key/value sequence length dimension (columns of K/V processed per tile).
// HEAD_DIM_MAX_BWD: Maximum head dimension supported by kernel versions with fixed shared memory.
constexpr int T_r_BWD_DEFAULT = 64; 
constexpr int T_c_BWD_DEFAULT = 64; 
constexpr int HEAD_DIM_MAX_BWD = 128;

/**
 * @brief CUDA kernel for the backward pass of FlashAttention.
 * 
 * This kernel computes gradients dQ, dK, dV given dO (gradient of output), Q, K, V, O (output),
 * and L (logsumexp from forward pass). It recomputes attention scores and probabilities
 * on-the-fly to save memory, a core idea of FlashAttention.
 * 
 * Template parameters:
 * @param T_r Tile size for the query sequence dimension.
 * @param T_c Tile size for the key/value sequence dimension.
 * @param HEAD_DIM Head dimension, used for sizing shared memory arrays.
 * 
 * Tensor arguments (accessed via PackedTensorAccessor):
 * @param Q_acc Query tensor from forward pass.
 * @param K_acc Key tensor from forward pass.
 * @param V_acc Value tensor from forward pass.
 * @param O_acc Output tensor from forward pass.
 * @param dO_acc Gradient of the output tensor O.
 * @param L_acc Logsumexp tensor (m_i + log(l_i)) from forward pass.
 * @param dQ_acc Output gradient tensor for Q.
 * @param dK_acc Output gradient tensor for K (accumulated using atomicAdd).
 * @param dV_acc Output gradient tensor for V (accumulated using atomicAdd).
 * 
 * Other arguments:
 * @param is_causal Boolean flag for causal masking (must match forward pass).
 * @param sm_scale Scaling factor for dot products (must match forward pass).
 */
template <int T_r, int T_c, int HEAD_DIM>
__global__ void flash_attention_backward_kernel(
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> Q_acc,    
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> K_acc,    
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> V_acc,    
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> O_acc,    
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> dO_acc,   
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> L_acc,    
    
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> dQ_acc,   
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> dK_acc,   
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> dV_acc,   
    
    bool is_causal,
    float sm_scale
) {
    // --- Block and Dimension Setup ---
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int q_block_start_row = blockIdx.x * T_r; // Each block processes T_r rows of Q.

    const int actual_head_dim = Q_acc.size(3);
    const int seq_len_q = Q_acc.size(2);
    const int seq_len_kv = K_acc.size(2);

    if (actual_head_dim > HEAD_DIM) return; // Prevent running if compiled for smaller HEAD_DIM.

    // --- Shared Memory Allocation ---
    // Tiles for Q, K, V, dO, recomputed S_ij, recomputed P_ij, and D_i values.
    __shared__ float q_tile_smem[T_r][HEAD_DIM];    // Stores the block of Q vectors.
    __shared__ float k_tile_smem[T_c][HEAD_DIM];    // Stores a tile of K vectors.
    __shared__ float v_tile_smem[T_c][HEAD_DIM];    // Stores a tile of V vectors.
    __shared__ float s_ij_tile_smem[T_r][T_c];      // Stores recomputed S_ij scores.
    __shared__ float p_ij_tile_smem[T_r][T_c];      // Stores recomputed P_ij probabilities.
    __shared__ float do_tile_smem[T_r][HEAD_DIM];   // Stores the block of dO vectors.
    __shared__ float d_i_shared_mem[T_r];           // Stores D_i = sum_h(O_ih * dO_ih) for each Q_i in the block.


    // --- Load Q-block and dO-block into Shared Memory ---
    // Threads cooperate to load T_r rows of Q and dO corresponding to the current block.
    // threadIdx.y iterates over rows within the T_r block.
    // threadIdx.x iterates over elements of the head dimension for each vector.
    for (int q_row = threadIdx.y; q_row < T_r; q_row += blockDim.y) {
        int q_abs_idx = q_block_start_row + q_row; // Absolute index in Q sequence.
        if (q_abs_idx < seq_len_q) { // Boundary check.
            for (int h_col = threadIdx.x; h_col < actual_head_dim; h_col += blockDim.x) {
                q_tile_smem[q_row][h_col] = Q_acc[batch_idx][head_idx][q_abs_idx][h_col];
                do_tile_smem[q_row][h_col] = dO_acc[batch_idx][head_idx][q_abs_idx][h_col];
            }
        } else { // Pad with zeros if past actual Q sequence length.
             for (int h_col = threadIdx.x; h_col < actual_head_dim; h_col += blockDim.x) {
                q_tile_smem[q_row][h_col] = 0.0f;
                do_tile_smem[q_row][h_col] = 0.0f;
            }
        }
    }
    // Note: dq_block_acc initialization and D_i calculation will provide necessary __syncthreads before this data is used broadly.

    // --- Initialize dQ Accumulator for the Block ---
    // Stored in registers (or potentially shared memory if T_r * HEAD_DIM is large).
    // Each Q_i in the block gets its own dQ_i accumulator.
    float dq_block_acc[T_r][HEAD_DIM]; 
    for (int q_row = 0; q_row < T_r; ++q_row) { 
        for (int h_col = 0; h_col < actual_head_dim; ++h_col) {
            dq_block_acc[q_row][h_col] = 0.0f;
        }
    }

    // --- Pre-calculate D_i = sum_h (O_ih * dO_ih) for each Q_i in the Block ---
    // D_i is a scalar value for each query Q_i. It's part of the dS_ij calculation.
    // Stored in d_i_shared_mem for access by all threads in the block.
    // threadIdx.y iterates over Q rows in the block.
    // threadIdx.x=0 computes D_i for its assigned q_row_offset.
    // More optimized: parallel reduction over actual_head_dim by threadIdx.x.
    for (int q_row_offset = threadIdx.y; q_row_offset < T_r; q_row_offset += blockDim.y) {
        int q_abs_idx = q_block_start_row + q_row_offset;
        if (q_abs_idx < seq_len_q) { // Boundary check.
            if (threadIdx.x == 0) { // Thread 0 of each y-group computes D_i for its q_row.
                float temp_D_i = 0.0f;
                for (int h_col = 0; h_col < actual_head_dim; ++h_col) {
                    temp_D_i += O_acc[batch_idx][head_idx][q_abs_idx][h_col] * do_tile_smem[q_row_offset][h_col];
                }
                d_i_shared_mem[q_row_offset] = temp_D_i;
            }
        } else { // For padded Q rows, D_i is 0.
            if (threadIdx.x == 0) {
                d_i_shared_mem[q_row_offset] = 0.0f;
            }
        }
    }
    // Synchronize to ensure Q, dO tiles and D_i values are fully written and visible to all threads.
    __syncthreads(); 

    // --- Outer Loop: Iterate over Key/Value Blocks (Tiles of K and V) ---
    // This loop processes K and V in tiles of size T_c.
    for (int kv_block_col_start = 0; kv_block_col_start < seq_len_kv; kv_block_col_start += T_c) {
        
        // --- Load K_j_tile and V_j_tile into Shared Memory ---
        // Threads cooperate to load T_c vectors of K and V.
        for (int kv_row = threadIdx.y; kv_row < T_c; kv_row += blockDim.y) { // Iterate K/V tile rows.
            int k_abs_idx = kv_block_col_start + kv_row; // Absolute index in K/V sequence.
            if (k_abs_idx < seq_len_kv) { // Boundary check.
                for (int h_col = threadIdx.x; h_col < actual_head_dim; h_col += blockDim.x) { // Iterate head dim.
                    k_tile_smem[kv_row][h_col] = K_acc[batch_idx][head_idx][k_abs_idx][h_col];
                    v_tile_smem[kv_row][h_col] = V_acc[batch_idx][head_idx][k_abs_idx][h_col];
                }
            } else { // Pad with zeros.
                for (int h_col = threadIdx.x; h_col < actual_head_dim; h_col += blockDim.x) {
                    k_tile_smem[kv_row][h_col] = 0.0f;
                    v_tile_smem[kv_row][h_col] = 0.0f;
                }
            }
        }
        __syncthreads(); // Ensure K_tile and V_tile are fully loaded.

        // --- Recompute S_ij_tile and P_ij_tile for current Q-block and K/V-tile ---
        // S_ij = (Q_i @ K_j.T) * sm_scale. Q_i from q_tile_smem, K_j from k_tile_smem.
        // Result (T_r x T_c) stored in s_ij_tile_smem.
        // Threads (threadIdx.y for q_row, threadIdx.x for k_col) compute elements of S_ij_tile.
        for (int q_row = threadIdx.y; q_row < T_r; q_row += blockDim.y) { 
            for (int k_col = threadIdx.x; k_col < T_c; k_col += blockDim.x) { 
                int q_abs_idx = q_block_start_row + q_row;
                if (q_abs_idx >= seq_len_q) { // Skip padded Q rows.
                    s_ij_tile_smem[q_row][k_col] = -std::numeric_limits<float>::infinity();
                    continue;
                }

                float sum_qk = 0.0f; // Dot product for S_ij.
                for (int d = 0; d < actual_head_dim; ++d) {
                    sum_qk += q_tile_smem[q_row][d] * k_tile_smem[k_col][d];
                }
                float s_val = sum_qk * sm_scale;

                if (is_causal) { // Apply causal masking.
                    int k_abs_idx = kv_block_col_start + k_col; 
                    if (k_abs_idx > q_abs_idx) {
                        s_val = -std::numeric_limits<float>::infinity();
                    }
                }
                s_ij_tile_smem[q_row][k_col] = s_val;
            }
        }
        __syncthreads(); // Ensure S_ij_tile_smem is fully computed.

        // P_ij = exp(S_ij - L_i), where L_i is logsumexp from forward pass.
        // Result (T_r x T_c) stored in p_ij_tile_smem.
        for (int q_row = threadIdx.y; q_row < T_r; q_row += blockDim.y) {
            int q_abs_idx = q_block_start_row + q_row;
            if (q_abs_idx >= seq_len_q) { // Skip padded Q rows.
                 for (int k_col = threadIdx.x; k_col < T_c; k_col += blockDim.x) {
                    p_ij_tile_smem[q_row][k_col] = 0.0f; // P_ij is 0 for padded Q.
                 }
                 continue;
            }
            float l_i_val = L_acc[batch_idx][head_idx][q_abs_idx]; // Fetch L_i for current Q_i.
            for (int k_col = threadIdx.x; k_col < T_c; k_col += blockDim.x) {
                 if (s_ij_tile_smem[q_row][k_col] > -std::numeric_limits<float>::infinity()) {
                    p_ij_tile_smem[q_row][k_col] = expf(s_ij_tile_smem[q_row][k_col] - l_i_val);
                 } else { // If S_ij was -inf (e.g. due to masking), P_ij is 0.
                    p_ij_tile_smem[q_row][k_col] = 0.0f;
                 }
            }
        }
        __syncthreads(); // Ensure P_ij_tile_smem is fully computed.

        // --- Gradient Calculation using Recomputed P_ij ---
        // 1. Compute dV_j += P_ij.T @ dO_i (accumulated globally using atomicAdd).
        //    dV_j_tile element (k_row, h_col) = sum_{q_row in T_r} (P_ij[q_row, k_row] * dO_i[q_row, h_col])
        for (int k_row = threadIdx.y; k_row < T_c; k_row += blockDim.y) { // Iterate K/V rows in tile.
            int k_abs_idx = kv_block_col_start + k_row;
            if (k_abs_idx >= seq_len_kv) continue; // Skip padded K/V rows.

            for (int h_col = threadIdx.x; h_col < actual_head_dim; h_col += blockDim.x) { // Iterate head_dim.
                float sum_dv_contrib = 0.0f; // Contribution to dV[k_abs_idx, h_col] from this Q-block.
                for (int q_row = 0; q_row < T_r; ++q_row) { // Sum over Q rows in the current block.
                    int q_abs_idx = q_block_start_row + q_row;
                    if (q_abs_idx < seq_len_q) { // Only include valid Q rows.
                         sum_dv_contrib += p_ij_tile_smem[q_row][k_row] * do_tile_smem[q_row][h_col];
                    }
                }
                atomicAdd(&dV_acc[batch_idx][head_idx][k_abs_idx][h_col], sum_dv_contrib);
            }
        }
        // A __syncthreads() here might be needed if dS computation immediately followed AND used V_acc directly,
        // but dS uses v_tile_smem, which is stable. dV updates are atomic to global.
        
        // 2. Compute dS_ij = (P_ij * ( (dO_i @ V_j.T) - D_i )) / sm_scale
        //    First, term1 = (dO_i @ V_j.T). This is a (T_r x T_c) matrix.
        //    Store it in dp_term_smem (reusing shared memory).
        __shared__ float dp_term_smem[T_r][T_c]; // Renamed from s_ij_tile_smem if used for dS later.
                                                // If s_ij_tile_smem is reused for dS_ij, this needs a new name or careful ordering.
                                                // Let's assume s_ij_tile_smem is reused for dS_ij. This intermediate needs its own SMEM.
        for (int q_row = threadIdx.y; q_row < T_r; q_row += blockDim.y) { // Iterate Q rows.
            for (int k_col = threadIdx.x; k_col < T_c; k_col += blockDim.x) { // Iterate K cols.
                float sum_dov = 0.0f; // Dot product for (dO_i @ V_j.T)_qk element.
                int q_abs_idx = q_block_start_row + q_row;
                if (q_abs_idx < seq_len_q) { // Check if Q row is valid for dO access.
                                             // K col (v_tile_smem[k_col]) is valid if k_col < T_c and not padded HBM.
                    for (int d = 0; d < actual_head_dim; ++d) {
                        sum_dov += do_tile_smem[q_row][d] * v_tile_smem[k_col][d];
                    }
                }
                // If q_abs_idx >= seq_len_q, sum_dov remains 0.0, which is correct.
                dp_term_smem[q_row][k_col] = sum_dov; 
            }
        }
        __syncthreads(); // Ensure dp_term_smem is fully computed.

        // Now, dS_ij = (P_ij * (dp_term_smem - D_i)) / sm_scale. Reuse s_ij_tile_smem for dS_ij.
        for (int q_row = threadIdx.y; q_row < T_r; q_row += blockDim.y) {
            int q_abs_idx = q_block_start_row + q_row;
            if (q_abs_idx >= seq_len_q) continue; // Skip padded Q rows.
            
            float current_D_i_val = d_i_shared_mem[q_row]; // D_i for this Q_i from shared memory.

            for (int k_col = threadIdx.x; k_col < T_c; k_col += blockDim.x) {
                // If P_ij is 0 (e.g. from masking or underflow), dS_ij is 0.
                // This check avoids NaN from 0 * inf if dp_term or D_i are inf.
                if (p_ij_tile_smem[q_row][k_col] == 0.0f) { 
                    s_ij_tile_smem[q_row][k_col] = 0.0f; // Store dS_ij here.
                    continue;
                }
                s_ij_tile_smem[q_row][k_col] = (p_ij_tile_smem[q_row][k_col] *
                                               (dp_term_smem[q_row][k_col] - current_D_i_val)) / sm_scale;
            }
        }
        __syncthreads(); // Ensure dS_ij_tile (now in s_ij_tile_smem) is fully computed.

        // 3. Compute dQ_i += dS_ij @ K_j. Accumulate in local dq_block_acc.
        //    dQ_i is (T_r x HEAD_DIM). dS_ij is (T_r x T_c). K_j is (T_c x HEAD_DIM).
        //    Each thread (threadIdx.y for q_row, threadIdx.x for h_col) computes elements of dQ.
        for (int q_row = threadIdx.y; q_row < T_r; q_row += blockDim.y) { 
            if (q_block_start_row + q_row >= seq_len_q) continue; // Skip padded Q rows.
            for (int h_col = threadIdx.x; h_col < actual_head_dim; h_col += blockDim.x) { 
                float sum_dsk_contrib = 0.0f; // Contribution to dq_block_acc[q_row][h_col].
                for (int k_col = 0; k_col < T_c; ++k_col) { // Sum over K columns in the tile.
                    sum_dsk_contrib += s_ij_tile_smem[q_row][k_col] * k_tile_smem[k_col][h_col];
                }
                dq_block_acc[q_row][h_col] += sum_dsk_contrib;
            }
        }

        // 4. Compute dK_j += dS_ij.T @ Q_i. Accumulate globally using atomicAdd.
        //    dK_j_tile element (k_row, h_col) = sum_{q_row in T_r} (dS_ij[q_row, k_row] * Q_i[q_row, h_col])
        for (int k_row = threadIdx.y; k_row < T_c; k_row += blockDim.y) { // Iterate K tile rows.
            int k_abs_idx = kv_block_col_start + k_row;
            if (k_abs_idx >= seq_len_kv) continue; // Skip padded K rows.

            for (int h_col = threadIdx.x; h_col < actual_head_dim; h_col += blockDim.x) { // Iterate head_dim.
                float sum_dsq_contrib = 0.0f; // Contribution to dK[k_abs_idx, h_col] from this Q-block.
                for (int q_row = 0; q_row < T_r; ++q_row) { // Sum over Q rows in the block.
                     int q_abs_idx = q_block_start_row + q_row;
                     if (q_abs_idx < seq_len_q) { // Only include valid Q rows.
                        sum_dsq_contrib += s_ij_tile_smem[q_row][k_row] * q_tile_smem[q_row][h_col];
                     }
                }
                atomicAdd(&dK_acc[batch_idx][head_idx][k_abs_idx][h_col], sum_dsq_contrib);
            }
        }
        // Synchronize before next K/V block iteration to ensure shared memory (k_tile, v_tile, etc.)
        // is not overwritten while still in use by gradient calculations dependent on dS_ij.
        __syncthreads(); 
    } // End of loop over K/V blocks (kv_block_col_start)

    // --- Write Accumulated dQ_block_acc to Global Memory dQ_acc ---
    // Each thread writes its assigned portion of the dQ_block_acc.
    for (int q_row = threadIdx.y; q_row < T_r; q_row += blockDim.y) {
        int q_abs_idx = q_block_start_row + q_row;
        if (q_abs_idx < seq_len_q) { // Boundary check.
            for (int h_col = threadIdx.x; h_col < actual_head_dim; h_col += blockDim.x) {
                dQ_acc[batch_idx][head_idx][q_abs_idx][h_col] = dq_block_acc[q_row][h_col];
            }
        }
    }
}


/**
 * @brief C++ dispatcher for the FlashAttention backward CUDA kernel.
 * 
 * This function handles tensor validation, determines kernel launch parameters,
 * and calls the appropriate templated version of `flash_attention_backward_kernel`
 * based on the head dimension. It assumes dK and dV are zero-initialized by the caller
 * as they are accumulated into using atomicAdd.
 * 
 * @param Q Query tensor from forward pass.
 * @param K Key tensor from forward pass.
 * @param V Value tensor from forward pass.
 * @param O Output tensor from forward pass.
 * @param dO Gradient of the output tensor O.
 * @param L Logsumexp tensor from forward pass.
 * @param dQ Output gradient tensor for Q.
 * @param dK Output gradient tensor for K.
 * @param dV Output gradient tensor for V.
 * @param is_causal Boolean flag for causal masking.
 * @param sm_scale Scaling factor for attention scores.
 */
void flash_attention_backward_cuda(
    const torch::Tensor& Q, const torch::Tensor& K, const torch::Tensor& V,
    const torch::Tensor& O, const torch::Tensor& dO, const torch::Tensor& L,
    torch::Tensor& dQ, torch::Tensor& dK, torch::Tensor& dV,
    bool is_causal, float sm_scale
) {
    // --- Input Tensor Validation ---
    // Device checks
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");
    TORCH_CHECK(O.is_cuda(), "O must be a CUDA tensor");
    TORCH_CHECK(dO.is_cuda(), "dO must be a CUDA tensor");
    TORCH_CHECK(L.is_cuda(), "L must be a CUDA tensor");
    TORCH_CHECK(dQ.is_cuda(), "dQ must be a CUDA tensor");
    TORCH_CHECK(dK.is_cuda(), "dK must be a CUDA tensor");
    TORCH_CHECK(dV.is_cuda(), "dV must be a CUDA tensor");
    
    // Device consistency checks
    auto current_device = Q.device();
    TORCH_CHECK(K.device() == current_device, "K not on same device as Q");
    TORCH_CHECK(V.device() == current_device, "V not on same device as Q");
    TORCH_CHECK(O.device() == current_device, "O not on same device as Q");
    TORCH_CHECK(dO.device() == current_device, "dO not on same device as Q");
    TORCH_CHECK(L.device() == current_device, "L not on same device as Q");
    TORCH_CHECK(dQ.device() == current_device, "dQ not on same device as Q");
    TORCH_CHECK(dK.device() == current_device, "dK not on same device as Q");
    TORCH_CHECK(dV.device() == current_device, "dV not on same device as Q");

    // Dimension checks
    TORCH_CHECK(Q.dim() == 4, "Q must be 4D");
    TORCH_CHECK(K.dim() == 4, "K must be 4D");
    TORCH_CHECK(V.dim() == 4, "V must be 4D");
    TORCH_CHECK(O.dim() == 4, "O must be 4D");
    TORCH_CHECK(dO.dim() == 4, "dO must be 4D");
    TORCH_CHECK(L.dim() == 3, "L must be 3D");
    TORCH_CHECK(dQ.dim() == 4, "dQ must be 4D");
    TORCH_CHECK(dK.dim() == 4, "dK must be 4D");
    TORCH_CHECK(dV.dim() == 4, "dV must be 4D");

    // Datatype checks (kernel currently supports Float32)
    TORCH_CHECK(Q.scalar_type() == torch::kFloat32, "Q must be Float32");
    TORCH_CHECK(K.scalar_type() == torch::kFloat32, "K must be Float32");
    TORCH_CHECK(V.scalar_type() == torch::kFloat32, "V must be Float32");
    TORCH_CHECK(O.scalar_type() == torch::kFloat32, "O must be Float32");
    TORCH_CHECK(dO.scalar_type() == torch::kFloat32, "dO must be Float32");
    TORCH_CHECK(L.scalar_type() == torch::kFloat32, "L must be Float32"); 
    TORCH_CHECK(dQ.scalar_type() == torch::kFloat32, "dQ must be Float32");
    TORCH_CHECK(dK.scalar_type() == torch::kFloat32, "dK must be Float32");
    TORCH_CHECK(dV.scalar_type() == torch::kFloat32, "dV must be Float32");

    // --- Shape Compatibility and Parameter Extraction ---
    const int batch_size = Q.size(0);
    const int num_heads = Q.size(1);
    const int seq_len_q = Q.size(2);
    const int head_dim = Q.size(3);
    const int seq_len_kv = K.size(2);

    TORCH_CHECK(K.size(0) == batch_size && K.size(1) == num_heads && K.size(3) == head_dim, "K shape mismatch");
    TORCH_CHECK(V.size(0) == batch_size && V.size(1) == num_heads && V.size(2) == seq_len_kv && V.size(3) == head_dim, "V shape mismatch");
    TORCH_CHECK(O.size(0) == batch_size && O.size(1) == num_heads && O.size(2) == seq_len_q && O.size(3) == head_dim, "O shape mismatch");
    TORCH_CHECK(dO.size(0) == batch_size && dO.size(1) == num_heads && dO.size(2) == seq_len_q && dO.size(3) == head_dim, "dO shape mismatch");
    TORCH_CHECK(L.size(0) == batch_size && L.size(1) == num_heads && L.size(2) == seq_len_q, "L shape mismatch");
    TORCH_CHECK(dQ.size(0) == batch_size && dQ.size(1) == num_heads && dQ.size(2) == seq_len_q && dQ.size(3) == head_dim, "dQ shape mismatch");
    TORCH_CHECK(dK.size(0) == batch_size && dK.size(1) == num_heads && dK.size(2) == seq_len_kv && dK.size(3) == head_dim, "dK shape mismatch");
    TORCH_CHECK(dV.size(0) == batch_size && dV.size(1) == num_heads && dV.size(2) == seq_len_kv && dV.size(3) == head_dim, "dV shape mismatch");
    
    TORCH_CHECK(head_dim <= HEAD_DIM_MAX_BWD, "Head dimension exceeds compiled maximum for BWD kernel.");

    // Note on gradient initialization:
    // dK and dV use atomicAdd in the kernel, so they MUST be zero-initialized by the caller (e.g., PyTorch's autograd engine).
    // dQ is written directly in this kernel version, so it doesn't strictly need pre-zeroing if this is its only computation.
    // If dQ were also accumulated atomically, it would also need zero-initialization.

    // --- Kernel Launch Configuration ---
    dim3 threads_per_block;
    // Heuristic for thread block dimensions. These should be tuned for optimal performance.
    if (head_dim <= 32) threads_per_block = dim3(32, 8, 1); // Example: 256 threads
    else if (head_dim <= 64) threads_per_block = dim3(64, 4, 1); // Example: 256 threads
    else threads_per_block = dim3(128, 2, 1); // Example: 256 threads
    // blockDim.x typically parallelizes over head_dim or columns of a tile (T_c).
    // blockDim.y typically parallelizes over rows of a tile (T_r or T_c rows).

    dim3 num_blocks((seq_len_q + T_r_BWD_DEFAULT - 1) / T_r_BWD_DEFAULT, num_heads, batch_size);

    // Get packed tensor accessors for efficient element access in CUDA.
    auto Q_acc_packed = Q.packed_accessor32<float,4,torch::RestrictPtrTraits>();
    auto K_acc_packed = K.packed_accessor32<float,4,torch::RestrictPtrTraits>();
    auto V_acc_packed = V.packed_accessor32<float,4,torch::RestrictPtrTraits>();
    auto O_acc_packed = O.packed_accessor32<float,4,torch::RestrictPtrTraits>();
    auto dO_acc_packed = dO.packed_accessor32<float,4,torch::RestrictPtrTraits>();
    auto L_acc_packed = L.packed_accessor32<float,3,torch::RestrictPtrTraits>();
    auto dQ_acc_packed = dQ.packed_accessor32<float,4,torch::RestrictPtrTraits>();
    auto dK_acc_packed = dK.packed_accessor32<float,4,torch::RestrictPtrTraits>();
    auto dV_acc_packed = dV.packed_accessor32<float,4,torch::RestrictPtrTraits>();

    // --- Dispatch to Templated Kernel based on Head Dimension ---
    if (head_dim <= 32) {
        flash_attention_backward_kernel<T_r_BWD_DEFAULT, T_c_BWD_DEFAULT, 32><<<num_blocks, threads_per_block>>>(
            Q_acc_packed, K_acc_packed, V_acc_packed, O_acc_packed, dO_acc_packed, L_acc_packed, 
            dQ_acc_packed, dK_acc_packed, dV_acc_packed, is_causal, sm_scale);
    } else if (head_dim <= 64) {
        flash_attention_backward_kernel<T_r_BWD_DEFAULT, T_c_BWD_DEFAULT, 64><<<num_blocks, threads_per_block>>>(
            Q_acc_packed, K_acc_packed, V_acc_packed, O_acc_packed, dO_acc_packed, L_acc_packed, 
            dQ_acc_packed, dK_acc_packed, dV_acc_packed, is_causal, sm_scale);
    } else if (head_dim <= HEAD_DIM_MAX_BWD) { // Max head dim supported by this build
        flash_attention_backward_kernel<T_r_BWD_DEFAULT, T_c_BWD_DEFAULT, HEAD_DIM_MAX_BWD><<<num_blocks, threads_per_block>>>(
            Q_acc_packed, K_acc_packed, V_acc_packed, O_acc_packed, dO_acc_packed, L_acc_packed, 
            dQ_acc_packed, dK_acc_packed, dV_acc_packed, is_causal, sm_scale);
    } else {
        // This case should be caught by the TORCH_CHECK for head_dim vs HEAD_DIM_MAX_BWD.
        AT_ERROR("Unsupported head_dimension for BWD kernel: ", head_dim, ". Max compiled is ", HEAD_DIM_MAX_BWD);
    }
    
    // Check for any CUDA errors during kernel launch.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed in flash_attention_backward_cuda: ", cudaGetErrorString(err));
    }
}
// Note on Pybind11 module definition:
// This file only contains the kernel and its C++ dispatcher.
// The PYBIND11_MODULE block that exposes functions to Python is expected to be
// in flash_attn_fwd.cu (or a separate central .cpp binding file) to ensure a single
// Python module ('flash_attn_cuda_lib') is defined for all CUDA functions.
// The `flash_attention_backward_cuda` function is forward-declared in `flash_attn_fwd.cu`
// and included in its PYBIND11_MODULE definition.I've added detailed comments and docstrings to `flash_attn/cuda/flash_attn_bwd.cu`, explaining the logic, shared memory usage, synchronization points, and the roles of different code sections in the backward pass. I also refined some existing comments for clarity and added more `TORCH_CHECK`s in the C++ dispatcher.
