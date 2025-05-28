#include <cuda_runtime.h>
#include <float.h> // For FLT_MAX
// #include <stdio.h> // For printf in kernel (debug) - remove for final

// Tile dimensions - These should be configured based on typical head_dim and desired occupancy/performance
#define Q_ROWS_PER_BLOCK 16      // Number of Q sequences processed by a block (M_TILE)
#define KV_SEQ_TILE_LEN 64       // Tile length for K/V sequence dimension (N_TILE)
#define MAX_HEAD_DIM 64          // Max head dimension this kernel is compiled for.
                                 // Shared memory for q, k, v will be based on this.
                                 // The actual head_dim passed to kernel must be <= MAX_HEAD_DIM.
#define WARP_SIZE 32

// Helper for warp-level reduction (sum)
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val; // Result in lane 0
}

// Helper for warp-level reduction (max)
__inline__ __device__ float warpReduceMax(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val; // Result in lane 0
}


__global__ void fused_attention_forward_kernel(
    const float* q_ptr,
    const float* k_ptr,
    const float* v_ptr,
    float* out_ptr,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim, // Actual head_dim for this call
    const float sm_scale,
    const bool is_causal) {

    // --- Thread and Block Indexing ---
    int batch_idx = blockIdx.y / num_heads;
    int head_idx = blockIdx.y % num_heads;

    int q_row_in_block = threadIdx.x; 
    int tid_in_warp = threadIdx.y;   

    int global_q_row_idx = blockIdx.x * Q_ROWS_PER_BLOCK + q_row_in_block;

    // Shared memory
    __shared__ float q_sdata[Q_ROWS_PER_BLOCK][MAX_HEAD_DIM];
    __shared__ float k_sdata[KV_SEQ_TILE_LEN][MAX_HEAD_DIM];
    __shared__ float v_sdata[KV_SEQ_TILE_LEN][MAX_HEAD_DIM];
    __shared__ float s_sdata[Q_ROWS_PER_BLOCK][KV_SEQ_TILE_LEN]; // Stores S_ij then P_ij


    // --- Load Q tile for the block ---
    if (global_q_row_idx < seq_len) { 
        for (int d_col = tid_in_warp; d_col < head_dim; d_col += WARP_SIZE) {
            q_sdata[q_row_in_block][d_col] = 
                q_ptr[batch_idx * num_heads * seq_len * head_dim +
                      head_idx * seq_len * head_dim +
                      global_q_row_idx * head_dim +
                      d_col];
        }
    }

    const int head_dim_elements_per_thread = (head_dim + WARP_SIZE - 1) / WARP_SIZE;
    float o_acc[(MAX_HEAD_DIM + WARP_SIZE - 1) / WARP_SIZE]; 

    for(int i=0; i < head_dim_elements_per_thread; ++i) {
        int dim_idx_to_init = tid_in_warp * head_dim_elements_per_thread + i;
        if (dim_idx_to_init < head_dim) { 
             o_acc[i] = 0.0f; 
        } else if (i < (MAX_HEAD_DIM + WARP_SIZE -1) / WARP_SIZE) {
             o_acc[i] = 0.0f;
        }
    }

    for (int kv_tile_start = 0; kv_tile_start < seq_len; kv_tile_start += KV_SEQ_TILE_LEN) {
        __syncthreads(); 

        int num_elements_in_kv_tile_slice = KV_SEQ_TILE_LEN * head_dim;
        int threads_in_block = Q_ROWS_PER_BLOCK * WARP_SIZE;
        int elements_per_thread_for_kv_load = (num_elements_in_kv_tile_slice + threads_in_block - 1) / threads_in_block;
        int flat_thread_idx_in_block = threadIdx.x * WARP_SIZE + threadIdx.y;

        for (int i = 0; i < elements_per_thread_for_kv_load; ++i) {
            int element_idx = flat_thread_idx_in_block * elements_per_thread_for_kv_load + i;
            if (element_idx < num_elements_in_kv_tile_slice) {
                int r = element_idx / head_dim; 
                int c = element_idx % head_dim; 

                if (kv_tile_start + r < seq_len) { 
                    k_sdata[r][c] = k_ptr[batch_idx * num_heads * seq_len * head_dim +
                                          head_idx * seq_len * head_dim +
                                          (kv_tile_start + r) * head_dim +
                                          c];
                    v_sdata[r][c] = v_ptr[batch_idx * num_heads * seq_len * head_dim +
                                          head_idx * seq_len * head_dim +
                                          (kv_tile_start + r) * head_dim +
                                          c];
                } else { 
                    k_sdata[r][c] = 0.0f; 
                    v_sdata[r][c] = 0.0f;
                }
            }
        }
        __syncthreads(); 

        if (global_q_row_idx < seq_len) { 
            for (int k_idx_in_tile = tid_in_warp; k_idx_in_tile < KV_SEQ_TILE_LEN; k_idx_in_tile += WARP_SIZE) {
                float score = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    score += q_sdata[q_row_in_block][d] * k_sdata[k_idx_in_tile][d];
                }
                score *= sm_scale;

                bool is_padded_k = (kv_tile_start + k_idx_in_tile) >= seq_len;
                if (is_causal && (kv_tile_start + k_idx_in_tile) > global_q_row_idx) {
                    score = -FLT_MAX; 
                }
                if (is_padded_k) { 
                    score = -FLT_MAX;
                }
                s_sdata[q_row_in_block][k_idx_in_tile] = score;
            }
        }
        __syncthreads(); 

        if (global_q_row_idx < seq_len) { 
            float max_val_thread = -FLT_MAX;
            for (int k_col = tid_in_warp; k_col < KV_SEQ_TILE_LEN; k_col += WARP_SIZE) {
                 max_val_thread = max(max_val_thread, s_sdata[q_row_in_block][k_col]);
            }
            float max_val_warp = warpReduceMax(max_val_thread); 
            max_val_warp = __shfl_sync(0xFFFFFFFF, max_val_warp, 0); 

            float sum_exp_thread = 0.0f;
            for (int k_col = tid_in_warp; k_col < KV_SEQ_TILE_LEN; k_col += WARP_SIZE) {
                float val = expf(s_sdata[q_row_in_block][k_col] - max_val_warp);
                if (s_sdata[q_row_in_block][k_col] <= -FLT_MAX) { 
                     val = 0.0f;
                }
                s_sdata[q_row_in_block][k_col] = val; 
                sum_exp_thread += val;
            }
            float sum_exp_warp = warpReduceSum(sum_exp_thread); 
            sum_exp_warp = __shfl_sync(0xFFFFFFFF, sum_exp_warp, 0); 
            
            if (sum_exp_warp == 0.0f) sum_exp_warp = 1e-6f; 

            for (int k_col = tid_in_warp; k_col < KV_SEQ_TILE_LEN; k_col += WARP_SIZE) {
                if (sum_exp_warp != 0.0f) { 
                    s_sdata[q_row_in_block][k_col] /= sum_exp_warp;
                } else { 
                    s_sdata[q_row_in_block][k_col] = 0.0f;
                }
            }
        }
        __syncthreads(); 

        if (global_q_row_idx < seq_len) { 
            for (int d_loop_idx = 0; d_loop_idx < head_dim_elements_per_thread; ++d_loop_idx) {
                int current_d_component = tid_in_warp * head_dim_elements_per_thread + d_loop_idx;
                                
                if (current_d_component < head_dim) { 
                    float pv_sum_for_d_component = 0.0f;
                    for (int k_col = 0; k_col < KV_SEQ_TILE_LEN; ++k_col) {
                        pv_sum_for_d_component += s_sdata[q_row_in_block][k_col] * v_sdata[k_col][current_d_component];
                    }
                    o_acc[d_loop_idx] += pv_sum_for_d_component;
                }
            }
        }
    } 

    if (global_q_row_idx < seq_len) { 
        for (int d_loop_idx = 0; d_loop_idx < head_dim_elements_per_thread; ++d_loop_idx) {
            int current_d_component = tid_in_warp * head_dim_elements_per_thread + d_loop_idx;
            if (current_d_component < head_dim) {
                 out_ptr[batch_idx * num_heads * seq_len * head_dim +
                         head_idx * seq_len * head_dim +
                         global_q_row_idx * head_dim +
                         current_d_component] = o_acc[d_loop_idx];
            }
        }
    }
}


__global__ void fused_attention_backward_kernel(
    const float* q_ptr,    // Q (forward)
    const float* k_ptr,    // K (forward)
    const float* v_ptr,    // V (forward)
    // const float* o_ptr, // O (forward) - not directly used
    const float* do_ptr,   // dL/dO
    float* dq_ptr,   // dL/dQ
    float* dk_ptr,   // dL/dK
    float* dv_ptr,   // dL/dV
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim,
    const float sm_scale,
    const bool is_causal) {

    // --- Thread and Block Indexing (same as forward) ---
    int batch_idx = blockIdx.y / num_heads;
    int head_idx = blockIdx.y % num_heads;
    int q_row_in_block = threadIdx.x; 
    int tid_in_warp = threadIdx.y;   
    int global_q_row_idx = blockIdx.x * Q_ROWS_PER_BLOCK + q_row_in_block;

    // --- Shared Memory Declarations ---
    __shared__ float q_sdata[Q_ROWS_PER_BLOCK][MAX_HEAD_DIM];
    __shared__ float k_sdata[KV_SEQ_TILE_LEN][MAX_HEAD_DIM];
    __shared__ float v_sdata[KV_SEQ_TILE_LEN][MAX_HEAD_DIM];
    __shared__ float do_sdata[Q_ROWS_PER_BLOCK][MAX_HEAD_DIM]; 

    __shared__ float s_sdata[Q_ROWS_PER_BLOCK][KV_SEQ_TILE_LEN]; 
    __shared__ float p_sdata[Q_ROWS_PER_BLOCK][KV_SEQ_TILE_LEN]; 

    __shared__ float dp_sdata[Q_ROWS_PER_BLOCK][KV_SEQ_TILE_LEN]; 
    __shared__ float D_rowsums_sdata[Q_ROWS_PER_BLOCK]; 

    __shared__ float dk_tile_acc_sdata[KV_SEQ_TILE_LEN][MAX_HEAD_DIM];
    __shared__ float dv_tile_acc_sdata[KV_SEQ_TILE_LEN][MAX_HEAD_DIM];

    // --- dQ Accumulator Registers ---
    const int head_dim_elements_per_thread = (head_dim + WARP_SIZE - 1) / WARP_SIZE;
    float dQ_acc[(MAX_HEAD_DIM + WARP_SIZE - 1) / WARP_SIZE];
    for (int i = 0; i < head_dim_elements_per_thread; ++i) {
        dQ_acc[i] = 0.0f;
    }

    // --- Load dO tile for the block ---
    if (global_q_row_idx < seq_len) {
        for (int d_col = tid_in_warp; d_col < head_dim; d_col += WARP_SIZE) {
            do_sdata[q_row_in_block][d_col] = 
                do_ptr[batch_idx * num_heads * seq_len * head_dim +
                       head_idx * seq_len * head_dim +
                       global_q_row_idx * head_dim +
                       d_col];
        }
    }
    
    // --- Load Q tile for the block (needed for dK) ---
    if (global_q_row_idx < seq_len) {
        for (int d_col = tid_in_warp; d_col < head_dim; d_col += WARP_SIZE) {
            q_sdata[q_row_in_block][d_col] = 
                q_ptr[batch_idx * num_heads * seq_len * head_dim +
                      head_idx * seq_len * head_dim +
                      global_q_row_idx * head_dim +
                      d_col];
        }
    }
    // __syncthreads(); // Ensure q_sdata and do_sdata are loaded. Done at start of loop.

    // --- Outer loop over Key/Value sequence length in tiles ---
    for (int kv_tile_start = 0; kv_tile_start < seq_len; kv_tile_start += KV_SEQ_TILE_LEN) {
        __syncthreads(); 

        // --- Initialize dk_tile_acc_sdata and dv_tile_acc_sdata to 0 for this K/V tile ---
        int num_elements_tile_kv_grad = KV_SEQ_TILE_LEN * head_dim; // Using actual head_dim
        int threads_in_block_count = Q_ROWS_PER_BLOCK * WARP_SIZE;
        int elems_per_thread_tile_init = (num_elements_tile_kv_grad + threads_in_block_count - 1) / threads_in_block_count;
        int flat_thread_idx = threadIdx.x * WARP_SIZE + threadIdx.y;

        for (int i = 0; i < elems_per_thread_tile_init; ++i) {
            int element_idx_in_tile = flat_thread_idx * elems_per_thread_tile_init + i;
            if (element_idx_in_tile < num_elements_tile_kv_grad) {
                int r = element_idx_in_tile / head_dim; 
                int c = element_idx_in_tile % head_dim;
                // Bounds check against actual SMEM allocation not strictly needed if head_dim <= MAX_HEAD_DIM
                // and KV_SEQ_TILE_LEN is the SMEM dim, but good for safety if head_dim < MAX_HEAD_DIM.
                if (r < KV_SEQ_TILE_LEN && c < MAX_HEAD_DIM) { 
                   dk_tile_acc_sdata[r][c] = 0.0f;
                   dv_tile_acc_sdata[r][c] = 0.0f;
                }
            }
        }
        __syncthreads(); 

        // --- Load K_tile and V_tile from global to shared memory ---
        int num_elements_in_kv_data_tile = KV_SEQ_TILE_LEN * head_dim;
        int elements_per_thread_kv_data_load = (num_elements_in_kv_data_tile + threads_in_block_count - 1) / threads_in_block_count;

        for (int i = 0; i < elements_per_thread_kv_data_load; ++i) {
            int element_idx = flat_thread_idx * elements_per_thread_kv_data_load + i;
            if (element_idx < num_elements_in_kv_data_tile) {
                int r = element_idx / head_dim; 
                int c = element_idx % head_dim; 

                if (kv_tile_start + r < seq_len) {
                    k_sdata[r][c] = k_ptr[batch_idx * num_heads * seq_len * head_dim +
                                          head_idx * seq_len * head_dim +
                                          (kv_tile_start + r) * head_dim +
                                          c];
                    v_sdata[r][c] = v_ptr[batch_idx * num_heads * seq_len * head_dim +
                                          head_idx * seq_len * head_dim +
                                          (kv_tile_start + r) * head_dim +
                                          c];
                } else { 
                    k_sdata[r][c] = 0.0f; 
                    v_sdata[r][c] = 0.0f;
                }
            }
        }
        __syncthreads(); 

        // --- Recompute S_ij and P_ij (Softmax) ---
        if (global_q_row_idx < seq_len) {
            for (int k_idx_in_tile = tid_in_warp; k_idx_in_tile < KV_SEQ_TILE_LEN; k_idx_in_tile += WARP_SIZE) {
                float score = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    score += q_sdata[q_row_in_block][d] * k_sdata[k_idx_in_tile][d];
                }
                score *= sm_scale;

                bool is_padded_k = (kv_tile_start + k_idx_in_tile) >= seq_len;
                if (is_causal && (kv_tile_start + k_idx_in_tile) > global_q_row_idx) {
                    score = -FLT_MAX; 
                }
                if (is_padded_k) {
                    score = -FLT_MAX;
                }
                s_sdata[q_row_in_block][k_idx_in_tile] = score;
            }
        }
        __syncthreads(); 

        if (global_q_row_idx < seq_len) {
            float max_val_thread = -FLT_MAX;
            for (int k_col = tid_in_warp; k_col < KV_SEQ_TILE_LEN; k_col += WARP_SIZE) {
                 max_val_thread = max(max_val_thread, s_sdata[q_row_in_block][k_col]);
            }
            float max_val_warp = warpReduceMax(max_val_thread);
            max_val_warp = __shfl_sync(0xFFFFFFFF, max_val_warp, 0); 

            float sum_exp_thread = 0.0f;
            for (int k_col = tid_in_warp; k_col < KV_SEQ_TILE_LEN; k_col += WARP_SIZE) {
                float val = expf(s_sdata[q_row_in_block][k_col] - max_val_warp);
                if (s_sdata[q_row_in_block][k_col] <= -FLT_MAX) {
                     val = 0.0f;
                }
                p_sdata[q_row_in_block][k_col] = val; 
                sum_exp_thread += val;
            }
            float sum_exp_warp = warpReduceSum(sum_exp_thread);
            sum_exp_warp = __shfl_sync(0xFFFFFFFF, sum_exp_warp, 0);
            if (sum_exp_warp == 0.0f) sum_exp_warp = 1e-6f;

            for (int k_col = tid_in_warp; k_col < KV_SEQ_TILE_LEN; k_col += WARP_SIZE) {
                p_sdata[q_row_in_block][k_col] /= sum_exp_warp; 
            }
        }
        __syncthreads(); 

        // --- Compute dP_ij = (dO @ V^T)_ij ---
        if (global_q_row_idx < seq_len) {
            for (int k_idx_in_tile = tid_in_warp; k_idx_in_tile < KV_SEQ_TILE_LEN; k_idx_in_tile += WARP_SIZE) {
                float dp_val = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    dp_val += do_sdata[q_row_in_block][d] * v_sdata[k_idx_in_tile][d];
                }
                dp_sdata[q_row_in_block][k_idx_in_tile] = dp_val;
            }
        }
        __syncthreads(); 

        // --- Compute dV contributions for this tile: dV_jd = P_ij^T dO_id ---
        // dv_tile_acc_sdata[k_idx_in_tile][d] = sum_{q_rows_in_block} (P_ij * dO_id)
        for (int k_target_offset = 0; k_target_offset < KV_SEQ_TILE_LEN; k_target_offset += Q_ROWS_PER_BLOCK) {
            int k_target_idx = q_row_in_block + k_target_offset; // Thread q_row_in_block works on k_target_idx row of dv_tile_acc
            if (k_target_idx < KV_SEQ_TILE_LEN) {
                 for (int d_target_offset = 0; d_target_offset < head_dim; d_target_offset += WARP_SIZE) {
                    int d_target_idx = tid_in_warp + d_target_offset; // Thread tid_in_warp works on d_target_idx col of dv_tile_acc
                    if (d_target_idx < head_dim) {
                        float dv_acc_val = 0.0f;
                        for (int i_q = 0; i_q < Q_ROWS_PER_BLOCK; ++i_q) {
                            if (blockIdx.x * Q_ROWS_PER_BLOCK + i_q < seq_len) { 
                                dv_acc_val += p_sdata[i_q][k_target_idx] * do_sdata[i_q][d_target_idx];
                            }
                        }
                        dv_tile_acc_sdata[k_target_idx][d_target_idx] = dv_acc_val; 
                    }
                }
            }
        }
        __syncthreads(); 

        // Atomically add dv_tile_acc_sdata to global dv_ptr
        for (int i = 0; i < elems_per_thread_tile_init; ++i) { 
            int element_idx_in_tile = flat_thread_idx * elems_per_thread_tile_init + i;
            if (element_idx_in_tile < num_elements_tile_kv_grad) { 
                int r = element_idx_in_tile / head_dim; 
                int c = element_idx_in_tile % head_dim;
                if (kv_tile_start + r < seq_len) { 
                    atomicAdd(&dv_ptr[batch_idx * num_heads * seq_len * head_dim +
                                     head_idx * seq_len * head_dim +
                                     (kv_tile_start + r) * head_dim +
                                     c], 
                              dv_tile_acc_sdata[r][c]);
                }
            }
        }
        
        // --- Compute dS_ij = P_ij * (dP_ij - D_i) ---
        if (global_q_row_idx < seq_len) {
            float d_sum_thread = 0.0f;
            for (int k_col = tid_in_warp; k_col < KV_SEQ_TILE_LEN; k_col += WARP_SIZE) {
                d_sum_thread += dp_sdata[q_row_in_block][k_col] * p_sdata[q_row_in_block][k_col];
            }
            float d_sum_warp = warpReduceSum(d_sum_thread); 
            if (tid_in_warp == 0) {
                D_rowsums_sdata[q_row_in_block] = d_sum_warp;
            }
        }
        __syncthreads(); 

        if (global_q_row_idx < seq_len) {
            float D_i = D_rowsums_sdata[q_row_in_block]; 
            for (int k_idx_in_tile = tid_in_warp; k_idx_in_tile < KV_SEQ_TILE_LEN; k_idx_in_tile += WARP_SIZE) {
                float p_val = p_sdata[q_row_in_block][k_idx_in_tile];
                if (p_val == 0.0f) { // If P_ij is 0 (masked), dS_ij is 0
                     s_sdata[q_row_in_block][k_idx_in_tile] = 0.0f;
                } else {
                     s_sdata[q_row_in_block][k_idx_in_tile] = 
                        p_val * (dp_sdata[q_row_in_block][k_idx_in_tile] - D_i);
                }
            }
        }
        __syncthreads(); 

        // --- Compute dQ contributions: dQ_id += (dS_ij * sm_scale) * K_jd ---
        if (global_q_row_idx < seq_len) {
            for (int d_loop_idx = 0; d_loop_idx < head_dim_elements_per_thread; ++d_loop_idx) {
                int current_d_component = tid_in_warp * head_dim_elements_per_thread + d_loop_idx;
                if (current_d_component < head_dim) {
                    float dq_sum_for_d_component = 0.0f;
                    for (int k_col = 0; k_col < KV_SEQ_TILE_LEN; ++k_col) {
                        if (kv_tile_start + k_col < seq_len) { 
                             dq_sum_for_d_component += s_sdata[q_row_in_block][k_col] * k_sdata[k_col][current_d_component];
                        }
                    }
                    dQ_acc[d_loop_idx] += dq_sum_for_d_component * sm_scale;
                }
            }
        }

        // --- Compute dK contributions: dK_jd += (dS_ij^T * sm_scale) * Q_id ---
        // dk_tile_acc_sdata[k_idx_in_tile][d] = sum_{q_rows_in_block} (s_sdata[q_row][k_idx] * q_sdata[q_row][d] * sm_scale)
        for (int k_target_offset = 0; k_target_offset < KV_SEQ_TILE_LEN; k_target_offset += Q_ROWS_PER_BLOCK) {
            int k_target_idx = q_row_in_block + k_target_offset;
            if (k_target_idx < KV_SEQ_TILE_LEN) {
                 for (int d_target_offset = 0; d_target_offset < head_dim; d_target_offset += WARP_SIZE) {
                    int d_target_idx = tid_in_warp + d_target_offset;
                    if (d_target_idx < head_dim) {
                        float dk_acc_val = 0.0f;
                        for (int i_q = 0; i_q < Q_ROWS_PER_BLOCK; ++i_q) {
                             if (blockIdx.x * Q_ROWS_PER_BLOCK + i_q < seq_len) { 
                                dk_acc_val += s_sdata[i_q][k_target_idx] * q_sdata[i_q][d_target_idx];
                             }
                        }
                        dk_tile_acc_sdata[k_target_idx][d_target_idx] = dk_acc_val * sm_scale; 
                    }
                }
            }
        }
        __syncthreads(); 
        
        // Atomically add dk_tile_acc_sdata to global dk_ptr
        for (int i = 0; i < elems_per_thread_tile_init; ++i) { 
            int element_idx_in_tile = flat_thread_idx * elems_per_thread_tile_init + i;
            if (element_idx_in_tile < num_elements_tile_kv_grad) {
                int r = element_idx_in_tile / head_dim;
                int c = element_idx_in_tile % head_dim;
                if (kv_tile_start + r < seq_len) { 
                     atomicAdd(&dk_ptr[batch_idx * num_heads * seq_len * head_dim +
                                     head_idx * seq_len * head_dim +
                                     (kv_tile_start + r) * head_dim +
                                     c], 
                              dk_tile_acc_sdata[r][c]);
                }
            }
        }
    } // End of loop over kv_tile_start

    // --- Write accumulated dQ_acc to global dq_ptr ---
    if (global_q_row_idx < seq_len) {
        for (int d_loop_idx = 0; d_loop_idx < head_dim_elements_per_thread; ++d_loop_idx) {
            int current_d_component = tid_in_warp * head_dim_elements_per_thread + d_loop_idx;
            if (current_d_component < head_dim) {
                 dq_ptr[batch_idx * num_heads * seq_len * head_dim +
                         head_idx * seq_len * head_dim +
                         global_q_row_idx * head_dim +
                         current_d_component] = dQ_acc[d_loop_idx];
            }
        }
    }
}


/*
Launch wrapper for backward pass
extern "C" void launch_fused_attention_backward(
    const float* q_ptr, const float* k_ptr, const float* v_ptr, const float* do_ptr,
    float* dq_ptr, float* dk_ptr, float* dv_ptr,
    int batch_size, int num_heads, int seq_len, int head_dim, float sm_scale, bool is_causal) {

    if (head_dim > MAX_HEAD_DIM) {
        // Handle error
        return;
    }
    // IMPORTANT: dk_ptr and dv_ptr must be zero-initialized by the caller ONCE 
    // before launching this kernel, as they are accumulated into using atomicAdd.

    dim3 blockDim(Q_ROWS_PER_BLOCK, WARP_SIZE); 
    dim3 gridDim((unsigned int)((seq_len + Q_ROWS_PER_BLOCK - 1) / Q_ROWS_PER_BLOCK), 
                 (unsigned int)(batch_size * num_heads));

    fused_attention_backward_kernel<<<gridDim, blockDim>>>(
        q_ptr, k_ptr, v_ptr, do_ptr,
        dq_ptr, dk_ptr, dv_ptr,
        batch_size, num_heads, seq_len, head_dim, sm_scale, is_causal);
    
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //    // Handle error
    // }
    // cudaDeviceSynchronize(); // For debugging
}
*/
