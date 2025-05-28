#include "../include/flash_attention.h"
#include <cuda_fp16.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Delta preprocessing kernel following Triton's _attn_bwd_preprocess
template<int BLOCK_M, int HEAD_DIM>
__global__ void attn_bwd_preprocess_kernel(
    const __half* __restrict__ O,
    const __half* __restrict__ DO,
    float* __restrict__ Delta,
    int Z, int H, int N_CTX) {
    
    int off_m = blockIdx.x * BLOCK_M + threadIdx.x;
    int off_hz = blockIdx.y;
    
    if (off_m >= N_CTX) return;
    
    // Load O and DO
    float delta = 0.0f;
    for (int d = 0; d < HEAD_DIM; ++d) {
        int idx = off_hz * HEAD_DIM * N_CTX + off_m * HEAD_DIM + d;
        float o_val = __half2float(O[idx]);
        float do_val = __half2float(DO[idx]);
        delta += o_val * do_val;
    }
    
    // Store delta
    Delta[off_hz * N_CTX + off_m] = delta;
}

// Device function following Triton's _attn_bwd_dkdv
template<int BLOCK_M1, int BLOCK_N1, int HEAD_DIM>
__device__ void attn_bwd_dkdv(
    float* dk, float* dv,
    const __half* Q, const __half* k, const __half* v, float sm_scale,
    const __half* DO,
    const float* M, const float* D,
    int stride_tok, int stride_d,
    int H, int N_CTX,
    int start_n, int start_m, int num_steps,
    bool MASK) {
    
    const float LN2 = 0.6931471824645996f; // ln(2)
    
    for (int blk_idx = 0; blk_idx < num_steps; ++blk_idx) {
        int curr_m = start_m + blk_idx * BLOCK_M1;
        
        // Load Q transpose and DO
        __half qT[HEAD_DIM][BLOCK_M1];
        __half do_vals[BLOCK_M1][HEAD_DIM];
        
        for (int i = 0; i < BLOCK_M1; ++i) {
            if (curr_m + i < N_CTX) {
                for (int d = 0; d < HEAD_DIM; ++d) {
                    int q_idx = (curr_m + i) * stride_tok + d * stride_d;
                    qT[d][i] = Q[q_idx];
                    do_vals[i][d] = DO[(curr_m + i) * stride_tok + d * stride_d];
                }
            }
        }
        
        // Load m (LSE values)
        float m_vals[BLOCK_M1];
        for (int i = 0; i < BLOCK_M1; ++i) {
            if (curr_m + i < N_CTX) {
                m_vals[i] = M[curr_m + i];
            }
        }
        
        // Compute QK^T
        float qkT[BLOCK_N1][BLOCK_M1];
        for (int j = 0; j < BLOCK_N1; ++j) {
            for (int i = 0; i < BLOCK_M1; ++i) {
                qkT[j][i] = 0.0f;
                for (int d = 0; d < HEAD_DIM; ++d) {
                    int k_idx = (start_n + j) * stride_tok + d * stride_d;
                    qkT[j][i] += __half2float(k[k_idx]) * __half2float(qT[d][i]);
                }
            }
        }
        
        // Compute P^T = exp2(QK^T - m)
        float pT[BLOCK_N1][BLOCK_M1];
        for (int j = 0; j < BLOCK_N1; ++j) {
            for (int i = 0; i < BLOCK_M1; ++i) {
                pT[j][i] = exp2f(qkT[j][i] - m_vals[i]);
                
                // Apply causal mask
                if (MASK && (curr_m + i) < (start_n + j)) {
                    pT[j][i] = 0.0f;
                }
            }
        }
        
        // Compute dV += P^T @ DO
        for (int j = 0; j < BLOCK_N1; ++j) {
            for (int d = 0; d < HEAD_DIM; ++d) {
                float dv_val = 0.0f;
                for (int i = 0; i < BLOCK_M1; ++i) {
                    if (curr_m + i < N_CTX) {
                        dv_val += pT[j][i] * __half2float(do_vals[i][d]);
                    }
                }
                dv[j * HEAD_DIM + d] += dv_val;
            }
        }
        
        // Load D (delta) values
        float Di[BLOCK_M1];
        for (int i = 0; i < BLOCK_M1; ++i) {
            if (curr_m + i < N_CTX) {
                Di[i] = D[curr_m + i];
            }
        }
        
        // Compute dP^T = V @ DO^T
        float dpT[BLOCK_N1][BLOCK_M1];
        for (int j = 0; j < BLOCK_N1; ++j) {
            for (int i = 0; i < BLOCK_M1; ++i) {
                dpT[j][i] = 0.0f;
                for (int d = 0; d < HEAD_DIM; ++d) {
                    int v_idx = (start_n + j) * stride_tok + d * stride_d;
                    if (curr_m + i < N_CTX) {
                        dpT[j][i] += __half2float(v[v_idx]) * __half2float(do_vals[i][d]);
                    }
                }
            }
        }
        
        // Compute dS^T = P^T * (dP^T - D)
        float dsT[BLOCK_N1][BLOCK_M1];
        for (int j = 0; j < BLOCK_N1; ++j) {
            for (int i = 0; i < BLOCK_M1; ++i) {
                dsT[j][i] = pT[j][i] * (dpT[j][i] - Di[i]);
            }
        }
        
        // Compute dK += dS^T @ Q^T
        for (int j = 0; j < BLOCK_N1; ++j) {
            for (int d = 0; d < HEAD_DIM; ++d) {
                float dk_val = 0.0f;
                for (int i = 0; i < BLOCK_M1; ++i) {
                    if (curr_m + i < N_CTX) {
                        dk_val += dsT[j][i] * __half2float(qT[d][i]);
                    }
                }
                dk[j * HEAD_DIM + d] += dk_val;
            }
        }
    }
}

// Device function following Triton's _attn_bwd_dq
template<int BLOCK_M2, int BLOCK_N2, int HEAD_DIM>
__device__ void attn_bwd_dq(
    float* dq,
    const __half* q, const __half* K, const __half* V,
    const __half* do_vals, const float* m, const float* D,
    int stride_tok, int stride_d,
    int H, int N_CTX,
    int start_m, int start_n, int num_steps,
    bool MASK) {
    
    for (int blk_idx = 0; blk_idx < num_steps; ++blk_idx) {
        int curr_n = start_n + blk_idx * BLOCK_N2;
        
        // Load K^T and V^T
        __half kT[HEAD_DIM][BLOCK_N2];
        __half vT[HEAD_DIM][BLOCK_N2];
        
        for (int j = 0; j < BLOCK_N2; ++j) {
            if (curr_n + j < N_CTX) {
                for (int d = 0; d < HEAD_DIM; ++d) {
                    int kv_idx = (curr_n + j) * stride_tok + d * stride_d;
                    kT[d][j] = K[kv_idx];
                    vT[d][j] = V[kv_idx];
                }
            }
        }
        
        // Compute QK
        float qk[BLOCK_M2][BLOCK_N2];
        for (int i = 0; i < BLOCK_M2; ++i) {
            for (int j = 0; j < BLOCK_N2; ++j) {
                qk[i][j] = 0.0f;
                for (int d = 0; d < HEAD_DIM; ++d) {
                    int q_idx = (start_m + i) * stride_tok + d * stride_d;
                    if (curr_n + j < N_CTX && start_m + i < N_CTX) {
                        qk[i][j] += __half2float(q[q_idx]) * __half2float(kT[d][j]);
                    }
                }
            }
        }
        
        // Compute P = exp2(QK - m)
        float p[BLOCK_M2][BLOCK_N2];
        for (int i = 0; i < BLOCK_M2; ++i) {
            for (int j = 0; j < BLOCK_N2; ++j) {
                p[i][j] = exp2f(qk[i][j] - m[i]);
                
                // Apply causal mask
                if (MASK && (start_m + i) < (curr_n + j)) {
                    p[i][j] = 0.0f;
                }
            }
        }
        
        // Compute dP = DO @ V^T
        float dp[BLOCK_M2][BLOCK_N2];
        for (int i = 0; i < BLOCK_M2; ++i) {
            for (int j = 0; j < BLOCK_N2; ++j) {
                dp[i][j] = 0.0f;
                for (int d = 0; d < HEAD_DIM; ++d) {
                    if (curr_n + j < N_CTX && start_m + i < N_CTX) {
                        dp[i][j] += __half2float(do_vals[i * HEAD_DIM + d]) * __half2float(vT[d][j]);
                    }
                }
            }
        }
        
        // Compute dS = P * (dP - D)
        float ds[BLOCK_M2][BLOCK_N2];
        for (int i = 0; i < BLOCK_M2; ++i) {
            for (int j = 0; j < BLOCK_N2; ++j) {
                ds[i][j] = p[i][j] * (dp[i][j] - D[i]);
            }
        }
        
        // Compute dQ += dS @ K^T
        for (int i = 0; i < BLOCK_M2; ++i) {
            for (int d = 0; d < HEAD_DIM; ++d) {
                float dq_val = 0.0f;
                for (int j = 0; j < BLOCK_N2; ++j) {
                    if (curr_n + j < N_CTX) {
                        dq_val += ds[i][j] * __half2float(kT[d][j]);
                    }
                }
                dq[i * HEAD_DIM + d] += dq_val;
            }
        }
    }
}

// Main backward kernel following Triton's _attn_bwd
template<int BLOCK_M1, int BLOCK_N1, int BLOCK_M2, int BLOCK_N2, int HEAD_DIM, int BLK_SLICE_FACTOR>
__global__ void flash_attention_bwd_kernel_impl(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    float sm_scale,
    const __half* __restrict__ DO,
    __half* __restrict__ DQ,
    __half* __restrict__ DK,
    __half* __restrict__ DV,
    const float* __restrict__ M,
    const float* __restrict__ D,
    int stride_z, int stride_h, int stride_tok, int stride_d,
    int H, int N_CTX) {
    
    extern __shared__ char smem_[];
    auto* smem = reinterpret_cast<BackwardSharedMemory<BLOCK_M, BLOCK_N, HEAD_DIM>*>(smem_);
    
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int kv_block_idx = blockIdx.x;
    
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    const int kv_start = kv_block_idx * BLOCK_N;
    const int kv_end = min(kv_start + BLOCK_N, seq_len);
    const int kv_size = kv_end - kv_start;
    
    if (kv_size <= 0) return;
    
    // Calculate tensor offsets
    const int batch_head_offset = (batch_idx * num_heads + head_idx) * seq_len * head_dim;
    const __half* q_batch = q + batch_head_offset;
    const __half* k_batch = k + batch_head_offset;
    const __half* v_batch = v + batch_head_offset;
    const __half* grad_out_batch = grad_out + batch_head_offset;
    const __half* out_batch = out + batch_head_offset;
    __half* grad_q_batch = grad_q + batch_head_offset;
    __half* grad_k_batch = grad_k + batch_head_offset;
    __half* grad_v_batch = grad_v + batch_head_offset;
    
    // Load K and V blocks to shared memory
    for (int i = tid; i < kv_size * head_dim; i += blockDim.x) {
        int row = i / head_dim;
        int col = i % head_dim;
        if (row < kv_size && col < head_dim) {
            smem->k_smem[row][col] = k_batch[(kv_start + row) * head_dim + col];
            smem->v_smem[row][col] = v_batch[(kv_start + row) * head_dim + col];
        }
    }
    __syncthreads();
    
    // Initialize grad_k and grad_v accumulators
    float grad_k_acc[BLOCK_N][HEAD_DIM];
    float grad_v_acc[BLOCK_N][HEAD_DIM];
    
    #pragma unroll
    for (int i = 0; i < kv_size; ++i) {
        #pragma unroll
        for (int j = 0; j < head_dim; ++j) {
            grad_k_acc[i][j] = 0.0f;
            grad_v_acc[i][j] = 0.0f;
        }
    }
    
    // Process Q blocks
    for (int q_block_start = 0; q_block_start < seq_len; q_block_start += BLOCK_M) {
        const int q_end = min(q_block_start + BLOCK_M, seq_len);
        const int q_size = q_end - q_block_start;
        
        // Load Q, grad_out, and out blocks to shared memory
        for (int i = tid; i < q_size * head_dim; i += blockDim.x) {
            int row = i / head_dim;
            int col = i % head_dim;
            if (row < q_size && col < head_dim) {
                smem->q_smem[row][col] = q_batch[(q_block_start + row) * head_dim + col];
                smem->do_smem[row][col] = grad_out_batch[(q_block_start + row) * head_dim + col];
                smem->o_smem[row][col] = out_batch[(q_block_start + row) * head_dim + col];
            }
        }
        __syncthreads();
        
        // Compute delta (rowsum of grad_out * out)
        for (int q_idx = 0; q_idx < q_size; ++q_idx) {
            float delta = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                delta += __half2float(smem->do_smem[q_idx][d]) * 
                        __half2float(smem->o_smem[q_idx][d]);
            }
            smem->delta_smem[q_idx] = delta;
        }
        __syncthreads();
        
        // Recompute attention scores Q @ K^T
        for (int q_idx = 0; q_idx < q_size; ++q_idx) {
            for (int kv_idx = tid; kv_idx < kv_size; kv_idx += blockDim.x) {
                float qk_val = 0.0f;
                #pragma unroll
                for (int d = 0; d < head_dim; ++d) {
                    qk_val += __half2float(smem->q_smem[q_idx][d]) * 
                             __half2float(smem->k_smem[kv_idx][d]);
                }
                qk_val *= scale;
                
                // Apply causal mask
                if (causal && (q_block_start + q_idx) < (kv_start + kv_idx)) {
                    qk_val = -INFINITY;
                }
                
                smem->qk_smem[q_idx][kv_idx] = qk_val;
            }
        }
        __syncthreads();
        
        // Recompute softmax probabilities
        for (int q_idx = 0; q_idx < q_size; ++q_idx) {
            int lse_offset = (batch_idx * num_heads + head_idx) * seq_len + (q_block_start + q_idx);
            float lse = softmax_lse[lse_offset];
            
            for (int kv_idx = 0; kv_idx < kv_size; ++kv_idx) {
                float prob = expf(smem->qk_smem[q_idx][kv_idx] - lse);
                if (causal && (q_block_start + q_idx) < (kv_start + kv_idx)) {
                    prob = 0.0f;
                }
                smem->softmax_smem[q_idx][kv_idx] = prob;
            }
        }
        __syncthreads();
        
        // Compute grad_v: P^T @ grad_out
        for (int kv_idx = 0; kv_idx < kv_size; ++kv_idx) {
            for (int d = 0; d < head_dim; ++d) {
                float grad_v_val = 0.0f;
                for (int q_idx = 0; q_idx < q_size; ++q_idx) {
                    grad_v_val += smem->softmax_smem[q_idx][kv_idx] * 
                                 __half2float(smem->do_smem[q_idx][d]);
                }
                grad_v_acc[kv_idx][d] += grad_v_val;
            }
        }
        
        // Compute dS = P * (grad_out @ V^T - delta)
        float ds_vals[BLOCK_M][BLOCK_N];
        for (int q_idx = 0; q_idx < q_size; ++q_idx) {
            for (int kv_idx = 0; kv_idx < kv_size; ++kv_idx) {
                float dp_val = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    dp_val += __half2float(smem->do_smem[q_idx][d]) * 
                             __half2float(smem->v_smem[kv_idx][d]);
                }
                
                float ds_val = smem->softmax_smem[q_idx][kv_idx] * 
                              (dp_val - smem->delta_smem[q_idx]);
                ds_vals[q_idx][kv_idx] = ds_val;
            }
        }
        
        // Compute grad_k: dS^T @ Q * scale
        for (int kv_idx = 0; kv_idx < kv_size; ++kv_idx) {
            for (int d = 0; d < head_dim; ++d) {
                float grad_k_val = 0.0f;
                for (int q_idx = 0; q_idx < q_size; ++q_idx) {
                    grad_k_val += ds_vals[q_idx][kv_idx] * 
                                 __half2float(smem->q_smem[q_idx][d]);
                }
                grad_k_acc[kv_idx][d] += grad_k_val * scale;
            }
        }
        
        __syncthreads();
    }
    
    // Write grad_k and grad_v
    for (int kv_idx = 0; kv_idx < kv_size; ++kv_idx) {
        for (int d = tid; d < head_dim; d += blockDim.x) {
            grad_k_batch[(kv_start + kv_idx) * head_dim + d] = 
                __float2half(grad_k_acc[kv_idx][d]);
            grad_v_batch[(kv_start + kv_idx) * head_dim + d] = 
                __float2half(grad_v_acc[kv_idx][d]);
        }
    }
}

// Separate kernel for computing grad_q
template<int BLOCK_M, int BLOCK_N, int HEAD_DIM>
__global__ void flash_attention_bwd_dq_kernel_impl(
    const __half* __restrict__ grad_out,
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    const float* __restrict__ softmax_lse,
    __half* __restrict__ grad_q,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale,
    bool causal) {
    
    extern __shared__ char smem_[];
    auto* smem = reinterpret_cast<BackwardSharedMemory<BLOCK_M, BLOCK_N, HEAD_DIM>*>(smem_);
    
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int q_block_idx = blockIdx.x;
    
    const int tid = threadIdx.x;
    
    const int q_start = q_block_idx * BLOCK_M;
    const int q_end = min(q_start + BLOCK_M, seq_len);
    const int q_size = q_end - q_start;
    
    if (q_size <= 0) return;
    
    // Calculate tensor offsets
    const int batch_head_offset = (batch_idx * num_heads + head_idx) * seq_len * head_dim;
    const __half* q_batch = q + batch_head_offset;
    const __half* k_batch = k + batch_head_offset;
    const __half* v_batch = v + batch_head_offset;
    const __half* grad_out_batch = grad_out + batch_head_offset;
    __half* grad_q_batch = grad_q + batch_head_offset;
    
    // Load Q and grad_out blocks
    for (int i = tid; i < q_size * head_dim; i += blockDim.x) {
        int row = i / head_dim;
        int col = i % head_dim;
        if (row < q_size && col < head_dim) {
            smem->q_smem[row][col] = q_batch[(q_start + row) * head_dim + col];
            smem->do_smem[row][col] = grad_out_batch[(q_start + row) * head_dim + col];
        }
    }
    __syncthreads();
    
    // Initialize grad_q accumulator
    float grad_q_acc[BLOCK_M][HEAD_DIM];
    #pragma unroll
    for (int i = 0; i < q_size; ++i) {
        #pragma unroll
        for (int j = 0; j < head_dim; ++j) {
            grad_q_acc[i][j] = 0.0f;
        }
    }
    
    // Process K,V blocks
    for (int kv_block_start = 0; kv_block_start < seq_len; kv_block_start += BLOCK_N) {
        const int kv_end = min(kv_block_start + BLOCK_N, seq_len);
        const int kv_size = kv_end - kv_block_start;
        
        // Load K and V blocks
        for (int i = tid; i < kv_size * head_dim; i += blockDim.x) {
            int row = i / head_dim;
            int col = i % head_dim;
            if (row < kv_size && col < head_dim) {
                smem->k_smem[row][col] = k_batch[(kv_block_start + row) * head_dim + col];
                smem->v_smem[row][col] = v_batch[(kv_block_start + row) * head_dim + col];
            }
        }
        __syncthreads();
        
        // Similar computation as in the backward pass...
        // This would follow the same pattern as above but compute grad_q
        
        __syncthreads();
    }
    
    // Write grad_q
    for (int q_idx = 0; q_idx < q_size; ++q_idx) {
        for (int d = tid; d < head_dim; d += blockDim.x) {
            grad_q_batch[(q_start + q_idx) * head_dim + d] = 
                __float2half(grad_q_acc[q_idx][d]);
        }
    }
}

void flash_attention_bwd_kernel(
    const __half* grad_out,
    const __half* q,
    const __half* k,
    const __half* v,
    const __half* out,
    const float* softmax_lse,
    __half* grad_q,
    __half* grad_k,
    __half* grad_v,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale,
    bool causal,
    cudaStream_t stream) {
    
    constexpr int BLOCK_M_CONST = 64;
    constexpr int BLOCK_N_CONST = 64;
    constexpr int HEAD_DIM_CONST = 64;
    
    // Initialize gradients to zero
    cudaMemsetAsync(grad_q, 0, batch_size * num_heads * seq_len * head_dim * sizeof(__half), stream);
    cudaMemsetAsync(grad_k, 0, batch_size * num_heads * seq_len * head_dim * sizeof(__half), stream);
    cudaMemsetAsync(grad_v, 0, batch_size * num_heads * seq_len * head_dim * sizeof(__half), stream);
    
    const int num_kv_blocks = (seq_len + BLOCK_N_CONST - 1) / BLOCK_N_CONST;
    const int num_q_blocks = (seq_len + BLOCK_M_CONST - 1) / BLOCK_M_CONST;
    
    dim3 grid_kv(num_kv_blocks, num_heads, batch_size);
    dim3 grid_q(num_q_blocks, num_heads, batch_size);
    dim3 block(256);
    
    const int smem_size = sizeof(BackwardSharedMemory<BLOCK_M_CONST, BLOCK_N_CONST, HEAD_DIM_CONST>);
    
    // Compute grad_k and grad_v
    flash_attention_bwd_kernel_impl<BLOCK_M_CONST, BLOCK_N_CONST, HEAD_DIM_CONST>
        <<<grid_kv, block, smem_size, stream>>>(
            grad_out, q, k, v, out, softmax_lse,
            grad_q, grad_k, grad_v,
            batch_size, num_heads, seq_len, head_dim,
            scale, causal
        );
    
    // Compute grad_q
    flash_attention_bwd_dq_kernel_impl<BLOCK_M_CONST, BLOCK_N_CONST, HEAD_DIM_CONST>
        <<<grid_q, block, smem_size, stream>>>(
            grad_out, q, k, v, softmax_lse, grad_q,
            batch_size, num_heads, seq_len, head_dim,
            scale, causal
        );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in flash_attention_bwd_kernel: %s\n", cudaGetErrorString(err));
    }
}
