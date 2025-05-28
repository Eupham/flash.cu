#include "flash_attention.h"
#include <cuda_fp16.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

template<int BLOCK_M, int BLOCK_N, int HEAD_DIM>
struct BackwardSharedMemory {
    __half q_smem[BLOCK_M][HEAD_DIM];
    __half k_smem[BLOCK_N][HEAD_DIM];
    __half v_smem[BLOCK_N][HEAD_DIM];
    __half do_smem[BLOCK_M][HEAD_DIM];
    __half o_smem[BLOCK_M][HEAD_DIM];
    float qk_smem[BLOCK_M][BLOCK_N];
    float softmax_smem[BLOCK_M][BLOCK_N];
    float delta_smem[BLOCK_M];
};

template<int BLOCK_M, int BLOCK_N, int HEAD_DIM>
__global__ void flash_attention_bwd_kernel_impl(
    const __half* __restrict__ grad_out,
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    const __half* __restrict__ out,
    const float* __restrict__ softmax_lse,
    __half* __restrict__ grad_q,
    __half* __restrict__ grad_k,
    __half* __restrict__ grad_v,
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
