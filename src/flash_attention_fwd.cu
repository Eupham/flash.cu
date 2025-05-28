#include "../include/flash_attention.h"
#include <cuda_fp16.h>
#include <mma.h>
#include <cooperative_groups.h>

using namespace nvcuda;
namespace cg = cooperative_groups;

// Shared memory for blocks
template<int BLOCK_M, int BLOCK_N, int HEAD_DIM>
struct SharedMemory {
    __half q_smem[BLOCK_M][HEAD_DIM];
    __half k_smem[BLOCK_N][HEAD_DIM];
    __half v_smem[BLOCK_N][HEAD_DIM];
    float qk_smem[BLOCK_M][BLOCK_N];
    float softmax_smem[BLOCK_M][BLOCK_N];
};

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template<int BLOCK_M, int BLOCK_N, int HEAD_DIM>
__global__ void flash_attention_fwd_kernel_impl(
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    __half* __restrict__ out,
    float* __restrict__ softmax_lse,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale,
    bool causal) {
    
    extern __shared__ char smem_[];
    auto* smem = reinterpret_cast<SharedMemory<BLOCK_M, BLOCK_N, HEAD_DIM>*>(smem_);
    
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int q_block_idx = blockIdx.x;
    
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    const int q_start = q_block_idx * BLOCK_M;
    const int q_end = min(q_start + BLOCK_M, seq_len);
    const int q_size = q_end - q_start;
    
    if (q_size <= 0) return;
    
    // Calculate tensor offsets
    const int batch_head_offset = (batch_idx * num_heads + head_idx) * seq_len * head_dim;
    const __half* q_batch = q + batch_head_offset;
    const __half* k_batch = k + batch_head_offset;
    const __half* v_batch = v + batch_head_offset;
    __half* out_batch = out + batch_head_offset;
    
    // Load Q block to shared memory
    for (int i = tid; i < q_size * head_dim; i += blockDim.x) {
        int row = i / head_dim;
        int col = i % head_dim;
        if (row < q_size && col < head_dim) {
            smem->q_smem[row][col] = q_batch[(q_start + row) * head_dim + col];
        }
    }
    __syncthreads();
    
    // Initialize accumulator and statistics
    float acc[BLOCK_M][HEAD_DIM];
    float max_vals[BLOCK_M];
    float sum_vals[BLOCK_M];
    
    #pragma unroll
    for (int i = 0; i < q_size; ++i) {
        max_vals[i] = -INFINITY;
        sum_vals[i] = 0.0f;
        #pragma unroll
        for (int j = 0; j < head_dim; ++j) {
            acc[i][j] = 0.0f;
        }
    }
    
    // Process K,V blocks
    for (int kv_block_start = 0; kv_block_start < seq_len; kv_block_start += BLOCK_N) {
        const int kv_end = min(kv_block_start + BLOCK_N, seq_len);
        const int kv_size = kv_end - kv_block_start;
        
        // Load K and V blocks to shared memory
        for (int i = tid; i < kv_size * head_dim; i += blockDim.x) {
            int row = i / head_dim;
            int col = i % head_dim;
            if (row < kv_size && col < head_dim) {
                smem->k_smem[row][col] = k_batch[(kv_block_start + row) * head_dim + col];
                smem->v_smem[row][col] = v_batch[(kv_block_start + row) * head_dim + col];
            }
        }
        __syncthreads();
        
        // Compute Q @ K^T
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
                if (causal && (q_start + q_idx) < (kv_block_start + kv_idx)) {
                    qk_val = -INFINITY;
                }
                
                smem->qk_smem[q_idx][kv_idx] = qk_val;
            }
        }
        __syncthreads();
        
        // Compute softmax and update statistics
        for (int q_idx = 0; q_idx < q_size; ++q_idx) {
            // Find max in this block
            float block_max = -INFINITY;
            for (int kv_idx = 0; kv_idx < kv_size; ++kv_idx) {
                block_max = fmaxf(block_max, smem->qk_smem[q_idx][kv_idx]);
            }
            
            // Update global max
            float new_max = fmaxf(max_vals[q_idx], block_max);
            float scale_factor = expf(max_vals[q_idx] - new_max);
            
            // Scale previous accumulator and sum
            for (int d = 0; d < head_dim; ++d) {
                acc[q_idx][d] *= scale_factor;
            }
            sum_vals[q_idx] *= scale_factor;
            
            // Compute softmax for current block
            float block_sum = 0.0f;
            for (int kv_idx = 0; kv_idx < kv_size; ++kv_idx) {
                float exp_val = expf(smem->qk_smem[q_idx][kv_idx] - new_max);
                smem->softmax_smem[q_idx][kv_idx] = exp_val;
                block_sum += exp_val;
            }
            
            // Update statistics
            max_vals[q_idx] = new_max;
            sum_vals[q_idx] += block_sum;
            
            // Update accumulator with V
            for (int d = 0; d < head_dim; ++d) {
                float sum_val = 0.0f;
                for (int kv_idx = 0; kv_idx < kv_size; ++kv_idx) {
                    sum_val += smem->softmax_smem[q_idx][kv_idx] * 
                              __half2float(smem->v_smem[kv_idx][d]);
                }
                acc[q_idx][d] += sum_val;
            }
        }
        __syncthreads();
    }
    
    // Write output
    for (int q_idx = 0; q_idx < q_size; ++q_idx) {
        // Normalize accumulator
        float norm_factor = 1.0f / sum_vals[q_idx];
        
        for (int d = tid; d < head_dim; d += blockDim.x) {
            out_batch[(q_start + q_idx) * head_dim + d] = 
                __float2half(acc[q_idx][d] * norm_factor);
        }
        
        // Store LSE (log-sum-exp)
        if (tid == 0) {
            int lse_offset = (batch_idx * num_heads + head_idx) * seq_len + (q_start + q_idx);
            softmax_lse[lse_offset] = max_vals[q_idx] + logf(sum_vals[q_idx]);
        }
    }
}

void flash_attention_fwd_kernel(
    const __half* q,
    const __half* k,
    const __half* v,
    __half* out,
    float* softmax_lse,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale,
    bool causal,
    cudaStream_t stream) {
    
    constexpr int BLOCK_M_CONST = 64;
    constexpr int BLOCK_N_CONST = 64;
    constexpr int HEAD_DIM_CONST = 64;  // Assuming 64 for now
    
    const int num_q_blocks = (seq_len + BLOCK_M_CONST - 1) / BLOCK_M_CONST;
    
    dim3 grid(num_q_blocks, num_heads, batch_size);
    dim3 block(256);  // Number of threads per block
    
    const int smem_size = sizeof(SharedMemory<BLOCK_M_CONST, BLOCK_N_CONST, HEAD_DIM_CONST>);
    
    flash_attention_fwd_kernel_impl<BLOCK_M_CONST, BLOCK_N_CONST, HEAD_DIM_CONST>
        <<<grid, block, smem_size, stream>>>(
            q, k, v, out, softmax_lse,
            batch_size, num_heads, seq_len, head_dim,
            scale, causal
        );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in flash_attention_fwd_kernel: %s\n", cudaGetErrorString(err));
    }
}
