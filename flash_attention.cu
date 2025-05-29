#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cmath>
#include <algorithm>

// Constants
constexpr float LN2 = 0.6931471824645996f;
constexpr float RCP_LN2 = 1.4426950408889634f;

// CUDA error checking macros
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// Warp-level primitives
__device__ __forceinline__ float warpReduceMax(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float blockReduceMax(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    val = warpReduceMax(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : -INFINITY;
    if (wid == 0) val = warpReduceMax(val);
    
    return val;
}

__device__ __forceinline__ float blockReduceSum(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    val = warpReduceSum(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    
    return val;
}

// Forward pass kernel
template<int BLOCK_M, int BLOCK_N, int HEAD_DIM, bool CAUSAL>
__global__ void flash_attention_forward_kernel(
    const half* Q,      // [B, H, N, D]
    const half* K,      // [B, H, N, D] 
    const half* V,      // [B, H, N, D]
    half* O,            // [B, H, N, D]
    float* M,           // [B, H, N] - max values
    const float sm_scale,
    const int B,        // batch size
    const int H,        // num heads
    const int N,        // sequence length
    const int D         // head dimension
) {
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int block_m_idx = blockIdx.x;
    
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    // Calculate offsets
    const int qo_offset = (batch_idx * H + head_idx) * N * D + block_m_idx * BLOCK_M * D;
    const int kv_offset = (batch_idx * H + head_idx) * N * D;
    
    // Shared memory for Q, K, V blocks
    __shared__ half q_smem[BLOCK_M * HEAD_DIM];
    __shared__ half k_smem[BLOCK_N * HEAD_DIM];
    __shared__ half v_smem[BLOCK_N * HEAD_DIM];
    __shared__ float qk_smem[BLOCK_M * BLOCK_N];
    
    // Load Q block
    for (int i = tid; i < BLOCK_M * HEAD_DIM; i += blockDim.x) {
        int row = i / HEAD_DIM;
        int col = i % HEAD_DIM;
        int global_row = block_m_idx * BLOCK_M + row;
        if (global_row < N) {
            q_smem[i] = Q[qo_offset + row * D + col];
        } else {
            q_smem[i] = __float2half(0.0f);
        }
    }
    __syncthreads();
    
    // Initialize output accumulator and statistics
    float acc[HEAD_DIM] = {0.0f};
    float m_i = -INFINITY;
    float l_i = 0.0f;
    
    // Loop over K, V blocks
    for (int block_n_start = 0; block_n_start < N; block_n_start += BLOCK_N) {
        // Load K and V blocks
        for (int i = tid; i < BLOCK_N * HEAD_DIM; i += blockDim.x) {
            int row = i / HEAD_DIM;
            int col = i % HEAD_DIM;
            int global_row = block_n_start + row;
            if (global_row < N) {
                k_smem[i] = K[kv_offset + global_row * D + col];
                v_smem[i] = V[kv_offset + global_row * D + col];
            } else {
                k_smem[i] = __float2half(0.0f);
                v_smem[i] = __float2half(0.0f);
            }
        }
        __syncthreads();
        
        // Compute QK^T for this thread's row
        if (warp_id < BLOCK_M) {
            int m_idx = warp_id;
            int global_m_idx = block_m_idx * BLOCK_M + m_idx;
            
            if (global_m_idx < N) {
                for (int n_idx = lane_id; n_idx < BLOCK_N; n_idx += 32) {
                    int global_n_idx = block_n_start + n_idx;
                    
                    if (global_n_idx < N) {
                        float qk_val = 0.0f;
                        
                        // Compute dot product Q[m_idx] * K[n_idx]^T
                        for (int d = 0; d < HEAD_DIM; d++) {
                            float q_val = __half2float(q_smem[m_idx * HEAD_DIM + d]);
                            float k_val = __half2float(k_smem[n_idx * HEAD_DIM + d]);
                            qk_val += q_val * k_val;
                        }
                        
                        qk_val *= sm_scale;
                        
                        // Apply causal mask
                        if (CAUSAL && global_m_idx < global_n_idx) {
                            qk_val = -INFINITY;
                        }
                        
                        qk_smem[m_idx * BLOCK_N + n_idx] = qk_val;
                    }
                }
            }
        }
        __syncthreads();
        
        // Compute softmax and update accumulator
        if (warp_id < BLOCK_M) {
            int m_idx = warp_id;
            int global_m_idx = block_m_idx * BLOCK_M + m_idx;
            
            if (global_m_idx < N) {
                // Find max in this row
                float m_ij = -INFINITY;
                for (int n_idx = 0; n_idx < BLOCK_N; n_idx++) {
                    int global_n_idx = block_n_start + n_idx;
                    if (global_n_idx < N) {
                        m_ij = fmaxf(m_ij, qk_smem[m_idx * BLOCK_N + n_idx]);
                    }
                }
                
                // Update global max
                float m_new = fmaxf(m_i, m_ij);
                float alpha = expf((m_i - m_new) * RCP_LN2);
                
                // Scale previous accumulator
                for (int d = lane_id; d < HEAD_DIM; d += 32) {
                    acc[d] *= alpha;
                }
                
                // Compute exp(qk - m_new) and sum
                float l_ij = 0.0f;
                for (int n_idx = 0; n_idx < BLOCK_N; n_idx++) {
                    int global_n_idx = block_n_start + n_idx;
                    if (global_n_idx < N) {
                        float exp_val = expf((qk_smem[m_idx * BLOCK_N + n_idx] - m_new) * RCP_LN2);
                        qk_smem[m_idx * BLOCK_N + n_idx] = exp_val;
                        l_ij += exp_val;
                    }
                }
                
                // Update accumulator with V
                for (int d = lane_id; d < HEAD_DIM; d += 32) {
                    float acc_update = 0.0f;
                    for (int n_idx = 0; n_idx < BLOCK_N; n_idx++) {
                        int global_n_idx = block_n_start + n_idx;
                        if (global_n_idx < N) {
                            float p_val = qk_smem[m_idx * BLOCK_N + n_idx];
                            float v_val = __half2float(v_smem[n_idx * HEAD_DIM + d]);
                            acc_update += p_val * v_val;
                        }
                    }
                    acc[d] += acc_update;
                }
                
                // Update statistics
                l_i = l_i * alpha + l_ij;
                m_i = m_new;
            }
        }
        __syncthreads();
    }
    
    // Write output and statistics
    if (warp_id < BLOCK_M) {
        int m_idx = warp_id;
        int global_m_idx = block_m_idx * BLOCK_M + m_idx;
        
        if (global_m_idx < N) {
            // Store max value for backward pass
            if (lane_id == 0) {
                M[(batch_idx * H + head_idx) * N + global_m_idx] = m_i + logf(l_i);
            }
            
            // Normalize and store output
            for (int d = lane_id; d < HEAD_DIM; d += 32) {
                float out_val = acc[d] / l_i;
                O[qo_offset + m_idx * D + d] = __float2half(out_val);
            }
        }
    }
}

// Backward pass - preprocess kernel
template<int BLOCK_M, int HEAD_DIM>
__global__ void flash_attention_backward_preprocess_kernel(
    const half* O,      // [B, H, N, D]
    const half* dO,     // [B, H, N, D]
    float* Delta,       // [B, H, N]
    const int B,
    const int H,
    const int N,
    const int D
) {
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int block_m_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    const int offset = (batch_idx * H + head_idx) * N * D;
    
    for (int m_idx = 0; m_idx < BLOCK_M; m_idx++) {
        int global_m_idx = block_m_idx * BLOCK_M + m_idx;
        if (global_m_idx >= N) break;
        
        float delta_sum = 0.0f;
        
        // Compute delta = sum(O * dO) for this row
        for (int d = tid; d < HEAD_DIM; d += blockDim.x) {
            float o_val = __half2float(O[offset + global_m_idx * D + d]);
            float do_val = __half2float(dO[offset + global_m_idx * D + d]);
            delta_sum += o_val * do_val;
        }
        
        // Reduce within block
        delta_sum = blockReduceSum(delta_sum);
        
        if (tid == 0) {
            Delta[(batch_idx * H + head_idx) * N + global_m_idx] = delta_sum;
        }
    }
}

// Backward pass - main kernel
template<int BLOCK_M1, int BLOCK_N1, int BLOCK_M2, int BLOCK_N2, int HEAD_DIM, bool CAUSAL>
__global__ void flash_attention_backward_kernel(
    const half* Q,      // [B, H, N, D]
    const half* K,      // [B, H, N, D]
    const half* V,      // [B, H, N, D]
    const half* dO,     // [B, H, N, D]
    half* dQ,           // [B, H, N, D]
    half* dK,           // [B, H, N, D]
    half* dV,           // [B, H, N, D]
    const float* M,     // [B, H, N]
    const float* Delta, // [B, H, N]
    const float sm_scale,
    const int B,
    const int H,
    const int N,
    const int D
) {
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int block_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    const int offset = (batch_idx * H + head_idx) * N * D;
    const int stat_offset = (batch_idx * H + head_idx) * N;
    
    // Shared memory
    __shared__ half q_smem[BLOCK_M1 * HEAD_DIM];
    __shared__ half k_smem[BLOCK_N1 * HEAD_DIM];
    __shared__ half v_smem[BLOCK_N1 * HEAD_DIM];
    __shared__ half do_smem[BLOCK_M1 * HEAD_DIM];
    __shared__ float qk_smem[BLOCK_M1 * BLOCK_N1];
    __shared__ float m_smem[BLOCK_M1];
    __shared__ float delta_smem[BLOCK_M1];
    
    // This is a simplified version - full implementation would need
    // more complex logic for handling different block patterns
    // and memory access patterns for dK, dV computation
    
    // Load K, V blocks for this thread block
    int block_n_start = block_idx * BLOCK_N1;
    for (int i = tid; i < BLOCK_N1 * HEAD_DIM; i += blockDim.x) {
        int row = i / HEAD_DIM;
        int col = i % HEAD_DIM;
        int global_row = block_n_start + row;
        if (global_row < N) {
            k_smem[i] = K[offset + global_row * D + col];
            v_smem[i] = V[offset + global_row * D + col];
        } else {
            k_smem[i] = __float2half(0.0f);
            v_smem[i] = __float2half(0.0f);
        }
    }
    __syncthreads();
    
    // Initialize dK, dV accumulators
    float dk_acc[HEAD_DIM] = {0.0f};
    float dv_acc[HEAD_DIM] = {0.0f};
    
    // Loop over Q blocks to compute dK, dV
    for (int block_m_start = 0; block_m_start < N; block_m_start += BLOCK_M1) {
        // Load Q, dO, M, Delta for this block
        for (int i = tid; i < BLOCK_M1 * HEAD_DIM; i += blockDim.x) {
            int row = i / HEAD_DIM;
            int col = i % HEAD_DIM;
            int global_row = block_m_start + row;
            if (global_row < N) {
                q_smem[i] = Q[offset + global_row * D + col];
                do_smem[i] = dO[offset + global_row * D + col];
            } else {
                q_smem[i] = __float2half(0.0f);
                do_smem[i] = __float2half(0.0f);
            }
        }
        
        for (int i = tid; i < BLOCK_M1; i += blockDim.x) {
            int global_row = block_m_start + i;
            if (global_row < N) {
                m_smem[i] = M[stat_offset + global_row];
                delta_smem[i] = Delta[stat_offset + global_row];
            } else {
                m_smem[i] = -INFINITY;
                delta_smem[i] = 0.0f;
            }
        }
        __syncthreads();
        
        // Compute QK^T and attention weights
        for (int m_idx = 0; m_idx < BLOCK_M1; m_idx++) {
            int global_m_idx = block_m_start + m_idx;
            if (global_m_idx >= N) break;
            
            for (int n_idx = tid; n_idx < BLOCK_N1; n_idx += blockDim.x) {
                int global_n_idx = block_n_start + n_idx;
                if (global_n_idx >= N) continue;
                
                // Skip if causal mask applies
                if (CAUSAL && global_m_idx < global_n_idx) {
                    qk_smem[m_idx * BLOCK_N1 + n_idx] = 0.0f;
                    continue;
                }
                
                float qk_val = 0.0f;
                for (int d = 0; d < HEAD_DIM; d++) {
                    float q_val = __half2float(q_smem[m_idx * HEAD_DIM + d]);
                    float k_val = __half2float(k_smem[n_idx * HEAD_DIM + d]);
                    qk_val += q_val * k_val;
                }
                
                qk_val = qk_val * sm_scale - m_smem[m_idx];
                float p_val = expf(qk_val * RCP_LN2);
                qk_smem[m_idx * BLOCK_N1 + n_idx] = p_val;
            }
        }
        __syncthreads();
        
        // Compute gradients for dK, dV
        if (tid < BLOCK_N1) {
            int n_idx = tid;
            int global_n_idx = block_n_start + n_idx;
            if (global_n_idx < N) {
                // Compute dV
                for (int d = 0; d < HEAD_DIM; d++) {
                    float dv_val = 0.0f;
                    for (int m_idx = 0; m_idx < BLOCK_M1; m_idx++) {
                        int global_m_idx = block_m_start + m_idx;
                        if (global_m_idx >= N) break;
                        if (CAUSAL && global_m_idx < global_n_idx) continue;
                        
                        float p_val = qk_smem[m_idx * BLOCK_N1 + n_idx];
                        float do_val = __half2float(do_smem[m_idx * HEAD_DIM + d]);
                        dv_val += p_val * do_val;
                    }
                    dv_acc[d] += dv_val;
                }
                
                // Compute dK
                for (int d = 0; d < HEAD_DIM; d++) {
                    float dk_val = 0.0f;
                    for (int m_idx = 0; m_idx < BLOCK_M1; m_idx++) {
                        int global_m_idx = block_m_start + m_idx;
                        if (global_m_idx >= N) break;
                        if (CAUSAL && global_m_idx < global_n_idx) continue;
                        
                        float p_val = qk_smem[m_idx * BLOCK_N1 + n_idx];
                        float do_val = __half2float(do_smem[m_idx * HEAD_DIM + d]);
                        float v_val = __half2float(v_smem[n_idx * HEAD_DIM + d]);
                        float delta_val = delta_smem[m_idx];
                        
                        float dp_val = 0.0f;
                        for (int d2 = 0; d2 < HEAD_DIM; d2++) {
                            dp_val += __half2float(do_smem[m_idx * HEAD_DIM + d2]) * 
                                     __half2float(v_smem[n_idx * HEAD_DIM + d2]);
                        }
                        
                        float ds_val = p_val * (dp_val - delta_val);
                        float q_val = __half2float(q_smem[m_idx * HEAD_DIM + d]);
                        dk_val += ds_val * q_val;
                    }
                    dk_acc[d] += dk_val * sm_scale;
                }
            }
        }
        __syncthreads();
    }
    
    // Write dK, dV results
    if (tid < BLOCK_N1) {
        int n_idx = tid;
        int global_n_idx = block_n_start + n_idx;
        if (global_n_idx < N) {
            for (int d = 0; d < HEAD_DIM; d++) {
                dK[offset + global_n_idx * D + d] = __float2half(dk_acc[d]);
                dV[offset + global_n_idx * D + d] = __float2half(dv_acc[d]);
            }
        }
    }
}

// Host wrapper functions
extern "C" {

void launch_flash_attention_forward(
    const half* Q, const half* K, const half* V, half* O, float* M,
    float sm_scale, int B, int H, int N, int D, bool causal, cudaStream_t stream) {
    
    constexpr int BLOCK_M = 64;
    constexpr int BLOCK_N = 64;
    constexpr int HEAD_DIM = 64; // This should match D
    
    dim3 grid(
        (N + BLOCK_M - 1) / BLOCK_M,  // blocks in M dimension
        H,                            // heads
        B                             // batches
    );
    dim3 block(128); // threads per block
    
    if (causal) {
        flash_attention_forward_kernel<BLOCK_M, BLOCK_N, HEAD_DIM, true>
            <<<grid, block, 0, stream>>>(Q, K, V, O, M, sm_scale, B, H, N, D);
    } else {
        flash_attention_forward_kernel<BLOCK_M, BLOCK_N, HEAD_DIM, false>
            <<<grid, block, 0, stream>>>(Q, K, V, O, M, sm_scale, B, H, N, D);
    }
    
    CUDA_CHECK(cudaGetLastError());
}

void launch_flash_attention_backward_preprocess(
    const half* O, const half* dO, float* Delta,
    int B, int H, int N, int D, cudaStream_t stream) {
    
    constexpr int BLOCK_M = 128;
    constexpr int HEAD_DIM = 64;
    
    dim3 grid(
        (N + BLOCK_M - 1) / BLOCK_M,
        H,
        B
    );
    dim3 block(256);
    
    flash_attention_backward_preprocess_kernel<BLOCK_M, HEAD_DIM>
        <<<grid, block, 0, stream>>>(O, dO, Delta, B, H, N, D);
    
    CUDA_CHECK(cudaGetLastError());
}

void launch_flash_attention_backward(
    const half* Q, const half* K, const half* V, const half* dO,
    half* dQ, half* dK, half* dV,
    const float* M, const float* Delta, float sm_scale,
    int B, int H, int N, int D, bool causal, cudaStream_t stream) {
    
    constexpr int BLOCK_M1 = 32;
    constexpr int BLOCK_N1 = 128;
    constexpr int BLOCK_M2 = 128;
    constexpr int BLOCK_N2 = 32;
    constexpr int HEAD_DIM = 64;
    
    dim3 grid(
        (N + BLOCK_N1 - 1) / BLOCK_N1,
        1,
        B * H
    );
    dim3 block(256);
    
    if (causal) {
        flash_attention_backward_kernel<BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2, HEAD_DIM, true>
            <<<grid, block, 0, stream>>>(
                Q, K, V, dO, dQ, dK, dV, M, Delta, sm_scale, B, H, N, D);
    } else {
        flash_attention_backward_kernel<BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2, HEAD_DIM, false>
            <<<grid, block, 0, stream>>>(
                Q, K, V, dO, dQ, dK, dV, M, Delta, sm_scale, B, H, N, D);
    }
    
    CUDA_CHECK(cudaGetLastError());
}

} // extern "C"
