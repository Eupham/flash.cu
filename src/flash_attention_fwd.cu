#include "../include/flash_attention.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

// Convert Triton's _attn_fwd_inner to CUDA
// This is the core algorithm that processes blocks of K,V for a given Q block
__device__ void attn_fwd_inner(
    float* acc,           // [BLOCK_M, HEAD_DIM] accumulator
    float* l_i,           // [BLOCK_M] normalization statistics  
    float* m_i,           // [BLOCK_M] max statistics
    const half* q,        // [BLOCK_M, HEAD_DIM] query block
    const half* K,        // Full K matrix
    const half* V,        // Full V matrix
    int offset_y,         // Base offset for this head
    int start_m,          // Starting M block index
    float qk_scale,       // Attention scale
    int BLOCK_M,
    int HEAD_DIM,
    int BLOCK_N,
    int STAGE,            // 1=off-band, 2=on-band, 3=non-causal
    const int* offs_m,    // [BLOCK_M] row indices
    const int* offs_n,    // [BLOCK_N] col indices  
    int N_CTX,
    int Z, int H,
    int tid
) {
    // Determine range of values handled by this stage
    int lo, hi;
    if (STAGE == 1) {
        lo = 0;
        hi = start_m * BLOCK_M;
    } else if (STAGE == 2) {
        lo = start_m * BLOCK_M;
        hi = (start_m + 1) * BLOCK_M;
        // Align to BLOCK_M boundary
        lo = (lo / BLOCK_M) * BLOCK_M;
    } else { // STAGE == 3, causal = False
        lo = 0; 
        hi = N_CTX;
    }
    
    int offsetkv_y = offset_y + lo;
    
    // Shared memory for this iteration
    __shared__ half k_shared[64 * 64];  // BLOCK_N x HEAD_DIM
    __shared__ half v_shared[64 * 64];  // BLOCK_N x HEAD_DIM
    __shared__ float qk_shared[64 * 64]; // BLOCK_M x BLOCK_N
    
    // Loop over k, v and update accumulator
    for (int start_n = lo; start_n < hi; start_n += BLOCK_N) {
        start_n = (start_n / BLOCK_N) * BLOCK_N; // multiple_of(start_n, BLOCK_N)
        
        // Load K block (transposed) - k = desc_k.load([offsetkv_y, 0]).T
        if (tid < BLOCK_N * HEAD_DIM) {
            int k_row = start_n + (tid / HEAD_DIM);
            int k_col = tid % HEAD_DIM;
            if (k_row < N_CTX) {
                int k_idx = offsetkv_y * HEAD_DIM + k_row * HEAD_DIM + k_col;
                k_shared[tid] = K[k_idx];
            } else {
                k_shared[tid] = __float2half(0.0f);
            }
        }
        
        // Load V block
        if (tid < BLOCK_N * HEAD_DIM) {
            int v_row = start_n + (tid / HEAD_DIM);
            int v_col = tid % HEAD_DIM;
            if (v_row < N_CTX) {
                int v_idx = offsetkv_y * HEAD_DIM + v_row * HEAD_DIM + v_col;
                v_shared[tid] = V[v_idx];
            } else {
                v_shared[tid] = __float2half(0.0f);
            }
        }
        __syncthreads();
        
        // Compute qk = dot(q, k) where k is transposed
        if (tid < BLOCK_M) {
            for (int n = 0; n < BLOCK_N; n++) {
                float qk_val = 0.0f;
                for (int d = 0; d < HEAD_DIM; d++) {
                    // q[tid, d] * k[n, d] (k is already loaded transposed)
                    qk_val += __half2float(q[tid * HEAD_DIM + d]) * 
                              __half2float(k_shared[n * HEAD_DIM + d]);
                }
                qk_shared[tid * BLOCK_N + n] = qk_val;
            }
        }
        __syncthreads();
        
        // Apply scaling and masking based on STAGE
        if (tid < BLOCK_M) {
            float m_ij;
            
            if (STAGE == 2) {
                // Apply causal mask: mask = offs_m[:, None] >= (start_n + offs_n[None, :])
                for (int n = 0; n < BLOCK_N; n++) {
                    int col_idx = start_n + n;
                    if (offs_m[tid] < col_idx) {
                        // Not causal, apply -inf
                        qk_shared[tid * BLOCK_N + n] = qk_shared[tid * BLOCK_N + n] * qk_scale + (-1.0e6f);
                    } else {
                        qk_shared[tid * BLOCK_N + n] = qk_shared[tid * BLOCK_N + n] * qk_scale;
                    }
                }
                
                // m_ij = maximum(m_i, max(qk, 1))
                float row_max = -INFINITY;
                for (int n = 0; n < BLOCK_N; n++) {
                    row_max = fmaxf(row_max, qk_shared[tid * BLOCK_N + n]);
                }
                m_ij = fmaxf(m_i[tid], row_max);
                
                // qk -= m_ij
                for (int n = 0; n < BLOCK_N; n++) {
                    qk_shared[tid * BLOCK_N + n] -= m_ij;
                }
            } else {
                // Non-causal or off-band
                // m_ij = maximum(m_i, max(qk, 1) * qk_scale)
                float row_max = -INFINITY;
                for (int n = 0; n < BLOCK_N; n++) {
                    row_max = fmaxf(row_max, qk_shared[tid * BLOCK_N + n]);
                }
                m_ij = fmaxf(m_i[tid], row_max * qk_scale);
                
                // qk = qk * qk_scale - m_ij
                for (int n = 0; n < BLOCK_N; n++) {
                    qk_shared[tid * BLOCK_N + n] = qk_shared[tid * BLOCK_N + n] * qk_scale - m_ij;
                }
            }
            
            // p = exp2(qk)
            float l_ij = 0.0f;
            for (int n = 0; n < BLOCK_N; n++) {
                float p_val = exp2f(qk_shared[tid * BLOCK_N + n]);
                qk_shared[tid * BLOCK_N + n] = p_val;
                l_ij += p_val;
            }
            
            // Compute correction factor: alpha = exp2(m_i - m_ij)
            float alpha = exp2f(m_i[tid] - m_ij);
            
            // Update output accumulator: acc = acc * alpha
            for (int d = 0; d < HEAD_DIM; d++) {
                acc[tid * HEAD_DIM + d] *= alpha;
            }
            
            // acc += dot(p, v)
            for (int d = 0; d < HEAD_DIM; d++) {
                float acc_val = 0.0f;
                for (int n = 0; n < BLOCK_N; n++) {
                    acc_val += qk_shared[tid * BLOCK_N + n] * 
                               __half2float(v_shared[n * HEAD_DIM + d]);
                }
                acc[tid * HEAD_DIM + d] += acc_val;
            }
            
            // Update statistics: l_i = l_i * alpha + l_ij, m_i = m_ij  
            l_i[tid] = l_i[tid] * alpha + l_ij;
            m_i[tid] = m_ij;
        }
        
        offsetkv_y += BLOCK_N;
        __syncthreads();
    }
}

// Main forward kernel following Triton's _attn_fwd
__global__ void flash_attention_fwd_kernel(
    const half* Q,
    const half* K,
    const half* V, 
    half* O,
    float* M,
    float sm_scale,
    int Z, int H, int N_CTX, int HEAD_DIM,
    int STAGE,
    bool warp_specialize
) {
    const int BLOCK_M = 64;
    const int BLOCK_N = 64;
    
    // Following Triton's grid and indexing
    int start_m = blockIdx.x;
    int off_hz = blockIdx.y;
    int off_z = off_hz / H;
    int off_h = off_hz % H;
    
    int tid = threadIdx.x;
    
    // Calculate tensor descriptor offsets following Triton
    int y_dim = Z * H * N_CTX;
    int offset_y = off_z * (N_CTX * H) + off_h * N_CTX;
    int qo_offset_y = offset_y + start_m * BLOCK_M;
    
    // Initialize offsets
    __shared__ int offs_m[64];
    __shared__ int offs_n[64];
    if (tid < BLOCK_M) {
        offs_m[tid] = start_m * BLOCK_M + tid;
    }
    if (tid < BLOCK_N) {
        offs_n[tid] = tid;
    }
    
    // Initialize statistics following Triton
    __shared__ float m_i[64];  // [BLOCK_M] - max stats
    __shared__ float l_i[64];  // [BLOCK_M] - normalization  
    __shared__ float acc[64 * 64]; // [BLOCK_M, HEAD_DIM] - accumulator
    
    if (tid < BLOCK_M) {
        m_i[tid] = -INFINITY;
        l_i[tid] = 1.0f;
        for (int d = 0; d < HEAD_DIM; d++) {
            acc[tid * HEAD_DIM + d] = 0.0f;
        }
    }
    __syncthreads();
    
    // Load scales: qk_scale *= 1.44269504 (1/log(2))
    float qk_scale = sm_scale * 1.44269504f;
    
    // Load Q block - it stays in SRAM throughout
    __shared__ half q_shared[64 * 64]; // [BLOCK_M, HEAD_DIM]
    if (tid < BLOCK_M * HEAD_DIM) {
        int q_row = tid / HEAD_DIM;
        int q_col = tid % HEAD_DIM;
        if (start_m * BLOCK_M + q_row < N_CTX) {
            int q_idx = off_z * (H * N_CTX * HEAD_DIM) + off_h * (N_CTX * HEAD_DIM) + 
                        (start_m * BLOCK_M + q_row) * HEAD_DIM + q_col;
            q_shared[tid] = Q[q_idx];
        } else {
            q_shared[tid] = __float2half(0.0f);
        }
    }
    __syncthreads();
    
    // Stage 1: off-band
    // For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    // For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if (STAGE & 1) {
        int inner_stage = 4 - STAGE; // Maps 3->1, 1->3
        attn_fwd_inner(acc, l_i, m_i, q_shared, K, V, offset_y, start_m, qk_scale,
                       BLOCK_M, HEAD_DIM, BLOCK_N, inner_stage, offs_m, offs_n, N_CTX,
                       Z, H, tid);
    }
    
    // Stage 2: on-band  
    if (STAGE & 2) {
        attn_fwd_inner(acc, l_i, m_i, q_shared, K, V, offset_y, start_m, qk_scale,
                       BLOCK_M, HEAD_DIM, BLOCK_N, 2, offs_m, offs_n, N_CTX,
                       Z, H, tid);
    }
    
    // Epilogue following Triton
    if (tid < BLOCK_M) {
        // m_i += log2(l_i)
        m_i[tid] += log2f(l_i[tid]);
        
        // acc = acc / l_i
        for (int d = 0; d < HEAD_DIM; d++) {
            acc[tid * HEAD_DIM + d] /= l_i[tid];
        }
        
        // Store M statistics
        if (start_m * BLOCK_M + tid < N_CTX) {
            M[off_hz * N_CTX + start_m * BLOCK_M + tid] = m_i[tid];
        }
        
        // Store output
        for (int d = 0; d < HEAD_DIM; d++) {
            if (start_m * BLOCK_M + tid < N_CTX) {
                int o_idx = off_z * (H * N_CTX * HEAD_DIM) + off_h * (N_CTX * HEAD_DIM) + 
                            (start_m * BLOCK_M + tid) * HEAD_DIM + d;
                O[o_idx] = __float2half(acc[tid * HEAD_DIM + d]);
            }
        }
    }
}
