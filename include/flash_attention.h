#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

// Forward declarations
torch::Tensor flash_attention_forward(
    torch::Tensor q,           // [batch, heads, seq_len, head_dim]
    torch::Tensor k,           // [batch, heads, seq_len, head_dim]
    torch::Tensor v,           // [batch, heads, seq_len, head_dim]
    float scale,
    bool causal
);

std::vector<torch::Tensor> flash_attention_backward(
    torch::Tensor grad_out,    // [batch, heads, seq_len, head_dim]
    torch::Tensor q,           // [batch, heads, seq_len, head_dim]
    torch::Tensor k,           // [batch, heads, seq_len, head_dim]
    torch::Tensor v,           // [batch, heads, seq_len, head_dim]
    torch::Tensor out,         // [batch, heads, seq_len, head_dim]
    torch::Tensor softmax_lse, // [batch, heads, seq_len]
    float scale,
    bool causal
);

// CUDA kernel declarations
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
    cudaStream_t stream
);

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
    cudaStream_t stream
);

// Constants
constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 64;
constexpr int WARP_SIZE = 32;
constexpr int MAX_THREADS_PER_BLOCK = 1024;
