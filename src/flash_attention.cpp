#include <torch/extension.h>
#include "flash_attention.h"

torch::Tensor flash_attention_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    float scale,
    bool causal) {
    
    TORCH_CHECK(q.device().is_cuda(), "q must be a CUDA tensor");
    TORCH_CHECK(k.device().is_cuda(), "k must be a CUDA tensor");
    TORCH_CHECK(v.device().is_cuda(), "v must be a CUDA tensor");
    TORCH_CHECK(q.dtype() == torch::kFloat16, "Only float16 is supported");
    
    auto sizes = q.sizes();
    int batch_size = sizes[0];
    int num_heads = sizes[1];
    int seq_len = sizes[2];
    int head_dim = sizes[3];
    
    TORCH_CHECK(k.sizes() == sizes, "k must have same shape as q");
    TORCH_CHECK(v.sizes() == sizes, "v must have same shape as q");
    TORCH_CHECK(head_dim <= 128, "head_dim must be <= 128");
    TORCH_CHECK(head_dim % 8 == 0, "head_dim must be divisible by 8");
    
    auto out = torch::empty_like(q);
    auto softmax_lse = torch::empty({batch_size, num_heads, seq_len}, 
                                   torch::dtype(torch::kFloat32).device(q.device()));
    
    const __half* q_ptr = reinterpret_cast<const __half*>(q.data_ptr<at::Half>());
    const __half* k_ptr = reinterpret_cast<const __half*>(k.data_ptr<at::Half>());
    const __half* v_ptr = reinterpret_cast<const __half*>(v.data_ptr<at::Half>());
    __half* out_ptr = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
    float* lse_ptr = softmax_lse.data_ptr<float>();
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    flash_attention_fwd_kernel(
        q_ptr, k_ptr, v_ptr, out_ptr, lse_ptr,
        batch_size, num_heads, seq_len, head_dim,
        scale, causal, stream
    );
    
    return out;
}

std::vector<torch::Tensor> flash_attention_backward(
    torch::Tensor grad_out,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor out,
    torch::Tensor softmax_lse,
    float scale,
    bool causal) {
    
    TORCH_CHECK(grad_out.device().is_cuda(), "grad_out must be a CUDA tensor");
    TORCH_CHECK(q.device().is_cuda(), "q must be a CUDA tensor");
    
    auto sizes = q.sizes();
    int batch_size = sizes[0];
    int num_heads = sizes[1];
    int seq_len = sizes[2];
    int head_dim = sizes[3];
    
    auto grad_q = torch::empty_like(q);
    auto grad_k = torch::empty_like(k);
    auto grad_v = torch::empty_like(v);
    
    const __half* grad_out_ptr = reinterpret_cast<const __half*>(grad_out.data_ptr<at::Half>());
    const __half* q_ptr = reinterpret_cast<const __half*>(q.data_ptr<at::Half>());
    const __half* k_ptr = reinterpret_cast<const __half*>(k.data_ptr<at::Half>());
    const __half* v_ptr = reinterpret_cast<const __half*>(v.data_ptr<at::Half>());
    const __half* out_ptr = reinterpret_cast<const __half*>(out.data_ptr<at::Half>());
    const float* lse_ptr = softmax_lse.data_ptr<float>();
    
    __half* grad_q_ptr = reinterpret_cast<__half*>(grad_q.data_ptr<at::Half>());
    __half* grad_k_ptr = reinterpret_cast<__half*>(grad_k.data_ptr<at::Half>());
    __half* grad_v_ptr = reinterpret_cast<__half*>(grad_v.data_ptr<at::Half>());
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    flash_attention_bwd_kernel(
        grad_out_ptr, q_ptr, k_ptr, v_ptr, out_ptr, lse_ptr,
        grad_q_ptr, grad_k_ptr, grad_v_ptr,
        batch_size, num_heads, seq_len, head_dim,
        scale, causal, stream
    );
    
    return {grad_q, grad_k, grad_v};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_attention_forward, "Flash Attention forward");
    m.def("backward", &flash_attention_backward, "Flash Attention backward");
}
