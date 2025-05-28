# Python wrapper for CUDA Flash Attention kernels
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os
import time
import numpy as np

# Compile CUDA kernels
def load_cuda_kernels():
    cuda_dir = os.path.dirname(os.path.abspath(__file__))
    cuda_sources = [os.path.join(cuda_dir, "flash_attention_cuda.cu")]
    
    return load(
        name="flash_attention_cuda",
        sources=cuda_sources,
        extra_cuda_cflags=["-O3", "-use_fast_math", "--expt-relaxed-constexpr"],
        verbose=True
    )

# Load CUDA kernels (will compile on first import)
try:
    flash_cuda = load_cuda_kernels()
    CUDA_AVAILABLE = True
except Exception as e:
    print(f"Failed to load CUDA kernels: {e}")
    CUDA_AVAILABLE = False

class CudaFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal=True, sm_scale=None):
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA kernels not available")
        
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        if sm_scale is None:
            sm_scale = 1.0 / (head_dim ** 0.5)
        
        # Ensure inputs are contiguous and half precision
        q = q.contiguous().half()
        k = k.contiguous().half()
        v = v.contiguous().half()
        
        # Allocate output tensors
        o = torch.empty_like(q)
        m = torch.empty((batch_size, num_heads, seq_len), dtype=torch.float32, device=q.device)
        
        # Launch CUDA kernel
        flash_cuda.launch_flash_attention_forward(
            q, k, v, o, m,
            batch_size, num_heads, seq_len, head_dim,
            sm_scale, causal,
            torch.cuda.current_stream().cuda_stream
        )
        
        # Save for backward
        ctx.save_for_backward(q, k, v, o, m)
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        
        return o
    
    @staticmethod
    def backward(ctx, do):
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA kernels not available")
            
        q, k, v, o, m = ctx.saved_tensors
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # Ensure grad_output is contiguous
        do = do.contiguous().half()
        
        # Compute delta = sum(o * do, dim=-1)
        delta = torch.sum(o * do, dim=-1, dtype=torch.float32)
        
        # Allocate gradient tensors
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        
        # Launch CUDA backward kernel
        flash_cuda.launch_flash_attention_backward(
            q, k, v, do, m, delta, dq, dk, dv,
            batch_size, num_heads, seq_len, head_dim,
            ctx.sm_scale, ctx.causal,
            torch.cuda.current_stream().cuda_stream
        )
        
        return dq, dk, dv, None, None

def cuda_flash_attention(q, k, v, causal=True, sm_scale=None):
    """
    CUDA implementation of Flash Attention
    
    Args:
        q, k, v: Input tensors of shape (batch, heads, seq_len, head_dim)
        causal: Whether to apply causal masking
        sm_scale: Scaling factor for attention scores
    
    Returns:
        Output tensor of same shape as q
    """
    return CudaFlashAttention.apply(q, k, v, causal, sm_scale)
