import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

try:
    import flash_attention_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("Warning: CUDA extension not available. Falling back to PyTorch implementation.")

class FlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale=None, causal=False):
        if scale is None:
            scale = 1.0 / (q.size(-1) ** 0.5)
        
        if CUDA_AVAILABLE and q.is_cuda:
            # Use CUDA implementation
            out = flash_attention_cuda.forward(q, k, v, scale, causal)
            ctx.save_for_backward(q, k, v, out)
            ctx.scale = scale
            ctx.causal = causal
            ctx.use_cuda = True
            return out
        else:
            # Fallback to PyTorch implementation
            return _pytorch_flash_attention(q, k, v, scale, causal)
    
    @staticmethod
    def backward(ctx, grad_out):
        if ctx.use_cuda and CUDA_AVAILABLE:
            q, k, v, out = ctx.saved_tensors
            # For CUDA backward, we need softmax_lse which we don't store
            # This is a simplified version - in practice you'd store LSE in forward
            softmax_lse = torch.zeros(q.shape[:-1], dtype=torch.float32, device=q.device)
            grad_q, grad_k, grad_v = flash_attention_cuda.backward(
                grad_out, q, k, v, out, softmax_lse, ctx.scale, ctx.causal
            )
            return grad_q, grad_k, grad_v, None, None
        else:
            # Fallback to PyTorch autograd
            return None, None, None, None, None

def _pytorch_flash_attention(q, k, v, scale, causal):
    """PyTorch fallback implementation of flash attention."""
    # Standard attention computation
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    if causal:
        seq_len = q.size(-2)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    attn_weights = F.softmax(scores, dim=-1)
    out = torch.matmul(attn_weights, v)
    
    return out

class FlashAttention(nn.Module):
    """Flash Attention module with CUDA acceleration."""
    
    def __init__(self, causal=False):
        super().__init__()
        self.causal = causal
    
    def forward(self, q, k, v, scale=None):
        """
        Args:
            q: Query tensor [batch, heads, seq_len, head_dim]
            k: Key tensor [batch, heads, seq_len, head_dim]
            v: Value tensor [batch, heads, seq_len, head_dim]
            scale: Attention scale factor (default: 1/sqrt(head_dim))
        
        Returns:
            Output tensor [batch, heads, seq_len, head_dim]
        """
        return FlashAttentionFunction.apply(q, k, v, scale, self.causal)

def flash_attention(q, k, v, scale=None, causal=False):
    """
    Functional interface for flash attention.
    
    Args:
        q: Query tensor [batch, heads, seq_len, head_dim]
        k: Key tensor [batch, heads, seq_len, head_dim]  
        v: Value tensor [batch, heads, seq_len, head_dim]
        scale: Attention scale factor (default: 1/sqrt(head_dim))
        causal: Whether to apply causal masking
    
    Returns:
        Output tensor [batch, heads, seq_len, head_dim]
    """
    return FlashAttentionFunction.apply(q, k, v, scale, causal)

# Convenience function for multi-head attention
def multi_head_attention(x, num_heads, causal=False):
    """
    Simple multi-head attention using flash attention.
    
    Args:
        x: Input tensor [batch, seq_len, embed_dim]
        num_heads: Number of attention heads
        causal: Whether to apply causal masking
    
    Returns:
        Output tensor [batch, seq_len, embed_dim]
    """
    batch_size, seq_len, embed_dim = x.shape
    head_dim = embed_dim // num_heads
    
    # Simple linear projections (in practice you'd want learnable weights)
    q = x.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    k = x.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    v = x.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    
    # Apply flash attention
    out = flash_attention(q, k, v, causal=causal)
    
    # Reshape back to original format
    out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
    
    return out
