import pytest
import torch
from flash_attention import flash_attention, _pytorch_flash_attention

def test_basic_functionality():
    """Test basic flash attention functionality."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    batch_size, num_heads, seq_len, head_dim = 2, 8, 64, 32
    device = 'cuda'
    dtype = torch.float16
    
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    
    scale = 1.0 / (head_dim ** 0.5)
    
    # Test forward pass
    out = flash_attention(q, k, v, scale, causal=True)
    
    assert out.shape == (batch_size, num_heads, seq_len, head_dim)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()

def test_causal_vs_non_causal():
    """Test that causal and non-causal produce different results."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        
    batch_size, num_heads, seq_len, head_dim = 1, 4, 32, 16
    device = 'cuda'
    dtype = torch.float16
    
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    
    scale = 1.0 / (head_dim ** 0.5)
    
    out_causal = flash_attention(q, k, v, scale, causal=True)
    out_non_causal = flash_attention(q, k, v, scale, causal=False)
    
    # Results should be different
    assert not torch.allclose(out_causal, out_non_causal, atol=1e-3)

def test_correctness_against_pytorch():
    """Test correctness against PyTorch reference."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        
    batch_size, num_heads, seq_len, head_dim = 1, 2, 64, 32
    device = 'cuda'
    dtype = torch.float16
    
    torch.manual_seed(42)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    
    scale = 1.0 / (head_dim ** 0.5)
    
    # Reference implementation
    ref_out = _pytorch_flash_attention(q, k, v, scale, causal=True)
    
    # Our implementation
    our_out = flash_attention(q, k, v, scale, causal=True)
    
    # Check that outputs are close (allowing for fp16 precision)
    torch.testing.assert_close(ref_out, our_out, atol=1e-2, rtol=1e-2)

def test_backward_pass():
    """Test that backward pass works."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        
    batch_size, num_heads, seq_len, head_dim = 1, 2, 32, 16
    device = 'cuda'
    dtype = torch.float16
    
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device, requires_grad=True)
    
    scale = 1.0 / (head_dim ** 0.5)
    
    out = flash_attention(q, k, v, scale, causal=True)
    loss = out.sum()
    loss.backward()
    
    # Check that gradients were computed
    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None
    
    # Check that gradients have the right shape
    assert q.grad.shape == q.shape
    assert k.grad.shape == k.shape
    assert v.grad.shape == v.shape

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
