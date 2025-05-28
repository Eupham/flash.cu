# Test suite for Flash Attention implementations
import pytest
import torch
import numpy as np
from attention import attention as triton_attention
from cuda_attention import cuda_flash_attention, CUDA_AVAILABLE

class TestFlashAttention:
    
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @pytest.fixture
    def test_configs(self):
        """Test configurations: (batch_size, num_heads, seq_len, head_dim)"""
        return [
            (1, 2, 128, 64),
            (2, 4, 256, 64),
            (1, 8, 512, 128),
            (4, 16, 1024, 64),
        ]
    
    def reference_attention(self, q, k, v, causal=True, sm_scale=None):
        """Reference PyTorch implementation"""
        if sm_scale is None:
            sm_scale = 1.0 / (q.shape[-1] ** 0.5)
            
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
        
        # Apply causal mask if needed
        if causal:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device))
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        
        return output
    
    def generate_tensors(self, batch_size, num_heads, seq_len, head_dim, device, dtype=torch.float16):
        """Generate test tensors with specific seed for reproducibility"""
        torch.manual_seed(42)
        
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                       dtype=dtype, device=device, requires_grad=True)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                       dtype=dtype, device=device, requires_grad=True)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                       dtype=dtype, device=device, requires_grad=True)
        
        return q, k, v
    
    @pytest.mark.parametrize("batch_size,num_heads,seq_len,head_dim", [
        (1, 2, 128, 64),
        (2, 4, 256, 64),
        (1, 8, 512, 128),
    ])
    @pytest.mark.parametrize("causal", [True, False])
    def test_triton_forward_accuracy(self, batch_size, num_heads, seq_len, head_dim, causal, device):
        """Test Triton forward pass accuracy"""
        q, k, v = self.generate_tensors(batch_size, num_heads, seq_len, head_dim, device)
        sm_scale = 1.0 / (head_dim ** 0.5)
        
        # Reference output
        ref_output = self.reference_attention(q, k, v, causal, sm_scale)
        
        # Triton output
        triton_output = triton_attention(q, k, v, causal, sm_scale)
        
        # Compare outputs
        torch.testing.assert_close(ref_output, triton_output, atol=1e-2, rtol=1e-2)
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA kernels not available")
    @pytest.mark.parametrize("batch_size,num_heads,seq_len,head_dim", [
        (1, 2, 128, 64),
        (2, 4, 256, 64),
    ])
    @pytest.mark.parametrize("causal", [True])  # CUDA implementation currently only supports causal
    def test_cuda_forward_accuracy(self, batch_size, num_heads, seq_len, head_dim, causal, device):
        """Test CUDA forward pass accuracy"""
        q, k, v = self.generate_tensors(batch_size, num_heads, seq_len, head_dim, device)
        sm_scale = 1.0 / (head_dim ** 0.5)
        
        # Reference output
        ref_output = self.reference_attention(q, k, v, causal, sm_scale)
        
        # CUDA output
        cuda_output = cuda_flash_attention(q, k, v, causal, sm_scale)
        
        # Compare outputs
        torch.testing.assert_close(ref_output, cuda_output, atol=1e-1, rtol=1e-1)  # Relaxed tolerance for CUDA
    
    @pytest.mark.parametrize("batch_size,num_heads,seq_len,head_dim", [
        (1, 2, 128, 64),
        (2, 4, 256, 64),
    ])
    def test_triton_backward_accuracy(self, batch_size, num_heads, seq_len, head_dim, device):
        """Test Triton backward pass accuracy"""
        q, k, v = self.generate_tensors(batch_size, num_heads, seq_len, head_dim, device)
        sm_scale = 1.0 / (head_dim ** 0.5)
        
        # Reference gradients
        q_ref = q.clone().detach().requires_grad_(True)
        k_ref = k.clone().detach().requires_grad_(True)
        v_ref = v.clone().detach().requires_grad_(True)
        
        ref_output = self.reference_attention(q_ref, k_ref, v_ref, True, sm_scale)
        grad_output = torch.randn_like(ref_output)
        ref_output.backward(grad_output)
        
        ref_dq, ref_dk, ref_dv = q_ref.grad, k_ref.grad, v_ref.grad
        
        # Triton gradients
        q_tri = q.clone().detach().requires_grad_(True)
        k_tri = k.clone().detach().requires_grad_(True)
        v_tri = v.clone().detach().requires_grad_(True)
        
        tri_output = triton_attention(q_tri, k_tri, v_tri, True, sm_scale)
        tri_output.backward(grad_output)
        
        tri_dq, tri_dk, tri_dv = q_tri.grad, k_tri.grad, v_tri.grad
        
        # Compare gradients
        torch.testing.assert_close(ref_dq, tri_dq, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(ref_dk, tri_dk, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(ref_dv, tri_dv, atol=1e-2, rtol=1e-2)
    
    def test_shape_consistency(self, device):
        """Test that output shapes are consistent across implementations"""
        batch_size, num_heads, seq_len, head_dim = 2, 4, 256, 64
        q, k, v = self.generate_tensors(batch_size, num_heads, seq_len, head_dim, device)
        sm_scale = 1.0 / (head_dim ** 0.5)
        
        # Test Triton
        triton_output = triton_attention(q, k, v, True, sm_scale)
        assert triton_output.shape == q.shape, f"Triton output shape mismatch: {triton_output.shape} vs {q.shape}"
        
        # Test CUDA (if available)
        if CUDA_AVAILABLE:
            cuda_output = cuda_flash_attention(q, k, v, True, sm_scale)
            assert cuda_output.shape == q.shape, f"CUDA output shape mismatch: {cuda_output.shape} vs {q.shape}"
    
    def test_gradient_flow(self, device):
        """Test that gradients flow correctly"""
        batch_size, num_heads, seq_len, head_dim = 1, 2, 128, 64
        q, k, v = self.generate_tensors(batch_size, num_heads, seq_len, head_dim, device)
        sm_scale = 1.0 / (head_dim ** 0.5)
        
        # Test Triton gradient flow
        q.grad = None
        k.grad = None
        v.grad = None
        
        output = triton_attention(q, k, v, True, sm_scale)
        loss = output.sum()
        loss.backward()
        
        assert q.grad is not None, "q.grad is None"
        assert k.grad is not None, "k.grad is None"
        assert v.grad is not None, "v.grad is None"
        
        # Check that gradients are not zero
        assert not torch.allclose(q.grad, torch.zeros_like(q.grad)), "q.grad is all zeros"
        assert not torch.allclose(k.grad, torch.zeros_like(k.grad)), "k.grad is all zeros"
        assert not torch.allclose(v.grad, torch.zeros_like(v.grad)), "v.grad is all zeros"
    
    def test_numerical_stability(self, device):
        """Test numerical stability with extreme values"""
        batch_size, num_heads, seq_len, head_dim = 1, 2, 64, 64
        
        # Test with large values
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device) * 10
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device) * 10
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device) * 10
        
        sm_scale = 1.0 / (head_dim ** 0.5)
        
        # Should not produce NaN or Inf
        output = triton_attention(q, k, v, True, sm_scale)
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"
    
    @pytest.mark.parametrize("seq_len", [64, 128, 256, 512, 1024])
    def test_sequence_length_scaling(self, seq_len, device):
        """Test that implementation works with different sequence lengths"""
        batch_size, num_heads, head_dim = 1, 4, 64
        q, k, v = self.generate_tensors(batch_size, num_heads, seq_len, head_dim, device)
        sm_scale = 1.0 / (head_dim ** 0.5)
        
        try:
            output = triton_attention(q, k, v, True, sm_scale)
            assert output.shape == q.shape
        except Exception as e:
            pytest.fail(f"Failed for seq_len={seq_len}: {e}")

if __name__ == "__main__":
    # Run tests directly
    import sys
    pytest.main([__file__] + sys.argv[1:])
