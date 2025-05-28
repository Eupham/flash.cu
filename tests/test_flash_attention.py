"""
Unit Tests for FlashAttention Implementation

This test suite verifies the correctness of the custom FlashAttention module by:
1.  Comparing its forward pass output with a manual PyTorch-based Multihead Attention (MHA) implementation.
2.  Performing gradient checks for the custom autograd function (`_FlashAttentionFunction`) using `torch.autograd.gradcheck`.
3.  Comparing the gradients computed by the custom FlashAttention module's backward pass
    with those from the manual MHA reference.

Tests cover various configurations including different batch sizes, sequence lengths,
number of heads, head dimensions, and causal/non-causal attention.
They are designed to run on a CUDA-enabled device.
"""
import unittest
import torch
import os
import sys
import math
from torch.autograd import gradcheck

# Add the project root to the Python path to allow importing flash_attn module
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

try:
    # Attempt to import the FlashAttention module and its autograd function
    from flash_attn.flash_attention import FlashAttention, _FlashAttentionFunction
    FLASH_ATTENTION_AVAILABLE = True
except ImportError as e:
    # If import fails, likely CUDA extension is not compiled or not found
    print(f"Could not import FlashAttention module: {e}. CUDA extension might not be compiled or installed.")
    FlashAttention = None # type: ignore
    _FlashAttentionFunction = None # type: ignore
    FLASH_ATTENTION_AVAILABLE = False

def manual_mha(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, head_dim: int, causal: bool = False) -> torch.Tensor:
    """
    Manually computes Multihead Attention using PyTorch operations.

    This function serves as a reference implementation to validate the output
    of the custom FlashAttention kernel.

    Args:
        Q (torch.Tensor): Query tensor of shape (batch_size, num_heads, q_seq_len, head_dim).
        K (torch.Tensor): Key tensor of shape (batch_size, num_heads, kv_seq_len, head_dim).
        V (torch.Tensor): Value tensor of shape (batch_size, num_heads, kv_seq_len, head_dim).
        head_dim (int): The dimension of each attention head.
        causal (bool, optional): If True, applies causal masking. Defaults to False.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, num_heads, q_seq_len, head_dim).
    """
    batch_size, num_heads, q_seq_len, _ = Q.shape
    kv_seq_len = K.shape[2]
    device = Q.device
    dtype = Q.dtype # Preserve input dtype for calculations

    # Calculate scaling factor
    scale = head_dim ** -0.5
    # Compute attention scores: Q @ K.T
    # Ensure K.transpose is of the same dtype as Q before matmul for mixed precision safety (though not used here)
    scores = torch.matmul(Q, K.transpose(-2, -1).to(dtype)) * scale

    if causal:
        # Create a causal mask: upper triangle of 1s (True means mask out)
        # Mask shape: (q_seq_len, kv_seq_len)
        causal_mask = torch.triu(torch.ones(q_seq_len, kv_seq_len, device=device, dtype=torch.bool), diagonal=1)
        # Apply mask by filling masked positions with -infinity before softmax
        # Expand mask to (1, 1, q_seq_len, kv_seq_len) for broadcasting over batch and heads
        scores.masked_fill_(causal_mask[None, None, :, :], float('-inf'))

    # Apply softmax to get attention weights
    attn_weights = torch.softmax(scores, dim=-1)
    # Compute output: AttentionWeights @ V
    # Ensure attn_weights is same dtype as V before matmul
    output_ref = torch.matmul(attn_weights.to(dtype), V)
    return output_ref

@unittest.skipIf(not FLASH_ATTENTION_AVAILABLE, "FlashAttention CUDA module not available or not compiled. Skipping tests.")
class TestFlashAttention(unittest.TestCase):
    """
    Test class for FlashAttention forward and backward passes.

    This class groups tests for numerical correctness of the forward pass output
    and the computed gradients from the backward pass. It also includes a structure
    for `gradcheck`.
    """

    # Tolerances for floating point comparisons, keyed by dtype
    TOLERANCES = {
        torch.float32: {"atol": 1e-5, "rtol": 1e-3},
        torch.float16: {"atol": 1e-3, "rtol": 1e-2}, # FP16 requires higher tolerances
        torch.float64: {"atol": 1e-7, "rtol": 1e-5}, # For gradcheck if using FP64 kernels
    }
    # Tolerances for gradient comparisons (can be looser than forward pass)
    GRAD_TOLERANCES = {
        torch.float32: {"atol": 1e-4, "rtol": 1e-2},
        torch.float16: {"atol": 1e-2, "rtol": 1e-1}, # FP16 grads can be less precise
    }

    def _get_tolerances(self, dtype: torch.dtype, grad_comparison: bool = False) -> dict:
        """Helper to retrieve appropriate tolerances based on dtype and comparison type."""
        source_tolerances = self.GRAD_TOLERANCES if grad_comparison else self.TOLERANCES
        return source_tolerances.get(dtype, {"atol": 1e-5, "rtol": 1e-3}) # Default if dtype not in map

    def _run_and_compare_forward(self, batch_size: int, num_heads: int, q_seq_len: int, kv_seq_len: int, 
                                 head_dim: int, causal: bool, dtype: torch.dtype = torch.float32, 
                                 device: str = 'cuda'):
        """
        Helper method to run forward pass for FlashAttention and manual MHA, then compare outputs.

        Args:
            batch_size, num_heads, q_seq_len, kv_seq_len, head_dim, causal: Configuration parameters.
            dtype: Data type for tensors (e.g., torch.float32).
            device: Device to run tensors on (e.g., 'cuda').
        """
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available, skipping test.")
        
        # Check if the head_dim is supported by the compiled CUDA kernels
        supported_head_dims = [32, 64, 128] # Based on current kernel template specializations
        if head_dim not in supported_head_dims:
            self.skipTest(f"head_dim={head_dim} not in supported list {supported_head_dims} for compiled kernel. Skipping.")

        # Current CUDA kernels are FP32 only. Skip FP16 tests for FlashAttention.
        if dtype == torch.float16:
             self.skipTest("FP16 tests are skipped as current CUDA kernel is FP32 only.")

        torch.manual_seed(0) # For reproducible random tensor generation
        
        # Create random input tensors
        Q = torch.randn(batch_size, num_heads, q_seq_len, head_dim, device=device, dtype=dtype)
        K = torch.randn(batch_size, num_heads, kv_seq_len, head_dim, device=device, dtype=dtype)
        V = torch.randn(batch_size, num_heads, kv_seq_len, head_dim, device=device, dtype=dtype)

        # Instantiate custom FlashAttention module
        flash_attn_custom = FlashAttention(head_dim=head_dim, causal=causal)
        # Run forward pass of custom FlashAttention
        output_custom = flash_attn_custom(Q.clone(), K.clone(), V.clone()) # Use .clone() to avoid in-place issues if any

        # Run reference manual MHA implementation
        output_ref = manual_mha(Q.clone(), K.clone(), V.clone(), head_dim, causal=causal)
        
        # Assert that shapes match
        self.assertTrue(output_custom.shape == output_ref.shape,
                        f"Shape mismatch: Custom {output_custom.shape}, Reference {output_ref.shape}")
        
        # Assert that outputs are numerically close within specified tolerances
        tolerances = self._get_tolerances(dtype)
        self.assertTrue(torch.allclose(output_custom, output_ref, **tolerances),
                        f"Output mismatch for dtype {dtype}. Max diff: {torch.max(torch.abs(output_custom - output_ref))}")
        
        print(f"Forward test passed: B={batch_size}, H={num_heads}, Q_S={q_seq_len}, KV_S={kv_seq_len}, D={head_dim}, causal={causal}, dtype={dtype}. Max diff: {torch.max(torch.abs(output_custom - output_ref)):.2e}")

    # --- Forward Pass Test Cases ---
    def test_forward_fp32_non_causal_eq_seqlen_d64(self):
        """Test forward pass: FP32, non-causal, equal seq_len, head_dim=64."""
        self._run_and_compare_forward(2, 3, 128, 128, 64, False, dtype=torch.float32)

    def test_forward_fp32_causal_eq_seqlen_d32(self):
        """Test forward pass: FP32, causal, equal seq_len, head_dim=32."""
        self._run_and_compare_forward(2, 2, 64, 64, 32, True, dtype=torch.float32)

    def test_forward_fp32_non_causal_neq_seqlen_d128(self):
        """Test forward pass: FP32, non-causal, unequal seq_len, head_dim=128."""
        self._run_and_compare_forward(1, 2, 128, 64, 128, False, dtype=torch.float32) # Q_seq_len > KV_seq_len
        self._run_and_compare_forward(1, 2, 64, 128, 128, False, dtype=torch.float32) # Q_seq_len < KV_seq_len

    @unittest.skip("FP16 forward tests skipped: Current CUDA kernel is FP32 only.")
    def test_forward_fp16_non_causal_eq_seqlen_d64(self):
        """Placeholder for FP16 forward pass test (currently skipped)."""
        self._run_and_compare_forward(2, 3, 128, 128, 64, False, dtype=torch.float16)


    # --- Gradcheck Test Cases ---
    @unittest.skip("Gradcheck tests skipped: Current CUDA kernel is FP32 only. Gradcheck requires FP64 support in kernel or very careful FP32 setup and loose tolerances.")
    def test_gradcheck_non_causal_d32(self):
        """Test gradients using torch.autograd.gradcheck (currently skipped)."""
        if not torch.cuda.is_available(): self.skipTest("CUDA not available.")
        # This test requires CUDA kernels to support float64 for reliable gradcheck.
        # Our current kernels are float32. This test is structured but expected to fail or need modification.
        
        # Small dimensions for gradcheck to run faster
        bs, nhead, q_sl, kv_sl, hdim = 1, 1, 4, 4, 32 
        
        # Gradcheck ideally needs double precision (torch.float64)
        # Using float32 here for structure, but it's less reliable for gradcheck.
        dtype_check = torch.float32 
        
        Q_gc = torch.randn(bs, nhead, q_sl, hdim, device='cuda', dtype=dtype_check, requires_grad=True)
        K_gc = torch.randn(bs, nhead, kv_sl, hdim, device='cuda', dtype=dtype_check, requires_grad=True)
        V_gc = torch.randn(bs, nhead, kv_sl, hdim, device='cuda', dtype=dtype_check, requires_grad=True)
        
        # Non-tensor inputs for _FlashAttentionFunction.apply
        head_dim_val = hdim
        is_causal_val = False
        sm_scale_val = 1.0 / math.sqrt(float(hdim))

        # Inputs tuple for gradcheck
        inputs_for_gradcheck = (Q_gc, K_gc, V_gc, head_dim_val, is_causal_val, sm_scale_val)
        
        # Perform gradcheck. Tolerances might need to be very loose for FP32.
        # `check_undefined_grad=True` is good practice.
        # `fast_mode=False` for more thorough checks (but slower).
        test_passed = gradcheck(_FlashAttentionFunction.apply, inputs_for_gradcheck, 
                                atol=1e-3, rtol=1e-2, # Example loose FP32 tolerances
                                check_undefined_grad=True, fast_mode=True) 
        self.assertTrue(test_passed, "Gradcheck failed for non-causal d32 configuration.")


    # --- Manual Gradient Comparison Test Cases ---
    def _run_and_compare_backward(self, batch_size: int, num_heads: int, q_seq_len: int, kv_seq_len: int, 
                                  head_dim: int, causal: bool, dtype: torch.dtype = torch.float32, 
                                  device: str = 'cuda'):
        """
        Helper method to compare gradients from FlashAttention with manual MHA.

        Args:
            batch_size, num_heads, q_seq_len, kv_seq_len, head_dim, causal: Configuration parameters.
            dtype: Data type for tensors.
            device: Device for tensors.
        """
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available, skipping test.")

        supported_head_dims = [32, 64, 128]
        if head_dim not in supported_head_dims:
            self.skipTest(f"head_dim={head_dim} not in supported list {supported_head_dims}. Skipping.")

        if dtype == torch.float16: # Current CUDA kernel is FP32 only.
            self.skipTest("FP16 backward tests are skipped as current CUDA kernel is FP32 only.")

        torch.manual_seed(123) # Use a different seed than forward pass tests for variety

        # Setup tensors for custom FlashAttention path
        Q_cust = torch.randn(batch_size, num_heads, q_seq_len, head_dim, device=device, dtype=dtype, requires_grad=True)
        K_cust = torch.randn(batch_size, num_heads, kv_seq_len, head_dim, device=device, dtype=dtype, requires_grad=True)
        V_cust = torch.randn(batch_size, num_heads, kv_seq_len, head_dim, device=device, dtype=dtype, requires_grad=True)

        # Setup tensors for reference manual MHA path (clone and detach for separate graph)
        Q_ref = Q_cust.clone().detach().requires_grad_(True)
        K_ref = K_cust.clone().detach().requires_grad_(True)
        V_ref = V_cust.clone().detach().requires_grad_(True)

        # --- Custom FlashAttention Backward Pass ---
        flash_attn_custom_module = FlashAttention(head_dim=head_dim, causal=causal)
        output_custom = flash_attn_custom_module(Q_cust, K_cust, V_cust)
        # Create a scalar loss for backward pass (e.g., sum of outputs)
        loss_custom = output_custom.sum() 
        loss_custom.backward()

        # --- Reference Manual MHA Backward Pass ---
        output_ref = manual_mha(Q_ref, K_ref, V_ref, head_dim, causal=causal)
        loss_ref = output_ref.sum()
        loss_ref.backward()

        # Get appropriate tolerances for gradient comparison
        tolerances = self._get_tolerances(dtype, grad_comparison=True)

        # Compare gradients for Q, K, V
        self.assertTrue(torch.allclose(Q_cust.grad, Q_ref.grad, **tolerances),
                        f"dQ mismatch for dtype {dtype}. Max diff: {torch.max(torch.abs(Q_cust.grad - Q_ref.grad)) if Q_cust.grad is not None and Q_ref.grad is not None else 'N/A'}")
        self.assertTrue(torch.allclose(K_cust.grad, K_ref.grad, **tolerances),
                        f"dK mismatch for dtype {dtype}. Max diff: {torch.max(torch.abs(K_cust.grad - K_ref.grad)) if K_cust.grad is not None and K_ref.grad is not None else 'N/A'}")
        self.assertTrue(torch.allclose(V_cust.grad, V_ref.grad, **tolerances),
                        f"dV mismatch for dtype {dtype}. Max diff: {torch.max(torch.abs(V_cust.grad - V_ref.grad)) if V_cust.grad is not None and V_ref.grad is not None else 'N/A'}")
        
        max_dq_diff = torch.max(torch.abs(Q_cust.grad - Q_ref.grad)) if Q_cust.grad is not None and Q_ref.grad is not None else -1
        print(f"Backward test passed: B={batch_size}, H={num_heads}, Q_S={q_seq_len}, KV_S={kv_seq_len}, D={head_dim}, causal={causal}, dtype={dtype}. Max dQ diff: {max_dq_diff:.2e}")

    # --- Backward Pass Gradient Comparison Test Cases ---
    def test_backward_fp32_non_causal_eq_seqlen_d32(self):
        """Test backward pass grads: FP32, non-causal, equal seq_len, head_dim=32."""
        self._run_and_compare_backward(1, 1, 16, 16, 32, False, dtype=torch.float32)
        self._run_and_compare_backward(2, 2, 32, 32, 32, False, dtype=torch.float32) # Slightly larger

    def test_backward_fp32_causal_eq_seqlen_d64(self):
        """Test backward pass grads: FP32, causal, equal seq_len, head_dim=64."""
        self._run_and_compare_backward(1, 1, 16, 16, 64, True, dtype=torch.float32)
        self._run_and_compare_backward(2, 2, 32, 32, 64, True, dtype=torch.float32)

    def test_backward_fp32_non_causal_neq_seqlen_d32(self):
        """Test backward pass grads: FP32, non-causal, unequal seq_len, head_dim=32."""
        self._run_and_compare_backward(1, 1, 16, 32, 32, False, dtype=torch.float32) # Q_seq_len < KV_seq_len
        self._run_and_compare_backward(1, 1, 32, 16, 32, False, dtype=torch.float32) # Q_seq_len > KV_seq_len
    
    def test_backward_fp32_causal_neq_seqlen_d64(self):
        """Test backward pass grads: FP32, causal, unequal seq_len, head_dim=64."""
        self._run_and_compare_backward(1, 1, 16, 32, 64, True, dtype=torch.float32)
        self._run_and_compare_backward(1, 1, 32, 16, 64, True, dtype=torch.float32)

    @unittest.skip("FP16 backward tests skipped: Current CUDA kernel is FP32 only.")
    def test_backward_fp16_non_causal_eq_seqlen_d64(self):
        """Placeholder for FP16 backward pass gradient test (currently skipped)."""
        self._run_and_compare_backward(1, 1, 16, 16, 64, False, dtype=torch.float16)


if __name__ == '__main__':
    # This allows running the tests directly from the script.
    if not FLASH_ATTENTION_AVAILABLE:
        print("Skipping FlashAttention tests as the CUDA module ('flash_attn_cuda_lib') is not available.")
        print("Please ensure the module is compiled, e.g., 'python setup.py build_ext --inplace'.")
        sys.exit(0) # Exit without running tests if module not found
    
    if not torch.cuda.is_available():
        print("Skipping FlashAttention tests as CUDA is not available on this system.")
        sys.exit(0) # Exit if CUDA not available
        
    print("Running FlashAttention tests (forward pass numerical comparisons and backward pass gradient comparisons)...")
    unittest.main()
