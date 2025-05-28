import torch
import pytest
import math

# Attempt to import from sibling modules
try:
    from fused_attention_module import FusedAttention, CUSTOM_ATTN_AVAILABLE, MAX_HEAD_DIM_CUDA
    # If benchmark_fused_attention is in the same directory or PYTHONPATH
    from benchmark_fused_attention import scaled_dot_product_attention_pytorch, SDPA_HAS_SCALE_ARG
except ImportError as e:
    print(f"Error importing dependent modules: {e}. Some tests may not run or might fail.")
    # Define placeholders if imports fail, to allow pytest to collect the file
    CUSTOM_ATTN_AVAILABLE = False
    MAX_HEAD_DIM_CUDA = 64 # Default assumption
    SDPA_HAS_SCALE_ARG = False
    class FusedAttention:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, *args, **kwargs):
            raise RuntimeError("FusedAttention module not available.")
    def scaled_dot_product_attention_pytorch(*args, **kwargs):
        raise RuntimeError("scaled_dot_product_attention_pytorch not available.")


# Constants and Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SM_SCALE_DEFAULT = lambda head_dim: 1.0 / math.sqrt(head_dim)
TOLERANCES = {'atol': 1e-5, 'rtol': 1e-3} # For float32 comparisons
# Looser tolerances for gradients, especially with fused kernels vs non-fused
GRAD_TOLERANCES = {'atol': 1e-4, 'rtol': 5e-3} 


# Skip all tests in this file if custom kernel isn't available or CUDA is not present
pytestmark = pytest.mark.skipif(not CUSTOM_ATTN_AVAILABLE or not torch.cuda.is_available(),
                                reason="Custom FusedAttention kernel not compiled/available or CUDA not found.")

@pytest.fixture(scope="module")
def fused_attn_module():
    """Pytest fixture to provide an instance of the FusedAttention module."""
    if not CUSTOM_ATTN_AVAILABLE or not torch.cuda.is_available():
        pytest.skip("Skipping tests, FusedAttention/CUDA not available.")
    return FusedAttention().to(DEVICE)

# Test Configurations: (Batch, Heads, SeqLen, HeadDim, is_causal)
# Q_ROWS_PER_BLOCK is 16 in the CUDA kernel.
test_params = [
    (1, 1, 16, 16, False),    # Minimal, N multiple of Q_ROWS_PER_BLOCK
    (2, 2, 32, 32, True),     # N multiple of Q_ROWS_PER_BLOCK
    (1, 4, 64, 64, False),    # Max head_dim, N multiple
    (1, 2, 24, 32, True),     # N not multiple of Q_ROWS_PER_BLOCK
    (2, 1, 50, 16, False),    # N not multiple
    (1, 1, 128, 64, True),    # Larger N
]

# Test Configurations specifically for head_dim limits
head_dim_test_params = [
    (1, 1, 16, MAX_HEAD_DIM_CUDA, False), # Max allowed head_dim
]


@pytest.mark.parametrize("B, H, N, D, is_causal", test_params)
def test_forward_pass(B, H, N, D, is_causal, fused_attn_module):
    """Test the forward pass for correctness against PyTorch's sdpa."""
    q = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float32)
    k = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float32)
    v = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float32)
    sm_scale = SM_SCALE_DEFAULT(D)

    custom_output = fused_attn_module(q, k, v, sm_scale=sm_scale, is_causal=is_causal)
    
    # scaled_dot_product_attention_pytorch expects sm_scale to be passed.
    # It handles the `scale` argument of F.sdpa internally.
    ref_output = scaled_dot_product_attention_pytorch(q, k, v, sm_scale=sm_scale, is_causal=is_causal)

    torch.testing.assert_close(custom_output, ref_output, **TOLERANCES)


@pytest.mark.parametrize("B, H, N, D, is_causal", test_params)
def test_backward_pass(B, H, N, D, is_causal, fused_attn_module):
    """Test the backward pass for correctness against PyTorch's sdpa."""
    q = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float32, requires_grad=True)
    k = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float32, requires_grad=True)
    v = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float32, requires_grad=True)
    sm_scale = SM_SCALE_DEFAULT(D)

    # Custom backward
    out_custom = fused_attn_module(q, k, v, sm_scale=sm_scale, is_causal=is_causal)
    dummy_grad = torch.randn_like(out_custom, device=DEVICE)
    (out_custom * dummy_grad).sum().backward()
    custom_q_grad, custom_k_grad, custom_v_grad = q.grad.clone(), k.grad.clone(), v.grad.clone()

    # Zero gradients for reference pass
    q.grad, k.grad, v.grad = None, None, None

    # Reference backward
    out_ref = scaled_dot_product_attention_pytorch(q, k, v, sm_scale=sm_scale, is_causal=is_causal)
    (out_ref * dummy_grad).sum().backward() # Use the same dummy_grad
    ref_q_grad, ref_k_grad, ref_v_grad = q.grad.clone(), k.grad.clone(), v.grad.clone()

    torch.testing.assert_close(custom_q_grad, ref_q_grad, **GRAD_TOLERANCES)
    torch.testing.assert_close(custom_k_grad, ref_k_grad, **GRAD_TOLERANCES)
    torch.testing.assert_close(custom_v_grad, ref_v_grad, **GRAD_TOLERANCES)


@pytest.mark.parametrize("B, H, N, D, is_causal", head_dim_test_params)
def test_head_dim_max_allowed(B, H, N, D, is_causal, fused_attn_module):
    """Test with MAX_HEAD_DIM_CUDA, which should pass."""
    q = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float32)
    k = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float32)
    v = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float32)
    sm_scale = SM_SCALE_DEFAULT(D)
    
    # This should not raise an error
    try:
        _ = fused_attn_module(q, k, v, sm_scale=sm_scale, is_causal=is_causal)
    except ValueError as e:
        pytest.fail(f"Test with head_dim={D} (MAX_HEAD_DIM_CUDA) failed unexpectedly: {e}")


def test_head_dim_too_large_validation(fused_attn_module):
    """Test that head_dim > MAX_HEAD_DIM_CUDA raises ValueError."""
    B, H, N = 1, 1, 16
    D_invalid = MAX_HEAD_DIM_CUDA + 1 # e.g., 65 if MAX_HEAD_DIM_CUDA is 64
    sm_scale = SM_SCALE_DEFAULT(D_invalid)
    
    q = torch.randn(B, H, N, D_invalid, device=DEVICE, dtype=torch.float32)
    k = torch.randn(B, H, N, D_invalid, device=DEVICE, dtype=torch.float32)
    v = torch.randn(B, H, N, D_invalid, device=DEVICE, dtype=torch.float32)

    with pytest.raises(ValueError, match=f"head_dim \\({D_invalid}\\) must be <= MAX_HEAD_DIM_CUDA \\({MAX_HEAD_DIM_CUDA}\\)"):
        _ = fused_attn_module(q, k, v, sm_scale=sm_scale, is_causal=False)


def test_input_shape_validation(fused_attn_module):
    """Test validation for incorrect input tensor shapes."""
    B, H, N, D = 1, 2, 16, 32
    sm_scale = SM_SCALE_DEFAULT(D)

    # Test with 3D input instead of 4D
    q_3d = torch.randn(B, N, D, device=DEVICE, dtype=torch.float32)
    k_4d = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float32)
    v_4d = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float32)
    with pytest.raises((ValueError, RuntimeError), match="Input tensors Q, K, V must be 4-dimensional"):
        _ = fused_attn_module(q_3d, k_4d, v_4d, sm_scale=sm_scale, is_causal=False)

    # Test with mismatched head_dim
    q_d64 = torch.randn(B, H, N, 64, device=DEVICE, dtype=torch.float32)
    k_d32 = torch.randn(B, H, N, 32, device=DEVICE, dtype=torch.float32)
    v_d32 = torch.randn(B, H, N, 32, device=DEVICE, dtype=torch.float32)
    with pytest.raises((ValueError, RuntimeError), match="Q, K, V must have matching batch_size, num_heads, and head_dim"):
         _ = fused_attn_module(q_d64, k_d32, v_d32, sm_scale=SM_SCALE_DEFAULT(32), is_causal=False)
    
    # Test with mismatched num_heads
    q_h2 = torch.randn(B, 2, N, D, device=DEVICE, dtype=torch.float32)
    k_h4 = torch.randn(B, 4, N, D, device=DEVICE, dtype=torch.float32)
    v_h4 = torch.randn(B, 4, N, D, device=DEVICE, dtype=torch.float32)
    with pytest.raises((ValueError, RuntimeError), match="Q, K, V must have matching batch_size, num_heads, and head_dim"):
        _ = fused_attn_module(q_h2, k_h4, v_h4, sm_scale=sm_scale, is_causal=False)


@pytest.mark.skipif(not SDPA_HAS_SCALE_ARG, reason="PyTorch sdpa needs scale argument for this test (version >= 2.2)")
def test_sm_scale_override(fused_attn_module):
    """Test using a non-default sm_scale value."""
    B, H, N, D = 1, 2, 16, 32
    q = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float32)
    k = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float32)
    v = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float32)
    
    custom_sm_scale = 0.75 # An arbitrary non-default scale

    custom_output = fused_attn_module(q, k, v, sm_scale=custom_sm_scale, is_causal=False)
    ref_output = scaled_dot_product_attention_pytorch(q, k, v, sm_scale=custom_sm_scale, is_causal=False)
    
    torch.testing.assert_close(custom_output, ref_output, **TOLERANCES)

    # Test backward pass with custom scale
    q.requires_grad_(True); k.requires_grad_(True); v.requires_grad_(True)
    
    out_custom = fused_attn_module(q, k, v, sm_scale=custom_sm_scale, is_causal=False)
    dummy_grad = torch.randn_like(out_custom, device=DEVICE)
    (out_custom * dummy_grad).sum().backward()
    custom_q_grad, custom_k_grad, custom_v_grad = q.grad.clone(), k.grad.clone(), v.grad.clone()

    q.grad, k.grad, v.grad = None, None, None

    out_ref = scaled_dot_product_attention_pytorch(q, k, v, sm_scale=custom_sm_scale, is_causal=False)
    (out_ref * dummy_grad).sum().backward()
    ref_q_grad, ref_k_grad, ref_v_grad = q.grad.clone(), k.grad.clone(), v.grad.clone()

    torch.testing.assert_close(custom_q_grad, ref_q_grad, **GRAD_TOLERANCES)
    torch.testing.assert_close(custom_k_grad, ref_k_grad, **GRAD_TOLERANCES)
    torch.testing.assert_close(custom_v_grad, ref_v_grad, **GRAD_TOLERANCES)

# Example of how to run with pytest:
# pytest test_fused_attention.py
```
