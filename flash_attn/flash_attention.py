"""
FlashAttention PyTorch Module

This module provides a PyTorch interface for a custom FlashAttention implementation
that leverages CUDA kernels for an efficient, memory-saving exact attention mechanism.
"""

import torch
import torch.nn as nn
import math

# Attempt to load the compiled CUDA extension
try:
    import flash_attn_cuda_lib
except ImportError:
    print("Failed to import CUDA extension 'flash_attn_cuda_lib'.")
    print("Ensure the extension has been compiled correctly (e.g., via 'python setup.py build_ext --inplace' or 'python setup.py install').")
    flash_attn_cuda_lib = None

class _FlashAttentionFunction(torch.autograd.Function):
    """
    PyTorch autograd Function for FlashAttention.

    This class defines the forward and backward passes for FlashAttention,
    linking the PyTorch operations with the custom CUDA kernels.
    The forward pass computes the attention output and saves necessary tensors
    for the backward pass. The backward pass computes gradients with respect
    to Q, K, and V.
    """
    @staticmethod
    def forward(ctx, q, k, v, head_dim, is_causal, sm_scale):
        """
        Forward pass for FlashAttention.

        Args:
            ctx: Context object for saving tensors and parameters for backward pass.
            q (torch.Tensor): Query tensor of shape (batch_size, num_heads, q_seq_len, head_dim).
            k (torch.Tensor): Key tensor of shape (batch_size, num_heads, kv_seq_len, head_dim).
            v (torch.Tensor): Value tensor of shape (batch_size, num_heads, kv_seq_len, head_dim).
            head_dim (int): The dimension of each attention head.
            is_causal (bool): If True, applies causal masking.
            sm_scale (float): Scaling factor for attention scores (1.0 / sqrt(head_dim)).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_heads, q_seq_len, head_dim).
        """
        # Ensure contiguous inputs for CUDA kernel
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        batch_size, num_heads, q_seq_len, _ = q.shape # Get dimensions from Q
        
        # Initialize output tensor 'o' with the same shape as Q
        o = torch.empty_like(q) 
        # Initialize logsumexp tensor 'L' for backward pass stability and correctness
        # Shape: (batch_size, num_heads, q_seq_len)
        L = torch.empty((batch_size, num_heads, q_seq_len), device=q.device, dtype=torch.float32)

        if flash_attn_cuda_lib is None:
            raise RuntimeError("FlashAttention CUDA extension not loaded. Cannot proceed with forward pass.")

        # Call the forward pass CUDA kernel
        flash_attn_cuda_lib.forward(
            q, k, v,    # Input tensors
            o, L,       # Output tensors
            is_causal, sm_scale # Parameters
        )

        # Save tensors and parameters needed for the backward pass
        ctx.save_for_backward(q, k, v, o, L)
        ctx.is_causal = is_causal
        ctx.sm_scale = sm_scale
        # head_dim is passed as an argument, but not typically saved on ctx if derivable or passed again.
        # However, it's good practice if backward kernel might need it and it's not easily part of tensor shapes.
        # In this case, head_dim is implicit in Q, K, V shapes.

        return o

    @staticmethod
    def backward(ctx, do):
        """
        Backward pass for FlashAttention.

        Args:
            ctx: Context object with saved tensors and parameters from forward pass.
            do (torch.Tensor): Gradient of the output tensor 'o', 
                               shape (batch_size, num_heads, q_seq_len, head_dim).

        Returns:
            Tuple[torch.Tensor, ...]: Gradients with respect to inputs of the forward function:
                                      (dQ, dK, dV, None_for_head_dim, None_for_is_causal, None_for_sm_scale).
        """
        # Retrieve saved tensors and parameters
        q, k, v, o, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        sm_scale = ctx.sm_scale
        
        # Ensure contiguous grad_output tensor 'do'
        do = do.contiguous()

        # Initialize gradient tensors for Q, K, V with zeros.
        # This is crucial because the backward CUDA kernel might use atomicAdd for dK and dV,
        # which requires initial zero values. dQ is written directly in the current kernel.
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        
        if flash_attn_cuda_lib is None:
            raise RuntimeError("FlashAttention CUDA extension not loaded. Cannot proceed with backward pass.")

        # Call the backward pass CUDA kernel
        flash_attn_cuda_lib.backward(
            q, k, v, o, do, L,  # Input tensors (Q, K, V, O, dO, L)
            dq, dk, dv,         # Output gradient tensors (dQ, dK, dV)
            is_causal, sm_scale # Parameters
        )

        # Return gradients in the same order as inputs to the forward function.
        # Non-tensor inputs (head_dim, is_causal, sm_scale) receive None as their gradient.
        return dq, dk, dv, None, None, None


class FlashAttention(nn.Module):
    """
    FlashAttention PyTorch Module.

    This module implements the FlashAttention mechanism, which provides an exact
    attention computation with significantly reduced memory usage compared to
    standard attention implementations, especially for long sequences. It achieves
    this by using tiling and on-the-fly softmax calculation in custom CUDA kernels.

    Args:
        head_dim (int): The dimension of each attention head. This implementation
                        supports specific head dimensions (e.g., 32, 64, 128)
                        for which CUDA kernels are compiled.
        causal (bool, optional): If True, applies causal masking to the attention scores,
                                 preventing attention to future tokens. Defaults to False.

    Example:
        >>> import torch
        >>> from flash_attn import FlashAttention # If installed
        >>>
        >>> B, H, S, D = 2, 4, 64, 32 # Batch, NumHeads, SeqLen, HeadDim
        >>> q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float32)
        >>> k = torch.randn(B, H, S, D, device='cuda', dtype=torch.float32)
        >>> v = torch.randn(B, H, S, D, device='cuda', dtype=torch.float32)
        >>>
        >>> flash_attn_op = FlashAttention(head_dim=D, causal=False)
        >>> output = flash_attn_op(q, k, v)
        >>> print(output.shape)
        torch.Size([2, 4, 64, 32])
    """
    def __init__(self, head_dim: int, causal: bool = False):
        """
        Initializes the FlashAttention module.

        Args:
            head_dim (int): Dimension of each attention head.
            causal (bool, optional): Whether to apply causal masking. Defaults to False.
        """
        super().__init__()
        if flash_attn_cuda_lib is None:
            # This check ensures that the module cannot be instantiated if the CUDA extension failed to load.
            raise RuntimeError("FlashAttention CUDA extension ('flash_attn_cuda_lib') not loaded. Cannot initialize FlashAttention module.")
        
        self.head_dim = head_dim
        self.causal = causal
        # Pre-calculate the scaling factor for attention scores
        self.sm_scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the FlashAttention module.

        Args:
            Q (torch.Tensor): Query tensor of shape (batch_size, num_heads, Q_seq_len, head_dim).
            K (torch.Tensor): Key tensor of shape (batch_size, num_heads, KV_seq_len, head_dim).
            V (torch.Tensor): Value tensor of shape (batch_size, num_heads, KV_seq_len, head_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_heads, Q_seq_len, head_dim).
        
        Raises:
            ValueError: If input tensors are not on CUDA, have incorrect dimensions,
                        or if head_dim does not match the initialized `self.head_dim`.
            RuntimeError: If the CUDA extension is not available.
        """

        # --- Input Validation ---
        if not all(t.is_cuda for t in [Q, K, V]):
            raise ValueError("Input tensors Q, K, V must all be CUDA tensors.")
        
        if not (Q.device == K.device == V.device): # Check for same CUDA device
            raise ValueError("Input tensors Q, K, V must be on the same CUDA device.")

        for tensor_name, tensor_val in [("Q", Q), ("K", K), ("V", V)]:
            if tensor_val.dim() != 4:
                raise ValueError(f"Tensor {tensor_name} must be a 4D tensor, but got {tensor_val.dim()} dimensions.")
            if tensor_val.shape[-1] != self.head_dim:
                raise ValueError(f"Tensor {tensor_name}'s last dimension (head_dim) must be {self.head_dim} (initialized), but got {tensor_val.shape[-1]}.")
        
        # Ensure correct data type (currently float32 for CUDA kernels)
        expected_dtype = torch.float32
        if Q.dtype != expected_dtype or K.dtype != expected_dtype or V.dtype != expected_dtype:
            # For simplicity, this implementation expects users to provide tensors of the correct type.
            # Alternatively, one could automatically cast them here, e.g.:
            # Q = Q.to(expected_dtype)
            # K = K.to(expected_dtype)
            # V = V.to(expected_dtype)
            # However, explicit casting by the user or in a higher-level model is often preferred.
            raise TypeError(f"Input tensors Q, K, V must have dtype {expected_dtype}, but got Q:{Q.dtype}, K:{K.dtype}, V:{V.dtype}. Please cast them explicitly.")
            
        # Call the autograd function to execute the custom CUDA kernels
        return _FlashAttentionFunction.apply(Q, K, V, self.head_dim, self.causal, self.sm_scale)

# Example Usage (primarily for demonstration and basic testing if run directly)
if __name__ == '__main__':
    # This block is for demonstration purposes and basic checks.
    # For comprehensive tests, run 'python -m unittest tests.test_flash_attention'.
    # For benchmarks, run 'python benchmarks/benchmark_flash_attention.py'.

    if flash_attn_cuda_lib is None:
        print("CUDA extension 'flash_attn_cuda_lib' not loaded. Skipping example usage in flash_attention.py.")
    elif not torch.cuda.is_available():
        print("CUDA is not available. Skipping example usage in flash_attention.py.")
    else:
        print("FlashAttention CUDA extension loaded and CUDA available. Running example...")
        
        # Example parameters
        batch_size, num_heads, q_seq_len, kv_seq_len, head_dim_example = 1, 1, 16, 16, 32 
        
        # Ensure head_dim is one supported by the compiled kernels (e.g., 32, 64, 128)
        if head_dim_example not in [32, 64, 128]: # Match kernel specializations
            print(f"Warning: Example head_dim {head_dim_example} might not be optimally supported by compiled kernels. Adjust if necessary.")

        # Create example tensors
        q_example = torch.randn(batch_size, num_heads, q_seq_len, head_dim_example, device='cuda', dtype=torch.float32, requires_grad=True)
        k_example = torch.randn(batch_size, num_heads, kv_seq_len, head_dim_example, device='cuda', dtype=torch.float32, requires_grad=True)
        v_example = torch.randn(batch_size, num_heads, kv_seq_len, head_dim_example, device='cuda', dtype=torch.float32, requires_grad=True)

        # Test non-causal FlashAttention
        print("\nTesting non-causal FlashAttention:")
        try:
            flash_attn_op_non_causal = FlashAttention(head_dim=head_dim_example, causal=False)
            output_non_causal = flash_attn_op_non_causal(q_example, k_example, v_example)
            print("Forward pass (non-causal) successful. Output shape:", output_non_causal.shape)
            
            # Basic backward pass test
            grad_output_example = torch.randn_like(output_non_causal)
            output_non_causal.backward(grad_output_example)
            print("Backward pass (non-causal) successful.")
            if q_example.grad is not None:
                print("q_example.grad shape:", q_example.grad.shape)
            else:
                print("q_example.grad is None (unexpected for this test setup).")

        except Exception as e:
            print(f"Error during non-causal FlashAttention example: {e}")
            import traceback
            traceback.print_exc()

        # Reset gradients for the causal test
        q_example.grad, k_example.grad, v_example.grad = None, None, None

        # Test causal FlashAttention
        print("\nTesting causal FlashAttention:")
        try:
            flash_attn_op_causal = FlashAttention(head_dim=head_dim_example, causal=True)
            output_causal = flash_attn_op_causal(q_example, k_example, v_example)
            print("Forward pass (causal) successful. Output shape:", output_causal.shape)

            grad_output_causal_example = torch.randn_like(output_causal)
            output_causal.backward(grad_output_causal_example)
            print("Backward pass (causal) successful.")
            if q_example.grad is not None:
                 print("q_example.grad shape (causal):", q_example.grad.shape)
            else:
                print("q_example.grad (causal) is None (unexpected for this test setup).")
        except Exception as e:
            print(f"Error during causal FlashAttention example: {e}")
            import traceback
            traceback.print_exc()
        
        print("\nNote: For rigorous gradient checking, refer to tests/test_flash_attention.py and use torch.autograd.gradcheck.")
        print("This example script provides basic forward/backward execution checks.")
