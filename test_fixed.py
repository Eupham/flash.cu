import torch
# Ensure the flash_attn module is importable (e.g., after build_ext --inplace from project root)
# Or after 'python setup.py install'
from flash_attn.flash_attention import FlashAttention 

# Example parameters
batch_size, num_heads, seq_len, head_dim = 4, 8, 512, 64 # Try head_dim in [32, 64, 128]

# Create random input tensors on CUDA device with requires_grad=True for autograd
q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float32, requires_grad=True)
k = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float32, requires_grad=True)
v = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float32, requires_grad=True)

# Initialize FlashAttention (non-causal by default)
# Ensure head_dim matches one of the compiled kernel versions (32, 64, or 128 for this implementation)
try:
    flash_attention_op = FlashAttention(head_dim=head_dim, causal=False)
    
    # Perform forward pass
    # Tensors must be on CUDA and of dtype torch.float32
    output = flash_attention_op(q, k, v)
    print("Output shape (non-causal):", output.shape) # Expected: [4, 8, 512, 64]

    # Example with autograd
    output.sum().backward() # Compute gradients
    print("Gradient for Q (sample):", q.grad[0,0,0,:5])


    # For causal attention:
    # Reset gradients if reusing tensors
    if q.grad is not None: q.grad.zero_()
    if k.grad is not None: k.grad.zero_()
    if v.grad is not None: v.grad.zero_()
            
    flash_attention_causal_op = FlashAttention(head_dim=head_dim, causal=True)
    output_causal = flash_attention_causal_op(q, k, v)
    print("Output shape (causal):", output_causal.shape) # Expected: [4, 8, 512, 64]
    
    output_causal.sum().backward()
    print("Gradient for Q (causal, sample):", q.grad[0,0,0,:5])

except RuntimeError as e:
    print(f"Error during FlashAttention usage: {e}")
    print("This might happen if the CUDA extension was not compiled correctly or if CUDA is unavailable.")
except ImportError:
    print("Could not import FlashAttention. Ensure the CUDA extension is built and accessible.")
