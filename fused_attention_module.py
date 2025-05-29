import torch
import torch.nn as nn
import torch.autograd
from torch.utils.cpp_extension import load
import os

# Define constants that match the CUDA kernel implementation
# These are based on the #define values in cuda_fused_attention.cu
Q_ROWS_PER_BLOCK = 16
WARP_SIZE = 32
MAX_HEAD_DIM_CUDA = 64 # Max head dimension the CUDA kernel is compiled for

# Get the directory of the current Python script
module_path = os.path.dirname(__file__)
cuda_source_path = os.path.join(module_path, "cuda_fused_attention.cu")

# Check if the CUDA source file exists
if not os.path.exists(cuda_source_path):
    # Fallback for environments where __file__ might not be in the expected location (e.g. some notebooks)
    # This assumes the .cu file is in the current working directory if not found next to the .py file.
    if os.path.exists("cuda_fused_attention.cu"):
        cuda_source_path = "cuda_fused_attention.cu"
    else:
        raise FileNotFoundError(
            f"cuda_fused_attention.cu not found in {module_path} or current working directory."
        )

# Load the CUDA kernel
# This will compile the .cu file when the module is imported for the first time.
try:
    fused_attention_kernels = load(
        name="jules_custom_fused_attention_kernels_v1",
        sources=[cuda_source_path],
        verbose=True, # Set to False for less output during compilation
        extra_cuda_cflags=['-O3'] # Example of adding compiler flags
    )
except Exception as e:
    print(f"Error compiling CUDA kernels: {e}")
    print("Please ensure CUDA toolkit is installed and nvcc is in your PATH.")
    # Provide a way for the program to continue if CUDA is not available,
    # by creating a placeholder object for fused_attention_kernels.
    # This allows importing the module but operations will fail.
    class PlaceholderKernels:
        def __getattr__(self, name):
            def _dummy_kernel(*args, **kwargs):
                raise RuntimeError(
                    "CUDA kernels not compiled. Fused attention cannot be used."
                )
            return _dummy_kernel
    fused_attention_kernels = PlaceholderKernels()


class _FusedAttentionCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, sm_scale, is_causal):
        if not query.is_cuda:
            raise ValueError("Input tensors must be CUDA tensors for fused attention.")
        if not query.is_contiguous() or not key.is_contiguous() or not value.is_contiguous():
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

        batch_size, num_heads, seq_len, head_dim = query.shape

        if head_dim > MAX_HEAD_DIM_CUDA:
            raise ValueError(
                f"head_dim ({head_dim}) must be <= MAX_HEAD_DIM_CUDA ({MAX_HEAD_DIM_CUDA}) "
                "for which the kernel was compiled."
            )

        out = torch.empty_like(query)

        # Define grid and block dimensions for the CUDA kernel
        # torch.cdiv computes ceiling division: (a + b - 1) // b
        grid_fwd = (torch.cdiv(seq_len, Q_ROWS_PER_BLOCK), batch_size * num_heads)
        block_fwd = (Q_ROWS_PER_BLOCK, WARP_SIZE) # (rows_per_block, threads_in_warp)

        fused_attention_kernels.fused_attention_forward_kernel(
            query, key, value, out,
            batch_size, num_heads, seq_len, head_dim,
            sm_scale, is_causal,
            grid=grid_fwd, block=block_fwd # Pass grid and block as named arguments
        )
        
        ctx.save_for_backward(query, key, value, out, sm_scale) # 'out' is not strictly needed if P is recomputed
                                                                # but dO is grad_output which is dL/d(out)
        ctx.is_causal = is_causal
        ctx.batch_size = batch_size
        ctx.num_heads = num_heads
        ctx.seq_len = seq_len
        ctx.head_dim = head_dim
        # sm_scale is saved as a tensor, no need to save it separately if it's already in save_for_backward

        return out

    @staticmethod
    def backward(ctx, grad_output): # grad_output is dL/dO
        if not grad_output.is_cuda:
            # This should ideally not happen if forward inputs were CUDA tensors
            raise ValueError("grad_output must be a CUDA tensor.")
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()

        query, key, value, out, sm_scale_tensor = ctx.saved_tensors # Retrieve sm_scale as a tensor
        sm_scale = sm_scale_tensor.item() # Convert to float if it was saved as a 0-dim tensor

        is_causal = ctx.is_causal
        batch_size = ctx.batch_size
        num_heads = ctx.num_heads
        seq_len = ctx.seq_len
        head_dim = ctx.head_dim

        # Initialize gradient tensors to zeros. This is crucial because the backward CUDA kernel
        # uses atomicAdd to accumulate gradients for dK and dV.
        grad_query = torch.zeros_like(query)
        grad_key = torch.zeros_like(key)
        grad_value = torch.zeros_like(value)
        
        # Define grid and block dimensions for the backward CUDA kernel
        grid_bwd = (torch.cdiv(seq_len, Q_ROWS_PER_BLOCK), batch_size * num_heads)
        block_bwd = (Q_ROWS_PER_BLOCK, WARP_SIZE)

        fused_attention_kernels.fused_attention_backward_kernel(
            query, key, value, grad_output, # dO
            grad_query, grad_key, grad_value,
            batch_size, num_heads, seq_len, head_dim,
            sm_scale, is_causal,
            grid=grid_bwd, block=block_bwd
        )

        # Gradients for: query, key, value, sm_scale, is_causal
        return grad_query, grad_key, grad_value, None, None


class FusedAttention(nn.Module):
    def __init__(self):
        super(FusedAttention, self).__init__()
        # MAX_HEAD_DIM_CUDA can be checked here or during forward pass if it's dynamic
        # For now, check is in _FusedAttentionCuda.forward

    def forward(self, query, key, value, sm_scale=None, is_causal=False):
        # Input validation
        if not (query.ndim == 4 and key.ndim == 4 and value.ndim == 4):
            raise ValueError("Input tensors Q, K, V must be 4-dimensional (batch, heads, seq_len, head_dim).")
        
        if not (query.device == key.device == value.device):
            raise ValueError("Input tensors Q, K, V must be on the same device.")
        
        if not (query.dtype == key.dtype == value.dtype):
            raise ValueError("Input tensors Q, K, V must have the same data type.")

        if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0] or \
           query.shape[1] != key.shape[1] or query.shape[1] != value.shape[1] or \
           query.shape[3] != key.shape[3] or query.shape[3] != value.shape[3]:
            raise ValueError("Q, K, V must have matching batch_size, num_heads, and head_dim.")

        # K and V can have different sequence lengths, but Q and K must match for QK^T
        # The CUDA kernel assumes Q, K, V have same seq_len for simplicity in this example.
        # If K/V can have different seq_len than Q, kernel logic (especially masking & loops) needs adjustment.
        # Current kernels assume seq_len_q == seq_len_k == seq_len_v = `seq_len` parameter.
        if key.shape[2] != value.shape[2]: # seq_len_k != seq_len_v
             raise ValueError("Key and Value sequence lengths must match for this implementation.")
        if query.shape[2] != key.shape[2]: # seq_len_q != seq_len_k
            # This is a common scenario (cross-attention), but current kernels might assume they are same.
            # The `seq_len` parameter to kernels refers to a single sequence length.
            # If this is to be supported, kernel needs distinct seq_len_q, seq_len_kv.
            # For now, enforce they are same.
            raise ValueError("Query and Key sequence lengths must match for this implementation.")


        head_dim = query.shape[3]
        if sm_scale is None:
            sm_scale = 1.0 / (head_dim ** 0.5)
        
        # Convert sm_scale to a 0-dim tensor to be saved by save_for_backward
        # This ensures it's on the same device as other tensors if needed by backward.
        # Or, just pass it as a float and also save it as a Python float on ctx.
        # For simplicity, current backward pass retrieves it from ctx.saved_tensors, so it should be a tensor.
        sm_scale_tensor = torch.tensor(sm_scale, device=query.device, dtype=query.dtype)

        return _FusedAttentionCuda.apply(query, key, value, sm_scale_tensor, is_causal)

if __name__ == '__main__':
    # Example Usage (requires CUDA device)
    if torch.cuda.is_available() and not isinstance(fused_attention_kernels, PlaceholderKernels):
        print("CUDA is available. Testing FusedAttention module.")
        
        batch_size = 2
        num_heads = 4
        seq_len = 128 # Try multiples of Q_ROWS_PER_BLOCK for easier debugging if issues arise
        head_dim = 64   # Must be <= MAX_HEAD_DIM_CUDA (64)

        # Ensure head_dim is compatible with MAX_HEAD_DIM_CUDA
        if head_dim > MAX_HEAD_DIM_CUDA:
            print(f"Skipping test: head_dim {head_dim} > MAX_HEAD_DIM_CUDA {MAX_HEAD_DIM_CUDA}")
        else:
            q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float32, requires_grad=True)
            k = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float32, requires_grad=True)
            v = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float32, requires_grad=True)

            fused_attn_layer = FusedAttention()

            # Test forward pass
            print("Testing forward pass...")
            try:
                output = fused_attn_layer(q, k, v, is_causal=False)
                print("Forward output shape:", output.shape)
                print("Forward output sample (first element):", output.flatten()[0])

                # Test backward pass
                print("\nTesting backward pass...")
                # Dummy gradient for the output
                grad_output_dummy = torch.randn_like(output)
                output.backward(grad_output_dummy)

                print("Backward pass completed.")
                if q.grad is not None:
                    print("Gradient for Q shape:", q.grad.shape)
                    print("Gradient for Q sample (first element):", q.grad.flatten()[0])
                else:
                    print("Gradient for Q is None.")
                
                if k.grad is not None:
                    print("Gradient for K shape:", k.grad.shape)
                    print("Gradient for K sample (first element):", k.grad.flatten()[0])
                else:
                    print("Gradient for K is None.")

                if v.grad is not None:
                    print("Gradient for V shape:", v.grad.shape)
                    print("Gradient for V sample (first element):", v.grad.flatten()[0])
                else:
                    print("Gradient for V is None.")

                # Test with causal masking
                print("\nTesting forward pass with causal masking...")
                output_causal = fused_attn_layer(q, k, v, is_causal=True)
                print("Forward output shape (causal):", output_causal.shape)
                print("Forward output sample (causal, first element):", output_causal.flatten()[0])
                grad_output_dummy_causal = torch.randn_like(output_causal)
                # Need to zero grads from previous backward if using same Q,K,V
                q.grad, k.grad, v.grad = None, None, None
                output_causal.backward(grad_output_dummy_causal)
                print("Backward pass (causal) completed.")
                if q.grad is not None:
                    print("Gradient for Q (causal) sample (first element):", q.grad.flatten()[0])


            except RuntimeError as e:
                print(f"An error occurred during testing: {e}")
                if "CUDA kernels not compiled" in str(e):
                    print("This is expected if CUDA compilation failed.")
                else:
                    # Potentially print more details for other runtime errors
                    import traceback
                    traceback.print_exc()
            except Exception as e:
                print(f"An unexpected error occurred during testing: {e}")
                import traceback
                traceback.print_exc()

    elif isinstance(fused_attention_kernels, PlaceholderKernels):
        print("CUDA kernels were not compiled. Skipping FusedAttention tests.")
    else:
        print("CUDA is not available. Skipping FusedAttention tests.")

```
