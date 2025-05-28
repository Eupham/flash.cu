# Custom FlashAttention Implementation in PyTorch with CUDA

## Overview

This repository provides a from-scratch implementation of FlashAttention, a memory-efficient exact attention mechanism, in PyTorch with custom CUDA kernels. The goal is to illustrate the core concepts of FlashAttention, including tiling, online softmax, and recomputation for the backward pass, to achieve memory savings compared to standard attention implementations, especially for long sequences.

This implementation is primarily for educational purposes to demonstrate how such a mechanism can be built.

## Features

-   **Tiled Forward Pass:** Custom CUDA kernel for the forward attention computation using tiling to reduce HBM reads/writes.
-   **Tiled Backward Pass:** Custom CUDA kernel for the backward pass, recomputing attention scores and probabilities to save memory that would otherwise be used for storing these large intermediate tensors.
-   **Online Softmax:** Softmax calculation is performed per tile, contributing to memory efficiency.
-   **Causal and Non-Causal Attention:** Supports both standard (non-causal) and causal attention masking.
-   **Support for Specific Head Dimensions:** CUDA kernels are templated and dispatched for head dimensions 32, 64, and 128.
-   **PyTorch Integration:** Implemented as a `torch.autograd.Function` for seamless integration with PyTorch's automatic differentiation system.
-   **Basic Tests:** Includes unit tests for numerical correctness against a standard PyTorch MHA implementation and gradient checks.
-   **Performance Benchmarks:** Includes scripts to benchmark against `torch.nn.MultiheadAttention`.

## Requirements

-   PyTorch (e.g., 1.10+ recommended, tested with versions around this)
-   CUDA Toolkit (e.g., 11.x or compatible, matching your PyTorch CUDA version)
-   C++ Compiler (e.g., g++ compatible with PyTorch's C++ extensions)
-   Python (e.g., 3.8+)

## Installation / Building

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/custom-flash-attention.git # Replace with actual URL if applicable
    cd custom-flash-attention
    ```

2.  **Build the CUDA extension:**
    You can build the extension in-place (recommended for development) or install it into your Python environment.

    *   **In-place build:**
        ```bash
        python setup.py build_ext --inplace
        ```
        This will create a `.so` file (e.g., `flash_attn_cuda_lib.cpython-38-x86_64-linux-gnu.so`) in the current directory structure, allowing you to import `flash_attn_cuda_lib` directly if your Python session is started from the project root.

    *   **Install:**
        ```bash
        python setup.py install
        ```
        This will install the package into your Python environment.

    **Note on Build Issues:**
    - Ensure `nvcc` (NVIDIA CUDA Compiler) is in your `PATH`.
    - Your PyTorch installation should be built with CUDA support.
    - Compiler versions should be compatible.

## Usage

Here's a basic example of how to use the `FlashAttention` module in Python:

```python
import torch
# Ensure the flash_attn module is importable (e.g., after build_ext --inplace from project root)
# Or after 'python setup.py install'
from flash_attn.flash_attention import FlashAttention 

# Example parameters
batch_size, num_heads, seq_len, head_dim = 4, 8, 512, 64 # Try head_dim in [32, 64, 128]

# Create random input tensors on CUDA device
q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float32)
k = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float32)
v = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float32)

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
```

## Running Tests

The tests verify the numerical correctness of the forward pass against a standard PyTorch MHA implementation and check the gradients computed by the backward pass.

To run the tests:
```bash
python -m unittest discover tests
# or directly:
# python tests/test_flash_attention.py
```
Ensure the CUDA extension has been built (e.g., via `python setup.py build_ext --inplace`) and you are in an environment where `flash_attn_cuda_lib` can be imported.

## Running Benchmarks

The benchmark script compares the performance (forward pass and optional forward+backward pass) of this custom FlashAttention implementation against `torch.nn.MultiheadAttention`.

Example command:
```bash
python benchmarks/benchmark_flash_attention.py --seq_lengths 512 1024 --head_dims 64 --batch_sizes 4 8 --num_iterations 50 --num_warmup 10
```

Key command-line arguments for `benchmark_flash_attention.py`:
-   `--seq_lengths`: List of sequence lengths.
-   `--batch_sizes`: List of batch sizes.
-   `--num_heads_list`: List of number of heads.
-   `--head_dims`: List of head dimensions (e.g., 32, 64, 128).
-   `--dtypes`: Data types to test (e.g., `fp32`, `fp16`). Note: FlashAttention kernels currently FP32.
-   `--causal_options`: `non-causal` or `causal`.
-   `--num_iterations`: Number of timed iterations.
-   `--num_warmup`: Number of warmup iterations.
-   `--models_to_run`: `flash` and/or `torch`.
-   `--measure_backward` / `--no_measure_backward`: Control if backward pass is timed.
-   `--allow_flash_fp16`: Use this flag to attempt running FlashAttention with FP16 (kernels are FP32, so this is for testing the Python path).

## Limitations / Current Status

-   **Data Type Support:** The custom CUDA kernels currently support `torch.float32` data type only. FP16/BF16 would require kernel modifications.
-   **Head Dimensions:** Kernels are templated and dispatched for head dimensions 32, 64, and 128. Other dimensions are not supported by this build.
-   **Hardware Specifics:** Performance characteristics, especially for atomic operations in the backward pass, may vary by GPU architecture. Tiling parameters (`T_r`, `T_c`) in the CUDA code are currently set to default values (e.g., 64) and may benefit from tuning for specific hardware.
-   **Torch MHA Comparison:** The benchmark against `torch.nn.MultiheadAttention` includes its internal linear projection layers for Q, K, V, and output, while this custom FlashAttention implementation operates on pre-projected Q, K, V. This is a common point of difference when comparing custom attention kernels with the full `nn.Module`.

## (Optional) TODO / Future Work

-   Add support for FP16 and BF16 data types in CUDA kernels.
-   Generalize CUDA kernels or add more template specializations for a wider range of head dimensions.
-   Further optimize CUDA kernels (e.g., more advanced reduction strategies, warp-level operations).
-   Explore support for variable sequence lengths within a batch (requires padding and modifications to masking).
-   More extensive testing, including different GPU architectures.
