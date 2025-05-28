"""
FlashAttention Benchmark Script

This script benchmarks the custom FlashAttention implementation against PyTorch's
standard `torch.nn.MultiheadAttention`. It measures the execution time for
both the forward pass and the combined forward + backward pass, along with
memory usage and numerical accuracy.

The script allows configuration of various parameters such as sequence length,
batch size, number of heads, head dimension, data type (FP32/FP16), and causality.
Results are printed in a formatted table, facilitating performance comparisons.

For a more comprehensive speed and accuracy comparison, use:
    python benchmarks/speed_accuracy_comparison.py

Note:
- The custom FlashAttention implementation currently supports FP32 for its CUDA kernels.
  FP16 benchmarks for FlashAttention might be skipped or require kernel modifications.
- `torch.nn.MultiheadAttention` includes linear projections for Q, K, V and output,
  whereas the custom FlashAttention implementation operates on pre-projected Q, K, V.
  This difference is inherent in the benchmark comparison against the standard module.
"""
import torch
import time
import argparse
import os
import sys
import math

# Add project root to Python path to allow importing `flash_attn`
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

try:
    from flash_attn.flash_attention import FlashAttention
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FlashAttention = None # type: ignore
    FLASH_ATTENTION_AVAILABLE = False

def benchmark_attention_op(
    op_name: str,
    op_instance: torch.nn.Module,
    q_input: torch.Tensor, 
    k_input: torch.Tensor, 
    v_input: torch.Tensor,
    num_iterations: int,
    num_warmup: int,
    device: str,
    is_torch_mha: bool = False, # Flag indicating if op_instance is torch.nn.MHA (not strictly needed with wrapper)
    requires_grad: bool = True
) -> tuple[float, float]:
    """
    Benchmarks a given attention operation for forward and forward+backward passes.

    Args:
        op_name (str): Name of the operation for logging.
        op_instance (torch.nn.Module): The attention module instance to benchmark.
        q_input (torch.Tensor): Query tensor.
        k_input (torch.Tensor): Key tensor.
        v_input (torch.Tensor): Value tensor.
        num_iterations (int): Number of iterations for the timing loop.
        num_warmup (int): Number of warmup iterations before timing.
        device (str): CUDA device string (e.g., 'cuda:0').
        is_torch_mha (bool, optional): True if the op is `torch.nn.MultiheadAttention`.
                                      This was used to handle its specific output tuple.
                                      With the wrapper, this might be less critical. Defaults to False.
        requires_grad (bool, optional): If True, also benchmarks the backward pass. Defaults to True.

    Returns:
        tuple[float, float]: Average forward pass time (ms), Average forward+backward pass time (ms).
                             Backward time is -1.0 if `requires_grad` is False.
    """
    if requires_grad:
        q_input.requires_grad_(True)
        k_input.requires_grad_(True)
        v_input.requires_grad_(True)
    else: # Ensure no grad for fwd-only measurement if specified
        q_input.requires_grad_(False)
        k_input.requires_grad_(False)
        v_input.requires_grad_(False)

    # --- Warmup for Forward Pass ---
    for _ in range(num_warmup):
        # The TorchMHAWrapper already handles the [0] indexing for output.
        _ = op_instance(q_input, k_input, v_input)
    
    # --- Timed Forward Pass ---
    torch.cuda.synchronize(device) # Ensure previous CUDA operations are complete
    fwd_start_time = time.perf_counter()
    for _ in range(num_iterations):
        _ = op_instance(q_input, k_input, v_input)
    torch.cuda.synchronize(device) # Wait for all iterations to complete
    fwd_end_time = time.perf_counter()
    avg_fwd_time_ms = ((fwd_end_time - fwd_start_time) / num_iterations) * 1000

    avg_fwd_bwd_time_ms = -1.0 # Default if not measuring backward pass

    if requires_grad:
        # Create gradient output tensor based on the actual output shape
        # Ensure this is done on the correct device and dtype
        output_for_grad = op_instance(q_input, k_input, v_input)
        grad_output = torch.randn_like(output_for_grad, device=device, dtype=output_for_grad.dtype)

        # --- Warmup for Forward + Backward Pass ---
        for _ in range(num_warmup):
            # Zero gradients before each backward pass in warmup
            if q_input.grad is not None: q_input.grad.zero_()
            if k_input.grad is not None: k_input.grad.zero_()
            if v_input.grad is not None: v_input.grad.zero_()
            
            output = op_instance(q_input, k_input, v_input)
            output.backward(grad_output, retain_graph=False) # retain_graph=False is typical for benchmarks

        # --- Timed Forward + Backward Pass ---
        torch.cuda.synchronize(device)
        fwd_bwd_start_time = time.perf_counter()
        for _ in range(num_iterations):
            # Zero gradients before each backward pass in timed loop
            if q_input.grad is not None: q_input.grad.zero_()
            if k_input.grad is not None: k_input.grad.zero_()
            if v_input.grad is not None: v_input.grad.zero_()

            output = op_instance(q_input, k_input, v_input)
            output.backward(grad_output, retain_graph=False)
        torch.cuda.synchronize(device)
        fwd_bwd_end_time = time.perf_counter()
        avg_fwd_bwd_time_ms = ((fwd_bwd_end_time - fwd_bwd_start_time) / num_iterations) * 1000
        
    return avg_fwd_time_ms, avg_fwd_bwd_time_ms


def main(args):
    """
    Main function to run the benchmarks.

    Parses arguments, sets up configurations, initializes models,
    and calls the benchmarking function for each case.
    Prints results in a formatted table.
    """
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping benchmarks.")
        return

    if not FLASH_ATTENTION_AVAILABLE and 'flash' in args.models_to_run:
        print("FlashAttention CUDA module not available. Skipping FlashAttention benchmarks.")
        # Remove 'flash' from models to run if it's not available
        if 'flash' in args.models_to_run: args.models_to_run.remove('flash')
        if not args.models_to_run: # Exit if no models are left to run
            return
            
    device = args.device
    
    # Print benchmark configuration summary
    print("Benchmarking Configurations:")
    print(f"  Sequence Lengths (S_q=S_kv): {args.seq_lengths}")
    print(f"  Batch Sizes (B): {args.batch_sizes}")
    print(f"  Number of Heads (H): {args.num_heads_list}")
    print(f"  Head Dimensions (D): {args.head_dims}")
    print(f"  Data Types: {args.dtypes}")
    print(f"  Causal Options: {args.causal_options}")
    print(f"  Iterations: {args.num_iterations}, Warmup Iterations: {args.num_warmup}")
    print(f"  Models to Run: {args.models_to_run}")
    print(f"  Measure Backward Pass: {args.measure_backward}")
    print(f"  Allow FlashAttention FP16: {args.allow_flash_fp16}")
    print("-" * 90) # Adjusted table width
    print(f"{'SeqLen':<7} | {'Batch':<6} | {'Heads':<6} | {'Hdim':<5} | {'Causal':<7} | {'Dtype':<7} | {'Model':<12} | {'Fwd (ms)':<10} | {'Fwd+Bwd (ms)':<12}")
    print("-" * 90) # Adjusted table width

    # Iterate through all specified configurations
    for seq_len in args.seq_lengths: # Assuming S_q = S_kv for these benchmarks
        for batch_size in args.batch_sizes:
            for num_heads in args.num_heads_list:
                for head_dim in args.head_dims:
                    for causal_str in args.causal_options:
                        causal = True if causal_str == 'causal' else False
                        for dtype_str in args.dtypes:
                            dtype = torch.float16 if dtype_str == 'fp16' else torch.float32
                            
                            # Prepare input tensors for FlashAttention format (Batch, NumHeads, SeqLen, HeadDim)
                            try:
                                q_fa_format = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
                                k_fa_format = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype) 
                                v_fa_format = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
                            except RuntimeError as e: # Catch potential CUDA OOM early
                                print(f"Skipping config B={batch_size}, H={num_heads}, S={seq_len}, D={head_dim}, causal={causal}, dtype={dtype_str} due to OOM during tensor creation: {e}")
                                continue

                            # --- Benchmark Custom FlashAttention ---
                            if 'flash' in args.models_to_run:
                                # Skip FlashAttention FP16 if not allowed (current kernels are FP32)
                                if dtype == torch.float16 and not args.allow_flash_fp16:
                                    print(f"{seq_len:<7} | {batch_size:<6} | {num_heads:<6} | {head_dim:<5} | {str(causal):<7} | {dtype_str:<7} | {'FlashAttn':<12} | {'SKIPPED':<10} | {'SKIPPED':<12}")
                                else:
                                    try:
                                        flash_model = FlashAttention(head_dim=head_dim, causal=causal).to(device).to(dtype)
                                        fwd_ms, fwd_bwd_ms = benchmark_attention_op(
                                            "FlashAttn", flash_model, 
                                            q_fa_format.clone(), k_fa_format.clone(), v_fa_format.clone(),
                                            args.num_iterations, args.num_warmup, device,
                                            requires_grad=args.measure_backward
                                        )
                                        fwd_bwd_str = f"{fwd_bwd_ms:<12.3f}" if args.measure_backward else "N/A"
                                        print(f"{seq_len:<7} | {batch_size:<6} | {num_heads:<6} | {head_dim:<5} | {str(causal):<7} | {dtype_str:<7} | {'FlashAttn':<12} | {fwd_ms:<10.3f} | {fwd_bwd_str:<12}")
                                    except Exception as e:
                                        print(f"Error benchmarking FlashAttention (B={batch_size},H={num_heads},S={seq_len},D={head_dim},causal={causal},dtype={dtype_str}): {e}")

                            # --- Benchmark PyTorch nn.MultiheadAttention ---
                            if 'torch' in args.models_to_run:
                                embed_dim = num_heads * head_dim
                                try:
                                    # Reshape Q, K, V for torch.nn.MultiheadAttention: (Batch, SeqLen, EmbedDim)
                                    q_mha_input = q_fa_format.permute(0, 2, 1, 3).reshape(batch_size, seq_len, embed_dim)
                                    k_mha_input = k_fa_format.permute(0, 2, 1, 3).reshape(batch_size, seq_len, embed_dim)
                                    v_mha_input = v_fa_format.permute(0, 2, 1, 3).reshape(batch_size, seq_len, embed_dim)

                                    torch_mha_model = torch.nn.MultiheadAttention(
                                        embed_dim=embed_dim, 
                                        num_heads=num_heads, 
                                        bias=False, # For a fairer comparison with custom kernels not using bias
                                        batch_first=True, # Input format (N, L, E)
                                        device=device,
                                        dtype=dtype
                                    )
                                    
                                    # Causal mask for torch.nn.MultiheadAttention
                                    # PyTorch MHA expects mask where True means "don't attend". Shape (L, S) or (N*H, L, S).
                                    # For (L,S) mask: L=target_seq_len, S=source_seq_len.
                                    attn_mask_mha = None
                                    if causal:
                                        attn_mask_mha = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
                                    
                                    # Wrapper to make nn.MHA interface compatible with benchmark_attention_op
                                    class TorchMHAWrapper(torch.nn.Module):
                                        def __init__(self, mha_instance, attn_mask_val):
                                            super().__init__()
                                            self.mha = mha_instance
                                            self.attn_mask = attn_mask_val
                                        def forward(self, q, k, v): # q,k,v are already reshaped for MHA
                                            # nn.MHA returns (output, weights), we only need output
                                            return self.mha(q, k, v, attn_mask=self.attn_mask, need_weights=False)[0]

                                    torch_mha_wrapped = TorchMHAWrapper(torch_mha_model, attn_mask_mha).to(device)

                                    fwd_ms, fwd_bwd_ms = benchmark_attention_op(
                                        "TorchMHA", torch_mha_wrapped, 
                                        q_mha_input.clone(), k_mha_input.clone(), v_mha_input.clone(),
                                        args.num_iterations, args.num_warmup, device,
                                        requires_grad=args.measure_backward
                                    )
                                    fwd_bwd_str = f"{fwd_bwd_ms:<12.3f}" if args.measure_backward else "N/A"
                                    print(f"{seq_len:<7} | {batch_size:<6} | {num_heads:<6} | {head_dim:<5} | {str(causal):<7} | {dtype_str:<7} | {'TorchMHA':<12} | {fwd_ms:<10.3f} | {fwd_bwd_str:<12}")
                                except Exception as e:
                                     print(f"Error benchmarking TorchMHA (B={batch_size},H={num_heads},S={seq_len},D={head_dim},causal={causal},dtype={dtype_str}): {e}")
                                     # import traceback # Uncomment for full trace during debugging
                                     # traceback.print_exc()
    print("-" * 90) # Adjusted table width


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark FlashAttention vs PyTorch MultiheadAttention.")
    
    # Configuration for benchmark dimensions
    parser.add_argument('--seq_lengths', nargs='+', type=int, default=[512, 1024, 2048], 
                        help="List of sequence lengths (S_q = S_kv) to benchmark.")
    parser.add_argument('--batch_sizes', nargs='+', type=int, default=[2, 4, 8],
                        help="List of batch sizes (B).")
    parser.add_argument('--num_heads_list', nargs='+', type=int, default=[8, 12],
                        help="List of number of attention heads (H).")
    parser.add_argument('--head_dims', nargs='+', type=int, default=[32, 64],
                        help="List of head dimensions (D). Note: FlashAttention kernels compiled for 32, 64, 128.")
    
    # Configuration for execution parameters
    parser.add_argument('--dtypes', nargs='+', type=str, default=['fp32'], choices=['fp32', 'fp16'],
                        help="List of data types to benchmark (fp32, fp16).")
    parser.add_argument('--causal_options', nargs='+', type=str, default=['non-causal', 'causal'], choices=['non-causal', 'causal'],
                        help="Causal options to test ('non-causal', 'causal').")
    parser.add_argument('--num_iterations', type=int, default=20,
                        help="Number of iterations for the main timing loop.")
    parser.add_argument('--num_warmup', type=int, default=5,
                        help="Number of warmup iterations before timing.")
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda'], # Removed 'cpu' as it's not relevant for CUDA ops
                        help="Device to run benchmarks on (currently only 'cuda').")
    parser.add_argument('--models_to_run', nargs='+', type=str, default=['flash', 'torch'], choices=['flash', 'torch'],
                        help="Which attention implementations to benchmark ('flash', 'torch').")
    parser.add_argument('--measure_backward', action='store_true', default=True,
                        help="Include backward pass in timing. Default: True.")
    parser.add_argument('--no_measure_backward', action='store_false', dest='measure_backward',
                        help="Only measure forward pass. Overrides --measure_backward if both used.")
    parser.add_argument('--allow_flash_fp16', action='store_true',
                        help="Allow attempting to run FlashAttention with FP16. May fail if CUDA kernels are not FP16-ready (current are FP32).")

    args = parser.parse_args()
    main(args)
