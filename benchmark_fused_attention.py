import torch
import torch.nn.functional as F
import time
import pandas as pd
import argparse
import math
import os # For checking module path related issues if any

# Attempt to import FusedAttention, handling potential compilation issues at import time
try:
    from fused_attention_module import FusedAttention, fused_attention_kernels
    CUSTOM_ATTN_AVAILABLE = not isinstance(fused_attention_kernels, fused_attention_module.PlaceholderKernels)
except ImportError as e:
    print(f"Could not import FusedAttention: {e}. Custom attention benchmarks will be skipped.")
    CUSTOM_ATTN_AVAILABLE = False
    # Define a placeholder if import fails, so the script can run parts of it
    class FusedAttention: 
        def __init__(self, *args, **kwargs): pass
        def __call__(self, *args, **kwargs):
            raise RuntimeError("FusedAttention module not available.")
except Exception as e: # Catch other errors like compilation issues during load
    print(f"Error loading FusedAttention module: {e}. Custom attention benchmarks will be skipped.")
    CUSTOM_ATTN_AVAILABLE = False
    class FusedAttention:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, *args, **kwargs):
            raise RuntimeError("FusedAttention module not available or failed to load.")

# Check PyTorch version for scaled_dot_product_attention 'scale' argument
TORCH_VERSION_MAJOR = int(torch.__version__.split('.')[0])
TORCH_VERSION_MINOR = int(torch.__version__.split('.')[1])
SDPA_HAS_SCALE_ARG = (TORCH_VERSION_MAJOR > 2) or (TORCH_VERSION_MAJOR == 2 and TORCH_VERSION_MINOR >= 2)


def scaled_dot_product_attention_pytorch(q, k, v, sm_scale, is_causal=False):
    """
    Reference implementation using torch.nn.functional.scaled_dot_product_attention.
    q, k, v: (B, H, N, D)
    sm_scale: scaling factor for QK^T. sdpa applies 1/sqrt(D) by default if scale is not set.
              If our sm_scale is different from 1/sqrt(D), and sdpa doesn't have 'scale' arg,
              this reference might not be perfectly equivalent for arbitrary sm_scale.
              However, for benchmarking, we typically use sm_scale = 1/sqrt(D).
    """
    N_q = q.size(2)
    N_kv = k.size(2)
    attn_mask = None
    if is_causal:
        # Create a causal mask for q_seq_len x kv_seq_len
        # For self-attention N_q == N_kv == N
        attn_mask = torch.triu(torch.ones(N_q, N_kv, device=q.device, dtype=torch.bool), diagonal=1)
        # sdpa expects mask where True means "don't attend"
    
    # scaled_dot_product_attention handles the scaling internally by default (1/sqrt(D)).
    # If a custom sm_scale is provided that ISN'T 1/sqrt(D), we'd ideally pass it via `scale` arg.
    # For this benchmark, we assume sm_scale passed IS 1/sqrt(D) for a fair comparison.
    if SDPA_HAS_SCALE_ARG:
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=sm_scale, is_causal=False) # is_causal in sdpa is different from our mask
    else:
        # If scale arg is not available, sdpa uses 1/sqrt(D_head).
        # We ensure our sm_scale matches this for fair comparison.
        # If sm_scale was crucially different, we'd need to pre-scale Q: q = q * (sm_scale * sqrt(D_head))
        # but this benchmark assumes standard scaling.
        if not math.isclose(sm_scale, 1.0 / math.sqrt(q.size(-1))):
            print(f"Warning: PyTorch sdpa pre-2.2 uses default 1/sqrt(D) scaling. Provided sm_scale {sm_scale} may not be fully effective for reference.")
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False) # is_causal in sdpa is different

    return output


def check_accuracy(config, custom_attention_fn, ref_attention_fn, is_causal, device='cuda'):
    print(f"\n--- Accuracy Check: causal={is_causal}, config={config} ---")
    B, H, N, D = config['B'], config['H'], config['N'], config['D']
    
    q = torch.randn(B, H, N, D, device=device, dtype=torch.float32, requires_grad=True)
    k = torch.randn(B, H, N, D, device=device, dtype=torch.float32, requires_grad=True)
    v = torch.randn(B, H, N, D, device=device, dtype=torch.float32, requires_grad=True)
    
    sm_scale = 1.0 / math.sqrt(D)

    # Custom Attention Forward
    custom_output = custom_attention_fn(q, k, v, sm_scale=sm_scale, is_causal=is_causal)
    
    # Reference Attention Forward
    ref_output = ref_attention_fn(q, k, v, sm_scale=sm_scale, is_causal=is_causal)

    fwd_allclose = torch.allclose(custom_output, ref_output, atol=1e-5, rtol=1e-3) # Higher tolerance for fused kernels
    print(f"Forward pass outputs {'match' if fwd_allclose else 'DO NOT match'}.")
    if not fwd_allclose:
        print(f"Custom output sample:\n{custom_output.flatten()[:8]}...")
        print(f"Reference output sample:\n{ref_output.flatten()[:8]}...")
        print(f"Max diff: {torch.max(torch.abs(custom_output - ref_output))}")


    # Backward Pass
    dummy_grad = torch.randn_like(custom_output)
    
    # Custom Backward
    loss_custom = (custom_output * dummy_grad).sum() # Multiply by dummy_grad to make loss scalar but grads meaningful
    loss_custom.backward()
    custom_grads = {'q': q.grad.clone(), 'k': k.grad.clone(), 'v': v.grad.clone()}
    
    q.grad, k.grad, v.grad = None, None, None # Zero grads

    # Reference Backward
    loss_ref = (ref_output * dummy_grad).sum()
    loss_ref.backward()
    ref_grads = {'q': q.grad.clone(), 'k': k.grad.clone(), 'v': v.grad.clone()}

    bwd_q_allclose = torch.allclose(custom_grads['q'], ref_grads['q'], atol=1e-5, rtol=1e-3)
    bwd_k_allclose = torch.allclose(custom_grads['k'], ref_grads['k'], atol=1e-5, rtol=1e-3)
    bwd_v_allclose = torch.allclose(custom_grads['v'], ref_grads['v'], atol=1e-5, rtol=1e-3)

    print(f"Backward pass Q gradients {'match' if bwd_q_allclose else 'DO NOT match'}.")
    print(f"Backward pass K gradients {'match' if bwd_k_allclose else 'DO NOT match'}.")
    print(f"Backward pass V gradients {'match' if bwd_v_allclose else 'DO NOT match'}.")

    if not (bwd_q_allclose and bwd_k_allclose and bwd_v_allclose):
        print(f"Custom Q grad sample: {custom_grads['q'].flatten()[:8]}...")
        print(f"Reference Q grad sample: {ref_grads['q'].flatten()[:8]}...")
        print(f"Max diff Q grad: {torch.max(torch.abs(custom_grads['q'] - ref_grads['q']))}")

        print(f"Custom K grad sample: {custom_grads['k'].flatten()[:8]}...")
        print(f"Reference K grad sample: {ref_grads['k'].flatten()[:8]}...")
        print(f"Max diff K grad: {torch.max(torch.abs(custom_grads['k'] - ref_grads['k']))}")

        print(f"Custom V grad sample: {custom_grads['v'].flatten()[:8]}...")
        print(f"Reference V grad sample: {ref_grads['v'].flatten()[:8]}...")
        print(f"Max diff V grad: {torch.max(torch.abs(custom_grads['v'] - ref_grads['v']))}")
    
    return fwd_allclose and bwd_q_allclose and bwd_k_allclose and bwd_v_allclose


def benchmark_speed(config, custom_attention_fn, ref_attention_fn, is_causal, 
                    warmup_runs=20, test_runs=100, device='cuda'):
    print(f"\n--- Speed Benchmark: causal={is_causal}, config={config} ---")
    B, H, N, D = config['B'], config['H'], config['N'], config['D']
    
    q = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
    k = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
    v = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
    sm_scale = 1.0 / math.sqrt(D)
    
    results = {'config': config, 'is_causal': is_causal}

    # --- Custom Attention Timing ---
    if custom_attention_fn:
        # Requires grad for backward pass
        q_c, k_c, v_c = q.clone().requires_grad_(), k.clone().requires_grad_(), v.clone().requires_grad_()
        
        # Warmup
        for _ in range(warmup_runs):
            _ = custom_attention_fn(q_c, k_c, v_c, sm_scale=sm_scale, is_causal=is_causal)
        
        # Forward Timing
        fwd_times = []
        for _ in range(test_runs):
            torch.cuda.synchronize(device=device)
            start_time = time.perf_counter()
            output_custom = custom_attention_fn(q_c, k_c, v_c, sm_scale=sm_scale, is_causal=is_causal)
            torch.cuda.synchronize(device=device)
            fwd_times.append(time.perf_counter() - start_time)
        results['custom_fwd_ms'] = (sum(fwd_times) / test_runs) * 1000
        
        # Backward Timing
        dummy_grad = torch.randn_like(output_custom)
        bwd_times = []
        for _ in range(test_runs):
            # Need to zero grads each time if not re-creating output_custom and graph
            if q_c.grad is not None: q_c.grad.zero_()
            if k_c.grad is not None: k_c.grad.zero_()
            if v_c.grad is not None: v_c.grad.zero_()
            # Re-run forward to ensure graph is live for backward, or use retain_graph=True (less clean for timing)
            output_custom_for_bwd = custom_attention_fn(q_c, k_c, v_c, sm_scale=sm_scale, is_causal=is_causal)

            torch.cuda.synchronize(device=device)
            start_time = time.perf_counter()
            output_custom_for_bwd.backward(dummy_grad, retain_graph=False) # No retain_graph needed if re-running fwd
            torch.cuda.synchronize(device=device)
            bwd_times.append(time.perf_counter() - start_time)
        results['custom_bwd_ms'] = (sum(bwd_times) / test_runs) * 1000
        print(f"Custom: Fwd {results['custom_fwd_ms']:.3f} ms, Bwd {results['custom_bwd_ms']:.3f} ms")

    # --- Reference Attention Timing ---
    q_r, k_r, v_r = q.clone().requires_grad_(), k.clone().requires_grad_(), v.clone().requires_grad_()
    
    # Warmup
    for _ in range(warmup_runs):
        _ = ref_attention_fn(q_r, k_r, v_r, sm_scale=sm_scale, is_causal=is_causal)
        
    # Forward Timing
    fwd_times = []
    for _ in range(test_runs):
        torch.cuda.synchronize(device=device)
        start_time = time.perf_counter()
        output_ref = ref_attention_fn(q_r, k_r, v_r, sm_scale=sm_scale, is_causal=is_causal)
        torch.cuda.synchronize(device=device)
        fwd_times.append(time.perf_counter() - start_time)
    results['ref_fwd_ms'] = (sum(fwd_times) / test_runs) * 1000
    
    # Backward Timing
    dummy_grad_ref = torch.randn_like(output_ref)
    bwd_times = []
    for _ in range(test_runs):
        if q_r.grad is not None: q_r.grad.zero_()
        if k_r.grad is not None: k_r.grad.zero_()
        if v_r.grad is not None: v_r.grad.zero_()
        output_ref_for_bwd = ref_attention_fn(q_r, k_r, v_r, sm_scale=sm_scale, is_causal=is_causal)

        torch.cuda.synchronize(device=device)
        start_time = time.perf_counter()
        output_ref_for_bwd.backward(dummy_grad_ref, retain_graph=False)
        torch.cuda.synchronize(device=device)
        bwd_times.append(time.perf_counter() - start_time)
    results['ref_bwd_ms'] = (sum(bwd_times) / test_runs) * 1000
    print(f"Reference: Fwd {results['ref_fwd_ms']:.3f} ms, Bwd {results['ref_bwd_ms']:.3f} ms")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark Fused Attention CUDA kernel against PyTorch reference.")
    parser.add_argument('--num_configs', type=int, default=None, help="Number of configurations to test from the predefined list.")
    parser.add_argument('--batch_size', type=int, default=None, help="Override batch size for all configs.")
    parser.add_argument('--num_heads', type=int, default=None, help="Override number of heads for all configs.")
    parser.add_argument('--head_dim', type=int, default=None, help="Override head dimension for all configs.")
    parser.add_argument('--seq_lens', type=int, nargs='+', 
                        default=[128, 256, 512, 1024, 2048, 4096], 
                        help="List of sequence lengths (N_CTX) to test.")
    parser.add_argument('--warmup_runs', type=int, default=20)
    parser.add_argument('--test_runs', type=int, default=100)

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        exit()
    
    device = 'cuda'
    
    # Define configurations: (B, H, N_CTX, HEAD_DIM)
    # MAX_HEAD_DIM_CUDA from fused_attention_module is 64 for the compiled kernel
    # Ensure HEAD_DIM <= 64
    base_configs = []
    default_B = args.batch_size if args.batch_size else 4
    default_H = args.num_heads if args.num_heads else 8
    default_D = args.head_dim if args.head_dim else 64

    if default_D > 64 and CUSTOM_ATTN_AVAILABLE: # MAX_HEAD_DIM_CUDA is 64 in the custom kernel
        print(f"Warning: Requested head_dim {default_D} > MAX_HEAD_DIM_CUDA (64). Custom kernel may error or be suboptimal.")
        print("Proceeding, but ensure custom kernel is compiled for this head_dim if issues occur.")
        # For this script, we assume MAX_HEAD_DIM_CUDA = 64 as defined in fused_attention_module.py for kernel loading.

    for n_ctx in args.seq_lens:
        base_configs.append({'B': default_B, 'H': default_H, 'N': n_ctx, 'D': default_D})

    test_configs = base_configs
    if args.num_configs is not None and args.num_configs < len(base_configs):
        test_configs = base_configs[:args.num_configs]

    all_results = []
    
    custom_attn_instance = None
    if CUSTOM_ATTN_AVAILABLE:
        try:
            custom_attn_instance = FusedAttention()
        except Exception as e:
            print(f"Failed to initialize FusedAttention: {e}. Skipping custom kernel benchmarks.")
            CUSTOM_ATTN_AVAILABLE = False # Ensure it's marked as unavailable

    ref_attn_fn = scaled_dot_product_attention_pytorch

    for config in test_configs:
        if CUSTOM_ATTN_AVAILABLE:
            # Accuracy check (non-causal)
            check_accuracy(config, custom_attn_instance.forward, ref_attn_fn, is_causal=False, device=device)
            # Accuracy check (causal)
            check_accuracy(config, custom_attn_instance.forward, ref_attn_fn, is_causal=True, device=device)
        else:
            print(f"Skipping accuracy checks for config {config} as custom attention is not available.")

        # Speed benchmark (non-causal)
        speed_res_non_causal = benchmark_speed(config, 
                                               custom_attn_instance.forward if CUSTOM_ATTN_AVAILABLE else None, 
                                               ref_attn_fn, 
                                               is_causal=False, 
                                               warmup_runs=args.warmup_runs, 
                                               test_runs=args.test_runs, 
                                               device=device)
        all_results.append(speed_res_non_causal)

        # Speed benchmark (causal)
        speed_res_causal = benchmark_speed(config, 
                                           custom_attn_instance.forward if CUSTOM_ATTN_AVAILABLE else None, 
                                           ref_attn_fn, 
                                           is_causal=True, 
                                           warmup_runs=args.warmup_runs, 
                                           test_runs=args.test_runs, 
                                           device=device)
        all_results.append(speed_res_causal)

    if all_results:
        df_results = pd.DataFrame(all_results)
        # Flatten the 'config' dict into separate columns
        df_config = pd.json_normalize(df_results['config'])
        df_results = pd.concat([df_config, df_results.drop('config', axis=1)], axis=1)
        
        print("\n--- Benchmark Results ---")
        print(df_results.to_string())
    else:
        print("\nNo benchmark results to display.")

```
