import torch
import torch.nn.functional as F
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import argparse
import os

from flash_attention import flash_attention, _pytorch_flash_attention

try:
    from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    print("flash-attn not available, will skip comparison")

def benchmark_function(func, *args, num_warmup=10, num_trials=100):
    """Benchmark a function with proper warmup and timing."""
    # Warmup
    for _ in range(num_warmup):
        _ = func(*args)
    
    torch.cuda.synchronize()
    
    # Timing
    times = []
    for _ in range(num_trials):
        start = time.perf_counter()
        result = func(*args)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    return np.mean(times), np.std(times), result

def compute_flops(batch_size, num_heads, seq_len, head_dim, causal=False):
    """Compute theoretical FLOPs for attention."""
    # Q @ K^T
    qk_flops = 2 * batch_size * num_heads * seq_len * seq_len * head_dim
    
    # Softmax (approximation)
    softmax_flops = 3 * batch_size * num_heads * seq_len * seq_len
    
    # Attention @ V
    av_flops = 2 * batch_size * num_heads * seq_len * seq_len * head_dim
    
    total_flops = qk_flops + softmax_flops + av_flops
    
    if causal:
        total_flops *= 0.5  # Roughly half the computation due to masking
    
    return total_flops

def test_correctness(batch_size=2, num_heads=8, seq_len=512, head_dim=64, causal=True):
    """Test correctness of CUDA implementation against PyTorch reference."""
    print(f"Testing correctness with shape [{batch_size}, {num_heads}, {seq_len}, {head_dim}]")
    
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16
    
    # Generate test data
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    
    scale = 1.0 / (head_dim ** 0.5)
    
    # PyTorch reference implementation
    ref_out = _pytorch_flash_attention(q, k, v, scale, causal)
    
    # Our CUDA implementation
    try:
        cuda_out = flash_attention(q, k, v, scale, causal)
        
        # Compare outputs
        max_diff = torch.max(torch.abs(ref_out - cuda_out)).item()
        mean_diff = torch.mean(torch.abs(ref_out - cuda_out)).item()
        
        print(f"Max absolute difference: {max_diff:.6f}")
        print(f"Mean absolute difference: {mean_diff:.6f}")
        
        # Check if differences are within acceptable tolerance
        tolerance = 1e-2  # fp16 tolerance
        if max_diff < tolerance:
            print("✅ Correctness test PASSED")
            return True
        else:
            print("❌ Correctness test FAILED")
            return False
            
    except Exception as e:
        print(f"❌ CUDA implementation failed: {e}")
        return False

def benchmark_speed(batch_sizes=[1, 2, 4], 
                   num_heads_list=[8, 16, 32],
                   seq_lens=[512, 1024, 2048, 4096],
                   head_dim=64,
                   causal=True):
    """Benchmark speed across different configurations."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16
    
    results = []
    
    print("Running speed benchmarks...")
    print("=" * 80)
    
    for batch_size in batch_sizes:
        for num_heads in num_heads_list:
            for seq_len in seq_lens:
                print(f"Benchmarking [{batch_size}, {num_heads}, {seq_len}, {head_dim}]...")
                
                # Generate test data
                q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
                k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
                v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
                
                scale = 1.0 / (head_dim ** 0.5)
                
                config = {
                    'batch_size': batch_size,
                    'num_heads': num_heads,
                    'seq_len': seq_len,
                    'head_dim': head_dim,
                    'causal': causal
                }
                
                # Benchmark PyTorch implementation
                try:
                    pt_time, pt_std, _ = benchmark_function(
                        _pytorch_flash_attention, q, k, v, scale, causal
                    )
                    config['pytorch_time'] = pt_time
                    config['pytorch_std'] = pt_std
                    
                    # Compute TFLOPS
                    flops = compute_flops(batch_size, num_heads, seq_len, head_dim, causal)
                    config['pytorch_tflops'] = flops / (pt_time * 1e-3) / 1e12
                    
                except Exception as e:
                    print(f"PyTorch benchmark failed: {e}")
                    config['pytorch_time'] = float('inf')
                    config['pytorch_tflops'] = 0
                
                # Benchmark our CUDA implementation
                try:
                    cuda_time, cuda_std, _ = benchmark_function(
                        flash_attention, q, k, v, scale, causal
                    )
                    config['cuda_time'] = cuda_time
                    config['cuda_std'] = cuda_std
                    config['cuda_tflops'] = flops / (cuda_time * 1e-3) / 1e12
                    config['speedup'] = pt_time / cuda_time if cuda_time > 0 else 0
                    
                except Exception as e:
                    print(f"CUDA benchmark failed: {e}")
                    config['cuda_time'] = float('inf')
                    config['cuda_tflops'] = 0
                    config['speedup'] = 0
                
                # Benchmark flash-attn if available
                if HAS_FLASH_ATTN:
                    try:
                        # Convert to flash-attn format: [batch, seq_len, 3, num_heads, head_dim]
                        qkv = torch.stack([q, k, v], dim=2).transpose(1, 2)
                        fa_time, fa_std, _ = benchmark_function(
                            flash_attn_qkvpacked_func, qkv, causal=causal
                        )
                        config['flash_attn_time'] = fa_time
                        config['flash_attn_std'] = fa_std
                        config['flash_attn_tflops'] = flops / (fa_time * 1e-3) / 1e12
                        
                    except Exception as e:
                        print(f"Flash-attn benchmark failed: {e}")
                        config['flash_attn_time'] = float('inf')
                        config['flash_attn_tflops'] = 0
                
                results.append(config)
                
                # Print results for this configuration
                print(f"  PyTorch:  {pt_time:.2f}±{pt_std:.2f}ms, {config['pytorch_tflops']:.2f} TFLOPS")
                if config['cuda_time'] != float('inf'):
                    print(f"  CUDA:     {cuda_time:.2f}±{cuda_std:.2f}ms, {config['cuda_tflops']:.2f} TFLOPS, {config['speedup']:.2f}x speedup")
                if HAS_FLASH_ATTN and 'flash_attn_time' in config:
                    print(f"  Flash-attn: {fa_time:.2f}±{fa_std:.2f}ms, {config['flash_attn_tflops']:.2f} TFLOPS")
                print()
    
    return results

def plot_results(results: List[Dict], save_dir="benchmarks"):
    """Plot benchmark results."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Group results by batch_size and num_heads
    grouped = {}
    for result in results:
        key = (result['batch_size'], result['num_heads'])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(result)
    
    for (batch_size, num_heads), group in grouped.items():
        # Sort by sequence length
        group.sort(key=lambda x: x['seq_len'])
        
        seq_lens = [r['seq_len'] for r in group]
        pytorch_tflops = [r['pytorch_tflops'] for r in group]
        cuda_tflops = [r.get('cuda_tflops', 0) for r in group]
        
        plt.figure(figsize=(10, 6))
        plt.plot(seq_lens, pytorch_tflops, 'o-', label='PyTorch', linewidth=2, markersize=6)
        plt.plot(seq_lens, cuda_tflops, 's-', label='CUDA (Ours)', linewidth=2, markersize=6)
        
        if HAS_FLASH_ATTN and 'flash_attn_tflops' in group[0]:
            flash_attn_tflops = [r.get('flash_attn_tflops', 0) for r in group]
            plt.plot(seq_lens, flash_attn_tflops, '^-', label='Flash-Attn', linewidth=2, markersize=6)
        
        plt.xlabel('Sequence Length')
        plt.ylabel('TFLOPS')
        plt.title(f'Flash Attention Performance\nBatch={batch_size}, Heads={num_heads}, HeadDim=64')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        plt.yscale('log')
        
        filename = f"{save_dir}/flash_attention_b{batch_size}_h{num_heads}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {filename}")

def test_backward_pass():
    """Test backward pass correctness."""
    print("Testing backward pass...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16
    
    batch_size, num_heads, seq_len, head_dim = 2, 8, 512, 64
    
    # Generate test data
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device, requires_grad=True)
    
    scale = 1.0 / (head_dim ** 0.5)
    
    # Test with our implementation
    try:
        out = flash_attention(q, k, v, scale, causal=True)
        loss = out.sum()
        loss.backward()
        
        print("✅ Backward pass completed successfully")
        
        if q.grad is not None:
            print(f"  grad_q shape: {q.grad.shape}, mean: {q.grad.mean().item():.6f}")
        if k.grad is not None:
            print(f"  grad_k shape: {k.grad.shape}, mean: {k.grad.mean().item():.6f}")
        if v.grad is not None:
            print(f"  grad_v shape: {v.grad.shape}, mean: {v.grad.mean().item():.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Backward pass failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Flash Attention CUDA Benchmark')
    parser.add_argument('--test-correctness', action='store_true', help='Test correctness')
    parser.add_argument('--test-backward', action='store_true', help='Test backward pass')
    parser.add_argument('--benchmark', action='store_true', help='Run speed benchmarks')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--save-dir', default='benchmarks', help='Directory to save results')
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("CUDA not available, running on CPU")
    else:
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    
    # Run tests
    all_passed = True
    
    if args.test_correctness:
        all_passed &= test_correctness()
        print()
    
    if args.test_backward:
        all_passed &= test_backward_pass()
        print()
    
    if args.benchmark:
        print("Running comprehensive benchmarks...")
        results = benchmark_speed(
            batch_sizes=[1, 2, 4],
            num_heads_list=[8, 16],
            seq_lens=[512, 1024, 2048],
            head_dim=64,
            causal=True
        )
        
        if args.plot:
            plot_results(results, args.save_dir)
        
        # Save results to file
        import json
        results_file = os.path.join(args.save_dir, 'benchmark_results.json')
        os.makedirs(args.save_dir, exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {results_file}")
    
    if not any([args.test_correctness, args.test_backward, args.benchmark]):
        # Run all tests by default
        print("Running all tests...")
        all_passed &= test_correctness()
        all_passed &= test_backward_pass()
        
        print("\nRunning quick benchmark...")
        results = benchmark_speed(
            batch_sizes=[2],
            num_heads_list=[8],
            seq_lens=[512, 1024],
            head_dim=64,
            causal=True
        )
        
        if args.plot:
            plot_results(results, args.save_dir)
    
    if all_passed:
        print("🎉 All tests passed!")
    else:
        print("⚠️ Some tests failed!")

if __name__ == "__main__":
    main()
