#!/usr/bin/env python3
"""
Speed and Accuracy Comparison: FlashAttention vs PyTorch nn.MultiheadAttention

This benchmark compares:
1. Forward pass speed
2. Backward pass speed  
3. Memory usage
4. Numerical accuracy between FlashAttention and PyTorch's native MHA

Requirements:
- CUDA-capable GPU
- FlashAttention CUDA extension built and installed
- PyTorch with CUDA support
"""

import torch
import torch.nn as nn
import time
import numpy as np
import gc
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

try:
    from flash_attn.flash_attention import FlashAttention
    FLASH_AVAILABLE = True
except ImportError:
    print("Warning: FlashAttention not available. Only PyTorch MHA will be benchmarked.")
    FLASH_AVAILABLE = False


class BenchmarkConfig:
    """Configuration for benchmark tests"""
    def __init__(self):
        # Test configurations: (batch_size, num_heads, seq_len, head_dim)
        self.test_configs = [
            (1, 8, 512, 64),      # Small
            (2, 8, 1024, 64),     # Medium
            (4, 8, 2048, 64),     # Large
            (1, 16, 512, 64),     # More heads
            (2, 8, 512, 128),     # Larger head_dim
            (1, 8, 4096, 64),     # Very long sequence
        ]
        
        self.warmup_iterations = 5
        self.benchmark_iterations = 20
        self.tolerance_atol = 1e-5
        self.tolerance_rtol = 1e-4


class PyTorchMHA:
    """Wrapper for PyTorch's MultiheadAttention to match FlashAttention interface"""
    
    def __init__(self, embed_dim: int, num_heads: int, causal: bool = False):
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.0,
            bias=False,
            batch_first=True
        ).cuda()
        self.causal = causal
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # Reshape from (B, H, S, D) to (B, S, H*D) for PyTorch MHA
        q_reshaped = q.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        k_reshaped = k.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        v_reshaped = v.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # Create causal mask if needed
        attn_mask = None
        if self.causal:
            attn_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().cuda()
        
        # Forward pass
        output, _ = self.mha(q_reshaped, k_reshaped, v_reshaped, attn_mask=attn_mask)
        
        # Reshape back to (B, H, S, D)
        output = output.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        
        return output


def measure_memory():
    """Measure current GPU memory usage"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1024**2  # MB
    return 0


def benchmark_forward_pass(model, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                          iterations: int = 20) -> Tuple[float, float]:
    """Benchmark forward pass speed and return mean and std of execution times"""
    times = []
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model(q, k, v)
    
    torch.cuda.synchronize()
    
    # Actual benchmark
    for _ in range(iterations):
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        with torch.no_grad():
            output = model(q, k, v)
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        times.append((end_time - start_time) * 1000)  # Convert to ms
    
    return np.mean(times), np.std(times)


def benchmark_backward_pass(model, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                           iterations: int = 20) -> Tuple[float, float]:
    """Benchmark backward pass speed and return mean and std of execution times"""
    times = []
    
    # Warmup
    for _ in range(5):
        q.grad = None
        k.grad = None  
        v.grad = None
        output = model(q, k, v)
        loss = output.sum()
        loss.backward()
    
    torch.cuda.synchronize()
    
    # Actual benchmark
    for _ in range(iterations):
        q.grad = None
        k.grad = None
        v.grad = None
        
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        output = model(q, k, v)
        loss = output.sum()
        loss.backward()
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        times.append((end_time - start_time) * 1000)  # Convert to ms
    
    return np.mean(times), np.std(times)


def check_accuracy(flash_output: torch.Tensor, pytorch_output: torch.Tensor,
                  atol: float = 1e-5, rtol: float = 1e-4) -> Dict[str, float]:
    """Compare accuracy between FlashAttention and PyTorch MHA outputs"""
    
    # Calculate various error metrics
    abs_diff = torch.abs(flash_output - pytorch_output)
    rel_diff = abs_diff / (torch.abs(pytorch_output) + 1e-8)
    
    max_abs_error = torch.max(abs_diff).item()
    mean_abs_error = torch.mean(abs_diff).item()
    max_rel_error = torch.max(rel_diff).item()
    mean_rel_error = torch.mean(rel_diff).item()
    
    # Check if tensors are close within tolerance
    is_close = torch.allclose(flash_output, pytorch_output, atol=atol, rtol=rtol)
    
    return {
        'max_abs_error': max_abs_error,
        'mean_abs_error': mean_abs_error, 
        'max_rel_error': max_rel_error,
        'mean_rel_error': mean_rel_error,
        'is_close': is_close,
        'cosine_similarity': torch.nn.functional.cosine_similarity(
            flash_output.flatten(), pytorch_output.flatten(), dim=0
        ).item()
    }


def run_single_benchmark(config: Tuple[int, int, int, int], causal: bool = False) -> Dict:
    """Run benchmark for a single configuration"""
    batch_size, num_heads, seq_len, head_dim = config
    embed_dim = num_heads * head_dim
    
    print(f"\nBenchmarking: B={batch_size}, H={num_heads}, S={seq_len}, D={head_dim}, Causal={causal}")
    
    # Create test tensors
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float32, requires_grad=True)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float32, requires_grad=True)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float32, requires_grad=True)
    
    results = {
        'config': config,
        'causal': causal,
        'seq_len': seq_len,
        'batch_size': batch_size,
        'num_heads': num_heads,
        'head_dim': head_dim
    }
    
    # Initialize models
    pytorch_mha = PyTorchMHA(embed_dim, num_heads, causal)
    
    if FLASH_AVAILABLE:
        flash_attn = FlashAttention(head_dim=head_dim, causal=causal)
    
    # Benchmark PyTorch MHA
    print("  Benchmarking PyTorch MHA...")
    torch.cuda.reset_peak_memory_stats()
    
    # Forward pass
    pytorch_fwd_mean, pytorch_fwd_std = benchmark_forward_pass(pytorch_mha, q, k, v)
    pytorch_fwd_memory = measure_memory()
    
    # Backward pass  
    pytorch_bwd_mean, pytorch_bwd_std = benchmark_backward_pass(pytorch_mha, q, k, v)
    pytorch_total_memory = measure_memory()
    
    results.update({
        'pytorch_fwd_time_ms': pytorch_fwd_mean,
        'pytorch_fwd_std_ms': pytorch_fwd_std,
        'pytorch_bwd_time_ms': pytorch_bwd_mean,
        'pytorch_bwd_std_ms': pytorch_bwd_std,
        'pytorch_memory_mb': pytorch_total_memory
    })
    
    # Get PyTorch reference output for accuracy comparison
    with torch.no_grad():
        pytorch_output = pytorch_mha(q, k, v)
    
    if FLASH_AVAILABLE:
        # Benchmark FlashAttention
        print("  Benchmarking FlashAttention...")
        torch.cuda.reset_peak_memory_stats()
        
        # Forward pass
        flash_fwd_mean, flash_fwd_std = benchmark_forward_pass(flash_attn, q, k, v)
        flash_fwd_memory = measure_memory()
        
        # Backward pass
        flash_bwd_mean, flash_bwd_std = benchmark_backward_pass(flash_attn, q, k, v)
        flash_total_memory = measure_memory()
        
        # Accuracy comparison
        with torch.no_grad():
            flash_output = flash_attn(q, k, v)
        
        accuracy_metrics = check_accuracy(flash_output, pytorch_output)
        
        results.update({
            'flash_fwd_time_ms': flash_fwd_mean,
            'flash_fwd_std_ms': flash_fwd_std,
            'flash_bwd_time_ms': flash_bwd_mean,
            'flash_bwd_std_ms': flash_bwd_std,
            'flash_memory_mb': flash_total_memory,
            'speedup_fwd': pytorch_fwd_mean / flash_fwd_mean,
            'speedup_bwd': pytorch_bwd_mean / flash_bwd_mean,
            'memory_reduction': (pytorch_total_memory - flash_total_memory) / pytorch_total_memory * 100,
            **accuracy_metrics
        })
        
        print(f"    Forward speedup: {results['speedup_fwd']:.2f}x")
        print(f"    Backward speedup: {results['speedup_bwd']:.2f}x") 
        print(f"    Memory reduction: {results['memory_reduction']:.1f}%")
        print(f"    Max abs error: {results['max_abs_error']:.2e}")
        print(f"    Accuracy check: {'PASS' if results['is_close'] else 'FAIL'}")
    else:
        print("  FlashAttention not available, skipping...")
    
    # Cleanup
    del q, k, v, pytorch_output
    if FLASH_AVAILABLE and 'flash_output' in locals():
        del flash_output
    torch.cuda.empty_cache()
    gc.collect()
    
    return results


def create_summary_table(all_results: List[Dict]):
    """Create a summary table of results"""
    table_data = []
    
    for result in all_results:
        if not FLASH_AVAILABLE:
            row = [
                f"B{result['batch_size']}_H{result['num_heads']}_S{result['seq_len']}_D{result['head_dim']}",
                "✓" if result['causal'] else "✗",
                f"{result['pytorch_fwd_time_ms']:.2f} ± {result['pytorch_fwd_std_ms']:.2f}",
                f"{result['pytorch_bwd_time_ms']:.2f} ± {result['pytorch_bwd_std_ms']:.2f}",
                f"{result['pytorch_memory_mb']:.1f}",
                "N/A", "N/A", "N/A", "N/A", "N/A"
            ]
        else:
            row = [
                f"B{result['batch_size']}_H{result['num_heads']}_S{result['seq_len']}_D{result['head_dim']}",
                "✓" if result['causal'] else "✗",
                f"{result['pytorch_fwd_time_ms']:.2f} ± {result['pytorch_fwd_std_ms']:.2f}",
                f"{result['flash_fwd_time_ms']:.2f} ± {result['flash_fwd_std_ms']:.2f}",
                f"{result['speedup_fwd']:.2f}x",
                f"{result['pytorch_bwd_time_ms']:.2f} ± {result['pytorch_bwd_std_ms']:.2f}",
                f"{result['flash_bwd_time_ms']:.2f} ± {result['flash_bwd_std_ms']:.2f}",
                f"{result['speedup_bwd']:.2f}x",
                f"{result['memory_reduction']:.1f}%",
                "✓" if result['is_close'] else "✗"
            ]
        table_data.append(row)
    
    headers = [
        "Config", "Causal", "PyTorch Fwd (ms)", "Flash Fwd (ms)", "Fwd Speedup",
        "PyTorch Bwd (ms)", "Flash Bwd (ms)", "Bwd Speedup", "Mem Reduction", "Accuracy"
    ]
    
    if not FLASH_AVAILABLE:
        headers = ["Config", "Causal", "PyTorch Fwd (ms)", "PyTorch Bwd (ms)", "Memory (MB)",
                  "Flash Fwd", "Flash Bwd", "Fwd Speedup", "Bwd Speedup", "Accuracy"]
    
    return tabulate(table_data, headers=headers, tablefmt="grid")


def plot_results(all_results: List[Dict], save_path: str = "benchmark_results.png"):
    """Create visualization plots of the benchmark results"""
    if not FLASH_AVAILABLE:
        print("Skipping plots as FlashAttention is not available")
        return
        
    # Filter results for plotting
    non_causal_results = [r for r in all_results if not r['causal'] and 'speedup_fwd' in r]
    
    if not non_causal_results:
        print("No FlashAttention results available for plotting")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    seq_lens = [r['seq_len'] for r in non_causal_results]
    fwd_speedups = [r['speedup_fwd'] for r in non_causal_results]
    bwd_speedups = [r['speedup_bwd'] for r in non_causal_results]
    memory_reductions = [r['memory_reduction'] for r in non_causal_results]
    max_errors = [r['max_abs_error'] for r in non_causal_results]
    
    # Forward speedup vs sequence length
    ax1.scatter(seq_lens, fwd_speedups, alpha=0.7, s=60)
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Forward Speedup (x)')
    ax1.set_title('Forward Pass Speedup vs Sequence Length')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # Backward speedup vs sequence length  
    ax2.scatter(seq_lens, bwd_speedups, alpha=0.7, s=60, color='orange')
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Backward Speedup (x)')
    ax2.set_title('Backward Pass Speedup vs Sequence Length')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    # Memory reduction vs sequence length
    ax3.scatter(seq_lens, memory_reductions, alpha=0.7, s=60, color='green')
    ax3.set_xlabel('Sequence Length')
    ax3.set_ylabel('Memory Reduction (%)')
    ax3.set_title('Memory Reduction vs Sequence Length')
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    
    # Accuracy (max absolute error) vs sequence length
    ax4.scatter(seq_lens, max_errors, alpha=0.7, s=60, color='red')
    ax4.set_xlabel('Sequence Length')
    ax4.set_ylabel('Max Absolute Error')
    ax4.set_title('Numerical Accuracy vs Sequence Length')
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plots saved to {save_path}")


def main():
    """Main benchmark function"""
    print("=" * 80)
    print("FlashAttention vs PyTorch MHA - Speed & Accuracy Benchmark")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This benchmark requires a GPU.")
        return
    
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"FlashAttention Available: {FLASH_AVAILABLE}")
    
    config = BenchmarkConfig()
    all_results = []
    
    # Run benchmarks for both causal and non-causal attention
    for causal in [False, True]:
        print(f"\n{'='*60}")
        print(f"Running {'Causal' if causal else 'Non-Causal'} Attention Benchmarks")
        print(f"{'='*60}")
        
        for test_config in config.test_configs:
            try:
                result = run_single_benchmark(test_config, causal=causal)
                all_results.append(result)
            except Exception as e:
                print(f"Error in config {test_config}: {e}")
                continue
    
    # Generate summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(create_summary_table(all_results))
    
    if FLASH_AVAILABLE and all_results:
        # Create plots
        plot_results(all_results)
        
        # Print key insights
        flash_results = [r for r in all_results if 'speedup_fwd' in r]
        if flash_results:
            avg_fwd_speedup = np.mean([r['speedup_fwd'] for r in flash_results])
            avg_bwd_speedup = np.mean([r['speedup_bwd'] for r in flash_results])
            avg_memory_reduction = np.mean([r['memory_reduction'] for r in flash_results])
            accuracy_pass_rate = np.mean([r['is_close'] for r in flash_results]) * 100
            
            print(f"\nKEY INSIGHTS:")
            print(f"• Average forward speedup: {avg_fwd_speedup:.2f}x")
            print(f"• Average backward speedup: {avg_bwd_speedup:.2f}x")
            print(f"• Average memory reduction: {avg_memory_reduction:.1f}%")
            print(f"• Accuracy pass rate: {accuracy_pass_rate:.1f}%")
    
    print(f"\nBenchmark completed! Results for {len(all_results)} configurations.")


if __name__ == "__main__":
    main()
