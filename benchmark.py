# Comprehensive benchmark for Flash Attention implementations
import torch
import torch.nn.functional as F
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import pytest

# Import our implementations
from attention import attention as triton_attention
from cuda_attention import cuda_flash_attention, CUDA_AVAILABLE

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AttentionBenchmark:
    def __init__(self, device=None):
        self.device = device or get_device()
        self.results = {}
        
    def reference_attention(self, q, k, v, causal=True, sm_scale=None):
        """Reference PyTorch implementation"""
        if sm_scale is None:
            sm_scale = 1.0 / (q.shape[-1] ** 0.5)
            
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
        
        # Apply causal mask if needed
        if causal:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device))
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        
        return output
    
    def generate_test_tensors(self, batch_size, num_heads, seq_len, head_dim, dtype=torch.float16):
        """Generate random test tensors"""
        torch.manual_seed(42)  # For reproducibility
        
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                       dtype=dtype, device=self.device, requires_grad=True)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                       dtype=dtype, device=self.device, requires_grad=True)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                       dtype=dtype, device=self.device, requires_grad=True)
        
        return q, k, v
    
    def benchmark_forward(self, impl_func, q, k, v, causal=True, sm_scale=None, warmup=10, trials=100):
        """Benchmark forward pass"""
        # Warmup
        for _ in range(warmup):
            output = impl_func(q, k, v, causal, sm_scale)
            torch.cuda.synchronize()
        
        # Benchmark
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(trials):
            output = impl_func(q, k, v, causal, sm_scale)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / trials
        return output, avg_time
    
    def benchmark_backward(self, impl_func, q, k, v, causal=True, sm_scale=None, warmup=10, trials=100):
        """Benchmark backward pass"""
        # Warmup
        for _ in range(warmup):
            q_copy = q.clone().detach().requires_grad_(True)
            k_copy = k.clone().detach().requires_grad_(True)
            v_copy = v.clone().detach().requires_grad_(True)
            
            output = impl_func(q_copy, k_copy, v_copy, causal, sm_scale)
            grad_output = torch.randn_like(output)
            output.backward(grad_output)
            torch.cuda.synchronize()
        
        # Benchmark
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(trials):
            q_copy = q.clone().detach().requires_grad_(True)
            k_copy = k.clone().detach().requires_grad_(True)
            v_copy = v.clone().detach().requires_grad_(True)
            
            output = impl_func(q_copy, k_copy, v_copy, causal, sm_scale)
            grad_output = torch.randn_like(output)
            output.backward(grad_output)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / trials
        return avg_time
    
    def accuracy_test(self, impl_func, q, k, v, causal=True, sm_scale=None, atol=1e-2, rtol=1e-2):
        """Test accuracy against reference implementation"""
        ref_output = self.reference_attention(q, k, v, causal, sm_scale)
        impl_output = impl_func(q, k, v, causal, sm_scale)
        
        try:
            torch.testing.assert_close(ref_output, impl_output, atol=atol, rtol=rtol)
            return True, 0.0
        except AssertionError as e:
            max_diff = torch.max(torch.abs(ref_output - impl_output)).item()
            return False, max_diff
    
    def compute_flops(self, batch_size, num_heads, seq_len, head_dim, causal=True):
        """Compute theoretical FLOPs for attention"""
        # QK^T: batch_size * num_heads * seq_len * seq_len * head_dim
        # Softmax: batch_size * num_heads * seq_len * seq_len (approx)
        # PV: batch_size * num_heads * seq_len * seq_len * head_dim
        
        flops_qk = batch_size * num_heads * seq_len * seq_len * head_dim
        flops_softmax = batch_size * num_heads * seq_len * seq_len * 5  # approx operations
        flops_pv = batch_size * num_heads * seq_len * seq_len * head_dim
        
        total_flops = 2 * (flops_qk + flops_pv) + flops_softmax
        
        if causal:
            total_flops *= 0.5  # Causal attention has half the operations
            
        return total_flops
    
    def run_benchmark_suite(self, configs: List[Tuple[int, int, int, int]]):
        """Run comprehensive benchmark suite"""
        implementations = {
            "Reference": self.reference_attention,
            "Triton": triton_attention,
        }
        
        if CUDA_AVAILABLE:
            implementations["CUDA"] = cuda_flash_attention
        
        results = {
            "configs": [],
            "forward_times": {name: [] for name in implementations},
            "backward_times": {name: [] for name in implementations},
            "accuracy": {name: [] for name in implementations},
            "tflops_forward": {name: [] for name in implementations},
            "tflops_backward": {name: [] for name in implementations},
        }
        
        for batch_size, num_heads, seq_len, head_dim in configs:
            print(f"Benchmarking config: B={batch_size}, H={num_heads}, S={seq_len}, D={head_dim}")
            
            q, k, v = self.generate_test_tensors(batch_size, num_heads, seq_len, head_dim)
            sm_scale = 1.0 / (head_dim ** 0.5)
            
            theoretical_flops = self.compute_flops(batch_size, num_heads, seq_len, head_dim)
            
            config_str = f"B{batch_size}_H{num_heads}_S{seq_len}_D{head_dim}"
            results["configs"].append(config_str)
            
            for name, impl_func in implementations.items():
                try:
                    # Forward benchmark
                    output, fwd_time = self.benchmark_forward(impl_func, q, k, v, True, sm_scale)
                    results["forward_times"][name].append(fwd_time * 1000)  # Convert to ms
                    results["tflops_forward"][name].append(theoretical_flops / (fwd_time * 1e12))
                    
                    # Backward benchmark (skip for reference to save time)
                    if name != "Reference":
                        bwd_time = self.benchmark_backward(impl_func, q, k, v, True, sm_scale)
                        results["backward_times"][name].append(bwd_time * 1000)
                        results["tflops_backward"][name].append(theoretical_flops * 2.5 / (bwd_time * 1e12))
                    else:
                        results["backward_times"][name].append(0)
                        results["tflops_backward"][name].append(0)
                    
                    # Accuracy test
                    if name != "Reference":
                        is_accurate, max_diff = self.accuracy_test(impl_func, q, k, v, True, sm_scale)
                        results["accuracy"][name].append(max_diff)
                        print(f"  {name}: Forward={fwd_time*1000:.2f}ms, "
                              f"TFLOPS={theoretical_flops/(fwd_time*1e12):.2f}, "
                              f"Accurate={is_accurate}, MaxDiff={max_diff:.6f}")
                    else:
                        results["accuracy"][name].append(0.0)
                        print(f"  {name}: Forward={fwd_time*1000:.2f}ms (reference)")
                        
                except Exception as e:
                    print(f"  {name}: FAILED - {e}")
                    results["forward_times"][name].append(float('inf'))
                    results["backward_times"][name].append(float('inf'))
                    results["tflops_forward"][name].append(0)
                    results["tflops_backward"][name].append(0)
                    results["accuracy"][name].append(float('inf'))
        
        return results
    
    def plot_results(self, results, save_path="benchmark_results.png"):
        """Plot benchmark results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        configs = results["configs"]
        x_pos = np.arange(len(configs))
        
        # Forward time comparison
        ax1 = axes[0, 0]
        for name in results["forward_times"]:
            if name != "Reference":  # Skip reference for clarity
                times = results["forward_times"][name]
                ax1.bar(x_pos + len(results["forward_times"]) * 0.1 * list(results["forward_times"].keys()).index(name), 
                       times, width=0.1, label=name)
        ax1.set_xlabel("Configuration")
        ax1.set_ylabel("Forward Time (ms)")
        ax1.set_title("Forward Pass Performance")
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(configs, rotation=45)
        ax1.legend()
        ax1.set_yscale('log')
        
        # TFLOPS comparison
        ax2 = axes[0, 1]
        for name in results["tflops_forward"]:
            if name != "Reference":
                tflops = results["tflops_forward"][name]
                ax2.bar(x_pos + len(results["tflops_forward"]) * 0.1 * list(results["tflops_forward"].keys()).index(name), 
                       tflops, width=0.1, label=name)
        ax2.set_xlabel("Configuration")
        ax2.set_ylabel("TFLOPS")
        ax2.set_title("Forward Pass Throughput")
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(configs, rotation=45)
        ax2.legend()
        
        # Accuracy comparison
        ax3 = axes[1, 0]
        for name in results["accuracy"]:
            if name != "Reference":
                accuracy = results["accuracy"][name]
                ax3.bar(x_pos + len(results["accuracy"]) * 0.1 * list(results["accuracy"].keys()).index(name), 
                       accuracy, width=0.1, label=name)
        ax3.set_xlabel("Configuration")
        ax3.set_ylabel("Max Difference from Reference")
        ax3.set_title("Accuracy (Lower is Better)")
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(configs, rotation=45)
        ax3.legend()
        ax3.set_yscale('log')
        
        # Backward time comparison
        ax4 = axes[1, 1]
        for name in results["backward_times"]:
            if name != "Reference":
                times = results["backward_times"][name]
                ax4.bar(x_pos + len(results["backward_times"]) * 0.1 * list(results["backward_times"].keys()).index(name), 
                       times, width=0.1, label=name)
        ax4.set_xlabel("Configuration")
        ax4.set_ylabel("Backward Time (ms)")
        ax4.set_title("Backward Pass Performance")
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(configs, rotation=45)
        ax4.legend()
        ax4.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig

def main():
    """Main benchmark function"""
    print("Flash Attention Benchmark Suite")
    print("=" * 50)
    
    benchmark = AttentionBenchmark()
    
    # Test configurations: (batch_size, num_heads, seq_len, head_dim)
    configs = [
        (1, 8, 512, 64),
        (1, 8, 1024, 64),
        (1, 8, 2048, 64),
        (4, 8, 512, 64),
        (4, 8, 1024, 64),
        (8, 16, 512, 64),
        (2, 12, 1024, 128),
    ]
    
    print(f"CUDA kernels available: {CUDA_AVAILABLE}")
    print(f"Device: {benchmark.device}")
    print(f"Configurations to test: {len(configs)}")
    print()
    
    # Run benchmark suite
    results = benchmark.run_benchmark_suite(configs)
    
    # Plot results
    print("\nGenerating plots...")
    benchmark.plot_results(results)
    
    # Print summary
    print("\nBenchmark Summary:")
    print("=" * 50)
    
    for i, config in enumerate(results["configs"]):
        print(f"\nConfiguration: {config}")
        for impl in results["forward_times"]:
            if impl != "Reference":
                fwd_time = results["forward_times"][impl][i]
                tflops = results["tflops_forward"][impl][i]
                accuracy = results["accuracy"][impl][i]
                print(f"  {impl:8s}: {fwd_time:6.2f}ms, {tflops:6.2f} TFLOPS, MaxDiff: {accuracy:.2e}")

if __name__ == "__main__":
    main()
