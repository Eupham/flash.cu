#!/usr/bin/env python3
"""
Comprehensive test suite for Flash Attention CUDA kernels.
Tests both forward and backward passes with various configurations.
"""

import os
import sys
import ctypes
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import pytest
import time

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class FlashAttentionCUDA:
    """
    Python wrapper for CUDA Flash Attention kernels.
    Provides interface to call the compiled CUDA kernels from Python.
    """
    
    def __init__(self, library_path: str = "./libflash_attention.so"):
        """Initialize the CUDA library wrapper."""
        if not os.path.exists(library_path):
            raise FileNotFoundError(f"CUDA library not found at {library_path}")
        
        self.lib = ctypes.CDLL(library_path)
        self._setup_function_signatures()
    
    def _setup_function_signatures(self):
        """Setup C function signatures for proper type checking."""
        # Forward pass
        self.lib.launch_flash_attention_forward.argtypes = [
            ctypes.c_void_p,  # Q
            ctypes.c_void_p,  # K
            ctypes.c_void_p,  # V
            ctypes.c_void_p,  # O
            ctypes.c_void_p,  # M
            ctypes.c_float,   # sm_scale
            ctypes.c_int,     # B
            ctypes.c_int,     # H
            ctypes.c_int,     # N
            ctypes.c_int,     # D
            ctypes.c_bool,    # causal
            ctypes.c_void_p   # stream
        ]
        
        # Backward preprocess
        self.lib.launch_flash_attention_backward_preprocess.argtypes = [
            ctypes.c_void_p,  # O
            ctypes.c_void_p,  # dO
            ctypes.c_void_p,  # Delta
            ctypes.c_int,     # B
            ctypes.c_int,     # H
            ctypes.c_int,     # N
            ctypes.c_int,     # D
            ctypes.c_void_p   # stream
        ]
        
        # Backward pass
        self.lib.launch_flash_attention_backward.argtypes = [
            ctypes.c_void_p,  # Q
            ctypes.c_void_p,  # K
            ctypes.c_void_p,  # V
            ctypes.c_void_p,  # dO
            ctypes.c_void_p,  # dQ
            ctypes.c_void_p,  # dK
            ctypes.c_void_p,  # dV
            ctypes.c_void_p,  # M
            ctypes.c_void_p,  # Delta
            ctypes.c_float,   # sm_scale
            ctypes.c_int,     # B
            ctypes.c_int,     # H
            ctypes.c_int,     # N
            ctypes.c_int,     # D
            ctypes.c_bool,    # causal
            ctypes.c_void_p   # stream
        ]
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                causal: bool = False, sm_scale: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of Flash Attention.
        
        Args:
            Q: Query tensor [B, H, N, D]
            K: Key tensor [B, H, N, D]
            V: Value tensor [B, H, N, D]
            causal: Whether to apply causal masking
            sm_scale: Scaling factor for attention (default: 1/sqrt(D))
            
        Returns:
            O: Output tensor [B, H, N, D]
            M: Max values for backward pass [B, H, N]
        """
        B, H, N, D = Q.shape
        assert K.shape == (B, H, N, D), f"K shape {K.shape} doesn't match Q shape {Q.shape}"
        assert V.shape == (B, H, N, D), f"V shape {V.shape} doesn't match Q shape {Q.shape}"
        assert Q.dtype == torch.float16, "Only float16 is supported"
        assert Q.device.type == "cuda", "CUDA tensors required"
        
        if sm_scale is None:
            sm_scale = 1.0 / np.sqrt(D)
        
        # Allocate output tensors
        O = torch.zeros_like(Q)
        M = torch.zeros((B, H, N), dtype=torch.float32, device=Q.device)
        
        # Get CUDA stream
        stream = torch.cuda.current_stream().cuda_stream
        
        # Call CUDA kernel
        self.lib.launch_flash_attention_forward(
            Q.data_ptr(), K.data_ptr(), V.data_ptr(), O.data_ptr(), M.data_ptr(),
            ctypes.c_float(sm_scale), B, H, N, D, causal, stream
        )
        
        torch.cuda.synchronize()
        return O, M
    
    def backward_preprocess(self, O: torch.Tensor, dO: torch.Tensor) -> torch.Tensor:
        """
        Preprocess step for backward pass.
        
        Args:
            O: Output from forward pass [B, H, N, D]
            dO: Gradient of output [B, H, N, D]
            
        Returns:
            Delta: Preprocessing result [B, H, N]
        """
        B, H, N, D = O.shape
        assert dO.shape == O.shape, f"dO shape {dO.shape} doesn't match O shape {O.shape}"
        
        Delta = torch.zeros((B, H, N), dtype=torch.float32, device=O.device)
        stream = torch.cuda.current_stream().cuda_stream
        
        self.lib.launch_flash_attention_backward_preprocess(
            O.data_ptr(), dO.data_ptr(), Delta.data_ptr(),
            B, H, N, D, stream
        )
        
        torch.cuda.synchronize()
        return Delta
    
    def backward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, dO: torch.Tensor,
                 M: torch.Tensor, Delta: torch.Tensor, causal: bool = False, 
                 sm_scale: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Backward pass of Flash Attention.
        
        Args:
            Q: Query tensor [B, H, N, D]
            K: Key tensor [B, H, N, D]
            V: Value tensor [B, H, N, D]
            dO: Gradient of output [B, H, N, D]
            M: Max values from forward pass [B, H, N]
            Delta: Preprocessing result [B, H, N]
            causal: Whether causal masking was used
            sm_scale: Scaling factor used in forward pass
            
        Returns:
            dQ: Gradient of Q [B, H, N, D]
            dK: Gradient of K [B, H, N, D]
            dV: Gradient of V [B, H, N, D]
        """
        B, H, N, D = Q.shape
        if sm_scale is None:
            sm_scale = 1.0 / np.sqrt(D)
        
        # Allocate gradient tensors
        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)
        
        stream = torch.cuda.current_stream().cuda_stream
        
        self.lib.launch_flash_attention_backward(
            Q.data_ptr(), K.data_ptr(), V.data_ptr(), dO.data_ptr(),
            dQ.data_ptr(), dK.data_ptr(), dV.data_ptr(),
            M.data_ptr(), Delta.data_ptr(),
            ctypes.c_float(sm_scale), B, H, N, D, causal, stream
        )
        
        torch.cuda.synchronize()
        return dQ, dK, dV


def reference_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                       causal: bool = False, sm_scale: Optional[float] = None) -> torch.Tensor:
    """
    Reference implementation of scaled dot-product attention using PyTorch.
    
    Args:
        Q: Query tensor [B, H, N, D]
        K: Key tensor [B, H, N, D]
        V: Value tensor [B, H, N, D]
        causal: Whether to apply causal masking
        sm_scale: Scaling factor (default: 1/sqrt(D))
        
    Returns:
        O: Output tensor [B, H, N, D]
    """
    B, H, N, D = Q.shape
    if sm_scale is None:
        sm_scale = 1.0 / np.sqrt(D)
    
    # Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) * sm_scale  # [B, H, N, N]
    
    # Apply causal mask if needed
    if causal:
        mask = torch.triu(torch.ones(N, N, device=Q.device, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(mask, float('-inf'))
    
    # Apply softmax
    attn_weights = F.softmax(scores, dim=-1)
    
    # Apply attention to values
    O = torch.matmul(attn_weights, V)
    
    return O


class TestFlashAttentionCUDA:
    """Test suite for Flash Attention CUDA kernels."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Compile CUDA kernels if library doesn't exist
        self.compile_cuda_kernels()
        
        # Initialize CUDA wrapper
        try:
            self.flash_attn = FlashAttentionCUDA()
        except FileNotFoundError:
            pytest.skip("Could not load CUDA library")
    
    def compile_cuda_kernels(self):
        """Compile CUDA kernels if not already compiled."""
        lib_path = "./libflash_attention.so"
        if os.path.exists(lib_path):
            return
        
        # Create CUDA source file
        cuda_source = """
        // Include the CUDA kernel code here
        // This would contain the entire CUDA kernel implementation
        // from the provided code
        """
        
        with open("flash_attention.cu", "w") as f:
            f.write(cuda_source)
        
        # Compile command (simplified - may need adjustment based on system)
        compile_cmd = (
            "nvcc -shared -o libflash_attention.so flash_attention.cu "
            "-lcuda -lcublas --compiler-options '-fPIC' "
            "-gencode arch=compute_80,code=sm_80"
        )
        
        print(f"Compiling CUDA kernels with: {compile_cmd}")
        result = os.system(compile_cmd)
        if result != 0:
            pytest.skip("Failed to compile CUDA kernels")
    
    def generate_test_tensors(self, B: int, H: int, N: int, D: int, dtype=torch.float16):
        """Generate test tensors with proper initialization."""
        torch.manual_seed(42)  # For reproducibility
        
        Q = torch.randn(B, H, N, D, dtype=dtype, device=DEVICE)
        K = torch.randn(B, H, N, D, dtype=dtype, device=DEVICE)
        V = torch.randn(B, H, N, D, dtype=dtype, device=DEVICE)
        
        # Normalize to prevent overflow in attention computation
        Q = Q / np.sqrt(D)
        K = K / np.sqrt(D)
        
        return Q, K, V
    
    @pytest.mark.parametrize("B,H,N,D", [
        (1, 1, 128, 64),
        (2, 8, 256, 64),
        (1, 4, 512, 64),
        (4, 16, 1024, 64),
    ])
    @pytest.mark.parametrize("causal", [False, True])
    def test_forward_pass_correctness(self, B, H, N, D, causal):
        """Test forward pass correctness against reference implementation."""
        Q, K, V = self.generate_test_tensors(B, H, N, D)
        sm_scale = 1.0 / np.sqrt(D)
        
        # CUDA implementation
        O_cuda, M_cuda = self.flash_attn.forward(Q, K, V, causal=causal, sm_scale=sm_scale)
        
        # Reference implementation
        Q_ref = Q.float()
        K_ref = K.float()
        V_ref = V.float()
        O_ref = reference_attention(Q_ref, K_ref, V_ref, causal=causal, sm_scale=sm_scale)
        
        # Compare outputs (allow for some numerical differences)
        O_cuda_float = O_cuda.float()
        max_diff = torch.max(torch.abs(O_cuda_float - O_ref))
        mean_diff = torch.mean(torch.abs(O_cuda_float - O_ref))
        
        print(f"Forward pass - Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")
        
        # Assertions with reasonable tolerances for half precision
        assert max_diff < 1e-2, f"Max difference too large: {max_diff}"
        assert mean_diff < 1e-3, f"Mean difference too large: {mean_diff}"
    
    @pytest.mark.parametrize("B,H,N,D", [
        (1, 1, 128, 64),
        (2, 4, 256, 64),
    ])
    @pytest.mark.parametrize("causal", [False, True])
    def test_backward_pass_correctness(self, B, H, N, D, causal):
        """Test backward pass correctness using gradient checking."""
        Q, K, V = self.generate_test_tensors(B, H, N, D)
        sm_scale = 1.0 / np.sqrt(D)
        
        # Enable gradients for reference computation
        Q_ref = Q.float().requires_grad_(True)
        K_ref = K.float().requires_grad_(True)
        V_ref = V.float().requires_grad_(True)
        
        # Forward pass with reference
        O_ref = reference_attention(Q_ref, K_ref, V_ref, causal=causal, sm_scale=sm_scale)
        
        # Create gradient output
        dO = torch.randn_like(O_ref)
        
        # Compute reference gradients
        O_ref.backward(dO, retain_graph=True)
        dQ_ref = Q_ref.grad.clone()
        dK_ref = K_ref.grad.clone()
        dV_ref = V_ref.grad.clone()
        
        # CUDA implementation
        O_cuda, M_cuda = self.flash_attn.forward(Q, K, V, causal=causal, sm_scale=sm_scale)
        dO_half = dO.half()
        Delta = self.flash_attn.backward_preprocess(O_cuda, dO_half)
        dQ_cuda, dK_cuda, dV_cuda = self.flash_attn.backward(
            Q, K, V, dO_half, M_cuda, Delta, causal=causal, sm_scale=sm_scale
        )
        
        # Compare gradients
        def compare_gradients(grad_cuda, grad_ref, name):
            grad_cuda_float = grad_cuda.float()
            max_diff = torch.max(torch.abs(grad_cuda_float - grad_ref))
            mean_diff = torch.mean(torch.abs(grad_cuda_float - grad_ref))
            rel_error = mean_diff / (torch.mean(torch.abs(grad_ref)) + 1e-8)
            
            print(f"{name} - Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}, Rel error: {rel_error:.6f}")
            
            assert max_diff < 1e-1, f"{name} max difference too large: {max_diff}"
            assert rel_error < 0.1, f"{name} relative error too large: {rel_error}"
        
        compare_gradients(dQ_cuda, dQ_ref, "dQ")
        compare_gradients(dK_cuda, dK_ref, "dK")
        compare_gradients(dV_cuda, dV_ref, "dV")
    
    def test_causal_mask_effect(self):
        """Test that causal masking produces different results."""
        B, H, N, D = 1, 1, 64, 64
        Q, K, V = self.generate_test_tensors(B, H, N, D)
        
        # Forward pass without causal mask
        O_non_causal, _ = self.flash_attn.forward(Q, K, V, causal=False)
        
        # Forward pass with causal mask
        O_causal, _ = self.flash_attn.forward(Q, K, V, causal=True)
        
        # Results should be different
        diff = torch.mean(torch.abs(O_non_causal - O_causal))
        assert diff > 1e-3, f"Causal and non-causal results too similar: {diff}"
        
        print(f"Causal mask effect - Mean difference: {diff:.6f}")
    
    def test_memory_efficiency(self):
        """Test memory usage doesn't grow excessively with sequence length."""
        B, H, D = 1, 8, 64
        
        memory_usage = []
        sequence_lengths = [128, 256, 512, 1024]
        
        for N in sequence_lengths:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            Q, K, V = self.generate_test_tensors(B, H, N, D)
            O, M = self.flash_attn.forward(Q, K, V)
            
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
            memory_usage.append(peak_memory)
            
            print(f"N={N}: Peak memory = {peak_memory:.3f} GB")
        
        # Memory should grow roughly linearly with sequence length (not quadratically)
        for i in range(1, len(memory_usage)):
            ratio = memory_usage[i] / memory_usage[i-1]
            seq_ratio = sequence_lengths[i] / sequence_lengths[i-1]
            # Memory growth should be closer to linear than quadratic
            assert ratio < seq_ratio * 1.5, f"Memory growth too steep: {ratio} vs {seq_ratio}"
    
    @pytest.mark.parametrize("N", [128, 256, 512, 1024])
    def test_performance_scaling(self, N):
        """Test performance scaling with sequence length."""
        B, H, D = 2, 8, 64
        Q, K, V = self.generate_test_tensors(B, H, N, D)
        
        # Warmup
        for _ in range(5):
            self.flash_attn.forward(Q, K, V)
        
        torch.cuda.synchronize()
        
        # Benchmark forward pass
        start_time = time.time()
        num_runs = 10
        
        for _ in range(num_runs):
            O, M = self.flash_attn.forward(Q, K, V)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs * 1000  # ms
        flops = 4 * B * H * N * N * D  # Approximate FLOPs for attention
        tflops = flops / (avg_time * 1e-3) / 1e12
        
        print(f"N={N}: {avg_time:.2f} ms, {tflops:.2f} TFLOPS")
        
        # Basic sanity check - should complete in reasonable time
        assert avg_time < 1000, f"Forward pass too slow: {avg_time} ms"
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        B, H, N, D = 1, 1, 64, 64
        
        # Test with large values
        Q = torch.randn(B, H, N, D, dtype=torch.float16, device=DEVICE) * 10
        K = torch.randn(B, H, N, D, dtype=torch.float16, device=DEVICE) * 10
        V = torch.randn(B, H, N, D, dtype=torch.float16, device=DEVICE) * 10
        
        O, M = self.flash_attn.forward(Q, K, V)
        
        # Check for NaN or Inf
        assert not torch.isnan(O).any(), "Output contains NaN"
        assert not torch.isinf(O).any(), "Output contains Inf"
        assert not torch.isnan(M).any(), "M contains NaN"
        assert not torch.isinf(M).any(), "M contains Inf"
        
        print("Numerical stability test passed")
    
    def test_different_head_dimensions(self):
        """Test with different head dimensions (if supported)."""
        B, H, N = 1, 4, 128
        
        # Note: The kernel is templated for HEAD_DIM=64
        # This test would need kernel modifications to support other dimensions
        for D in [64]:  # Currently only 64 is supported
            Q, K, V = self.generate_test_tensors(B, H, N, D)
            O, M = self.flash_attn.forward(Q, K, V)
            
            assert O.shape == Q.shape, f"Output shape mismatch for D={D}"
            print(f"Head dimension D={D} test passed")


def benchmark_flash_attention():
    """Benchmark Flash Attention against reference implementation."""
    print("\n" + "="*60)
    print("FLASH ATTENTION BENCHMARK")
    print("="*60)
    
    try:
        flash_attn = FlashAttentionCUDA()
    except FileNotFoundError:
        print("CUDA library not found. Please compile the kernels first.")
        return
    
    configs = [
        (1, 8, 512, 64),
        (2, 16, 1024, 64),
        (4, 32, 2048, 64),
    ]
    
    for B, H, N, D in configs:
        print(f"\nTesting B={B}, H={H}, N={N}, D={D}")
        
        # Generate test data
        Q = torch.randn(B, H, N, D, dtype=torch.float16, device=DEVICE)
        K = torch.randn(B, H, N, D, dtype=torch.float16, device=DEVICE)
        V = torch.randn(B, H, N, D, dtype=torch.float16, device=DEVICE)
        
        # Warmup
        for _ in range(3):
            flash_attn.forward(Q, K, V)
        
        # Benchmark Flash Attention
        torch.cuda.synchronize()
        start = time.time()
        num_runs = 10
        
        for _ in range(num_runs):
            O, M = flash_attn.forward(Q, K, V)
        
        torch.cuda.synchronize()
        flash_time = (time.time() - start) / num_runs * 1000
        
        # Benchmark reference implementation
        Q_ref = Q.float()
        K_ref = K.float()
        V_ref = V.float()
        
        torch.cuda.synchronize()
        start = time.time()
        
        for _ in range(num_runs):
            O_ref = reference_attention(Q_ref, K_ref, V_ref)
        
        torch.cuda.synchronize()
        ref_time = (time.time() - start) / num_runs * 1000
        
        # Calculate metrics
        flops = 4 * B * H * N * N * D
        flash_tflops = flops / (flash_time * 1e-3) / 1e12
        ref_tflops = flops / (ref_time * 1e-3) / 1e12
        speedup = ref_time / flash_time
        
        print(f"Flash Attention: {flash_time:.2f} ms, {flash_tflops:.2f} TFLOPS")
        print(f"Reference:       {ref_time:.2f} ms, {ref_tflops:.2f} TFLOPS")
        print(f"Speedup:         {speedup:.2f}x")


if __name__ == "__main__":
    # Run tests
    print("Running Flash Attention CUDA tests...")
    
    # Basic functionality test
    if torch.cuda.is_available():
        test_suite = TestFlashAttentionCUDA()
        test_suite.setup()
        
        try:
            # Run a simple test
            test_suite.test_forward_pass_correctness(1, 1, 128, 64, False)
            print("✓ Forward pass test passed")
            
            test_suite.test_backward_pass_correctness(1, 1, 128, 64, False)
            print("✓ Backward pass test passed")
            
            test_suite.test_causal_mask_effect()
            print("✓ Causal mask test passed")
            
            test_suite.test_numerical_stability()
            print("✓ Numerical stability test passed")
            
            print("\n✓ All basic tests passed!")
            
        except Exception as e:
            print(f"✗ Test failed: {e}")
        
        # Run benchmark
        benchmark_flash_attention()
    
    else:
        print("CUDA not available. Skipping tests.")
    
    # Run with pytest for full test suite
    print("\nTo run full test suite with pytest:")
    print("pytest test_flash_attention_cuda.py -v")
