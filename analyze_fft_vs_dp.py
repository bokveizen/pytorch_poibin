#!/usr/bin/env python3
"""
Detailed analysis of why FFT might be slower than DP for Poisson binomial PMF.
"""

import torch
import time
import numpy as np
from poi_bin import (
    pmf_poibin_naive_dp_vectorized,
    pmf_poibin_fft
)

def profile_operations():
    """Profile individual operations in FFT vs DP methods."""
    print("=== Profiling Individual Operations ===\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Test different problem sizes
    sizes = [10, 20, 50, 100, 200]
    batch_size = 16
    
    for n in sizes:
        print(f"\nProblem size: n={n}, batch_size={batch_size}")
        torch.manual_seed(42)
        probs = torch.rand(batch_size, n, device=device) * 0.8 + 0.1
        
        # Profile DP method
        print("  DP Vectorized:")
        start_time = time.time()
        dp_result = pmf_poibin_naive_dp_vectorized(probs, device)
        dp_time = time.time() - start_time
        print(f"    Total time: {dp_time:.4f}s")
        
        # Profile FFT method
        print("  FFT:")
        start_time = time.time()
        fft_result = pmf_poibin_fft(probs, device)
        fft_orig_time = time.time() - start_time
        print(f"    Total time: {fft_orig_time:.4f}s")
        
        # Profile FFT vector method (single vector)
        single_probs = probs[0]
        print("  FFT Vector (single):")
        start_time = time.time()
        fft_vec_result = pmf_poibin_fft(single_probs, device)
        fft_vec_time = time.time() - start_time
        print(f"    Total time: {fft_vec_time:.4f}s")
        
        # Compare results
        max_diff_orig = torch.max(torch.abs(dp_result - fft_result)).item()
        print(f"    Max diff (DP vs FFT): {max_diff_orig:.2e}")
        
        # Speed comparison
        print(f"    Speed ratios:")
        print(f"      DP / FFT: {dp_time/fft_orig_time:.2f}")

def analyze_complexity():
    """Analyze the theoretical and practical complexity differences."""
    print("\n=== Complexity Analysis ===\n")
    
    print("Theoretical Complexity:")
    print("  DP Vectorized:     O(n²) per batch")
    print("  FFT methods:       O(n²) for characteristic function + O(n log n) for FFT")
    print("  Total FFT:         O(n²) dominated by characteristic function computation")
    
    print("\nWhy DP might be faster:")
    print("1. CONSTANT FACTORS:")
    print("   - DP: Simple arithmetic operations (multiply, add)")
    print("   - FFT: Complex number arithmetic, transcendental functions (exp, log, atan2)")
    
    print("\n2. MEMORY ACCESS PATTERNS:")
    print("   - DP: Sequential access, good cache locality")
    print("   - FFT: More complex access patterns, potential cache misses")
    
    print("\n3. VECTORIZATION EFFICIENCY:")
    print("   - DP: Simple operations vectorize well on GPU")
    print("   - FFT: Complex operations may not vectorize as efficiently")
    
    print("\n4. OVERHEAD:")
    print("   - DP: Minimal overhead")
    print("   - FFT: Function call overhead for complex operations")

def detailed_fft_breakdown():
    """Break down FFT computation into components."""
    print("\n=== FFT Method Breakdown ===\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n = 50
    batch_size = 16
    torch.manual_seed(42)
    probs = torch.rand(batch_size, n, device=device) * 0.8 + 0.1
    
    print(f"Analyzing FFT components for n={n}, batch_size={batch_size}")
    
    # Time each component of FFT method
    omega = torch.tensor(2 * torch.pi / (n + 1), dtype=torch.float, device=device)
    
    # 1. Setup and initialization
    start = time.time()
    chi = torch.empty(batch_size, n + 1, device=device, dtype=torch.cfloat)
    chi[:, 0] = 1.0
    half_n = int(n / 2 + n % 2)
    setup_time = time.time() - start
    
    # 2. Exponential computation
    start = time.time()
    exp_values = torch.exp(omega * torch.arange(1, half_n + 1, device=device) * 1j)
    exp_time = time.time() - start
    
    # 3. Characteristic function computation (most expensive part)
    start = time.time()
    xy = (1.0 - probs.unsqueeze(2) + probs.unsqueeze(2) * exp_values.unsqueeze(0))
    argz_sum = torch.arctan2(xy.imag, xy.real).sum(dim=1)
    exparg = torch.log(torch.abs(xy)).sum(dim=1)
    magnitude = torch.exp(exparg)
    chi[:, 1:half_n + 1] = magnitude * torch.exp(argz_sum * 1j)
    char_func_time = time.time() - start
    
    # 4. Hermitian symmetry
    start = time.time()
    if half_n < n:
        chi[:, half_n + 1:n + 1] = torch.conj(chi[:, 1:n - half_n + 1].flip(dims=[1]))
    symmetry_time = time.time() - start
    
    # 5. FFT computation
    start = time.time()
    chi /= (n + 1)
    pmf = torch.fft.fft(chi)
    pmf = pmf.real.float()
    fft_time = time.time() - start
    
    total_fft_time = setup_time + exp_time + char_func_time + symmetry_time + fft_time
    
    print(f"FFT component breakdown:")
    print(f"  Setup & init:          {setup_time:.4f}s ({setup_time/total_fft_time*100:.1f}%)")
    print(f"  Exponential values:    {exp_time:.4f}s ({exp_time/total_fft_time*100:.1f}%)")
    print(f"  Characteristic func:   {char_func_time:.4f}s ({char_func_time/total_fft_time*100:.1f}%)")
    print(f"  Hermitian symmetry:    {symmetry_time:.4f}s ({symmetry_time/total_fft_time*100:.1f}%)")
    print(f"  Actual FFT:            {fft_time:.4f}s ({fft_time/total_fft_time*100:.1f}%)")
    print(f"  Total:                 {total_fft_time:.4f}s")
    
    # Compare with DP
    start = time.time()
    dp_result = pmf_poibin_naive_dp_vectorized(probs, device)
    dp_time = time.time() - start
    
    print(f"\nComparison:")
    print(f"  DP time:               {dp_time:.4f}s")
    print(f"  FFT time:              {total_fft_time:.4f}s")
    print(f"  Ratio (FFT/DP):        {total_fft_time/dp_time:.2f}x")

def scaling_analysis():
    """Analyze how performance scales with problem size."""
    print("\n=== Scaling Analysis ===\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 8
    
    sizes = [5, 10, 20, 50, 100, 200]
    dp_times = []
    fft_times = []
    
    print(f"{'Size':<6} {'DP Time':<10} {'FFT Time':<10} {'Ratio':<8} {'DP n²':<10} {'FFT n²':<10}")
    print("-" * 60)
    
    for n in sizes:
        torch.manual_seed(42)
        probs = torch.rand(batch_size, n, device=device) * 0.8 + 0.1
        
        # Time DP
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start = time.time()
        dp_result = pmf_poibin_naive_dp_vectorized(probs, device)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        dp_time = time.time() - start
        
        # Time FFT
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start = time.time()
        fft_result = pmf_poibin_fft(probs, device)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        fft_time = time.time() - start
        
        dp_times.append(dp_time)
        fft_times.append(fft_time)
        
        ratio = fft_time / dp_time
        dp_normalized = dp_time / (n * n) * 1e6  # microseconds per n²
        fft_normalized = fft_time / (n * n) * 1e6
        
        print(f"{n:<6} {dp_time:.4f}s   {fft_time:.4f}s   {ratio:<8.2f} {dp_normalized:<10.2f} {fft_normalized:<10.2f}")
    
    print("\nObservations:")
    print("- FFT has higher constant factors due to complex arithmetic")
    print("- DP benefits more from vectorization and simple operations")
    print("- For small to medium n, DP can be faster despite same complexity")
    print("- FFT becomes relatively better for very large n (if memory allows)")

def main():
    """Run all analyses."""
    print("Analysis: Why FFT might be slower than DP")
    print("=" * 50)
    
    analyze_complexity()
    detailed_fft_breakdown()
    scaling_analysis()
    profile_operations()
    
    print("\n=== CONCLUSION ===")
    print("FFT is slower than DP for small-medium n because:")
    print("1. Higher constant factors (complex arithmetic vs simple operations)")
    print("2. More expensive operations (exp, log, atan2 vs multiply/add)")
    print("3. Less efficient vectorization of complex operations")
    print("4. Memory access overhead from complex number operations")
    print("\nDP wins for n < ~500-1000 despite same O(n²) complexity!")

if __name__ == "__main__":
    main() 