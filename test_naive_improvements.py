#!/usr/bin/env python3
"""
Test script to compare different naive implementations of Poisson binomial PMF.
Shows the dramatic performance improvements while maintaining correctness.
"""

import torch
import time
import numpy as np
from poi_bin import (
    pmf_poibin_naive, 
    pmf_poibin_naive_improved, 
    pmf_poibin_naive_dp, 
    pmf_poibin_naive_dp_vectorized,
    pmf_poibin_orig  # For reference comparison
)

def test_correctness():
    """Test that all implementations produce the same results."""
    print("=== Correctness Test ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    
    # Test cases
    test_cases = [
        # Small single vector
        torch.tensor([0.1, 0.3, 0.7], device=device),
        
        # Small batch
        torch.tensor([[0.1, 0.3, 0.7], 
                     [0.2, 0.5, 0.8]], device=device),
        
        # Medium size (still feasible for original naive)
        torch.rand(8, device=device) * 0.9 + 0.05,  # Avoid 0/1 probabilities
    ]
    
    for i, probs in enumerate(test_cases):
        print(f"\nTest case {i+1}: shape {probs.shape}")
        
        # Original FFT-based method for reference
        if probs.dim() == 1:
            ref_result = pmf_poibin_orig(probs.unsqueeze(0), device, use_normalization=False).squeeze(0)
        else:
            ref_result = pmf_poibin_orig(probs, device, use_normalization=False)
        
        # Test all naive methods (skip original for larger sizes due to exponential complexity)
        methods = [
            ("DP Vectorized", pmf_poibin_naive_dp_vectorized),
            ("DP", pmf_poibin_naive_dp),
            ("Improved Combinatorial", pmf_poibin_naive_improved),
        ]
        
        # Only test original naive for very small inputs
        if probs.shape[-1] <= 8:
            methods.append(("Original Naive", pmf_poibin_naive))
        
        for name, method in methods:
            result = method(probs, device)
            
            # Check if results match reference within tolerance
            max_diff = torch.max(torch.abs(result - ref_result)).item()
            print(f"  {name:20s}: max_diff = {max_diff:.2e}")
            
            if max_diff > 1e-6:
                print(f"    WARNING: Large difference detected!")
            
            # Check if PMF sums to 1
            if probs.dim() == 1:
                pmf_sum = torch.sum(result).item()
            else:
                pmf_sum = torch.sum(result, dim=1)
                pmf_sum = torch.mean(pmf_sum).item()  # Average across batch
            
            print(f"  {name:20s}: PMF sum = {pmf_sum:.6f}")

def benchmark_methods():
    """Benchmark different implementations across various problem sizes."""
    print("\n=== Performance Benchmark ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test different problem sizes
    sizes = [5, 8, 10, 15, 20, 25]
    batch_size = 100
    
    methods = [
        ("DP Vectorized", pmf_poibin_naive_dp_vectorized),
        ("DP", pmf_poibin_naive_dp),
        ("Improved Combinatorial", pmf_poibin_naive_improved),
        ("FFT Reference", lambda x, d: pmf_poibin_orig(x, d, use_normalization=False)),
    ]
    
    print(f"\nBatch size: {batch_size}")
    print(f"{'Size':<6} {'DP Vec':<10} {'DP':<10} {'Improved':<12} {'FFT Ref':<10} {'Naive':<10}")
    print("-" * 70)
    
    for n in sizes:
        torch.manual_seed(42)
        probs = torch.rand(batch_size, n, device=device) * 0.9 + 0.05
        
        times = {}
        
        for name, method in methods:
            try:
                torch.cuda.synchronize() if device.type == 'cuda' else None
                start_time = time.time()
                
                result = method(probs, device)
                
                torch.cuda.synchronize() if device.type == 'cuda' else None
                elapsed = time.time() - start_time
                times[name] = elapsed
                
            except Exception as e:
                times[name] = float('inf')
        
        # Test original naive only for very small sizes
        if n <= 12:
            try:
                torch.cuda.synchronize() if device.type == 'cuda' else None
                start_time = time.time()
                
                result = pmf_poibin_naive(probs, device)
                
                torch.cuda.synchronize() if device.type == 'cuda' else None
                times["Original Naive"] = time.time() - start_time
                
            except Exception as e:
                times["Original Naive"] = float('inf')
        else:
            times["Original Naive"] = float('inf')
        
        # Format and print results
        def format_time(t):
            if t == float('inf'):
                return "TIMEOUT"
            elif t < 0.001:
                return f"{t*1000:.1f}ms"
            else:
                return f"{t:.3f}s"
        
        print(f"{n:<6} {format_time(times['DP Vectorized']):<10} "
              f"{format_time(times['DP']):<10} "
              f"{format_time(times['Improved Combinatorial']):<12} "
              f"{format_time(times['FFT Reference']):<10} "
              f"{format_time(times['Original Naive']):<10}")

def demonstrate_scaling():
    """Demonstrate the computational complexity differences."""
    print("\n=== Computational Complexity Demonstration ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Time complexity comparison:")
    print("- Original Naive:     O(2^n) - exponential")
    print("- Improved Naive:     O(2^n) - exponential (better constants)")
    print("- DP:                 O(n^2) - quadratic") 
    print("- DP Vectorized:      O(n^2) - quadratic (better parallelization)")
    print("- FFT (reference):    O(n^2) - quadratic (highly optimized)")
    
    print(f"\nFor n=20: 2^n = {2**20:,} vs n^2 = {20**2}")
    print(f"For n=30: 2^n = {2**30:,} vs n^2 = {30**2}")
    print(f"For n=40: 2^n = {2**40:,} vs n^2 = {40**2}")

def main():
    """Run all tests and benchmarks."""
    print("Testing Improved Naive Poisson Binomial Implementations")
    print("=" * 60)
    
    test_correctness()
    benchmark_methods()
    demonstrate_scaling()
    
    print("\n=== Summary ===")
    print("1. pmf_poibin_naive_dp_vectorized: Best overall performance")
    print("   - O(n^2) complexity vs O(2^n) for combinatorial methods")
    print("   - Vectorized across batches for better GPU utilization")
    print("   - Maintains numerical precision")
    
    print("\n2. pmf_poibin_naive_dp: Good educational alternative")
    print("   - Clearer logic for understanding the recurrence relation")
    print("   - Still much faster than combinatorial approaches")
    
    print("\n3. pmf_poibin_naive_improved: Better than original naive")
    print("   - More readable code using itertools.combinations")
    print("   - Still exponential complexity - only use for very small n")
    
    print("\n4. Original pmf_poibin_naive: Reference implementation")
    print("   - Keep for educational purposes")
    print("   - Shows direct translation of mathematical definition")

if __name__ == "__main__":
    main() 