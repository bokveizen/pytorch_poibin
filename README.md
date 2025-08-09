# PyTorch Poisson Binomial

> **⚠️ Development Status**: This repository is currently under active development. APIs may change and some planned features are not yet implemented. Use with caution in production environments.

Efficient computation of Poisson binomial distributions in PyTorch with multiple implementation strategies.

## Features

* **Multiple implementation approaches** with different performance characteristics
* **Dynamic Programming method** (`pmf_poibin_naive_dp_vectorized`): O(n²) time, educational and surprisingly fast
* **FFT-based characteristic function** (`pmf_poibin_fft`): O(n²) time for setup + O(n log n) FFT
* **Legacy naive methods** for educational purposes and small problems
* **GPU acceleration** via native PyTorch tensors
* **Batch processing** of multiple probability vectors in parallel
* **Comprehensive numerical verification** ensuring accuracy across all methods

## Current Implementations

### 1. Dynamic Programming
```python
from poi_bin import pmf_poibin_naive_dp_vectorized

# Fast O(n²) method using recurrence relation
pmf = pmf_poibin_naive_dp_vectorized(prob_vector, device)
```

### 2. FFT-based Method
```python
from poi_bin import pmf_poibin_fft

# FFT-based characteristic function approach
pmf = pmf_poibin_fft(prob_vector, device)
```

### 3. Educational Implementations
```python
from poi_bin import pmf_poibin_naive, pmf_poibin_naive_improved, pmf_poibin_naive_dp

# Original naive O(2^n) - for understanding only
pmf = pmf_poibin_naive(prob_vector, device)

# Improved naive using itertools.combinations
pmf = pmf_poibin_naive_improved(prob_vector, device)

# Non-vectorized DP for educational purposes  
pmf = pmf_poibin_naive_dp(prob_vector, device)
```

## Quick Start

```python
import torch
from poi_bin import pmf_poibin_naive_dp_vectorized

# Single probability vector
n = 100
probs = torch.rand(n, device='cuda') * 0.8 + 0.1
pmf = pmf_poibin_naive_dp_vectorized(probs, device=torch.device('cuda'))
print(pmf.shape)  # -> (n+1,)

# Batch processing
batch_size = 32
prob_batch = torch.rand(batch_size, n, device='cuda') * 0.8 + 0.1
pmf_batch = pmf_poibin_naive_dp_vectorized(prob_batch, device=torch.device('cuda'))
print(pmf_batch.shape)  # -> (batch_size, n+1)
```

## Performance Characteristics

| Method | Time Complexity | Current Use Case | Memory | Notes |
|--------|----------------|------------------|---------|-------|
| `pmf_poibin_naive_dp_vectorized` | O(n²) | **Currently best for n < 100** | O(n) | Fast, simple operations |
| `pmf_poibin_fft` | O(n²) + O(n log n) | Interim implementation | O(n) | Will be optimized with tree convolution |
| `pmf_poibin_naive_improved` | O(2^n) | n ≤ 12 only | O(2^n) | Educational |
| `pmf_poibin_naive` | O(2^n) | n ≤ 10 only | O(2^n) | Reference implementation |

**Key Insight**: Despite having the same theoretical complexity, the DP method often outperforms FFT for moderate n due to:
- Lower constant factors (simple arithmetic vs complex number operations)
- Better GPU vectorization of basic operations
- Superior cache locality and memory access patterns

## Testing and Verification

```python
# Run comprehensive numerical verification
python verify_numerical_accuracy.py

# Performance benchmarks
python test_naive_improvements.py

# Detailed FFT vs DP analysis
python analyze_fft_vs_dp.py
```

## Planned Optimizations

### 1. Divide-and-Conquer FFT Tree Convolution
**Target**: O(n log² n) time complexity - **this will become the primary method**
- Break down convolution into smaller subproblems
- Use tree structure to minimize FFT overhead
- Expected to outperform DP for n > 1000 and become the recommended approach
- Will make FFT-based computation the default choice for all problem sizes

### 2. Mixed-Precision Support
- FP16/FP32 support for reduced GPU memory usage
- Maintain numerical accuracy while improving performance

### 3. Specialized Kernels
- Custom CUDA kernels for DP method
- Fused operations to reduce memory bandwidth

### 4. Adaptive Method Selection
```python
# Planned API: automatically choose best method based on problem size
pmf = pmf_poibin_auto(prob_vector, device)  # Chooses optimal implementation
```

## API Reference

### Core Functions

#### `pmf_poibin_naive_dp_vectorized(prob_matrix_raw, device)`
- **Arguments**:
  - `prob_matrix_raw` (`Tensor[B, n]` or `Tensor[n]`): probability values in [0,1]
  - `device` (`torch.device`): target device for computation
- **Returns**: `Tensor[B, n+1]` or `Tensor[n+1]`
- **Time**: O(n²), **Space**: O(n)

#### `pmf_poibin_fft(prob_matrix_raw, device)`
**FFT-based method**
- Same signature as above
- **Time**: O(n²) + O(n log n), **Space**: O(n)

## Mathematical Background

The Poisson binomial distribution generalizes the binomial distribution to the case where each trial has a different success probability p_i. For n independent Bernoulli trials with probabilities [p₁, p₂, ..., pₙ], the PMF gives P(X = k) for k = 0, 1, ..., n.

### Dynamic Programming Approach
Uses the recurrence relation:
```
P_i(k) = P_{i-1}(k) × (1 - p_i) + P_{i-1}(k-1) × p_i
```

### FFT Approach  
Computes the characteristic function and applies inverse DFT.

## Contributing

Contributions welcome! Areas of interest:
- Performance optimizations
- Numerical stability improvements  
- Additional test cases
- Documentation improvements

## License

This project is licensed under the MIT License.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{pytorch_poibin,
  title = {PyTorch Poisson Binomial: Efficient computation with multiple algorithms},
  author = {Bu, Fanchen},
  year = {2025},
  url = {https://github.com/bokveizen/pytorch_poibin}
}

@article{hong2013computing,
  title={On computing the distribution function for the Poisson binomial distribution},
  author={Hong, Yili},
  journal={Computational Statistics \& Data Analysis},
  volume={59},
  pages={41--51},
  year={2013},
  publisher={Elsevier}
}
```

Please also consider citing the following paper:
```bibtex
@inproceedings{Bu2024UCom2,
  title={Tackling Prevalent Conditions in Unsupervised Combinatorial Optimization: Cardinality, Minimum, Covering, and More},
  author={Bu, Fanchen and Jo, Hyeonsoo and Lee, Soo Yong and Ahn, Sungsoo and Shin, Kijung},
  booktitle={ICML},
  year={2024}
}
```
See also the GitHub repository of the paper: [https://github.com/ai4co/unsupervised-CO-ucom2](https://github.com/ai4co/unsupervised-CO-ucom2)