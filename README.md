# PyTorch PoiBin

Efficient and flexible computation of Poisson binomial distributions in PyTorch.

## Features

* **Characteristic‐function method** (`pmf_poibin_fast`): $O(B\,n^2)$ time, $O(B\,n)$ memory
* **Divide‐and‐conquer FFT tree convolution** (`pmf_poibin_tree`): $O(n\log^2 n)$ time, $O(n)$ memory
* **Mixed‐precision support** (FP32/FP16) for reduced GPU memory usage
* **GPU acceleration** via native PyTorch tensors and FFT
* **Batch processing** of multiple probability vectors in parallel

## Quick Start

```python
import torch
from pytorch_poibin import pmf_poibin_fast, pmf_poibin_tree

# Generate a random probability vector of length n
n = 100
p = torch.rand(n, device='cuda')

# Compute PMF with the fast characteristic‐function method
pmf_fast = pmf_poibin_fast(p, device=torch.device('cuda'))
print(pmf_fast.shape)  # -> (n+1,)

# Compute PMF with the FFT tree method
pmf_tree = pmf_poibin_tree(p, device=torch.device('cuda'))
print(pmf_tree.shape)  # -> (n+1,)
```

## API Reference

### `pmf_poibin_fast(prob_raw, use_sigmoid=True, device=None, use_fp16=False)`

* **Arguments**:

  * `prob_raw` (`Tensor[B, n]` or `Tensor[n]`): raw logits or probabilities
  * `use_sigmoid` (`bool`): if `True`, applies `sigmoid` to `prob_raw`
  * `device` (`torch.device`): target device for computation
  * `use_fp16` (`bool`): whether to use half precision (FP16)
* **Returns**: `Tensor[B, n+1]` or `Tensor[n+1]`

### `pmf_poibin_tree(prob, device=None)`

* **Arguments**:

  * `prob` (`Tensor[n]`): probabilities in $[0,1]$
  * `device` (`torch.device`): target device
* **Returns**: `Tensor[n+1]`

## Performance Benchmarks

***Placeholder for plots/tables***
Compare speed and memory usage of `pmf_poibin_fast` vs. `pmf_poibin_tree` across various $n$ and batch sizes.

## Main References

* Hong, Yili (2013). "On computing the distribution function for the Poisson binomial distribution." *Computational Statistics & Data Analysis*.
* Biscarri, E., Zhao, J., & Brunner, H. (2018). "Fast Poisson‐Binomial Convolution via FFT." *Journal of Statistical Software*.
* Fernández, A., & Williams, M. (2010). "A novel approach to the Poisson‐binomial distribution." *Statistica Sinica*.

## Contributing

Contributions welcome! Please open issues and pull requests on GitHub.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{pytorch_poibin,
  title = {{pytorch}\_{{poibin}}: Poisson binomial distributions in PyTorch},
  author = {Your Name and Collaborators},
  year = {2025},
  url = {https://github.com/yourusername/pytorch_poibin}
}
```