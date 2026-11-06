# PyTorch Poisson Binomial

Batched, differentiable Poisson binomial PMFs and tail probabilities on CPU and CUDA.
For independent Bernoulli trials with probabilities `p_i`, compute the distribution
of their sum even when each trial has a different probability.

Version 0.3.0 is prepared for release; this change does not publish to PyPI.
The package replaces the original research prototype with a clean API.
See [migration](docs/migration.md) and [release notes](CHANGELOG.md).

## Install

Requires Python 3.10+ and PyTorch 2.6+. Install the appropriate CPU or CUDA build
of PyTorch, then install from a checkout:

```bash
git clone https://github.com/bokveizen/pytorch_poibin.git
cd pytorch_poibin
python -m pip install .

# Development, tests, and benchmarks:
python -m pip install -e '.[dev]'
```

PyTorch is the only runtime dependency. SciPy is an independent test and benchmark
reference, not a library dependency.

## Quick start

```python
import torch
from pytorch_poibin import pmf, cdf, sf, proportional_tail

probs = torch.tensor([0.2, 0.7], dtype=torch.float64)
print(pmf(probs))       # [0.24, 0.62, 0.14]
print(cdf(probs, 0))    # P(X <= 0) = 0.24
print(sf(probs, 1))     # P(X > 1)  = 0.14

batch = torch.tensor([[0.2, 0.7], [0.4, 0.5]], dtype=torch.float64)
assert pmf(batch).shape == (2, 3)
assert cdf(batch, 0).shape == (2,)

interesting = torch.tensor([True, False])
print(proportional_tail(batch, 0.5, interesting))
# P(interesting successes / total successes < 0.5) = [0.56, 0.30]
```

Inputs have shape `[..., n]` with any number of batch dimensions. PMFs have shape
`[..., n+1]`, ordered by success count; tails have shape `[...]`. Unbatched tails
are scalar tensors. Zero trials have PMF `[1]`.

The input determines device and precision. Float32 and float64 are preserved;
float16/bfloat16 compute and return float32. Integer and complex inputs are
rejected. Probabilities must be finite and in `[0, 1]`.

## Gradients and CUDA

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
probs = torch.tensor([0.2, 0.3, 0.7], device=device,
                     dtype=torch.float64, requires_grad=True)
loss = sf(probs, 1)
loss.backward()
assert probs.grad is not None
```

DP supports derivatives through probabilities, including zero and one. FFT
methods support autograd on interior inputs, but output cleanup can change
derivatives at zero. Library backends never convert probabilities to NumPy or
move them to the CPU.

## API

```python
pmf(probs, *, method="dp", validate_args=True, block_size=32)
cdf(probs, k, *, method="dp", validate_args=True, block_size=32)
sf(probs, k, *, method="dp", validate_args=True, block_size=32)
logpmf(probs, *, method="dp", validate_args=True)
logcdf(probs, k, *, method="dp", validate_args=True)
logsf(probs, k, *, method="dp", validate_args=True)
proportional_tail(probs, beta, interesting, *, side="left",
                  method="dp", validate_args=True, block_size=32)
```

- `cdf` returns `P(X <= k)` and `sf` returns `P(X > k)`. `k` must be a scalar
  Python integer. Thresholds outside support return exact zeros or ones.
- Proportional tails use strict `< beta` for `side="left"` and strict `> beta`
  for `side="right"`. `beta` is a finite real scalar in `[0, 1]`; equality enters
  neither event. Interpret beta as the exact supplied Python float, including
  for values such as `1/3`.
- `interesting` is a binary tensor on the input device, broadcastable to the
  probability shape. Its complement defines the uninteresting subset. Boolean,
  binary integer, and binary floating masks are accepted; masks receive no gradients.
- Both proportional events exclude outcomes with no successes. Results remain
  unconditional. Beta endpoints, empty subsets, and zero trials are supported.
- `validate_args=False` skips probability-value checks. Type, shape, mask, and
  method checks remain. Validation and FFT output checks can synchronize CUDA.

## Algorithms and accuracy

| Method | Behavior | Forward time | Forward workspace |
|---|---|---|---|
| `dp` (default) | Nonnegative tensor recurrence | O(B n²) | O(B n) |
| `fft_cf` | Chunked characteristic-function inversion | O(B n² + B n log n) | O(B n), frequency chunks capped at 128 |
| `fft_tree` | Balanced polynomial convolution | O(B n log² n) | O(B n) |
| `fft_hybrid` | DP blocks followed by an FFT tree; tune `block_size` | O(B n b + B n log²(n/b)) for block size b | O(B n) |
| `auto` | Opt-in measured DP/tree selection | Selected method | Selected method |

`B` is the product of batch dimensions. Autograd retains additional state: DP and
characteristic-function backward storage can be quadratic; tree storage grows
across levels. These are not constant-memory training kernels.

DP tails use truncated recurrence. CDF retains at most `k+1` counts; SF retains
at most `n-k` failure counts and never subtracts from one. Auto count tails also
use these paths; proportional auto uses DP subset distributions.

**Use DP for tiny probabilities or gradient-sensitive work.** FFT methods clip
negative roundoff and normalize the final PMF without adding epsilon. FFT
internals use float64/complex128, then return the requested output dtype. Cleanup
cannot recover probabilities obscured by FFT roundoff. DP still underflows below
its floating-point range. Use `logpmf`, `logcdf`, or `logsf` to compute directly
in log space; impossible events return `-inf`. Their dedicated backward supports
first derivatives; undefined derivatives of impossible log events use zero.
See [numerical behavior](docs/numerics.md).

Log functions use float64 internals and preserve the public output dtype, reducing
accumulated error in long recurrences. A float32 output still quantizes a large
logarithm; choose float64 when log-probability precision matters.

### Rare tails with exponential tilting

```python
from pytorch_poibin import logsf

p = torch.full((2000,), 0.5, dtype=torch.float64)
print(logsf(p, 1999))                      # direct log DP: about -1386.29436
print(logsf(p, 1999, method="shift_fft"))  # exponential tilt + FFT
```

`shift_fft` is an explicit method for `cdf`, `sf`, `logcdf`, and `logsf`; it is
not a full-PMF or proportional-tail method. It moves the queried region near
the tilted mean, convolves, and reconstructs the tail logarithm. Large requested
tails use stable logarithmic complementation. Its bracketed solver and output
checks stay on the input device but synchronize CUDA.

Shifted FFT first derivatives use the log-DP adjoint, recomputing its prefix
states. This deliberately avoids differentiating the numerical root or FFT
cleanup; backward therefore has DP cost and memory, and is not promised to be
faster. The method raises on solver or numerical-check failures without fallback.
See [the algorithm and limitations](docs/numerics.md) and
[experiment reproduction](benchmarks/README.md).

`auto` explicitly opts into approximate PMFs. It uses tree convolution only for
exact workloads with checked-in accuracy-qualified speedups of at least 1.25x.
Unknown hardware, runtime versions, thread settings, dtypes, or shapes fall back
to DP. Inputs requiring gradients use DP. Ordinary calls never benchmark.
See [benchmark results and reproduction](benchmarks/README.md).

## Verify and build

```bash
python -m pytest -q                  # CPU and available CUDA tests
python -m pytest -q -m 'not cuda'    # CPU-only selection
python -m pytest -q -m cuda          # CUDA selection; skips without a device
python scripts/smoke.py              # Executable examples and gradients
python -m build                     # Wheel and source distribution in dist/
python -m twine check dist/*
```

Tests cover independent SciPy/enumeration references, analytical distributions,
batch independence, deterministic probabilities, empty dimensions, rare tails,
dtype promotion, and gradients. Failed checks exit nonzero. Linux CPU CI tests
minimum and current PyTorch versions and installs the wheel outside the checkout.
CUDA verification is opt-in on a GPU machine.

## Citation

```bibtex
@software{pytorch_poibin,
  title = {PyTorch Poisson Binomial},
  author = {Bu, Fanchen},
  year = {2025},
  url = {https://github.com/bokveizen/pytorch_poibin}
}

@article{hong2013computing,
  title = {On computing the distribution function for the Poisson binomial distribution},
  author = {Hong, Yili},
  journal = {Computational Statistics & Data Analysis},
  volume = {59},
  pages = {41--51},
  year = {2013}
}

@inproceedings{Bu2024UCom2,
  title = {Tackling Prevalent Conditions in Unsupervised Combinatorial Optimization: Cardinality, Minimum, Covering, and More},
  author = {Bu, Fanchen and Jo, Hyeonsoo and Lee, Soo Yong and Ahn, Sungsoo and Shin, Kijung},
  booktitle = {ICML},
  year = {2024}
}
```

See also [the UCom2 repository](https://github.com/ai4co/unsupervised-CO-ucom2).
Licensed under the [MIT License](LICENSE).
