# Benchmarks and auto-selection profiles

Install the development extra and run from the repository root:

```bash
# Default grid: CPU, seven sizes through n=8192, B=1/16/128, both precisions.
python benchmarks/benchmark.py --output benchmarks/results.json

# Reproduce the original v0.2 baseline (exclude the new hybrid candidates).
python benchmarks/benchmark.py --devices cpu cuda --sizes 32 128 512 --batches 1 16 \
  --methods dp fft_cf fft_tree \
  --output benchmarks/results-forward.json --profiles src/pytorch_poibin/profiles.json

# Separate training measurements; never used for auto dispatch.
python benchmarks/benchmark.py --devices cpu cuda --sizes 32 128 --batches 1 \
  --methods dp fft_cf fft_tree \
  --backward --output benchmarks/results-backward.json

# Tail-only measurements retaining three lower or upper counts.
python benchmarks/benchmark.py --devices cpu cuda --sizes 128 512 --batches 1 \
  --methods dp fft_cf fft_tree \
  --dtypes float64 --operations cdf sf --tail-width 3 --output benchmarks/results-tails.json
```

Every algorithm/workload runs in a subprocess with a default 30-second deadline.
Estimated tensor workspace is capped at 256 MiB; oversized and timed-out cases
are recorded as skipped. The estimate excludes the interpreter, libraries, FFT
plans, and allocator caches; it is not a hard process-memory cap. Unexpected
failures and accuracy failures make the command exit nonzero.

`torch.utils.benchmark.Timer` supplies warmup and accelerator synchronization,
using a default 0.5-second measurement window and one CPU thread. Reports record
runtime versions, hardware, threads, seed, shape, dtype, validation setting,
operation, forward/backward mode, median/IQR, reference error, and memory metrics.
Value validation is disabled by default; `--validate-args` includes it.

CUDA memory is incremental peak allocated tensor memory for a warmed-up call,
including backward when requested. CPU memory is process peak RSS including
runtime and oracle. These metrics are not directly comparable. Backward timing
includes forward computation.

Every workload is checked against SciPy for random, fair, low-probability,
high-probability, and deterministic trials. Maximum absolute error must be <=1e-5
for float32 and <=1e-10 for float64. Backward reports also compare the measured
operation's squared-output loss gradient with DP, using tolerances 1e-4 for
float32 and 1e-8 for float64. These checks do not establish relative rare-tail accuracy.

## Auto profiles

Profiles admit tree convolution when both algorithms pass accuracy checks,
tree median latency is at least 1.25x faster, and both timing IQRs are <=20% of
their medians. Profiles use exact batch/trial sizes with no extrapolation. They
match hardware, platform, exact PyTorch/CUDA versions, thread count, and dtype.
Gradient-bearing inputs and auto tails stay on DP.

Profile generation replaces its destination. Review the accompanying report and
rerun pytest before including new profiles. Unknown environments use DP; ordinary
calls perform no tuning. Common validation overhead may reduce end-to-end speedup
relative to the measured kernels with probability-value checks disabled.

The checked-in `results-forward.json`, `results-backward.json`, and
`results-tails.json` archive the v0.2 verification run. Results are
workload-specific, rather than universal crossover thresholds.

## Original v0.2 measured example

The 72 forward cases (CPU/CUDA, both precisions, B=1/16, n=32/128/512) all
passed the five-regime accuracy checks. Fifteen workloads qualified for auto
selection after the speed and timing-noise gates.

For float64, B=16, n=512, one CPU thread, PyTorch 2.6.0+cu118:

| Hardware | DP | Characteristic-function FFT | Tree FFT |
|---|---:|---:|---:|
| Intel Xeon Gold 6426Y | 17.48 ms | 35.81 ms | 1.51 ms |
| NVIDIA RTX A6000 | 28.65 ms | 1.68 ms | 1.70 ms |

These are synchronized forward medians with probability-value checks disabled.
Both FFT methods still validate their computed output. The output error across
the five distributions in these cases was below 2e-13 for characteristic-function
FFT and below 8e-15 for tree FFT.

## Log tails and hybrid research (v0.3)

Run these phases serially to avoid competing benchmark processes. Both
precisions and all five block sizes are selected by default:

```bash
python benchmarks/benchmark.py --devices cpu cuda --sizes 32 512 2048 --batches 1 16 \
  --min-run-time 0.3 --output benchmarks/results-hybrid-forward.json

python benchmarks/benchmark.py --devices cpu cuda --sizes 32 512 2048 --batches 1 16 \
  --methods dp fft_tree fft_hybrid --backward --min-run-time 0.3 \
  --output benchmarks/results-hybrid-backward.json

python -m benchmarks.log_tails --devices cpu cuda --sizes 128 2048 --batches 1 16 \
  --output benchmarks/results-log-tails-forward.json

python -m benchmarks.log_tails --devices cpu cuda --sizes 128 2048 --batches 1 16 \
  --regimes fair_upper high_lower --backward \
  --output benchmarks/results-log-tails-backward.json

python -m benchmarks.summarize_research
```

`--block-sizes` tunes the hybrid candidates (default 8, 16, 32, 64, 128).
Memory guards distinguish saved quadratic DP/CF states, FFT tree levels, and
hybrid block states. Estimates remain conservative rather than hard allocations.
A 1 MiB minimum estimate covers allocator granularity in small workloads.
Missing CUDA devices and cases exceeding time/memory budgets are recorded as
skipped; unexpected exceptions and failed accuracy checks exit nonzero.

The log-tail harness has a 45-second per-case deadline and 256 MiB estimated
tensor budget. It compares log DP and shifted FFT with independent binomial sums
and derivatives evaluated at 100 decimal digits. Timed queries retain three lower
or upper counts: fair upper tails, low-probability upper tails, and high-probability
lower tails. Probability values are converted from their actual input dtype for
the reference. Reports record absolute log error, relative probability error
computed through `expm1`, ordinary-probability underflow, and gradient error.
Float64 log error must be <=5e-8; float32 uses an output-quantization tolerance
of `2*eps*max(1,abs(reference_log))`. Gradient tolerances scale with the reference
derivative: 1e-7 for float64 and 2e-4 for float32.

The ordinary PMF benchmark uses seed 2026 and five distribution regimes; log-tail
inputs are deterministic homogeneous vectors. Unit tests separately use a
400-digit recurrence for heterogeneous cases and analytic extreme derivatives.
No new hybrid auto profiles are generated by these commands. Inspect the
[generated findings](../docs/research-results.md) and raw JSON when deciding
whether a method or block size suits a particular workload.
