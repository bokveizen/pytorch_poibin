# Changelog

## 0.3.0 — prepared, unreleased

- Add direct logpmf/logcdf/logsf recurrences with float64 internals and public
  dtype preservation, including probabilities below ordinary floating-point range.
- Add signed-log coefficient adjoints for defined first derivatives, including
  deterministic boundary cases; document the impossible-event convention.
- Add explicit shift_fft count tails with bracketed exponential tilting and
  logarithmic reconstruction. Its first-order backward uses the log-DP adjoint.
- Add experimental fft_hybrid block DP/FFT convolution and bounded experiments
  across block sizes, devices, dtypes, forward/backward latency, and memory.

## 0.2.0 — prepared, unreleased

This first packaged release combines the planned v0.1 correctness/packaging
foundation with the v0.2 algorithms. No historical v0.1 release or publication
is implied.

- Add `pytorch_poibin.pmf`, `cdf`, `sf`, and `proportional_tail` with arbitrary
  batches, device inference, dtype preservation, validation, and gradients.
- Replace count-by-count DP loops with tensor operations and add truncated tails.
- Repair batch mixing, negative thresholds, masks, beta endpoints, and empty
  proportional outcomes.
- Add chunked characteristic-function inversion and balanced FFT tree convolution
  with double-precision internals, output dtype preservation, and final cleanup
  instead of epsilon injection.
- Add conservative opt-in auto dispatch from measured exact-workload profiles.
- Add SciPy/enumeration references, property tests, and CPU/CUDA gradient checks.
- Add bounded benchmarks, smoke checks, source/wheel builds, and CI.
- Remove prototype imports and exponential public functions. See
  [migration](docs/migration.md) for replacement calls.

## Prototype — August 2025

Standalone PMF/tail functions and exploratory verification/profiling scripts.
