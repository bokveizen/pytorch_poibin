# Migration from the prototype

Version 0.2.0 makes a clean API break. The root `poi_bin` and `tails` modules and
the exponential educational functions are removed from the installed API.

| Prototype call | Package call |
|---|---|
| `pmf_poibin_naive_dp_vectorized(p, device)` | `pmf(p.to(device), method="dp")` |
| `pmf_poibin_naive_dp(p, device)` | `pmf(p.to(device), method="dp")` |
| `pmf_poibin_fft(p, device)` | `pmf(p.to(device), method="fft_cf")` |
| `left_tail_probs(p, t, device)` | `cdf(p.to(device), t-1)` |
| `right_tail_probs(p, t, device)` | `sf(p.to(device), t)` |
| `proportional_left_tail_probs(p, beta, S, U, device)` | `proportional_tail(p.to(device), beta, S.to(device))` |
| `proportional_right_tail_probs(p, beta, S, U, device)` | `proportional_tail(p.to(device), beta, S.to(device), side="right")` |

Import functions from `pytorch_poibin`. Input device determines execution; masks
must share it. The uninteresting mask is inferred as the complement. Overlapping
or incomplete old S/U pairs have no direct equivalent.

Tails return a result per batch member, rather than summing the batch. Arbitrary
batch dimensions and unbatched tensors are supported. Float64 is preserved;
float16/bfloat16 promote to float32. The unused `use_normalization` flag is removed.
FFT outputs are always cleaned and normalized without epsilon; DP uses the
recurrence directly.

The old verification and profiling scripts are replaced by pytest,
`scripts/smoke.py`, and `benchmarks/benchmark.py`.
