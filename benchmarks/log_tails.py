"""Bounded log-DP/ShiftConvolve measurements with arbitrary-precision oracles.

Run with python -m benchmarks.log_tails from a development checkout.
"""

import argparse
import datetime
import gc
import json
import os
import platform
import resource
import subprocess
import sys
from pathlib import Path

import mpmath as mp
import torch
from torch.utils.benchmark import Timer

from pytorch_poibin import __version__, cdf, logcdf, logsf, sf
from pytorch_poibin._selection import hardware_key

REGIMES = {
    "fair_upper": (0.5, True),
    "low_upper": (0.001, True),
    "high_lower": (0.999, False),
}


def reference(n, probability, k, upper):
    with mp.workdps(100):
        p = mp.mpf(probability)
        counts = range(k + 1, n + 1) if upper else range(k + 1)
        total = mp.fsum(mp.binomial(n, j) * p**j * (1 - p) ** (n - j) for j in counts)
        density = mp.binomial(n - 1, k) * p**k * (1 - p) ** (n - 1 - k)
        return float(mp.log(total)), float((1 if upper else -1) * density / total)


def worker(case):
    torch.set_num_threads(case["threads"])
    device = torch.device(case["device"])
    dtype = getattr(torch, case["dtype"])
    value, upper = REGIMES[case["regime"]]
    p = torch.full((case["batch_size"], case["n"]), value, device=device, dtype=dtype)
    k = case["n"] - 3 if upper else 2
    expected, expected_gradient = reference(case["n"], float(p[0, 0]), k, upper)
    function = logsf if upper else logcdf
    ordinary = sf if upper else cdf
    underflow = bool((ordinary(p, k) == 0).all())
    p.requires_grad_(case["backward"])

    def run():
        result = function(p, k, method=case["method"], validate_args=False)
        if case["backward"]:
            torch.autograd.grad(result.sum(), p)
        return result

    output = run()
    error = (output.detach().double() - expected).abs().max().item()
    relative_error = torch.expm1(output.detach().double() - expected).abs().max().item()
    gradient_error = None
    if case["backward"]:
        gradient = torch.autograd.grad(
            function(p, k, method=case["method"], validate_args=False).sum(), p
        )[0]
        gradient_error = (gradient.double() - expected_gradient).abs().max().item()
    # Float32 outputs necessarily quantize logs; report the actual error as well
    # as an acceptance threshold accounting for that output quantization.
    tolerance = (
        5e-8
        if dtype == torch.float64
        else 2 * torch.finfo(dtype).eps * max(1, abs(expected))
    )
    gradient_tolerance = (1e-7 if dtype == torch.float64 else 2e-4) * max(
        1, abs(expected_gradient)
    )
    measurement = Timer(
        "run()", globals={"run": run}, num_threads=case["threads"]
    ).blocked_autorange(min_run_time=case["min_run_time"])
    del output
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        run()
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device) - baseline
        kind = "cuda_tensor_allocated_delta"
    else:
        run()
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        kind = "process_peak_rss_including_runtime"
    return {
        **case,
        "status": "ok",
        "runtime": hardware_key(device),
        "threshold": k,
        "reference_log_probability": expected,
        "max_log_error": error,
        "max_relative_probability_error": relative_error,
        "ordinary_underflow": underflow,
        "max_abs_gradient_error": gradient_error,
        "reference_gradient": expected_gradient,
        "accuracy_passed": error <= tolerance
        and (gradient_error is None or gradient_error <= gradient_tolerance),
        "log_error_tolerance": tolerance,
        "median_seconds": measurement.median,
        "iqr_seconds": measurement.iqr,
        "measurements": len(measurement.raw_times),
        "peak_bytes": peak,
        "peak_kind": kind,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument(
        "--devices", nargs="+", choices=["cpu", "cuda"], default=["cpu"]
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=["float32", "float64"],
        default=["float32", "float64"],
    )
    parser.add_argument("--sizes", nargs="+", type=int, default=[128, 2048])
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 16])
    parser.add_argument(
        "--regimes", nargs="+", choices=list(REGIMES), default=list(REGIMES)
    )
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--memory-mib", type=int, default=256)
    parser.add_argument("--min-run-time", type=float, default=0.3)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/results-log-tails.json")
    )
    args = parser.parse_args()
    if args.worker:
        try:
            record = worker(json.loads(args.worker))
        except Exception as exc:
            record = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(record, allow_nan=False))
        return
    if (
        min(args.sizes) < 3
        or min(
            [
                *args.batches,
                args.threads,
                args.memory_mib,
                args.timeout,
                args.min_run_time,
            ]
        )
        <= 0
    ):
        parser.error("sizes must be >=3; batches and resource limits must be positive")
    records = []
    for device in args.devices:
        for dtype in args.dtypes:
            for n in args.sizes:
                for batch in args.batches:
                    for regime in args.regimes:
                        for method in ["dp", "shift_fft"]:
                            case = {
                                "device": device,
                                "dtype": dtype,
                                "n": n,
                                "batch_size": batch,
                                "regime": regime,
                                "method": method,
                                "backward": args.backward,
                                "threads": args.threads,
                                "min_run_time": args.min_run_time,
                            }
                            # All measured tails retain three counts in the log-DP
                            # reference/adjoint. Shift additionally uses a full FFT.
                            estimate = 128 * batch * (n + 1) * 8 + (
                                32 * batch * (n + 1) * 3 * 8 if args.backward else 0
                            )
                            if device == "cuda" and not torch.cuda.is_available():
                                record = {
                                    **case,
                                    "status": "skipped",
                                    "reason": "CUDA unavailable",
                                }
                            elif estimate > args.memory_mib * 1024**2:
                                record = {
                                    **case,
                                    "status": "skipped",
                                    "reason": "estimated workspace exceeds budget",
                                }
                            else:
                                try:
                                    result = subprocess.run(
                                        [
                                            sys.executable,
                                            "-m",
                                            "benchmarks.log_tails",
                                            "--worker",
                                            json.dumps(case),
                                        ],
                                        capture_output=True,
                                        text=True,
                                        check=True,
                                        timeout=args.timeout,
                                        env={
                                            **os.environ,
                                            "OMP_NUM_THREADS": str(args.threads),
                                        },
                                    )
                                    record = {**case, **json.loads(result.stdout)}
                                except subprocess.TimeoutExpired:
                                    record = {
                                        **case,
                                        "status": "skipped",
                                        "reason": "execution timeout",
                                    }
                                except (
                                    subprocess.CalledProcessError,
                                    json.JSONDecodeError,
                                ) as exc:
                                    record = {
                                        **case,
                                        "status": "error",
                                        "reason": str(exc),
                                    }
                            records.append(record)
                            print(
                                f"{device} {dtype} B={batch} n={n} {regime} {method}: {record['status']}",
                                flush=True,
                            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "package_version": __version__,
        "python": platform.python_version(),
        "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "reference": "mpmath, 100 decimal digits",
        "memory_budget_mib": args.memory_mib,
        "timeout_seconds": args.timeout,
        "records": records,
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(
        r["status"] == "error" or (r["status"] == "ok" and not r["accuracy_passed"])
        for r in records
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
