#!/usr/bin/env python3
"""Bounded, reproducible CPU/CUDA benchmarks and accuracy-gated profiles.

Each case runs in a subprocess so a slow algorithm cannot stall the grid.
Run from a checkout with the development dependencies installed.
"""

import argparse
import datetime
import gc
import json
import math
import os
import platform
import resource
import subprocess
import sys
from pathlib import Path

import torch
from scipy.stats import poisson_binom
from torch.utils.benchmark import Timer

from pytorch_poibin import __version__, cdf, pmf, sf
from pytorch_poibin._selection import hardware_key

METHODS = ("dp", "fft_cf", "fft_tree", "fft_hybrid")


def workspace_estimate(case):
    # Allocator granularity dominates tiny cases: measured n=32/B=1 DP CUDA
    # training peaks are ~37 KiB despite smaller raw tensor byte counts.
    return max(1024**2, _tensor_workspace_estimate(case))


def _tensor_workspace_estimate(case):
    """Conservative tensor-workspace guard, excluding the Python/Torch runtime."""
    b, n = case["batch_size"], case["n"]
    size = 8 if case["dtype"] == "float64" else 4
    if case["method"] != "dp":
        size = 8  # FFT internals use float64, even for float32 output.
    if case["backward"]:
        if case["method"] == "fft_tree":
            return 32 * b * (n + 1) * (n + 1).bit_length() * size
        if case["method"] == "fft_hybrid":
            block = min(case.get("block_size", 32), n)
            return (
                4
                * b
                * (n + 1)
                * (block + 8 * (n // max(1, block) + 1).bit_length())
                * size
            )
        # Saved recurrence states or characteristic-function factors can be quadratic.
        return 4 * b * (n + 1) ** 2 * size
    if case["method"] == "fft_cf":
        return 12 * b * max(1, n) * min(128, n + 1) * size
    return 32 * b * (n + 1) * size


def evaluate(p, method, operation, threshold, validate_args=False, block_size=32):
    if operation == "pmf":
        return pmf(p, method=method, validate_args=validate_args, block_size=block_size)
    function = cdf if operation == "cdf" else sf
    return function(
        p, threshold, method=method, validate_args=validate_args, block_size=block_size
    )


def accuracy_check(p, method, operation="pmf", threshold=2, block_size=32):
    """Check diverse distributions for this exact shape, not just the timed sample."""
    generator = torch.Generator(device=p.device).manual_seed(2026)
    regimes = [
        torch.rand(p.shape, generator=generator, device=p.device, dtype=p.dtype),
        torch.full_like(p, 0.5),
        torch.rand(p.shape, generator=generator, device=p.device, dtype=p.dtype) * 0.05,
        1
        - torch.rand(p.shape, generator=generator, device=p.device, dtype=p.dtype)
        * 0.05,
        (
            torch.rand(p.shape, generator=generator, device=p.device, dtype=p.dtype)
            > 0.5
        ).to(p.dtype),
    ]
    worst = 0.0
    for sample in regimes:
        reference = poisson_binom.pmf(
            torch.arange(p.shape[-1] + 1).numpy()[:, None],
            sample.cpu().double().numpy(),
        ).T.copy()
        expected = torch.as_tensor(reference, device=p.device, dtype=torch.float64)
        if operation == "cdf":
            expected = expected[..., : max(0, threshold + 1)].sum(-1)
        elif operation == "sf":
            expected = expected[..., max(0, threshold + 1) :].sum(-1)
        with torch.no_grad():
            actual = evaluate(
                sample, method, operation, threshold, block_size=block_size
            )
        error = (actual.double() - expected).abs().max().item()
        if not math.isfinite(error):
            return float("inf")
        worst = max(worst, error)
    return worst


def worker(case):
    torch.set_num_threads(case["threads"])
    torch.manual_seed(2026)
    device = torch.device(case["device"])
    dtype = getattr(torch, case["dtype"])
    p = torch.rand(case["batch_size"], case["n"], device=device, dtype=dtype)
    block_size = case.get("block_size", 32)
    error = accuracy_check(
        p, case["method"], case["operation"], case["threshold"], block_size
    )
    tolerance = 1e-10 if dtype == torch.float64 else 1e-5
    p.requires_grad_(case["backward"])
    method = case["method"]
    gradient_error = None
    if case["backward"]:
        actual_gradient = torch.autograd.grad(
            evaluate(
                p, method, case["operation"], case["threshold"], block_size=block_size
            )
            .square()
            .sum(),
            p,
        )[0]
        expected_gradient = torch.autograd.grad(
            evaluate(p, "dp", case["operation"], case["threshold"]).square().sum(), p
        )[0]
        gradient_error = (actual_gradient - expected_gradient).abs().max().item()

    def run():
        q = evaluate(
            p,
            method,
            case["operation"],
            case["threshold"],
            case["validate_args"],
            block_size,
        )
        if case["backward"]:
            torch.autograd.grad(q.square().sum(), p)
        return q

    run()  # Warm up allocation and FFT plans before timing.
    measurement = Timer(
        "run()", globals={"run": run}, num_threads=case["threads"]
    ).blocked_autorange(min_run_time=case["min_run_time"])
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        run()
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device) - baseline
        peak_kind = "cuda_tensor_allocated_delta"
    else:
        # RSS includes the interpreter, numerical libraries and allocator caches.
        run()
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        peak_kind = "process_peak_rss_including_runtime"
    return {
        **case,
        "status": "ok",
        "runtime": hardware_key(device),
        "median_seconds": measurement.median,
        "iqr_seconds": measurement.iqr,
        "measurements": len(measurement.raw_times),
        "max_abs_error": error,
        "accuracy_passed": error <= tolerance
        and (
            gradient_error is None
            or gradient_error <= (1e-8 if dtype == torch.float64 else 1e-4)
        ),
        "max_abs_gradient_error": gradient_error,
        "peak_bytes": peak,
        "peak_kind": peak_kind,
    }


def qualified_profiles(records):
    """Only exact measured workloads with >=1.25x speedup may use the tree."""
    profiles = []
    for tree in records:
        if (
            tree.get("status") != "ok"
            or tree["method"] != "fft_tree"
            or tree["backward"]
            or tree["operation"] != "pmf"
            or not tree["accuracy_passed"]
            or tree["validate_args"]
        ):
            continue
        for baseline in records:
            if baseline.get("status") != "ok" or baseline["method"] != "dp":
                continue
            keys = (
                "runtime",
                "dtype",
                "n",
                "batch_size",
                "backward",
                "operation",
                "validate_args",
            )
            if not all(tree[key] == baseline[key] for key in keys):
                continue
            speedup = baseline["median_seconds"] / tree["median_seconds"]
            if not baseline["accuracy_passed"] or speedup < 1.25:
                continue
            # Do not qualify a highly noisy timing sample.
            if any(
                r["iqr_seconds"] > r["median_seconds"] * 0.2 for r in [tree, baseline]
            ):
                continue
            profiles.append(
                {
                    "runtime": tree["runtime"],
                    "dtype": "torch." + tree["dtype"],
                    "batch_size": tree["batch_size"],
                    "n": tree["n"],
                    "speedup": speedup,
                    "max_abs_error": tree["max_abs_error"],
                }
            )
            break
    return {"schema_version": 1, "profiles": profiles}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument(
        "--devices", nargs="+", choices=["cpu", "cuda"], default=["cpu"]
    )
    parser.add_argument(
        "--sizes", nargs="+", type=int, default=[1, 8, 32, 128, 512, 2048, 8192]
    )
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 16, 128])
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=["float32", "float64"],
        default=["float32", "float64"],
    )
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument(
        "--block-sizes", nargs="+", type=int, default=[8, 16, 32, 64, 128]
    )
    parser.add_argument(
        "--operations", nargs="+", choices=["pmf", "cdf", "sf"], default=["pmf"]
    )
    parser.add_argument("--threshold", type=int, default=2)
    parser.add_argument(
        "--tail-width",
        type=int,
        help="Retain this many counts for CDF and failure counts for SF",
    )
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--validate-args", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--memory-mib", type=int, default=256)
    parser.add_argument("--min-run-time", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results.json"))
    parser.add_argument(
        "--profiles", type=Path, help="Write qualifying exact-workload auto profiles"
    )
    args = parser.parse_args()
    if args.worker:
        try:
            result = worker(json.loads(args.worker))
        except Exception as exc:
            result = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(result, allow_nan=False))
        return
    if any(
        x <= 0
        for x in [
            *args.sizes,
            *args.batches,
            *args.block_sizes,
            args.threads,
            args.memory_mib,
            args.timeout,
            args.min_run_time,
        ]
    ):
        parser.error(
            "sizes, batches, threads, memory, timeout and run time must be positive"
        )
    if args.tail_width is not None and args.tail_width <= 0:
        parser.error("tail width must be positive")
    results = []
    for device in args.devices:
        for dtype in args.dtypes:
            for batch in args.batches:
                for n in args.sizes:
                    for operation in args.operations:
                        for method, block_size in [
                            (m, width)
                            for m in args.methods
                            for width in (
                                args.block_sizes if m == "fft_hybrid" else [32]
                            )
                        ]:
                            threshold = args.threshold
                            if args.tail_width is not None:
                                if operation == "cdf":
                                    threshold = args.tail_width - 1
                                elif operation == "sf":
                                    threshold = n - args.tail_width
                            case = {
                                "device": device,
                                "dtype": dtype,
                                "batch_size": batch,
                                "n": n,
                                "method": method,
                                "block_size": block_size,
                                "operation": operation,
                                "threshold": threshold,
                                "backward": args.backward,
                                "validate_args": args.validate_args,
                                "threads": args.threads,
                                "min_run_time": args.min_run_time,
                            }
                            reason = None
                            if device == "cuda" and not torch.cuda.is_available():
                                reason = "CUDA is unavailable"
                            if workspace_estimate(case) > args.memory_mib * 1024**2:
                                reason = "estimated tensor workspace exceeds budget"
                            if reason:
                                record = {**case, "status": "skipped", "reason": reason}
                            else:
                                try:
                                    completed = subprocess.run(
                                        [
                                            sys.executable,
                                            str(Path(__file__).resolve()),
                                            "--worker",
                                            json.dumps(case),
                                        ],
                                        capture_output=True,
                                        text=True,
                                        timeout=args.timeout,
                                        check=True,
                                        env={
                                            **os.environ,
                                            "OMP_NUM_THREADS": str(args.threads),
                                        },
                                    )
                                    record = {**case, **json.loads(completed.stdout)}
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
                            results.append(record)
                            print(
                                f"{device} {dtype} B={batch} n={n} {operation} {method} block={block_size}: {record['status']}",
                                flush=True,
                            )
    report = {
        "schema_version": 1,
        "package_version": __version__,
        "seed": 2026,
        "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python": platform.python_version(),
        "memory_budget_mib": args.memory_mib,
        "timeout_seconds": args.timeout,
        "records": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.profiles:
        args.profiles.parent.mkdir(parents=True, exist_ok=True)
        args.profiles.write_text(
            json.dumps(qualified_profiles(results), indent=2, allow_nan=False) + "\n"
        )
    if any(
        r["status"] == "error" or (r["status"] == "ok" and not r["accuracy_passed"])
        for r in results
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
