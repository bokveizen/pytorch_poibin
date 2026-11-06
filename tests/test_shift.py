import math

import pytest
import torch
from scipy.stats import poisson_binom

from pytorch_poibin import (
    _backends,
    cdf,
    logcdf,
    logpmf,
    logsf,
    pmf,
    proportional_tail,
    sf,
)
from pytorch_poibin._shift import solve_tilt

from .test_logs import high_precision_logs


@pytest.mark.parametrize(
    "values,k",
    [
        ([0.2, 0.3, 0.7], 1),
        ([0.0, 1.0, 0.2, 0.8], 1),
        ([0.0, 0.0, 1.0], 1),
        ([1e-30], 0),
        ([1e-300], 0),
        ([1e-250] * 8, 3),
        ([0.999] * 160, 2),
        ([0.001] * 160, 157),
        ([0.0, 1.0, 1.0, 0.0], 1),
        ([0.02, 0.04, 0.01, 0.03, 0.2, 0.0, 1.0], 4),
    ],
)
def test_shift_arbitrary_precision(device, values, k):
    p = torch.tensor(values, device=device, dtype=torch.float64)
    _, lc, ls = high_precision_logs(values, k)
    for fn, reference, normal in [(logcdf, lc, cdf), (logsf, ls, sf)]:
        actual = fn(p, k, method="shift_fft")
        torch.testing.assert_close(
            actual, p.new_tensor(reference), atol=3e-9, rtol=1e-12
        )
        torch.testing.assert_close(
            normal(p, k, method="shift_fft"), actual.exp(), atol=0, rtol=1e-12
        )


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
def test_shift_batched_noncontiguous(device, dtype):
    p = torch.tensor(
        [[[0.0, 1.0, 0.2], [1.0, 0.0, 0.8]], [[0.1, 0.3, 0.7], [1.0, 1.0, 1.0]]],
        device=device,
        dtype=dtype,
    ).transpose(0, 1)
    for k in [-1, 0, 1, 2, 3, 5]:
        for fn in [logcdf, logsf]:
            actual = fn(p, k, method="shift_fft")
            expected = fn(p, k)
            assert actual.shape == (2, 2) and actual.device == p.device
            assert actual.dtype == (
                torch.float64 if dtype == torch.float64 else torch.float32
            )
            torch.testing.assert_close(actual, expected)
            rows = torch.stack([fn(row, k, method="shift_fft") for row in p])
            torch.testing.assert_close(actual, rows)


@pytest.mark.parametrize("shape", [(0,), (2, 0), (0, 3), (2, 0, 4)])
def test_shift_empty(device, shape):
    p = torch.empty(shape, device=device, dtype=torch.float64, requires_grad=True)
    for fn in [logcdf, logsf]:
        result = fn(p, 0, method="shift_fft")
        assert result.shape == shape[:-1]
        torch.autograd.grad(result.sum(), p)


def test_shift_underflow_and_gradients(device):
    p = torch.full((2000,), 0.5, device=device, dtype=torch.float64, requires_grad=True)
    for fn, k, sign in [(logcdf, 0, -1), (logsf, 1999, 1)]:
        result = fn(p, k, method="shift_fft")
        torch.testing.assert_close(
            result, p.new_tensor(-2000 * math.log(2)), atol=1e-9, rtol=0
        )
        gradient = torch.autograd.grad(result, p)[0]
        torch.testing.assert_close(
            gradient, torch.full_like(p, sign * 2), atol=1e-9, rtol=1e-9
        )
    tiny = torch.tensor([1e-300], device=device, dtype=torch.float64)
    torch.testing.assert_close(
        logcdf(tiny, 0, method="shift_fft"), -tiny[0], atol=0, rtol=1e-12
    )


def test_shift_gradcheck_and_endpoints(device):
    p = torch.tensor(
        [0.2, 0.3, 0.7], device=device, dtype=torch.float64, requires_grad=True
    )
    for fn in [logcdf, logsf, cdf, sf]:
        assert torch.autograd.gradcheck(lambda p: fn(p, 1, method="shift_fft"), (p,))
        # The favorable-tail choice switches here; the probability and its
        # first derivative remain continuous across that implementation branch.
        balanced = torch.tensor(
            [0.2, 0.5, 0.8], device=device, dtype=torch.float64, requires_grad=True
        )
        assert torch.autograd.gradcheck(
            lambda p: fn(p, 1, method="shift_fft"), (balanced,)
        )
    for values in [[0.0, 0.3, 0.7], [1.0, 0.2, 0.8], [0.0, 0.0, 1.0]]:
        p = torch.tensor(values, device=device, dtype=torch.float64, requires_grad=True)
        for fn in [logcdf, logsf]:
            for k in [0, 1, 2]:
                expected = fn(p, k)
                if torch.isfinite(expected):
                    actual = fn(p, k, method="shift_fft")
                    ga = torch.autograd.grad(actual, p)[0]
                    ge = torch.autograd.grad(expected, p)[0]
                    torch.testing.assert_close(ga, ge)


def test_large_near_mean_queries(device):
    generator = torch.Generator().manual_seed(911)
    p = (torch.rand(3, 513, generator=generator, dtype=torch.float64) * 0.9 + 0.05).to(
        device
    )
    p[:, 0] = 0
    p[:, 1] = 1
    coefficients = poisson_binom.pmf(
        torch.arange(514).numpy()[:, None], p.cpu().numpy()
    ).T.copy()
    expected = torch.as_tensor(coefficients, device=device, dtype=p.dtype)
    for k in [255, 256, 257]:
        torch.testing.assert_close(
            logcdf(p, k, method="shift_fft"),
            expected[:, : k + 1].sum(-1).log(),
            atol=1e-10,
            rtol=1e-10,
        )
        torch.testing.assert_close(
            logsf(p, k, method="shift_fft"),
            expected[:, k + 1 :].sum(-1).log(),
            atol=1e-10,
            rtol=1e-10,
        )


def test_solver_extreme_bracket(device):
    logits = torch.tensor(
        [[-1000.0, 0, 1000.0]] * 3, device=device, dtype=torch.float64
    )
    target = logits.new_tensor([0.5, 1.5, 2.5])
    theta = solve_tilt(logits, target, torch.ones(3, device=device, dtype=torch.bool))
    torch.testing.assert_close(
        torch.sigmoid(logits + theta[:, None]).sum(-1), target, atol=1e-12, rtol=0
    )
    with pytest.raises(RuntimeError, match="bracket"):
        solve_tilt(logits, target + 10, torch.ones_like(target, dtype=torch.bool))


def test_shift_failure_is_not_hidden(monkeypatch):
    monkeypatch.setattr(
        _backends,
        "convolve_nodes",
        lambda nodes, degree: nodes.new_full((degree + 1,), float("nan")),
    )
    with pytest.raises(RuntimeError, match="roundoff"):
        logsf(torch.tensor([0.2, 0.7]), 0, method="shift_fft")


def test_shift_rejects_non_tail_apis():
    p = torch.tensor([0.2, 0.7])
    with pytest.raises(ValueError, match="tail-only"):
        pmf(p, method="shift_fft")
    with pytest.raises(ValueError, match="tail-only"):
        logpmf(p, method="shift_fft")
    with pytest.raises(ValueError, match="proportional"):
        proportional_tail(p, 0.5, torch.tensor([1, 0]), method="shift_fft")
