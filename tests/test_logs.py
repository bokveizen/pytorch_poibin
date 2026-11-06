import math

import mpmath as mp
import pytest
import torch
from scipy.stats import poisson_binom

from pytorch_poibin import cdf, logcdf, logpmf, logsf, pmf, sf


def high_precision_logs(values, k):
    """Independent arbitrary-precision coefficient recurrence, not float PMF logs."""
    with mp.workdps(100):
        coefficients = [mp.mpf(1)]
        for value in values:
            p = mp.mpf(float(value))
            updated = [mp.mpf(0)] * (len(coefficients) + 1)
            for j, coefficient in enumerate(coefficients):
                updated[j] += coefficient * (1 - p)
                updated[j + 1] += coefficient * p
            coefficients = updated
        return (
            [float(mp.log(value)) for value in coefficients],
            float(mp.log(mp.fsum(coefficients[: max(0, k + 1)]))),
            float(mp.log(mp.fsum(coefficients[max(0, k + 1) :]))),
        )


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
def test_log_shape_dtype_and_scipy(device, dtype):
    p = torch.tensor(
        [[[0.2, 0.3, 0.7], [0.1, 0.5, 0.9]], [[0.3, 0.6, 0.8], [0.4, 0.7, 0.9]]],
        device=device,
        dtype=dtype,
    ).transpose(0, 1)
    assert not p.is_contiguous()
    target_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    q = logpmf(p)
    assert q.shape == (2, 2, 4) and q.dtype == target_dtype and q.device == p.device
    reference = poisson_binom.pmf(
        torch.arange(4).numpy()[:, None, None], p.cpu().double().numpy()
    ).transpose(1, 2, 0)
    torch.testing.assert_close(q, p.new_tensor(reference, dtype=target_dtype).log())
    torch.testing.assert_close(logcdf(p, 1).exp(), cdf(p, 1))
    torch.testing.assert_close(logsf(p, 1).exp(), sf(p, 1))
    torch.testing.assert_close(q, logpmf(p, method="auto"))
    torch.testing.assert_close(logcdf(p, 1), logcdf(p, 1, method="auto"))
    torch.testing.assert_close(logsf(p, 1), logsf(p, 1, method="auto"))


@pytest.mark.parametrize(
    "values,k",
    [
        ([0.2, 0.3, 0.7], 1),
        ([0.0, 1.0, 0.2, 0.8], 1),
        ([0.0, 0.0, 1.0], 1),
        ([1e-30], 0),
        ([1e-250] * 8, 3),
        ([0.999] * 160, 2),
        ([0.001] * 160, 157),
    ],
)
def test_arbitrary_precision_logs(device, values, k):
    p = torch.tensor(values, device=device, dtype=torch.float64)
    expected_pmf, expected_cdf, expected_sf = high_precision_logs(values, k)
    torch.testing.assert_close(
        logpmf(p), p.new_tensor(expected_pmf), atol=2e-10, rtol=1e-12
    )
    torch.testing.assert_close(
        logcdf(p, k), p.new_tensor(expected_cdf), atol=2e-10, rtol=1e-12
    )
    torch.testing.assert_close(
        logsf(p, k), p.new_tensor(expected_sf), atol=2e-10, rtol=1e-12
    )


def test_analytic_underflow(device):
    p = torch.full((2000,), 0.5, device=device, dtype=torch.float64)
    expected = p.new_tensor(-2000 * math.log(2))
    assert pmf(p)[0] == 0
    torch.testing.assert_close(logcdf(p, 0), expected, atol=1e-9, rtol=0)
    torch.testing.assert_close(logsf(p, 1999), expected, atol=1e-9, rtol=0)
    # One direct log-PMF calculation covers both extreme coefficients.
    torch.testing.assert_close(
        logpmf(p)[[0, -1]], expected.expand(2), atol=1e-9, rtol=0
    )


@pytest.mark.parametrize("shape", [(0,), (2, 0), (0, 3), (2, 0, 4)])
def test_empty_log_inputs(device, shape):
    p = torch.empty(shape, device=device, dtype=torch.float64, requires_grad=True)
    q = logpmf(p)
    assert q.shape == shape[:-1] + (shape[-1] + 1,)
    q.sum().backward()
    assert p.grad is not None and p.grad.shape == p.shape
    for fn in [logcdf, logsf]:
        for k in [-1, 0, 10]:
            assert fn(p, k).shape == shape[:-1]


@pytest.mark.parametrize("k", [-5, -1, 0, 1, 2, 7])
def test_support_and_batch_independence(device, k):
    p = torch.tensor([[0.0, 1.0], [0.2, 0.7]], device=device, dtype=torch.float64)
    for fn in [logcdf, logsf]:
        actual = fn(p, k)
        torch.testing.assert_close(actual, torch.stack([fn(row, k) for row in p]))
        assert not bool(torch.isnan(actual).any())
    torch.testing.assert_close(logpmf(p).exp(), pmf(p))


def test_log_gradcheck(device):
    p = torch.tensor(
        [[0.2, 0.3, 0.7], [0.1, 0.6, 0.8]],
        device=device,
        dtype=torch.float64,
        requires_grad=True,
    )
    for fn in [logpmf, lambda p: logcdf(p, 1), lambda p: logsf(p, 1)]:
        assert torch.autograd.gradcheck(fn, (p,))


@pytest.mark.parametrize(
    "values", [[0.0, 0.2, 0.7], [1.0, 0.3, 0.8], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
)
def test_defined_endpoint_derivatives(device, values):
    p = torch.tensor(values, device=device, dtype=torch.float64, requires_grad=True)
    q = pmf(p)
    actual_logs = logpmf(p)
    # Each finite event's boundary derivative is compared with the ordinary
    # polynomial recurrence, which also includes paths through zero coefficients.
    for j in range(q.shape[-1]):
        if q[j] > 0:
            expected = torch.autograd.grad(q[j].log(), p, retain_graph=True)[0]
            actual = torch.autograd.grad(actual_logs[j], p, retain_graph=True)[0]
            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
    for normal, logged in [(cdf, logcdf), (sf, logsf)]:
        for k in [0, 1, 2]:
            probability = normal(p, k)
            if probability > 0:
                expected = torch.autograd.grad(probability.log(), p, retain_graph=True)[
                    0
                ]
                actual = torch.autograd.grad(logged(p, k), p, retain_graph=True)[0]
                torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)


def test_masked_impossible_events_have_no_nans(device):
    p = torch.tensor(
        [[0.0, 0.0], [0.2, 0.7]], device=device, dtype=torch.float64, requires_grad=True
    )
    loss = torch.where(torch.isfinite(logsf(p, 0)), logsf(p, 0), 0).sum()
    loss.backward()
    assert bool(torch.isfinite(p.grad).all())
    q = torch.tensor(
        [0.0, 0.3, 1.0], device=device, dtype=torch.float64, requires_grad=True
    )
    weights = torch.arange(4, device=device)
    supported = pmf(q).detach() > 0
    expected = torch.autograd.grad((pmf(q)[supported] * weights[supported]).sum(), q)[0]
    (logpmf(q).exp() * weights).sum().backward()
    torch.testing.assert_close(q.grad, expected)


def test_underflow_gradient(device):
    p = torch.full((2000,), 0.5, device=device, dtype=torch.float64, requires_grad=True)
    logsf(p, 1999).backward()
    torch.testing.assert_close(p.grad, torch.full_like(p, 2), atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize(
    "fn",
    [logpmf, lambda p, **kw: logcdf(p, 0, **kw), lambda p, **kw: logsf(p, 0, **kw)],
)
def test_log_validation(fn):
    with pytest.raises(ValueError):
        fn(torch.tensor([-0.2]))
    with pytest.raises(ValueError):
        fn(torch.tensor([0.2]), method="fft_tree")
    with pytest.raises(TypeError):
        fn(torch.tensor([1]))
    with pytest.raises(ValueError):
        fn(torch.tensor(0.2))
    assert fn(torch.tensor([0.2]), validate_args=False).numel() > 0
