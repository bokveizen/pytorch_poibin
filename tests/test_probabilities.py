import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import binom, poisson_binom

from pytorch_poibin import _backends, cdf, pmf, proportional_tail, sf

from ._reference import enumerate_pmf, enumerate_proportion

METHODS = ["dp", "fft_cf", "fft_tree", "auto"]


@pytest.mark.parametrize("method", ["fft_cf", "fft_tree", "auto"])
def test_float32_deterministic_large(device, method):
    generator = torch.Generator().manual_seed(2026)
    p = (torch.rand(16, 512, generator=generator) > 0.5).float().to(device)
    q = pmf(p, method=method)
    expected = torch.nn.functional.one_hot(p.sum(-1).long(), 513).to(p.dtype)
    torch.testing.assert_close(q, expected, atol=1e-5, rtol=0)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("n", [0, 1, 2, 3, 7, 8, 9, 31, 32, 33, 127, 128, 129, 512])
def test_scipy_and_invariants(device, method, dtype, n):
    generator = torch.Generator().manual_seed(17 + n)
    p = torch.rand(2, 2, n, generator=generator, dtype=dtype).to(device)
    actual = pmf(p, method=method)
    reference = (
        torch.as_tensor(
            poisson_binom.pmf(
                torch.arange(n + 1).numpy()[:, None, None], p.cpu().double().numpy()
            ).transpose(1, 2, 0),
            device=device,
            dtype=dtype,
        )
        if n
        else torch.ones(2, 2, 1, device=device, dtype=dtype)
    )
    atol, rtol = (1e-10, 1e-10) if dtype == torch.float64 else (1e-5, 1e-4)
    if method == "dp":
        atol = 1e-12 if dtype == torch.float64 else 1e-6
    torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)
    assert actual.shape == (2, 2, n + 1)
    assert actual.device == p.device and actual.dtype == dtype
    assert bool((actual >= 0).all())
    torch.testing.assert_close(actual.sum(-1), p.new_ones(2, 2), atol=atol, rtol=rtol)
    counts = torch.arange(n + 1, device=device, dtype=dtype)
    mean = (actual * counts).sum(-1)
    variance = (actual * (counts - mean.unsqueeze(-1)).square()).sum(-1)
    torch.testing.assert_close(mean, p.sum(-1), atol=atol * max(1, n), rtol=rtol)
    torch.testing.assert_close(
        variance, (p * (1 - p)).sum(-1), atol=atol * max(1, n * n), rtol=rtol
    )


@pytest.mark.parametrize("method", METHODS)
def test_enumeration_and_binomial(device, method):
    p = torch.tensor([0.01, 0.99], device=device, dtype=torch.float64)
    torch.testing.assert_close(
        pmf(p, method=method), p.new_tensor([0.0099, 0.9802, 0.0099])
    )
    p = torch.tensor([0.1, 0.3, 0.7, 0.9], device=device, dtype=torch.float64)
    torch.testing.assert_close(
        pmf(p, method=method), p.new_tensor(enumerate_pmf(p.tolist()))
    )
    p = torch.full((30,), 0.3, device=device, dtype=torch.float64)
    torch.testing.assert_close(
        pmf(p, method=method), p.new_tensor(binom.pmf(range(31), 30, 0.3))
    )


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "values", [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0, 0.5], [1e-12, 1 - 1e-12]]
)
def test_endpoints(device, method, values):
    p = torch.tensor(values, device=device, dtype=torch.float64)
    torch.testing.assert_close(
        pmf(p, method=method),
        p.new_tensor(enumerate_pmf(values)),
        atol=1e-12,
        rtol=1e-10,
    )


@pytest.mark.parametrize("method", METHODS)
def test_batch_independence_noncontiguous_and_symmetry(device, method):
    p = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4], [0.7, 0.8, 0.9, 0.6]], device=device, dtype=torch.float64
    ).T
    assert not p.is_contiguous()
    actual = pmf(p, method=method)
    torch.testing.assert_close(
        actual, torch.stack([pmf(row, method=method) for row in p])
    )
    torch.testing.assert_close(actual, pmf(p.flip(-1), method=method))
    torch.testing.assert_close(actual, pmf(1 - p, method=method).flip(-1))


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("shape", [(0,), (2, 0), (0, 3), (2, 0, 4)])
def test_empty_shapes(device, method, shape):
    p = torch.empty(shape, device=device, dtype=torch.float64, requires_grad=True)
    q = pmf(p, method=method)
    assert q.shape == shape[:-1] + (shape[-1] + 1,)
    q.sum().backward()
    assert p.grad is not None and p.grad.shape == p.shape
    assert cdf(p, 0, method=method).shape == shape[:-1]
    assert sf(p, 0, method=method).shape == shape[:-1]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("k", [-5, -1, 0, 1, 2, 3, 9])
def test_tails(device, method, k):
    p = torch.tensor([[0.2, 0.7], [0.4, 0.5]], device=device, dtype=torch.float64)
    reference = torch.tensor(
        [enumerate_pmf(row) for row in p.tolist()], device=device, dtype=p.dtype
    )
    expected_cdf = reference[..., : max(0, k + 1)].sum(-1)
    expected_sf = reference[..., max(0, k + 1) :].sum(-1)
    torch.testing.assert_close(cdf(p, k, method=method), expected_cdf.to(p.dtype))
    torch.testing.assert_close(sf(p, k, method=method), expected_sf.to(p.dtype))
    assert cdf(p[0], k, method=method).ndim == 0


def test_rare_tails(device):
    p = torch.full((100,), 0.5, device=device, dtype=torch.float64)
    expected = p.new_tensor(2.0**-100)
    for method in ["dp", "auto"]:
        torch.testing.assert_close(
            cdf(p, 0, method=method), expected, atol=0, rtol=1e-12
        )
        torch.testing.assert_close(
            sf(p, 99, method=method), expected, atol=0, rtol=1e-12
        )
    torch.testing.assert_close(pmf(p)[0], expected, atol=0, rtol=1e-12)
    # Do not round tiny successes away by computing 1-(1-p) in failure DP.
    tiny = torch.tensor([1e-30], device=device, dtype=torch.float64)
    torch.testing.assert_close(sf(tiny, 0), tiny[0], atol=0, rtol=1e-12)
    torch.testing.assert_close(
        proportional_tail(tiny, 0, torch.ones_like(tiny), side="right"),
        tiny[0],
        atol=0,
        rtol=1e-12,
    )


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("beta", [0.0, 0.1, 1 / 3, 0.5, 0.9, 1.0])
@pytest.mark.parametrize("mask", [[0, 0, 0], [1, 1, 1], [1, 0, 1]])
def test_proportions(device, method, side, beta, mask):
    p = torch.tensor(
        [[0.2, 0.7, 0.4], [0.4, 0.5, 0.8]], device=device, dtype=torch.float64
    )
    m = torch.tensor(mask, device=device)
    expected = p.new_tensor(
        [enumerate_proportion(row, mask, beta, side) for row in p.tolist()]
    )
    torch.testing.assert_close(
        proportional_tail(p, beta, m, side=side, method=method), expected
    )
    batched_mask = torch.stack((m, 1 - m)).float()
    actual = proportional_tail(p, beta, batched_mask, side=side, method=method)
    expected = p.new_tensor(
        [
            enumerate_proportion(row, mask, beta, side)
            for row, mask in zip(p.tolist(), batched_mask.tolist())
        ]
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("method", METHODS)
def test_proportion_regression_and_empty(device, method):
    p = torch.tensor([[0.2, 0.7], [0.4, 0.5]], device=device, dtype=torch.float64)
    m = torch.tensor([1, 0], device=device)
    torch.testing.assert_close(
        proportional_tail(p, 0.5, m, method=method), p.new_tensor([0.56, 0.30])
    )
    torch.testing.assert_close(
        proportional_tail(p, 0.5, m, side="right", method=method),
        p.new_tensor([0.06, 0.20]),
    )
    for n in [0, 3]:
        p = torch.zeros(2, n, device=device, dtype=torch.float64)
        m = torch.zeros(n, device=device, dtype=torch.bool)
        for beta in [0, 0.5, 1]:
            for side in ["left", "right"]:
                torch.testing.assert_close(
                    proportional_tail(p, beta, m, side=side, method=method),
                    p.new_zeros(2),
                    atol=1e-12,
                    rtol=0,
                )


@pytest.mark.parametrize("method", METHODS)
def test_proportions_multiple_batch_dimensions(device, method):
    p = torch.tensor(
        [[[0.2, 0.7], [0.4, 0.5]], [[0.6, 0.3], [0.8, 0.1]]],
        device=device,
        dtype=torch.float64,
    )
    mask = torch.tensor([[[1, 0]], [[0, 1]]], device=device)
    actual = proportional_tail(p, 0.5, mask, method=method)
    expected = torch.stack(
        [proportional_tail(row, 0.5, m, method=method) for row, m in zip(p, mask)]
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("method", METHODS)
def test_half_promotion(device, dtype, method):
    p = torch.tensor([0.25, 0.5, 0.75], device=device, dtype=dtype, requires_grad=True)
    q = pmf(p, method=method)
    assert q.dtype == torch.float32 and q.device == p.device
    q[1].backward()
    assert p.grad is not None and bool(torch.isfinite(p.grad).all())


@pytest.mark.parametrize("kernel", [_backends.fft_cf_raw, _backends.fft_tree_raw])
def test_raw_fft_roundoff(device, kernel):
    p = torch.linspace(0.05, 0.95, 129, device=device, dtype=torch.float64)
    raw = kernel(p)
    assert raw.min() >= -1e-12
    torch.testing.assert_close(raw.sum(), p.new_tensor(1.0), atol=1e-12, rtol=0)


@given(
    st.lists(
        st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False),
        max_size=7,
    )
)
@settings(max_examples=35, deadline=None)
def test_property_dp_matches_enumeration(values):
    p = torch.tensor(values, dtype=torch.float64)
    torch.testing.assert_close(
        pmf(p), p.new_tensor(enumerate_pmf(values)), atol=1e-12, rtol=1e-10
    )
