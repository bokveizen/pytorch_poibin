import pytest
import torch

from pytorch_poibin import cdf, pmf, proportional_tail, sf


@pytest.mark.parametrize("block_size", [8, 16, 32, 64, 128])
@pytest.mark.parametrize("n", [0, 1, 7, 8, 9, 31, 32, 33, 129, 512])
def test_hybrid_against_dp(device, block_size, n):
    generator = torch.Generator().manual_seed(123 + n)
    p = (
        torch.rand(2, 3, n, generator=generator, dtype=torch.float64)
        .to(device)
        .transpose(0, 1)
    )
    actual = pmf(p, method="fft_hybrid", block_size=block_size)
    torch.testing.assert_close(actual, pmf(p), atol=1e-10, rtol=1e-10)
    assert actual.shape == p.shape[:-1] + (n + 1,)
    for fn in [cdf, sf]:
        torch.testing.assert_close(
            fn(p, 2, method="fft_hybrid", block_size=block_size), fn(p, 2)
        )


@pytest.mark.parametrize("block_size", [8, 16, 32, 64, 128])
def test_hybrid_gradients_and_proportions(device, block_size):
    p = torch.linspace(
        0.1, 0.9, 17, device=device, dtype=torch.float64
    ).requires_grad_()
    weights = torch.linspace(-1, 1, 18, device=device, dtype=p.dtype)
    actual = pmf(p, method="fft_hybrid", block_size=block_size)
    ga = torch.autograd.grad((actual * weights).sum(), p)[0]
    ge = torch.autograd.grad((pmf(p) * weights).sum(), p)[0]
    torch.testing.assert_close(ga, ge, atol=1e-10, rtol=1e-10)
    mask = torch.arange(17, device=device) % 2
    torch.testing.assert_close(
        proportional_tail(p, 0.5, mask, method="fft_hybrid", block_size=block_size),
        proportional_tail(p, 0.5, mask),
    )


def test_hybrid_gradcheck(device):
    p = torch.linspace(0.2, 0.8, 9, device=device, dtype=torch.float64).requires_grad_()
    assert torch.autograd.gradcheck(
        lambda p: pmf(p, method="fft_hybrid", block_size=8), (p,)
    )


def test_hybrid_empty_and_half(device):
    p = torch.empty(2, 0, 4, device=device, dtype=torch.float64, requires_grad=True)
    q = pmf(p, method="fft_hybrid")
    assert q.shape == (2, 0, 5)
    q.sum().backward()
    assert p.grad is not None
    p = torch.tensor(
        [0.25, 0.5, 0.75], device=device, dtype=torch.float16, requires_grad=True
    )
    q = pmf(p, method="fft_hybrid")
    assert q.dtype == torch.float32
    q[1].backward()
    assert bool(torch.isfinite(p.grad).all())


@pytest.mark.parametrize("block_size", [0, -1, True, 1.0, "8"])
def test_invalid_block_size(block_size):
    p = torch.tensor([0.2, 0.7])
    with pytest.raises(ValueError, match="block_size"):
        pmf(p, method="fft_hybrid", block_size=block_size)
