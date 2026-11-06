import pytest
import torch

from pytorch_poibin import cdf, pmf, proportional_tail, sf


@pytest.mark.parametrize("method", ["dp", "fft_cf", "fft_tree", "auto"])
def test_gradcheck(device, method):
    p = torch.tensor(
        [[0.21, 0.37, 0.72], [0.14, 0.43, 0.68]],
        device=device,
        dtype=torch.float64,
        requires_grad=True,
    )
    for fn in [
        lambda p: pmf(p, method=method),
        lambda p: cdf(p, 1, method=method),
        lambda p: sf(p, 1, method=method),
    ]:
        assert torch.autograd.gradcheck(fn, (p,))


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("method", ["dp", "fft_cf", "fft_tree", "auto"])
def test_proportional_gradcheck(device, side, method):
    p = torch.tensor(
        [0.2, 0.3, 0.7], device=device, dtype=torch.float64, requires_grad=True
    )
    mask = torch.tensor([1, 0, 1], device=device)
    assert torch.autograd.gradcheck(
        lambda p: proportional_tail(p, 0.5, mask, side=side, method=method), (p,)
    )


@pytest.mark.parametrize("values", [[0.0, 0.0, 1.0], [1.0, 1.0, 0.0], [0.2, 0.3, 0.7]])
def test_dp_endpoint_derivative(device, values):
    p = torch.tensor(values, device=device, dtype=torch.float64, requires_grad=True)
    weights = torch.arange(len(values) + 1, device=device, dtype=p.dtype).square()
    (pmf(p) * weights).sum().backward()
    # E[X^2] = sum(p) + 2*sum_{i<j} p_i*p_j.
    expected = 1 + 2 * (p.sum() - p)
    torch.testing.assert_close(p.grad, expected)


@pytest.mark.cuda
@pytest.mark.parametrize("method", ["dp", "fft_cf", "fft_tree", "auto"])
def test_cpu_cuda_gradients(method):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    gradients = []
    for device in ["cpu", "cuda"]:
        p = torch.tensor(
            [0.2, 0.3, 0.7], device=device, dtype=torch.float64, requires_grad=True
        )
        pmf(p, method=method)[1].backward()
        gradients.append(p.grad.cpu())
    torch.testing.assert_close(*gradients, atol=1e-10, rtol=1e-10)


def test_boundary_tails_are_connected():
    p = torch.tensor([0.2, 0.3], dtype=torch.float64, requires_grad=True)
    for fn, k in [(cdf, -1), (cdf, 3), (sf, -1), (sf, 3)]:
        gradient = torch.autograd.grad(fn(p, k), p)[0]
        torch.testing.assert_close(gradient, torch.zeros_like(p))
