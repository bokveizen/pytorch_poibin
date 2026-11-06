"""Executable documentation and wheel-installation smoke checks."""

import torch

from pytorch_poibin import cdf, pmf, proportional_tail, sf


def main():
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for device in devices:
        p = torch.tensor(
            [[0.2, 0.7], [0.4, 0.5]],
            device=device,
            dtype=torch.float64,
            requires_grad=True,
        )
        torch.testing.assert_close(
            pmf(p), p.new_tensor([[0.24, 0.62, 0.14], [0.3, 0.5, 0.2]])
        )
        torch.testing.assert_close(cdf(p, 0), p.new_tensor([0.24, 0.3]))
        torch.testing.assert_close(sf(p, 1), p.new_tensor([0.14, 0.2]))
        mask = torch.tensor([1, 0], device=device)
        torch.testing.assert_close(
            proportional_tail(p, 0.5, mask), p.new_tensor([0.56, 0.3])
        )
        pmf(p)[..., 1].sum().backward()
        assert p.grad is not None and bool(torch.isfinite(p.grad).all())
        for method in ["fft_cf", "fft_tree", "auto"]:
            torch.testing.assert_close(pmf(p, method=method), pmf(p))
        print(f"{device}: package examples and gradients passed")


if __name__ == "__main__":
    main()
