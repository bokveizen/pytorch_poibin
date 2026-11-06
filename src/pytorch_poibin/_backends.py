"""Numerical kernels. Inputs have already been validated and promoted."""

import math

import torch
from torch import Tensor
from torch.nn.functional import pad


def dp(probs: Tensor, limit: int | None = None) -> Tensor:
    """Nonnegative recurrence, optionally retaining only counts through limit."""
    # Keep even the zero-trial result connected to autograd.
    result = probs.sum(dim=-1, keepdim=True) * 0 + 1
    for trial in probs.unbind(-1):
        p = trial.unsqueeze(-1)
        result = pad(result * (1 - p), (0, 1)) + pad(result * p, (1, 0))
        if limit is not None:
            result = result[..., : limit + 1]
    return result


def dp_failures(probs: Tensor, limit: int) -> Tensor:
    """Failure-count recurrence using original p, avoiding 1-(1-p)."""
    result = probs.sum(dim=-1, keepdim=True) * 0 + 1
    for trial in probs.unbind(-1):
        p = trial.unsqueeze(-1)
        result = (pad(result * p, (0, 1)) + pad(result * (1 - p), (1, 0)))[
            ..., : limit + 1
        ]
    return result


def fft_cf_raw(probs: Tensor) -> Tensor:
    """Characteristic-function inversion, with bounded frequency workspace."""
    n = probs.shape[-1]
    if n == 0 or probs.numel() == 0:
        return dp(probs)
    chunks = []
    for start in range(0, n + 1, 128):
        frequencies = torch.arange(
            start, min(start + 128, n + 1), device=probs.device, dtype=probs.dtype
        )
        roots = torch.exp((-2j * math.pi / (n + 1)) * frequencies)
        factors = (1 - probs.unsqueeze(-1)) + probs.unsqueeze(-1) * roots
        chunks.append(factors.prod(dim=-2))
    return torch.fft.ifft(torch.cat(chunks, dim=-1), dim=-1).real


def fft_tree_raw(probs: Tensor) -> Tensor:
    """Balanced polynomial multiplication with real FFT linear convolutions."""
    n = probs.shape[-1]
    if n == 0 or probs.numel() == 0:
        return dp(probs)
    nodes = torch.stack((1 - probs, probs), dim=-1)
    # An odd node is carried without FFT padding until the next level.
    while nodes.shape[-2] > 1:
        count = nodes.shape[-2]
        carry = nodes[..., -1:, :] if count % 2 else None
        left = nodes[..., : count - count % 2 : 2, :]
        right = nodes[..., 1 : count - count % 2 : 2, :]
        length = left.shape[-1] + right.shape[-1] - 1
        fft_length = 1 << (length - 1).bit_length()
        product = torch.fft.rfft(left, n=fft_length, dim=-1) * torch.fft.rfft(
            right, n=fft_length, dim=-1
        )
        nodes = torch.fft.irfft(product, n=fft_length, dim=-1)[..., :length]
        if carry is not None:
            nodes = torch.cat(
                (nodes, pad(carry, (0, length - carry.shape[-1]))), dim=-2
            )
    return nodes.squeeze(-2)[..., : n + 1]


def clean_fft(result: Tensor) -> Tensor:
    """Clean roundoff without adding mass to individual outcomes."""
    finite = torch.isfinite(result).all()
    result = result.clamp_min(0)
    mass = result.sum(dim=-1, keepdim=True)
    if not bool(finite & torch.isfinite(mass).all() & (mass > 0).all()):
        raise RuntimeError("FFT produced nonfinite probabilities or nonpositive mass")
    return result / mass
