"""Exponential tilting and logarithmic FFT tail reconstruction.

Peres, Lee, Keich (2020), https://arxiv.org/abs/2004.07429, Algorithm 1.
We use a real-FFT tree rather than the paper's optional frugal transform engine.
The root solve is not differentiated: the exact unshifted probability is
independent of the chosen tilt. Leave-one-out log densities supply first derivatives
without differentiating through FFT clipping or ill-conditioned cancellation.
"""

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

from . import _backends
from ._log_dp import log_factors, tail_gradient


def solve_tilt(logits: Tensor, target: Tensor, active: Tensor) -> Tensor:
    """Bracket by doubling, then bisect; all operations stay on the input device."""
    lower = torch.full_like(target, -1)
    upper = torch.full_like(target, 1)

    def mean(theta):
        return torch.sigmoid(logits + theta.unsqueeze(-1)).sum(-1)

    for _ in range(12):
        too_high = active & (mean(lower) > target)
        too_low = active & (mean(upper) < target)
        if not bool((too_high | too_low).any()):
            break
        lower = torch.where(too_high, lower * 2, lower)
        upper = torch.where(too_low, upper * 2, upper)
    else:
        raise RuntimeError("could not bracket the exponential tilt")
    tolerance = 32 * torch.finfo(logits.dtype).eps * max(1, logits.shape[-1])
    theta = (lower + upper) / 2
    for _ in range(64):
        theta = (lower + upper) / 2
        current = mean(theta)
        if bool((~active | ((current - target).abs() <= tolerance)).all()):
            break
        lower = torch.where(current < target, theta, lower)
        upper = torch.where(current >= target, theta, upper)
    else:
        raise RuntimeError("exponential tilt solver failed to converge")
    return torch.where(active, theta, 0)


def shifted_logtail(probs: Tensor, k: int, upper: bool) -> Tensor:
    """Compute log P(X>k) or log P(X<=k); probs use float64 internally."""
    if probs.numel() == 0:
        return probs.sum(-1) * 0
    n = probs.shape[-1]
    minimum = (probs == 1).sum(-1)
    maximum = (probs > 0).sum(-1)
    log_p, log_q = log_factors(probs)
    mean = probs.sum(-1)
    # Compute the numerically favorable tail in each batch row. Recover a large
    # requested tail with log1mexp, preserving e.g. log P(X=0)=-1e-300.
    direct_upper = k + 0.5 >= mean
    boundary = torch.where(direct_upper, k + 1, k).to(probs.dtype)
    target = torch.where(
        direct_upper,
        torch.minimum(boundary, maximum - 0.5),
        torch.maximum(boundary, minimum + 0.5),
    )
    active = torch.where(direct_upper, target > mean, target < mean) & (
        maximum > minimum
    )
    impossible = torch.where(direct_upper, boundary > maximum, boundary < minimum)
    certain = torch.where(direct_upper, boundary <= minimum, boundary >= maximum)
    active = active & ~impossible & ~certain
    theta = solve_tilt(log_p - log_q, target, active)
    normalizers = torch.logaddexp(log_q, log_p + theta.unsqueeze(-1))
    # Compute both coefficients separately: 1-sigmoid(logit+theta) can round to
    # zero while the failure probability is still needed in the tilted PMF.
    nodes = torch.stack(
        (
            torch.exp(log_q - normalizers),
            torch.exp(log_p + theta.unsqueeze(-1) - normalizers),
        ),
        -1,
    )
    raw = _backends.convolve_nodes(nodes, n)
    mass = raw.sum(-1)
    if not bool(
        torch.isfinite(raw).all()
        & (raw >= -1e-10).all()
        & ((mass - 1).abs() <= 1e-8).all()
    ):
        raise RuntimeError("shifted FFT failed its finite-mass/roundoff checks")
    tilted = _backends.clean_fft(raw)
    counts = torch.arange(n + 1, device=probs.device)
    support = (counts >= minimum.unsqueeze(-1)) & (counts <= maximum.unsqueeze(-1))
    log_tilted = torch.where(support & (tilted > 0), tilted.log(), -torch.inf)
    selected = torch.where(direct_upper.unsqueeze(-1), counts >= k + 1, counts <= k)
    offsets = counts - boundary.unsqueeze(-1)
    # Offsets always point away from the boundary in the direction of the tail,
    # so -theta*offset <= 0. Roundoff far across the mean cannot be amplified.
    log_probability = (
        # Cancel guaranteed successes analytically before subtracting the shift.
        torch.where(probs == 1, 0, normalizers).sum(-1)
        - theta * (boundary - minimum)
        + torch.logsumexp(
            torch.where(
                selected, log_tilted - theta.unsqueeze(-1) * offsets, -torch.inf
            ),
            -1,
        )
    )
    partial = ~impossible & ~certain
    if not bool(
        (~partial | (torch.isfinite(log_probability) & (log_probability <= 1e-8))).all()
    ):
        raise RuntimeError("shifted FFT could not reconstruct a valid tail")
    direct = torch.where(
        impossible, -torch.inf, torch.where(certain, 0, log_probability.clamp_max(0))
    )
    complement = torch.where(
        direct < -0.6931471805599453,
        torch.log1p(-direct.exp()),
        torch.log(-torch.expm1(direct)),
    )
    return torch.where(direct_upper == upper, direct, complement)


class _ShiftTail(torch.autograd.Function):
    @staticmethod
    def forward(ctx, probs: Tensor, k: int, upper: bool):
        ctx.k, ctx.upper = k, upper
        result = shifted_logtail(probs, k, upper)
        ctx.save_for_backward(probs, result)
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: Tensor):
        probs, result = ctx.saved_tensors
        gradient = tail_gradient(probs, ctx.k, ctx.upper, result, grad_output)
        return gradient, None, None


def shift_logtail(probs: Tensor, k: int, *, upper: bool) -> Tensor:
    return _ShiftTail.apply(probs.double(), k, upper).to(probs.dtype)
