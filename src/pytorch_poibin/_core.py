"""Public contracts, exact threshold handling, and backend dispatch."""

import math
from numbers import Real
from typing import Literal

import torch
from torch import Tensor

from . import _backends, _selection

Method = Literal["dp", "fft_cf", "fft_tree", "auto"]
_METHODS = {"dp", "fft_cf", "fft_tree", "auto"}


def _prepare(probs: Tensor, method: str, validate_args: bool) -> Tensor:
    if not isinstance(probs, Tensor):
        raise TypeError("probs must be a floating PyTorch tensor")
    if probs.ndim < 1:
        raise ValueError("probs must have at least one dimension (the trial axis)")
    if probs.dtype not in {torch.float16, torch.bfloat16, torch.float32, torch.float64}:
        raise TypeError("probs must have float16, bfloat16, float32, or float64 dtype")
    if not isinstance(method, str) or method not in _METHODS:
        raise ValueError(f"method must be one of {sorted(_METHODS)}")
    if not isinstance(validate_args, bool):
        raise TypeError("validate_args must be a bool")
    if validate_args and not bool(
        (torch.isfinite(probs) & (probs >= 0) & (probs <= 1)).all()
    ):
        raise ValueError("probs must contain finite probabilities in [0, 1]")
    return probs.float() if probs.dtype in {torch.float16, torch.bfloat16} else probs


def _pmf(probs: Tensor, method: str) -> Tensor:
    if method == "auto":
        method = _selection.select_method(probs)
    if method == "dp":
        return _backends.dp(probs)
    kernel = _backends.fft_cf_raw if method == "fft_cf" else _backends.fft_tree_raw
    # Float32 phase errors and off-support roundoff accumulate on deterministic
    # inputs. Work in double, then preserve the public output dtype.
    working = probs.double() if probs.dtype == torch.float32 else probs
    return _backends.clean_fft(kernel(working)).to(probs.dtype)


def pmf(probs: Tensor, *, method: Method = "dp", validate_args: bool = True) -> Tensor:
    """Return P(X=k), k=0..n, with shape [..., n+1] for probs shaped [..., n].

    DP is the accuracy-first default. FFT and opt-in auto may lose tiny
    probabilities; FFT clipping/normalization can affect gradients at zero.
    Float32/float64 precision and device are preserved; half inputs use float32.
    """
    return _pmf(_prepare(probs, method, validate_args), method)


def _count(k: int) -> None:
    if not isinstance(k, int) or isinstance(k, bool):
        raise TypeError("k must be a scalar Python integer")


def cdf(
    probs: Tensor, k: int, *, method: Method = "dp", validate_args: bool = True
) -> Tensor:
    """Return P(X <= k). DP/auto retain only counts through k."""
    probs = _prepare(probs, method, validate_args)
    _count(k)
    zero = probs.sum(-1) * 0
    if k < 0:
        return zero
    if k >= probs.shape[-1]:
        return zero + 1
    if method in {"dp", "auto"}:
        return _backends.dp(probs, limit=k).sum(-1)
    return _pmf(probs, method)[..., : k + 1].sum(-1)


def sf(
    probs: Tensor, k: int, *, method: Method = "dp", validate_args: bool = True
) -> Tensor:
    """Return P(X > k), directly (never as 1 - cdf).

    DP/auto run a truncated recurrence on failures, retaining n-k entries.
    """
    probs = _prepare(probs, method, validate_args)
    _count(k)
    zero = probs.sum(-1) * 0
    if k < 0:
        return zero + 1
    if k >= probs.shape[-1]:
        return zero
    if method in {"dp", "auto"}:
        return _backends.dp_failures(probs, limit=probs.shape[-1] - k - 1).sum(-1)
    return _pmf(probs, method)[..., k + 1 :].sum(-1)


def proportional_tail(
    probs: Tensor,
    beta: float,
    interesting: Tensor,
    *,
    side: Literal["left", "right"] = "left",
    method: Method = "dp",
    validate_args: bool = True,
) -> Tensor:
    """Return P(S/(S+U) < beta) or P(S/(S+U) > beta), per batch.

    S counts successes selected by the binary interesting mask and U counts
    its complement. Both strict events exclude S+U=0, without conditioning.
    Beta is interpreted as the exact value of the supplied Python float.
    Masks must broadcast to probs.shape and share its device.
    Auto uses DP for both subset distributions.
    """
    probs = _prepare(probs, method, validate_args)
    if not isinstance(beta, Real) or isinstance(beta, bool):
        raise TypeError("beta must be a real Python scalar")
    beta = float(beta)
    if not math.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError("beta must be finite and in [0, 1]")
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")
    if not isinstance(interesting, Tensor):
        raise TypeError("interesting must be a binary PyTorch tensor")
    if interesting.device != probs.device:
        raise ValueError("interesting must be on the same device as probs")
    if interesting.is_complex() or not bool(
        ((interesting == 0) | (interesting == 1)).all()
    ):
        raise ValueError("interesting must contain only 0 and 1")
    try:
        mask = torch.broadcast_to(interesting, probs.shape).bool()
    except RuntimeError as exc:
        raise ValueError("interesting must broadcast to probs.shape") from exc
    selected = probs * mask
    other = probs * ~mask
    zero = probs.sum(-1) * 0
    backend = "dp" if method == "auto" else method
    # Endpoint events also exclude the empty outcome, with no subtraction.
    if beta == 0:
        return (
            zero
            if side == "left"
            else sf(selected, 0, method=backend, validate_args=False)
        )
    if beta == 1:
        return (
            sf(other, 0, method=backend, validate_args=False)
            if side == "left"
            else zero
        )

    pmf_s = _pmf(selected, backend)
    pmf_u = _pmf(other, backend)
    n = probs.shape[-1]
    numerator, denominator = beta.as_integer_ratio()
    denominator -= numerator
    # (1-beta)*S < beta*U iff S <= ceil(beta*U/(1-beta))-1.
    if side == "left":
        counts = [
            min(n, (numerator * t + denominator - 1) // denominator - 1)
            for t in range(n + 1)
        ]
        cumulative = pmf_s.cumsum(-1)
    else:
        counts = [min(n, numerator * t // denominator) for t in range(n + 1)]
        # Reverse accumulation avoids cancellation for small right tails.
        cumulative = torch.cat(
            (pmf_s[..., 1:].flip(-1).cumsum(-1).flip(-1), zero.unsqueeze(-1)), -1
        )
    indices = torch.tensor(counts, device=probs.device, dtype=torch.long)
    conditional = cumulative.index_select(-1, indices.clamp_min(0))
    if side == "left":
        conditional = conditional * (indices >= 0)
    return (conditional * pmf_u).sum(-1)
