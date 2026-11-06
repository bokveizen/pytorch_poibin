"""Log probabilities computed directly, without ordinary-probability underflow."""

from typing import Literal

import torch
from torch import Tensor

from ._core import _count, _prepare
from ._log_dp import log_dp

LogMethod = Literal["dp", "auto", "shift_fft"]


def _prepare_log(probs: Tensor, method: str, validate_args: bool) -> Tensor:
    probs = _prepare(probs, method, validate_args)
    if method not in {"dp", "auto", "shift_fft"}:
        raise ValueError(
            "log probabilities support method='dp', 'auto', or 'shift_fft'"
        )
    return probs


def logpmf(
    probs: Tensor, *, method: LogMethod = "dp", validate_args: bool = True
) -> Tensor:
    """Return log P(X=j), j=0..n, directly using log-domain DP.

    Impossible counts return -inf. First derivatives are supported wherever
    finite, including deterministic input boundaries. Undefined derivatives of
    impossible events use a zero convention; second derivatives are unsupported.
    Auto stays on log DP, never taking a logarithm of an FFT result.
    """
    probs = _prepare_log(probs, method, validate_args)
    if method == "shift_fft":
        raise ValueError("shift_fft is a tail-only method; use logcdf or logsf")
    return log_dp(probs.double()).to(probs.dtype)


def logcdf(
    probs: Tensor, k: int, *, method: LogMethod = "dp", validate_args: bool = True
) -> Tensor:
    """Return log P(X <= k), retaining at most k+1 log-DP coefficients."""
    probs = _prepare_log(probs, method, validate_args)
    _count(k)
    zero = probs.sum(-1) * 0
    if k < 0:
        return zero - torch.inf
    if k >= probs.shape[-1]:
        return zero
    if method == "shift_fft":
        from ._shift import shift_logtail

        return shift_logtail(probs, k, upper=False)
    return log_dp(probs.double(), k, reduce=True).to(probs.dtype)


def logsf(
    probs: Tensor, k: int, *, method: LogMethod = "dp", validate_args: bool = True
) -> Tensor:
    """Return log P(X > k), directly using a failure-count log recurrence."""
    probs = _prepare_log(probs, method, validate_args)
    _count(k)
    zero = probs.sum(-1) * 0
    if k < 0:
        return zero
    if k >= probs.shape[-1]:
        return zero - torch.inf
    if method == "shift_fft":
        from ._shift import shift_logtail

        return shift_logtail(probs, k, upper=True)
    return log_dp(
        probs.double(), probs.shape[-1] - k - 1, failures=True, reduce=True
    ).to(probs.dtype)
