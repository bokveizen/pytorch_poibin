"""Log probabilities computed directly, without ordinary-probability underflow."""

from typing import Literal

import torch
from torch import Tensor

from ._core import _count, _prepare
from ._log_dp import log_dp

LogMethod = Literal["dp", "auto"]


def _prepare_log(probs: Tensor, method: str, validate_args: bool) -> Tensor:
    probs = _prepare(probs, method, validate_args)
    if method not in {"dp", "auto"}:
        raise ValueError("log probabilities support method='dp' or 'auto'")
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
    return log_dp(_prepare_log(probs, method, validate_args))


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
    return log_dp(probs, k, reduce=True)


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
    return log_dp(probs, probs.shape[-1] - k - 1, failures=True, reduce=True)
