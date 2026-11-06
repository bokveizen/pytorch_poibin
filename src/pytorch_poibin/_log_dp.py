"""Log recurrences, signed-log PMF adjoints, and nonnegative tail derivatives.

The adjoint differentiates probability coefficients, rather than log states.
This retains paths through zero coefficients at deterministic input boundaries.
"""

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable
from torch.nn.functional import pad


def log_factors(probs: Tensor) -> tuple[Tensor, Tensor]:
    """Logs of p and 1-p, without forming a double complement."""
    log_p = torch.where(probs > 0, probs, 1).log()
    log_q = torch.log1p(-torch.where(probs < 1, probs, 0))
    return (
        torch.where(probs > 0, log_p, -torch.inf),
        torch.where(probs < 1, log_q, -torch.inf),
    )


def logsumexp(values: Tensor) -> Tensor:
    """Reduce counts; impossible rows have a zero-gradient convention."""
    possible = torch.isfinite(values).any(-1)
    safe = torch.where(possible.unsqueeze(-1), values, 0)
    return torch.where(possible, torch.logsumexp(safe, -1), -torch.inf)


def _signed_difference(positive: Tensor, negative: Tensor) -> Tensor:
    same = positive == negative
    high = torch.where(same, 0, torch.maximum(positive, negative))
    low = torch.where(same, 0, torch.minimum(positive, negative))
    magnitude = torch.exp(high + torch.log(-torch.expm1(low - high)))
    return torch.where(positive > negative, magnitude, -magnitude)


def _step(result: Tensor, log_p: Tensor, log_q: Tensor, limit: int) -> Tensor:
    return torch.logaddexp(
        pad(result + log_q.unsqueeze(-1), (0, 1), value=-torch.inf),
        pad(result + log_p.unsqueeze(-1), (1, 0), value=-torch.inf),
    )[..., : limit + 1]


def tail_gradient(
    probs: Tensor, k: int, upper: bool, total: Tensor, incoming: Tensor
) -> Tensor:
    """Use dF/dp_i = -P(X without i = k), with the opposite sign for SF.

    Prefix/suffix log convolutions compute nonnegative leave-one-out densities.
    Unlike subtracting adjoints, this retains tiny derivatives of near-one tails.
    """
    n = probs.shape[-1]
    if n == 0:
        return torch.zeros_like(probs)
    failures = k > (n - 1) // 2
    count = n - 1 - k if failures else k
    log_p, log_q = log_factors(probs)
    if failures:
        log_p, log_q = log_q, log_p
    state = probs.sum(-1, keepdim=True) * 0
    prefixes = [state]
    for i in range(n - 1):
        state = _step(state, log_p[..., i], log_q[..., i], count)
        prefixes.append(state)
    suffix = probs.sum(-1, keepdim=True) * 0
    nonzero = (incoming != 0) & torch.isfinite(total)
    scale = torch.where(nonzero, incoming.abs(), 1).log() - torch.where(
        nonzero, total, 0
    )
    scale = torch.where(nonzero, scale, -torch.inf)
    sign = incoming.sign() * (1 if upper else -1)
    derivatives = []
    for i in range(n - 1, -1, -1):
        prefix = prefixes[i]
        start = max(0, count - suffix.shape[-1] + 1)
        end = min(count + 1, prefix.shape[-1])
        if start >= end:
            density = probs.sum(-1) * 0 - torch.inf
        else:
            density = torch.logsumexp(
                prefix[..., start:end]
                + suffix[..., count - end + 1 : count - start + 1].flip(-1),
                -1,
            )
        derivatives.append(sign * torch.exp(density + scale))
        if i:
            suffix = _step(suffix, log_p[..., i], log_q[..., i], count)
    return torch.stack(derivatives[::-1], -1)


class _LogTail(torch.autograd.Function):
    @staticmethod
    def forward(ctx, probs: Tensor, limit: int, failures: bool):
        # Nested Function.forward is inference: no prefixes are kept here.
        result = logsumexp(log_dp(probs, limit, failures=failures)).clamp_max(0)
        ctx.k = probs.shape[-1] - limit - 1 if failures else limit
        ctx.upper = failures
        ctx.save_for_backward(probs, result)
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, incoming):
        probs, result = ctx.saved_tensors
        return tail_gradient(probs, ctx.k, ctx.upper, result, incoming), None, None


class _LogDP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, probs: Tensor, limit: int, failures: bool):
        log_p, log_q = log_factors(probs)
        if failures:
            log_p, log_q = log_q, log_p
        result = probs.sum(-1, keepdim=True) * 0
        states = [result] if ctx.needs_input_grad[0] else []
        for success, failure in zip(log_p.unbind(-1), log_q.unbind(-1)):
            result = torch.logaddexp(
                pad(result + failure.unsqueeze(-1), (0, 1), value=-torch.inf),
                pad(result + success.unsqueeze(-1), (1, 0), value=-torch.inf),
            )[..., : limit + 1]
            if ctx.needs_input_grad[0]:
                states.append(result)
        ctx.failures = failures
        ctx.save_for_backward(probs, *states)
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: Tensor):
        probs, *states = ctx.saved_tensors
        if probs.shape[-1] == 0:
            return torch.zeros_like(probs), None, None
        final = states[-1]
        nonzero = (grad_output != 0) & torch.isfinite(final)
        log_abs = torch.where(nonzero, grad_output.abs(), 1).log()
        seed = log_abs - torch.where(nonzero, final, 0)
        positive = torch.where(nonzero & (grad_output > 0), seed, -torch.inf)
        negative = torch.where(nonzero & (grad_output < 0), seed, -torch.inf)
        log_p, log_q = log_factors(probs)
        if ctx.failures:
            log_p, log_q = log_q, log_p
        derivatives = []
        for i in range(probs.shape[-1] - 1, -1, -1):
            old = states[i]
            length = positive.shape[-1]
            stay = pad(old, (0, length - old.shape[-1]), value=-torch.inf)
            increment = pad(old, (1, 0), value=-torch.inf)[..., :length]
            plus = torch.logsumexp(
                torch.cat((positive + increment, negative + stay), -1), -1
            )
            minus = torch.logsumexp(
                torch.cat((negative + increment, positive + stay), -1), -1
            )
            derivative = _signed_difference(plus, minus)
            derivatives.append(-derivative if ctx.failures else derivative)
            old_length = old.shape[-1]
            success = log_p[..., i : i + 1]
            failure = log_q[..., i : i + 1]
            positive = torch.logaddexp(
                positive[..., :old_length] + failure,
                pad(positive, (0, 1), value=-torch.inf)[..., 1 : old_length + 1]
                + success,
            )
            negative = torch.logaddexp(
                negative[..., :old_length] + failure,
                pad(negative, (0, 1), value=-torch.inf)[..., 1 : old_length + 1]
                + success,
            )
        return torch.stack(derivatives[::-1], -1), None, None


def log_dp(
    probs: Tensor,
    limit: int | None = None,
    *,
    failures: bool = False,
    reduce: bool = False,
) -> Tensor:
    # Function.forward runs with grad disabled, so decide whether to retain
    # prefixes here. A requires-grad input inside no_grad is still inference.
    source = probs if torch.is_grad_enabled() else probs.detach()
    if reduce:
        return _LogTail.apply(
            source, probs.shape[-1] if limit is None else limit, failures
        )
    return _LogDP.apply(source, probs.shape[-1] if limit is None else limit, failures)
