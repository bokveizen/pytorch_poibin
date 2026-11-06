"""Batched, differentiable Poisson binomial PMFs and tail probabilities."""

from ._core import cdf, pmf, proportional_tail, sf

__all__ = ["pmf", "cdf", "sf", "proportional_tail"]
__version__ = "0.2.0"
