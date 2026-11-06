import pytest
import torch

from pytorch_poibin import _backends, cdf, pmf, proportional_tail, sf


@pytest.mark.parametrize("value", [[-0.1], [1.1], [float("nan")], [float("inf")]])
def test_invalid_probabilities(value):
    with pytest.raises(ValueError, match="finite probabilities"):
        pmf(torch.tensor(value))


@pytest.mark.parametrize("value", [[0.2], torch.tensor([1]), torch.tensor([0.2 + 0j])])
def test_probability_types(value):
    with pytest.raises(TypeError):
        pmf(value)


def test_structural_validation():
    with pytest.raises(ValueError):
        pmf(torch.tensor(0.2))
    with pytest.raises(ValueError):
        pmf(torch.tensor([0.2]), method="unknown")
    with pytest.raises(ValueError):
        pmf(torch.tensor([0.2]), method=[])
    with pytest.raises(TypeError):
        pmf(torch.tensor([0.2]), validate_args=1)
    with pytest.raises(ValueError):
        pmf(torch.tensor(0.2), validate_args=False)
    for fn in [cdf, sf]:
        for k in [True, 1.0, torch.tensor(1)]:
            with pytest.raises(TypeError):
                fn(torch.tensor([0.2]), k)


def test_value_checks_can_be_disabled():
    p = torch.tensor([-0.1, 1.1])
    assert pmf(p, validate_args=False).shape == (3,)


@pytest.mark.parametrize("beta", [-0.1, 1.1, float("inf"), float("nan")])
def test_beta_range(beta):
    with pytest.raises(ValueError):
        proportional_tail(torch.tensor([0.2]), beta, torch.tensor([1]))


@pytest.mark.parametrize("beta", [True, ".5", torch.tensor(0.5)])
def test_beta_type(beta):
    with pytest.raises(TypeError):
        proportional_tail(torch.tensor([0.2]), beta, torch.tensor([1]))


@pytest.mark.parametrize(
    "mask",
    [
        torch.tensor([0.3]),
        torch.tensor([float("nan")]),
        torch.ones(2, 2),
        torch.tensor([1 + 0j]),
    ],
)
def test_invalid_masks(mask):
    with pytest.raises(ValueError):
        proportional_tail(torch.tensor([0.2]), 0.5, mask)


def test_invalid_proportional_structure():
    p = torch.tensor([0.2])
    with pytest.raises(TypeError):
        proportional_tail(p, 0.5, [1])
    with pytest.raises(ValueError):
        proportional_tail(p, 0.5, torch.tensor([1]), side="invalid")
    if torch.cuda.is_available():
        with pytest.raises(ValueError, match="same device"):
            proportional_tail(p, 0.5, torch.tensor([1], device="cuda"))


@pytest.mark.parametrize(
    "values",
    [
        [float("nan"), 1.0],
        [float("inf"), 1.0],
        [-float("inf"), 1.0],
        [0.0, 0.0],
        [-1.0, 0.0],
    ],
)
def test_fft_invalid_mass(values):
    with pytest.raises(RuntimeError):
        _backends.clean_fft(torch.tensor(values))
