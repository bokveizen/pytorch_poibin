import torch

from pytorch_poibin import _backends, _selection, cdf, pmf, proportional_tail, sf


def test_dispatch_exact_profile_and_fallback(monkeypatch):
    p = torch.full((2, 8), 0.3, dtype=torch.float64)
    profile = {
        "runtime": _selection.hardware_key(p.device),
        "dtype": str(p.dtype),
        "batch_size": 2,
        "n": 8,
    }
    monkeypatch.setattr(_selection, "load_profiles", lambda: [profile])
    assert _selection.select_method(p) == "fft_tree"
    torch.testing.assert_close(pmf(p, method="auto"), pmf(p, method="fft_tree"))
    for other in [p.float(), p[:1], p[:, :7], p.clone().requires_grad_()]:
        assert _selection.select_method(other) == "dp"
    profile["runtime"] = {**profile["runtime"], "torch": "unmeasured"}
    assert _selection.select_method(p) == "dp"


def test_auto_tails_do_not_select_fft(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("auto tails must stay on DP")

    monkeypatch.setattr(_selection, "select_method", forbidden)
    p = torch.tensor([0.2, 0.3, 0.7], dtype=torch.float64)
    torch.testing.assert_close(cdf(p, 1, method="auto"), cdf(p, 1))
    torch.testing.assert_close(sf(p, 1, method="auto"), sf(p, 1))
    torch.testing.assert_close(
        proportional_tail(p, 0.5, torch.tensor([1, 0, 1]), method="auto"),
        proportional_tail(p, 0.5, torch.tensor([1, 0, 1])),
    )


def test_truncation_limit(monkeypatch):
    original = _backends.dp
    original_failures = _backends.dp_failures
    calls = []

    def record(probs, limit=None):
        calls.append((probs, limit))
        return original(probs, limit)

    monkeypatch.setattr(_backends, "dp", record)

    def record_failures(probs, limit):
        calls.append((probs, limit))
        return original_failures(probs, limit)

    monkeypatch.setattr(_backends, "dp_failures", record_failures)
    p = torch.full((100,), 0.5)
    cdf(p, 2)
    sf(p, 98)
    assert [limit for _, limit in calls] == [2, 1]


def test_proportional_distributions_computed_once(monkeypatch):
    original = _backends.dp
    calls = []

    def record(probs, limit=None):
        calls.append(probs)
        return original(probs, limit)

    monkeypatch.setattr(_backends, "dp", record)
    proportional_tail(torch.full((2, 8), 0.3), 0.5, torch.tensor([0, 1] * 4))
    assert len(calls) == 2
