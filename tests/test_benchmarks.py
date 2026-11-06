import copy

from benchmarks.benchmark import qualified_profiles, workspace_estimate


def record(method):
    return {
        "status": "ok",
        "method": method,
        "backward": False,
        "operation": "pmf",
        "accuracy_passed": True,
        "validate_args": False,
        "runtime": {"device": "cpu"},
        "dtype": "float64",
        "n": 128,
        "batch_size": 1,
        "median_seconds": 0.01 if method == "dp" else 0.005,
        "iqr_seconds": 0.0001,
        "max_abs_error": 1e-14,
    }


def test_profile_qualification():
    dp, tree = record("dp"), record("fft_tree")
    profiles = qualified_profiles([dp, tree])["profiles"]
    assert len(profiles) == 1 and profiles[0]["speedup"] == 2
    assert profiles[0]["dtype"] == "torch.float64"
    for update in [
        {"accuracy_passed": False},
        {"median_seconds": 0.009},
        {"iqr_seconds": 0.01},
        {"runtime": {"device": "other"}},
        {"backward": True},
        {"operation": "cdf"},
        {"validate_args": True},
        {"status": "skipped"},
    ]:
        changed = {**copy.deepcopy(tree), **update}
        assert not qualified_profiles([dp, changed])["profiles"]
    assert not qualified_profiles([{**dp, "accuracy_passed": False}, tree])["profiles"]


def test_workspace_budget_is_conservative_for_backward():
    case = {
        "batch_size": 64,
        "n": 8192,
        "dtype": "float64",
        "method": "dp",
        "backward": False,
    }
    assert workspace_estimate(case) < 256 * 1024**2
    assert workspace_estimate({**case, "backward": True}) > 256 * 1024**2
    assert workspace_estimate({**case, "method": "fft_cf"}) > 256 * 1024**2


def test_training_budget_allows_useful_comparisons():
    case = {
        "batch_size": 1,
        "n": 2048,
        "dtype": "float64",
        "method": "dp",
        "backward": True,
    }
    assert workspace_estimate(case) < 256 * 1024**2
    assert workspace_estimate({**case, "batch_size": 16}) > 256 * 1024**2
    for block in [8, 16, 32, 64, 128]:
        assert (
            workspace_estimate(
                {**case, "method": "fft_hybrid", "batch_size": 16, "block_size": block}
            )
            < 256 * 1024**2
        )


def test_log_tail_reference_retains_underflow():
    import math

    from benchmarks.log_tails import reference

    probability, gradient = reference(2000, 0.5, 1999, True)
    assert abs(probability + 2000 * math.log(2)) < 1e-10
    assert gradient == 2
