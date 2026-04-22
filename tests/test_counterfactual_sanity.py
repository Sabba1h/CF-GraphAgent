"""Tests for oracle counterfactual sanity helpers."""

from scripts.run_counterfactual_sanity import run_sanity, summarize_deltas


def test_summarize_deltas_reports_min_max_mean_and_nonzero_count() -> None:
    stats = summarize_deltas([0.0, 1.0, 2.0])

    assert stats["count"] == 3
    assert stats["nonzero_count"] == 2
    assert stats["min"] == 0.0
    assert stats["max"] == 2.0
    assert stats["mean"] == 1.0


def test_counterfactual_sanity_runs_toy_modes_with_stepwise_delta() -> None:
    results = run_sanity(counterfactual_modes=["remove", "replace", "null"])

    assert results["baseline"]["final_answer"] == "Chicago"
    assert set(results["modes"].keys()) == {"remove", "replace", "null"}
    assert any(payload["stats"]["nonzero_count"] > 0 for payload in results["modes"].values())
    for payload in results["modes"].values():
        assert payload["steps"]
        assert {"original_score", "counterfactual_score", "score_delta", "counterfactual_reward"} <= set(
            payload["steps"][0].keys()
        )
