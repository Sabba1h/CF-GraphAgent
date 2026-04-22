"""Counterfactual reward delta utilities."""

from __future__ import annotations

from dataclasses import dataclass

from core.episode_result import CounterfactualComparisonResult


@dataclass(slots=True)
class CounterfactualRewardResult:
    """Counterfactual reward scalar plus metrics."""

    counterfactual_reward: float
    metrics: dict[str, float | str | int]


class CounterfactualRewardEngine:
    """Compute counterfactual reward from comparison results."""

    def __init__(self, *, clamp_range: tuple[float, float] | None = None) -> None:
        self.clamp_range = clamp_range

    def compute(self, comparison: CounterfactualComparisonResult) -> CounterfactualRewardResult:
        """Return score_delta as the first counterfactual reward rule."""

        reward = comparison.score_delta
        if self.clamp_range is not None:
            lower, upper = self.clamp_range
            reward = min(max(reward, lower), upper)
        return CounterfactualRewardResult(
            counterfactual_reward=reward,
            metrics={
                "mode": comparison.mode,
                "step_idx": comparison.step_idx,
                "original_score": comparison.original_score,
                "counterfactual_score": comparison.counterfactual_score,
                "score_delta": comparison.score_delta,
                "counterfactual_reward": reward,
            },
        )
