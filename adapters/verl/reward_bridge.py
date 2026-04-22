"""Reward bridge for verl-facing code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.rollout_types import RewardMode
from core.episode_result import CounterfactualComparisonResult, RewardBreakdown
from core.experiment_config import ExperimentConfig


@dataclass(slots=True)
class VerlRewardOutput:
    """Scalar reward plus logging metrics for adapter consumers."""

    reward: float
    metrics: dict[str, Any]


class VerlRewardBridge:
    """Convert structured RewardBreakdown into scalar reward and metrics."""

    def to_scalar(self, breakdown: RewardBreakdown, *, reward_mode: RewardMode = "baseline") -> float:
        """Return the unchanged scalar reward value."""

        self._validate_reward_mode(reward_mode)
        return breakdown.total_reward

    def to_metrics(
        self,
        breakdown: RewardBreakdown,
        *,
        reward_mode: RewardMode = "baseline",
        comparison: CounterfactualComparisonResult | None = None,
        config: ExperimentConfig | None = None,
    ) -> dict[str, Any]:
        """Return reward components as flat metrics."""

        if config is not None:
            reward_mode = config.reward_mode
        self._validate_reward_mode(reward_mode)
        metrics: dict[str, Any] = {
            "reward_mode": reward_mode,
            "task_reward": breakdown.task_reward,
            "process_reward": breakdown.process_reward,
            "constraint_penalty": breakdown.constraint_penalty,
            "counterfactual_reward": breakdown.counterfactual_reward,
            "total_reward": breakdown.total_reward,
        }
        if config is not None:
            metrics.update(
                {
                    "counterfactual_mode": config.counterfactual_mode,
                    "use_counterfactual_merge": config.use_counterfactual_merge,
                }
            )
        if reward_mode == "oracle_counterfactual" and comparison is not None:
            metrics.update(
                {
                    "resolved_counterfactual_mode": comparison.mode,
                    "counterfactual_step_idx": comparison.step_idx,
                    "original_score": comparison.original_score,
                    "counterfactual_score": comparison.counterfactual_score,
                    "score_delta": comparison.score_delta,
                }
            )
        return metrics

    def bridge(
        self,
        breakdown: RewardBreakdown,
        *,
        reward_mode: RewardMode = "baseline",
        comparison: CounterfactualComparisonResult | None = None,
        config: ExperimentConfig | None = None,
    ) -> VerlRewardOutput:
        """Return scalar reward and metrics for one transition."""

        if config is not None:
            reward_mode = config.reward_mode
        return VerlRewardOutput(
            reward=self.to_scalar(breakdown, reward_mode=reward_mode),
            metrics=self.to_metrics(breakdown, reward_mode=reward_mode, comparison=comparison, config=config),
        )

    def _validate_reward_mode(self, reward_mode: RewardMode) -> None:
        if reward_mode not in {"baseline", "oracle_counterfactual"}:
            raise ValueError("reward_mode must be either 'baseline' or 'oracle_counterfactual'.")
