"""Reward bridge for verl-facing code."""

from __future__ import annotations

from dataclasses import dataclass

from core.episode_result import RewardBreakdown


@dataclass(slots=True)
class VerlRewardOutput:
    """Scalar reward plus logging metrics for adapter consumers."""

    reward: float
    metrics: dict[str, float]


class VerlRewardBridge:
    """Convert structured RewardBreakdown into scalar reward and metrics."""

    def to_scalar(self, breakdown: RewardBreakdown) -> float:
        """Return the unchanged scalar reward value."""

        return breakdown.total_reward

    def to_metrics(self, breakdown: RewardBreakdown) -> dict[str, float]:
        """Return reward components as flat metrics."""

        return {
            "task_reward": breakdown.task_reward,
            "process_reward": breakdown.process_reward,
            "constraint_penalty": breakdown.constraint_penalty,
            "counterfactual_reward": breakdown.counterfactual_reward,
            "total_reward": breakdown.total_reward,
        }

    def bridge(self, breakdown: RewardBreakdown) -> VerlRewardOutput:
        """Return scalar reward and metrics for one transition."""

        return VerlRewardOutput(reward=self.to_scalar(breakdown), metrics=self.to_metrics(breakdown))
