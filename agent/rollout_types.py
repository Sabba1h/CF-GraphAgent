"""Dataclasses for the minimal single-environment rollout layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.episode_result import CounterfactualComparisonResult, RewardBreakdown
from core.experiment_config import RewardMode


@dataclass(slots=True)
class ParsedGraphAction:
    """Parsed candidate-id action."""

    candidate_id: int
    raw_action: Any


@dataclass(slots=True)
class RolloutStep:
    """One synchronous rollout step."""

    observation: dict[str, Any]
    action: Any
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)
    reward_mode: RewardMode = "baseline"
    base_reward: float | None = None
    reward_breakdown: RewardBreakdown | None = None
    counterfactual_reward: float = 0.0
    counterfactual_comparison: CounterfactualComparisonResult | None = None


@dataclass(slots=True)
class RolloutResult:
    """Result of a single-environment rollout."""

    steps: list[RolloutStep] = field(default_factory=list)
    reward_mode: RewardMode = "baseline"

    @property
    def total_reward(self) -> float:
        """Return accumulated rollout reward."""

        return sum(step.reward for step in self.steps)

    @property
    def base_total_reward(self) -> float:
        """Return accumulated environment reward before optional oracle deltas."""

        return sum(step.base_reward if step.base_reward is not None else step.reward for step in self.steps)
