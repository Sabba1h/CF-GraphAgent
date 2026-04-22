"""Unified experiment result schemas."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from core.episode_result import CounterfactualComparisonResult, EvalResult, RewardBreakdown
from core.experiment_config import ExperimentConfig, RewardMode


@dataclass(slots=True)
class ExperimentStepTrace:
    """One step in a rollout experiment trace."""

    step_idx: int
    action: Any
    reward_mode: RewardMode
    base_reward: float
    reward: float
    reward_breakdown: RewardBreakdown | None
    counterfactual_reward: float = 0.0
    counterfactual_comparison: CounterfactualComparisonResult | None = None
    terminated: bool = False
    truncated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary copy."""

        return asdict(self)


@dataclass(slots=True)
class ExperimentResult:
    """Unified result format for baseline and oracle rollout experiments."""

    config: ExperimentConfig
    final_answer: str | None = None
    final_eval: EvalResult | None = None
    total_reward: float = 0.0
    base_total_reward: float = 0.0
    step_traces: list[ExperimentStepTrace] = field(default_factory=list)
    reward_summaries: list[dict[str, Any]] = field(default_factory=list)
    counterfactual_summaries: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary copy."""

        return asdict(self)
