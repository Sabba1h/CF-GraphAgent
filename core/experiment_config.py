"""Experiment configuration for rollout and adapter layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


RewardMode = Literal["baseline", "oracle_counterfactual"]
CounterfactualMode = Literal["remove", "replace", "null", "mixed"]


@dataclass(slots=True)
class ExperimentConfig:
    """Unified experiment-layer config for rollout and adapter paths."""

    reward_mode: RewardMode = "baseline"
    counterfactual_mode: CounterfactualMode = "mixed"
    use_counterfactual_merge: bool = False
    max_steps: int | None = None
    record_step_traces: bool = True
    record_counterfactual_metadata: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.reward_mode not in {"baseline", "oracle_counterfactual"}:
            raise ValueError("reward_mode must be either 'baseline' or 'oracle_counterfactual'.")
        if self.counterfactual_mode not in {"remove", "replace", "null", "mixed"}:
            raise ValueError("counterfactual_mode must be one of {'remove', 'replace', 'null', 'mixed'}.")
        if self.max_steps is not None and self.max_steps < 0:
            raise ValueError("max_steps must be None or a non-negative integer.")

    @property
    def resolved_counterfactual_mode(self) -> Literal["remove", "replace", "null"]:
        """Return the concrete counterfactual mode used by this stage.

        Stage-8 keeps "mixed" as a config placeholder. It is deliberately
        mapped to "remove" for now instead of introducing strategy selection.
        """

        if self.counterfactual_mode == "mixed":
            return "remove"
        return self.counterfactual_mode
