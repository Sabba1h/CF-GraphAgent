"""Dataclasses for the minimal single-environment rollout layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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


@dataclass(slots=True)
class RolloutResult:
    """Result of a single-environment rollout."""

    steps: list[RolloutStep] = field(default_factory=list)

    @property
    def total_reward(self) -> float:
        """Return accumulated rollout reward."""

        return sum(step.reward for step in self.steps)
