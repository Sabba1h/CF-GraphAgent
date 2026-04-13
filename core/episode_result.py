"""Episode, evaluation, and reward result schemas."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from core.actions import CandidateAction


@dataclass(slots=True)
class EvalResult:
    """Evaluator output kept separate from answer generation."""

    score: float
    is_correct: bool | None
    reason: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary copy."""

        return asdict(self)


@dataclass(slots=True)
class RewardBreakdown:
    """Structured reward decomposition for stage-1 and later extensions."""

    task_reward: float = 0.0
    process_reward: float = 0.0
    constraint_penalty: float = 0.0
    counterfactual_reward: float = 0.0
    total_reward: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary copy."""

        return asdict(self)


@dataclass(slots=True)
class AnswerResult:
    """Answer payload returned by AnswerEngine.

    The reward/is_correct fields are retained for compatibility with the
    original stage-1 API. New code should use EvalResult for scoring.
    """

    answer: str
    evidence_edge_ids: list[str] = field(default_factory=list)
    reasoning: str = ""
    reward: float = 0.0
    is_correct: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary copy."""

        return asdict(self)


@dataclass(slots=True)
class RewardResult:
    """Reward payload returned by RewardEngine."""

    reward: float
    reason: str
    counterfactual_bonus: float = 0.0
    breakdown: RewardBreakdown | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary copy."""

        return asdict(self)


@dataclass(slots=True)
class TransitionResult:
    """Result returned by TransitionEngine.step."""

    candidate_id: int
    candidate_actions: list[dict[str, Any]]
    selected_action: dict[str, Any] | None
    reward_result: RewardResult
    terminated: bool
    truncated: bool
    info: dict[str, Any]
    final_answer: str | None = None
    termination_reason: str | None = None


@dataclass(slots=True)
class StepRecord:
    """Step-level trajectory log entry."""

    step_index: int
    candidate_actions: list[dict[str, Any]]
    selected_action: dict[str, Any] | None
    reward: float
    reward_reason: str
    working_subgraph_summary: dict[str, Any]
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary copy."""

        return asdict(self)


@dataclass(slots=True)
class EpisodeSummary:
    """Episode-level trajectory summary."""

    query: str
    ground_truth: str | None = None
    steps: list[StepRecord] = field(default_factory=list)
    termination_reason: str | None = None
    final_answer: str | None = None
    final_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary copy."""

        return {
            "query": self.query,
            "ground_truth": self.ground_truth,
            "steps": [step.to_dict() for step in self.steps],
            "termination_reason": self.termination_reason,
            "final_answer": self.final_answer,
            "final_score": self.final_score,
        }
