"""Dataset examples for future proxy reward training."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.experiment_result import ExperimentResult


@dataclass(slots=True)
class CounterfactualDatasetExample:
    """One JSON-serializable supervised example from a counterfactual rollout."""

    task_id: str
    query: str
    step_idx: int
    reward_mode: str
    counterfactual_mode: str
    working_subgraph_summary: dict[str, Any]
    candidate_actions_summary: list[dict[str, Any]]
    action: Any
    action_type: str | None
    original_score: float
    counterfactual_score: float
    score_delta: float
    base_reward: float
    counterfactual_reward: float
    final_reward: float
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "v1"
    history_summary: list[dict[str, Any]] = field(default_factory=list)
    final_answer: str | None = None
    counterfactual_answer: str | None = None
    trajectory_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a plain JSON-serializable dictionary."""

        return asdict(self)


def examples_from_experiment_result(
    *,
    result: ExperimentResult,
    task_id: str,
    trajectory_id: str | None = None,
) -> list[CounterfactualDatasetExample]:
    """Convert an ExperimentResult into proxy-reward dataset examples."""

    examples: list[CounterfactualDatasetExample] = []
    for trace in result.step_traces:
        observation = trace.metadata.get("observation") or {}
        comparison = trace.counterfactual_comparison
        selected_action = _selected_action(observation=observation, action=trace.action)
        examples.append(
            CounterfactualDatasetExample(
                task_id=task_id,
                trajectory_id=trajectory_id,
                query=str(observation.get("query", result.metadata.get("query", ""))),
                step_idx=trace.step_idx,
                reward_mode=trace.reward_mode,
                counterfactual_mode=(
                    str(comparison.metadata.get("resolved_counterfactual_mode", comparison.mode))
                    if comparison is not None
                    else result.config.counterfactual_mode
                ),
                working_subgraph_summary=dict(observation.get("working_subgraph_summary", {})),
                candidate_actions_summary=list(observation.get("candidate_actions", [])),
                history_summary=list(observation.get("history_summary", [])),
                action=trace.action,
                action_type=selected_action.get("action_type"),
                original_score=comparison.original_score if comparison is not None else 0.0,
                counterfactual_score=comparison.counterfactual_score if comparison is not None else 0.0,
                score_delta=comparison.score_delta if comparison is not None else 0.0,
                base_reward=trace.base_reward,
                counterfactual_reward=trace.counterfactual_reward,
                final_reward=trace.reward,
                final_answer=result.final_answer,
                counterfactual_answer=(
                    comparison.metadata.get("counterfactual_final_answer") if comparison is not None else None
                ),
                metadata={
                    "selected_action": selected_action,
                    "experiment_total_reward": result.total_reward,
                    "experiment_base_total_reward": result.base_total_reward,
                    "comparison": comparison.to_dict() if comparison is not None else None,
                },
            )
        )
    return examples


def write_jsonl(examples: list[CounterfactualDatasetExample], output_path: str | Path) -> Path:
    """Write dataset examples as JSONL and return the output path."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict(), sort_keys=True) + "\n")
    return path


def _selected_action(*, observation: dict[str, Any], action: Any) -> dict[str, Any]:
    for candidate in observation.get("candidate_actions", []):
        if candidate.get("candidate_id") == action:
            return dict(candidate)
    return {"candidate_id": action}
