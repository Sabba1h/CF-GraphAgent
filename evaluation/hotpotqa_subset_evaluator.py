"""Subset evaluator for HotpotQA graph-backed experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

from core.experiment_config import RewardMode
from core.experiment_result import ExperimentResult
from data.benchmarks.hotpotqa import load_hotpotqa
from evaluation.hotpotqa_metrics import (
    exact_match,
    normalize_answer_source_type,
    summarize_hotpotqa_records,
    token_f1,
)
from graph.hotpotqa_graph_runtime import run_hotpotqa_graph_experiment_for_example

HotpotQAEvalMode = Literal["baseline", "oracle_counterfactual", "both"]


@dataclass(slots=True)
class HotpotQAGraphEvalRecord:
    """One graph-backed HotpotQA evaluation record."""

    question_id: str
    graph_id: str
    reward_mode: RewardMode
    gold_answer: str
    raw_graph_answer: str | None
    selected_graph_answer: str | None
    projected_answer: str
    normalized_projected_answer: str
    projected_eval_score: float
    exact_match: float
    f1: float
    answer_source_type: str
    base_total_reward: float
    total_reward: float
    step_count: int
    graph_node_count: int
    graph_edge_count: int
    policy_name: str = "baseline"
    answer_selector_name: str = "raw_final_node"
    answer_extractor_name: str = "full_sentence"
    answer_mapper_name: str = "identity"
    candidate_generator_name: str = "baseline_generator"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""

        return asdict(self)


@dataclass(slots=True)
class HotpotQASubsetEvalResult:
    """Evaluation output for a HotpotQA graph-backed subset."""

    records: list[HotpotQAGraphEvalRecord]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""

        return {
            "records": [record.to_dict() for record in self.records],
            "summary": self.summary,
        }


def evaluate_hotpotqa_graph_subset(
    *,
    path: str | Path,
    split: str | None = None,
    limit: int | None = None,
    indices: Iterable[int] | None = None,
    reward_mode: HotpotQAEvalMode = "baseline",
    max_steps: int = 4,
    candidate_top_k: int = 5,
    min_expand_steps: int = 1,
    policy_factory: Callable[[], Any] | None = None,
    policy_name: str = "baseline",
    answer_selector_factory: Callable[[], Any] | None = None,
    answer_selector_name: str = "raw_final_node",
    answer_extractor_factory: Callable[[], Any] | None = None,
    answer_extractor_name: str = "full_sentence",
    answer_mapper_factory: Callable[[], Any] | None = None,
    answer_mapper_name: str = "identity",
    yesno_mapper_factory: Callable[[], Any] | None = None,
    yesno_mapper_name: str | None = None,
    relation_span_mapper_factory: Callable[[], Any] | None = None,
    relation_span_mapper_name: str | None = None,
    relation_span_discovery_factory: Callable[[], Any] | None = None,
    relation_span_discovery_name: str | None = None,
    relation_span_ranker_factory: Callable[[], Any] | None = None,
    relation_span_ranker_name: str | None = None,
    relation_span_proposal_factory: Callable[[], Any] | None = None,
    relation_span_proposal_name: str | None = None,
    candidate_generator_factory: Callable[[], Any] | None = None,
    candidate_generator_name: str = "baseline_generator",
) -> HotpotQASubsetEvalResult:
    """Run graph-backed experiments and aggregate HotpotQA subset metrics."""

    examples = load_hotpotqa(path, split=split, limit=limit, indices=indices)
    modes: list[RewardMode] = (
        ["baseline", "oracle_counterfactual"] if reward_mode == "both" else [reward_mode]
    )
    records: list[HotpotQAGraphEvalRecord] = []
    for mode in modes:
        for example in examples:
            _, experiment_result = run_hotpotqa_graph_experiment_for_example(
                example,
                reward_mode=mode,
                max_steps=max_steps,
                candidate_top_k=candidate_top_k,
                min_expand_steps=min_expand_steps,
                policy=policy_factory() if policy_factory is not None else None,
                policy_name=policy_name,
                answer_selector=answer_selector_factory() if answer_selector_factory is not None else None,
                answer_selector_name=answer_selector_name,
                answer_extractor=answer_extractor_factory() if answer_extractor_factory is not None else None,
                answer_extractor_name=answer_extractor_name,
                answer_mapper=answer_mapper_factory() if answer_mapper_factory is not None else None,
                answer_mapper_name=answer_mapper_name,
                yesno_mapper=yesno_mapper_factory() if yesno_mapper_factory is not None else None,
                yesno_mapper_name=yesno_mapper_name,
                relation_span_mapper=relation_span_mapper_factory() if relation_span_mapper_factory is not None else None,
                relation_span_mapper_name=relation_span_mapper_name,
                relation_span_discovery=(
                    relation_span_discovery_factory() if relation_span_discovery_factory is not None else None
                ),
                relation_span_discovery_name=relation_span_discovery_name,
                relation_span_ranker=relation_span_ranker_factory() if relation_span_ranker_factory is not None else None,
                relation_span_ranker_name=relation_span_ranker_name,
                relation_span_proposal=(
                    relation_span_proposal_factory() if relation_span_proposal_factory is not None else None
                ),
                relation_span_proposal_name=relation_span_proposal_name,
                candidate_generator=candidate_generator_factory() if candidate_generator_factory is not None else None,
                candidate_generator_name=candidate_generator_name,
            )
            records.append(record_from_experiment_result(experiment_result))

    summary = summarize_hotpotqa_eval_records(records)
    return HotpotQASubsetEvalResult(records=records, summary=summary)


def record_from_experiment_result(result: ExperimentResult) -> HotpotQAGraphEvalRecord:
    """Convert one graph-backed ExperimentResult into an eval record."""

    metadata = result.metadata
    graph_stats = metadata.get("graph_stats") or {}
    gold_answer = str(metadata.get("gold_answer") or "")
    projected_answer = str(metadata.get("projected_answer") or "")
    normalized_projected_answer = str(metadata.get("normalized_projected_answer") or "")
    projected_eval_score = float(metadata.get("projected_eval_score") or 0.0)
    answer_source_type = _extract_answer_source_type(metadata)
    has_nonzero_oracle_delta = _has_nonzero_oracle_delta(result)
    path_summary = _build_path_summary(result)
    record_metadata = {
        "dataset_name": metadata.get("dataset_name"),
        "question": metadata.get("query") or result.config.metadata.get("query"),
        "policy_name": metadata.get("policy_name", "baseline"),
        "answer_selector_name": metadata.get("answer_selector_name", "raw_final_node"),
        "answer_extractor_name": metadata.get("answer_extractor_name", "full_sentence"),
        "answer_mapper_name": metadata.get("answer_mapper_name", "identity"),
        "base_answer_mapper_name": metadata.get("base_answer_mapper_name"),
        "yesno_mapper_name": metadata.get("yesno_mapper_name"),
        "relation_span_mapper_name": metadata.get("relation_span_mapper_name"),
        "relation_span_discovery_name": metadata.get("relation_span_discovery_name"),
        "relation_span_ranker_name": metadata.get("relation_span_ranker_name"),
        "relation_span_proposal_name": metadata.get("relation_span_proposal_name"),
        "answer_type_label": metadata.get("answer_type_label"),
        "yesno_mapper_applied": metadata.get("yesno_mapper_applied", False),
        "relation_span_mapper_applied": metadata.get("relation_span_mapper_applied", False),
        "relation_span_discovery_applied": metadata.get("relation_span_discovery_applied", False),
        "relation_span_ranker_applied": metadata.get("relation_span_ranker_applied", False),
        "relation_span_proposal_applied": metadata.get("relation_span_proposal_applied", False),
        "candidate_generator_name": metadata.get("candidate_generator_name", "baseline_generator"),
        "normalized_gold_answer": metadata.get("normalized_gold_answer"),
        "final_answer": result.final_answer,
        "graph_stats": graph_stats,
        "graph_sentence_nodes": metadata.get("graph_sentence_nodes", []),
        "selected_graph_answer": metadata.get("selected_graph_answer"),
        "answer_selection": metadata.get("answer_selection"),
        "answer_mapping": metadata.get("answer_mapping"),
        "yesno_mapping": metadata.get("yesno_mapping"),
        "relation_span_mapping": metadata.get("relation_span_mapping"),
        "relation_span_discovery": metadata.get("relation_span_discovery"),
        "answer_extraction": metadata.get("answer_extraction"),
        "answer_projection": metadata.get("answer_projection"),
        "reward_summaries": result.reward_summaries,
        "counterfactual_summaries": result.counterfactual_summaries,
        "merged_counterfactual_reward_sum": metadata.get("merged_counterfactual_reward_sum", 0.0),
        "has_nonzero_oracle_delta": has_nonzero_oracle_delta,
        "path_summary": path_summary,
        "candidate_stage_summaries": _candidate_stage_summaries(result),
    }
    return HotpotQAGraphEvalRecord(
        question_id=str(metadata.get("question_id") or ""),
        graph_id=str(metadata.get("graph_id") or ""),
        reward_mode=result.config.reward_mode,
        gold_answer=gold_answer,
        raw_graph_answer=metadata.get("raw_graph_answer"),
        selected_graph_answer=metadata.get("selected_graph_answer"),
        projected_answer=projected_answer,
        normalized_projected_answer=normalized_projected_answer,
        projected_eval_score=projected_eval_score,
        exact_match=exact_match(projected_answer, gold_answer),
        f1=token_f1(projected_answer, gold_answer),
        answer_source_type=answer_source_type,
        base_total_reward=float(result.base_total_reward),
        total_reward=float(result.total_reward),
        step_count=len(result.step_traces),
        graph_node_count=int(graph_stats.get("node_count", 0)),
        graph_edge_count=int(graph_stats.get("edge_count", 0)),
        policy_name=str(metadata.get("policy_name", "baseline")),
        answer_selector_name=str(metadata.get("answer_selector_name", "raw_final_node")),
        answer_extractor_name=str(metadata.get("answer_extractor_name", "full_sentence")),
        answer_mapper_name=str(metadata.get("answer_mapper_name", "identity")),
        candidate_generator_name=str(metadata.get("candidate_generator_name", "baseline_generator")),
        metadata=record_metadata,
    )


def summarize_hotpotqa_eval_records(records: list[HotpotQAGraphEvalRecord]) -> dict[str, Any]:
    """Build overall and per-mode summary metrics."""

    by_mode: dict[str, dict[str, Any]] = {}
    for mode in sorted({record.reward_mode for record in records}):
        by_mode[mode] = summarize_hotpotqa_records([record for record in records if record.reward_mode == mode])
    summary = summarize_hotpotqa_records(records)
    summary["by_reward_mode"] = by_mode
    return summary


def save_hotpotqa_eval_outputs(result: HotpotQASubsetEvalResult, output_dir: str | Path) -> tuple[Path, Path]:
    """Save records JSONL and summary JSON files."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    records_path = output_path / "hotpotqa_graph_eval_records.jsonl"
    summary_path = output_path / "hotpotqa_graph_eval_summary.json"
    with records_path.open("w", encoding="utf-8") as handle:
        for record in result.records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    summary_path.write_text(
        json.dumps(result.summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return records_path, summary_path


def _extract_answer_source_type(metadata: dict[str, Any]) -> str:
    alignment = metadata.get("answer_projection") or {}
    source = None
    if isinstance(alignment, dict):
        alignment_metadata = alignment.get("metadata") or {}
        projection = alignment.get("projection") or {}
        source = alignment_metadata.get("projection_source") or projection.get("projection_source")
        if source is None and _has_projection_fallback(alignment):
            source = "fallback"
    return normalize_answer_source_type(source)


def _has_projection_fallback(alignment: dict[str, Any]) -> bool:
    projection = alignment.get("projection") or {}
    projection_metadata = projection.get("metadata") or {}
    alignment_metadata = alignment.get("metadata") or {}
    return bool(
        projection_metadata.get("projection_fallback_reason")
        or alignment_metadata.get("projection_fallback_reason")
    )


def _has_nonzero_oracle_delta(result: ExperimentResult) -> bool:
    if result.config.reward_mode != "oracle_counterfactual":
        return False
    return any(abs(float(trace.counterfactual_reward)) > 1e-12 for trace in result.step_traces)


def _build_path_summary(result: ExperimentResult) -> dict[str, Any]:
    """Build a lightweight action-path summary from recorded step traces."""

    action_types: list[str] = []
    selected_actions: list[dict[str, Any]] = []
    for trace in result.step_traces:
        action_type = _trace_action_type(trace)
        action_types.append(action_type)
        selected_actions.append(
            {
                "step_idx": trace.step_idx,
                "candidate_id": trace.action,
                "action_type": action_type,
                "candidate_stage_summary": _trace_candidate_stage_summary(trace),
            }
        )

    expand_count = sum(1 for action_type in action_types if action_type == "EXPAND_EDGE")
    answer_step_idx = _first_index(action_types, "ANSWER")
    stop_step_idx = _first_index(action_types, "STOP")
    expand_count_before_answer = (
        sum(1 for action_type in action_types[:answer_step_idx] if action_type == "EXPAND_EDGE")
        if answer_step_idx is not None
        else expand_count
    )
    return {
        "step_count": len(result.step_traces),
        "selected_action_types": action_types,
        "selected_actions": selected_actions,
        "expand_count": expand_count,
        "answer_step_idx": answer_step_idx,
        "stop_step_idx": stop_step_idx,
        "expand_count_before_answer": expand_count_before_answer,
    }


def _candidate_stage_summaries(result: ExperimentResult) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for trace in result.step_traces:
        summary = _trace_candidate_stage_summary(trace)
        if isinstance(summary, dict):
            summaries.append(summary)
    return summaries


def _trace_candidate_stage_summary(trace: Any) -> dict[str, Any] | None:
    observation = trace.metadata.get("observation", {}) if trace.metadata else {}
    candidates = observation.get("candidate_actions", []) if isinstance(observation, dict) else []
    trace_candidate_id = _trace_candidate_id(trace.action)
    for candidate in candidates:
        if int(candidate.get("candidate_id", -1)) != trace_candidate_id:
            continue
        metadata = candidate.get("metadata") or {}
        summary = metadata.get("candidate_stage_summary") if isinstance(metadata, dict) else None
        if isinstance(summary, dict):
            return summary
    return None


def _trace_action_type(trace: Any) -> str:
    observation = trace.metadata.get("observation", {}) if trace.metadata else {}
    candidates = observation.get("candidate_actions", []) if isinstance(observation, dict) else []
    trace_candidate_id = _trace_candidate_id(trace.action)
    for candidate in candidates:
        if int(candidate.get("candidate_id", -1)) == trace_candidate_id:
            return str(candidate.get("action_type", "UNKNOWN"))

    info = trace.metadata.get("info", {}) if trace.metadata else {}
    trajectory = info.get("trajectory", {}) if isinstance(info, dict) else {}
    steps = trajectory.get("steps", []) if isinstance(trajectory, dict) else []
    if steps:
        selected_action = steps[-1].get("selected_action") or {}
        return str(selected_action.get("action_type", "UNKNOWN"))
    return "UNKNOWN"


def _trace_candidate_id(action: Any) -> int:
    if isinstance(action, dict):
        return int(action.get("candidate_id", -1))
    return int(action)


def _first_index(items: list[str], value: str) -> int | None:
    for index, item in enumerate(items):
        if item == value:
            return index
    return None
