"""Minimal HotpotQA graph-backed evaluation metrics."""

from __future__ import annotations

from collections import Counter
from typing import Any

from data.benchmarks.answer_normalization import normalize_answer

ANSWER_SOURCE_TYPES = {"title", "sentence", "fallback", "unknown"}


def exact_match(prediction: str | None, gold: str | None) -> float:
    """Return exact match over normalized answer strings."""

    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def token_f1(prediction: str | None, gold: str | None) -> float:
    """Return a transparent token-level F1 score."""

    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    overlap = sum((pred_counts & gold_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def normalize_answer_source_type(source_type: object | None) -> str:
    """Map raw projection source strings into a fixed enum."""

    value = "" if source_type is None else str(source_type)
    if value in {"title", "sentence"}:
        return value
    if value == "fallback":
        return "fallback"
    return "unknown"


def summarize_hotpotqa_records(records: list[Any]) -> dict[str, Any]:
    """Aggregate subset-level metrics from per-example eval records."""

    count = len(records)
    if count == 0:
        return {
            "sample_count": 0,
            "avg_exact_match": 0.0,
            "avg_f1": 0.0,
            "avg_projected_eval_score": 0.0,
            "avg_base_total_reward": 0.0,
            "avg_total_reward": 0.0,
            "avg_step_count": 0.0,
            "avg_graph_node_count": 0.0,
            "avg_graph_edge_count": 0.0,
            "answer_source_type_distribution": {source_type: 0 for source_type in sorted(ANSWER_SOURCE_TYPES)},
            "nonzero_oracle_delta_count": 0,
            "reward_modes": [],
        }

    source_counts = Counter(record.answer_source_type for record in records)
    return {
        "sample_count": count,
        "avg_exact_match": _average(record.exact_match for record in records),
        "avg_f1": _average(record.f1 for record in records),
        "avg_projected_eval_score": _average(record.projected_eval_score for record in records),
        "avg_base_total_reward": _average(record.base_total_reward for record in records),
        "avg_total_reward": _average(record.total_reward for record in records),
        "avg_step_count": _average(record.step_count for record in records),
        "avg_graph_node_count": _average(record.graph_node_count for record in records),
        "avg_graph_edge_count": _average(record.graph_edge_count for record in records),
        "answer_source_type_distribution": {
            source_type: source_counts.get(source_type, 0) for source_type in sorted(ANSWER_SOURCE_TYPES)
        },
        "nonzero_oracle_delta_count": sum(1 for record in records if record.metadata.get("has_nonzero_oracle_delta")),
        "reward_modes": sorted({record.reward_mode for record in records}),
    }


def _average(values: Any) -> float:
    items = [float(value) for value in values]
    if not items:
        return 0.0
    return sum(items) / len(items)
