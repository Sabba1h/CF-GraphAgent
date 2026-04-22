"""Sentence-hit diagnostics for HotpotQA graph-backed results.

Gold answers are used here only for offline diagnostics. This module must not
feed any gold-derived signal back into policy or selector decisions.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from data.benchmarks.answer_normalization import normalize_answer
from evaluation.hotpotqa_subset_evaluator import HotpotQAGraphEvalRecord


@dataclass(slots=True)
class HotpotQASentenceHitRecord:
    """One sentence-hit diagnostic record for a HotpotQA graph run."""

    question_id: str
    graph_id: str
    policy_name: str
    answer_selector_name: str
    answer_extractor_name: str
    reward_mode: str
    gold_answer: str
    gold_answer_type: str
    raw_graph_answer: str | None
    selected_graph_answer: str | None
    projected_answer: str
    exact_match: float
    f1: float
    projected_eval_score: float
    touched_sentence: bool
    touched_sentence_count: int
    first_sentence_step_idx: int | None
    any_touched_sentence_contains_gold: bool | None
    selected_sentence_contains_gold: bool | None
    gold_in_any_sentence: bool | None
    diagnostic_bucket: str
    touched_sentences: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""

        return asdict(self)


@dataclass(slots=True)
class HotpotQASentenceHitDiagnosticResult:
    """Sentence-hit diagnostics for a HotpotQA subset."""

    records: list[HotpotQASentenceHitRecord]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""

        return {
            "records": [record.to_dict() for record in self.records],
            "summary": self.summary,
        }


def analyze_hotpotqa_sentence_hits(records: list[HotpotQAGraphEvalRecord]) -> HotpotQASentenceHitDiagnosticResult:
    """Build sentence-hit diagnostic records and aggregate summaries."""

    diagnostic_records = [sentence_hit_record_from_eval_record(record) for record in records]
    return HotpotQASentenceHitDiagnosticResult(
        records=diagnostic_records,
        summary=summarize_sentence_hit_records(diagnostic_records),
    )


def sentence_hit_record_from_eval_record(record: HotpotQAGraphEvalRecord) -> HotpotQASentenceHitRecord:
    """Convert one eval record into a sentence-hit diagnostic record."""

    answer_selection = record.metadata.get("answer_selection") or {}
    candidate_nodes = answer_selection.get("candidate_nodes") if isinstance(answer_selection, dict) else []
    if not isinstance(candidate_nodes, list):
        candidate_nodes = []

    path_nodes = [node for node in candidate_nodes if isinstance(node, dict) and node.get("source") == "path"]
    touched_sentences = [node for node in path_nodes if node.get("node_type") == "sentence"]
    touched_sentence_count = len(touched_sentences)
    first_sentence_step_idx = _first_sentence_step_idx(touched_sentences)

    graph_sentence_nodes = record.metadata.get("graph_sentence_nodes") or []
    if not isinstance(graph_sentence_nodes, list):
        graph_sentence_nodes = []

    gold_answer_type = classify_gold_answer(record.gold_answer)
    any_touched_contains = _any_contains_gold(touched_sentences, record.gold_answer, gold_answer_type)
    selected_contains = _selected_sentence_contains_gold(record, candidate_nodes, gold_answer_type)
    graph_contains = _any_contains_gold(graph_sentence_nodes, record.gold_answer, gold_answer_type)
    diagnostic_bucket = classify_sentence_hit_bucket(
        touched_sentence=bool(touched_sentences),
        gold_answer_type=gold_answer_type,
        any_touched_sentence_contains_gold=any_touched_contains,
        selected_sentence_contains_gold=selected_contains,
        projected_eval_score=float(record.projected_eval_score),
        gold_in_any_sentence=graph_contains,
    )

    return HotpotQASentenceHitRecord(
        question_id=record.question_id,
        graph_id=record.graph_id,
        policy_name=record.policy_name,
        answer_selector_name=record.answer_selector_name,
        answer_extractor_name=record.answer_extractor_name,
        reward_mode=record.reward_mode,
        gold_answer=record.gold_answer,
        gold_answer_type=gold_answer_type,
        raw_graph_answer=record.raw_graph_answer,
        selected_graph_answer=record.selected_graph_answer,
        projected_answer=record.projected_answer,
        exact_match=float(record.exact_match),
        f1=float(record.f1),
        projected_eval_score=float(record.projected_eval_score),
        touched_sentence=bool(touched_sentences),
        touched_sentence_count=touched_sentence_count,
        first_sentence_step_idx=first_sentence_step_idx,
        any_touched_sentence_contains_gold=any_touched_contains,
        selected_sentence_contains_gold=selected_contains,
        gold_in_any_sentence=graph_contains,
        diagnostic_bucket=diagnostic_bucket,
        touched_sentences=[_sentence_summary(node) for node in touched_sentences],
        metadata={
            "path_node_count": len(path_nodes),
            "gold_answer_type": gold_answer_type,
            "yes_no_contains_gold_not_applicable": gold_answer_type == "yes_no",
        },
    )


def summarize_sentence_hit_records(records: list[HotpotQASentenceHitRecord]) -> dict[str, Any]:
    """Aggregate sentence-hit diagnostic metrics."""

    applicable_records = [record for record in records if record.gold_answer_type != "yes_no"]
    return {
        "sample_count": len(records),
        "sentence_touch_count": sum(1 for record in records if record.touched_sentence),
        "sentence_touch_rate": _safe_rate(sum(1 for record in records if record.touched_sentence), len(records)),
        "avg_touched_sentence_count": _average(record.touched_sentence_count for record in records),
        "gold_answer_type_distribution": dict(Counter(record.gold_answer_type for record in records)),
        "gold_sentence_applicable_count": len(applicable_records),
        "gold_in_any_sentence_count": _true_count(record.gold_in_any_sentence for record in applicable_records),
        "gold_in_any_sentence_rate": _true_rate(record.gold_in_any_sentence for record in applicable_records),
        "gold_sentence_touched_count": _true_count(
            record.any_touched_sentence_contains_gold for record in applicable_records
        ),
        "gold_sentence_touched_rate": _true_rate(
            record.any_touched_sentence_contains_gold for record in applicable_records
        ),
        "selected_sentence_contains_gold_count": _true_count(
            record.selected_sentence_contains_gold for record in applicable_records
        ),
        "selected_sentence_contains_gold_rate": _true_rate(
            record.selected_sentence_contains_gold for record in applicable_records
        ),
        "diagnostic_buckets": bucket_summary(records, lambda record: record.diagnostic_bucket),
        "policy_names": sorted({record.policy_name for record in records}),
        "answer_selector_names": sorted({record.answer_selector_name for record in records}),
        "answer_extractor_names": sorted({record.answer_extractor_name for record in records}),
        "reward_modes": sorted({record.reward_mode for record in records}),
    }


def classify_gold_answer(answer: str | None) -> str:
    """Classify gold answer for sentence-hit diagnostics."""

    normalized = normalize_answer(answer)
    if normalized in {"yes", "no"}:
        return "yes_no"
    if not normalized:
        return "empty"
    return "span"


def contains_gold(sentence_text: str | None, gold_answer: str | None, gold_answer_type: str | None = None) -> bool | None:
    """Return whether a sentence contains the normalized gold answer.

    yes/no gold answers are marked not-applicable to avoid short-token
    substring pollution in sentence-hit diagnostics.
    """

    resolved_type = gold_answer_type or classify_gold_answer(gold_answer)
    if resolved_type == "yes_no":
        return None
    normalized_gold = normalize_answer(gold_answer)
    normalized_sentence = normalize_answer(sentence_text)
    if not normalized_gold or not normalized_sentence:
        return False
    return normalized_gold in normalized_sentence


def classify_sentence_hit_bucket(
    *,
    touched_sentence: bool,
    gold_answer_type: str,
    any_touched_sentence_contains_gold: bool | None,
    selected_sentence_contains_gold: bool | None,
    projected_eval_score: float,
    gold_in_any_sentence: bool | None,
) -> str:
    """Classify where sentence-level diagnosis says the failure sits."""

    if gold_answer_type == "yes_no":
        return "yes_no_not_applicable"
    if not touched_sentence:
        return "no_sentence_touched"
    if gold_in_any_sentence is False:
        return "no_gold_sentence_in_graph"
    if any_touched_sentence_contains_gold is False:
        return "sentence_touched_but_no_gold_sentence"
    if any_touched_sentence_contains_gold and not selected_sentence_contains_gold:
        return "gold_sentence_touched_but_not_selected"
    if selected_sentence_contains_gold and projected_eval_score <= 0.0:
        return "selected_sentence_contains_gold_but_eval_still_zero"
    if selected_sentence_contains_gold and projected_eval_score > 0.0:
        return "selected_sentence_and_eval_positive"
    return "unknown"


def bucket_summary(records: list[HotpotQASentenceHitRecord], bucket_fn) -> dict[str, dict[str, float | int]]:
    """Aggregate count and quality metrics by a bucket function."""

    grouped: dict[str, list[HotpotQASentenceHitRecord]] = defaultdict(list)
    for record in records:
        grouped[str(bucket_fn(record))].append(record)
    return {bucket: _metric_summary(items) for bucket, items in sorted(grouped.items())}


def save_sentence_hit_outputs(
    result: HotpotQASentenceHitDiagnosticResult,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Save sentence-hit records JSONL and summary JSON."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    records_path = output_path / "hotpotqa_sentence_hit_records.jsonl"
    summary_path = output_path / "hotpotqa_sentence_hit_summary.json"
    with records_path.open("w", encoding="utf-8") as handle:
        for record in result.records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    summary_path.write_text(
        json.dumps(result.summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return records_path, summary_path


def _any_contains_gold(
    sentence_nodes: list[dict[str, Any]],
    gold_answer: str,
    gold_answer_type: str,
) -> bool | None:
    if gold_answer_type == "yes_no":
        return None
    return any(bool(contains_gold(str(node.get("text") or ""), gold_answer, gold_answer_type)) for node in sentence_nodes)


def _selected_sentence_contains_gold(
    record: HotpotQAGraphEvalRecord,
    candidate_nodes: list[Any],
    gold_answer_type: str,
) -> bool | None:
    if gold_answer_type == "yes_no":
        return None
    if record.answer_source_type != "sentence":
        return False
    selected_id = record.selected_graph_answer
    for node in candidate_nodes:
        if isinstance(node, dict) and node.get("node_id") == selected_id and node.get("node_type") == "sentence":
            return contains_gold(str(node.get("text") or ""), record.gold_answer, gold_answer_type)
    return contains_gold(record.projected_answer, record.gold_answer, gold_answer_type)


def _first_sentence_step_idx(touched_sentences: list[dict[str, Any]]) -> int | None:
    step_indices = [node.get("step_idx") for node in touched_sentences if node.get("step_idx") is not None]
    if not step_indices:
        return None
    return min(int(step_idx) for step_idx in step_indices)


def _sentence_summary(node: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_id": node.get("node_id"),
        "text": node.get("text"),
        "step_idx": node.get("step_idx"),
        "metadata": dict(node.get("metadata") or {}),
    }


def _metric_summary(records: list[HotpotQASentenceHitRecord]) -> dict[str, float | int]:
    return {
        "count": len(records),
        "exact_match": _average(record.exact_match for record in records),
        "avg_f1": _average(record.f1 for record in records),
        "avg_projected_eval_score": _average(record.projected_eval_score for record in records),
        "sentence_touch_rate": _safe_rate(sum(1 for record in records if record.touched_sentence), len(records)),
    }


def _true_count(values: Any) -> int:
    return sum(1 for value in values if value is True)


def _true_rate(values: Any) -> float:
    items = [value for value in values if value is not None]
    return _safe_rate(sum(1 for value in items if value is True), len(items))


def _average(values: Any) -> float:
    items = [float(value) for value in values]
    if not items:
        return 0.0
    return sum(items) / len(items)


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
