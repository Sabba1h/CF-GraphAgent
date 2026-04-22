"""Answer type-aware diagnostics for HotpotQA graph-backed results.

Gold answers are used here only for offline bucketed diagnostics. This module
does not feed answer-type information back into policy, selector, or extractor
decisions.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from data.benchmarks.answer_normalization import normalize_answer
from evaluation.hotpotqa_error_analysis import classify_projected_answer
from evaluation.hotpotqa_metrics import ANSWER_SOURCE_TYPES, normalize_answer_source_type
from evaluation.hotpotqa_sentence_hit_diagnostic import (
    HotpotQASentenceHitRecord,
    sentence_hit_record_from_eval_record,
)
from evaluation.hotpotqa_subset_evaluator import HotpotQAGraphEvalRecord

ANSWER_TYPE_BUCKETS = (
    "yes_no",
    "numeric_or_date",
    "single_token_entity_like",
    "multi_token_entity_or_title_like",
    "descriptive_span_or_relation",
)

_MONTH_TOKENS = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}
_DATE_CONNECTOR_TOKENS = {"ad", "bc", "ce", "bce"}
_DIGIT_RE = re.compile(r"^\d+$")


@dataclass(slots=True)
class HotpotQAAnswerTypeRecord:
    """One eval record enriched with answer type and sentence-hit diagnostics."""

    question_id: str
    graph_id: str
    reward_mode: str
    policy_name: str
    answer_selector_name: str
    answer_extractor_name: str
    answer_mapper_name: str
    gold_answer: str
    normalized_gold_answer: str
    answer_type: str
    raw_graph_answer: str | None
    selected_graph_answer: str | None
    projected_answer: str
    normalized_projected_answer: str
    exact_match: float
    f1: float
    projected_eval_score: float
    answer_source_type: str
    projected_answer_type: str
    touched_sentence: bool
    touched_sentence_count: int
    first_sentence_step_idx: int | None
    gold_in_any_sentence: bool | None
    any_touched_sentence_contains_gold: bool | None
    selected_sentence_contains_gold: bool | None
    sentence_hit_applicability: str
    sentence_hit_bucket: str
    base_total_reward: float
    total_reward: float
    step_count: int
    graph_node_count: int
    graph_edge_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""

        return asdict(self)


@dataclass(slots=True)
class HotpotQAAnswerTypeDiagnosticResult:
    """Answer type-aware diagnostics for a fixed HotpotQA graph-backed config."""

    records: list[HotpotQAAnswerTypeRecord]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""

        return {
            "records": [record.to_dict() for record in self.records],
            "summary": self.summary,
        }


def analyze_hotpotqa_answer_types(
    records: list[HotpotQAGraphEvalRecord],
    *,
    require_fixed_config: bool = True,
) -> HotpotQAAnswerTypeDiagnosticResult:
    """Build answer type records and aggregate fixed-config summaries."""

    if require_fixed_config:
        _validate_fixed_config(records)
    type_records = [
        answer_type_record_from_eval_record(
            record,
            sentence_hit_record=sentence_hit_record_from_eval_record(record),
        )
        for record in records
    ]
    return HotpotQAAnswerTypeDiagnosticResult(
        records=type_records,
        summary=summarize_answer_type_records(type_records),
    )


def answer_type_record_from_eval_record(
    record: HotpotQAGraphEvalRecord,
    *,
    sentence_hit_record: HotpotQASentenceHitRecord | None = None,
) -> HotpotQAAnswerTypeRecord:
    """Convert one eval record into an answer type-aware diagnostic record."""

    sentence_hit = sentence_hit_record or sentence_hit_record_from_eval_record(record)
    answer_type = classify_hotpotqa_answer_type(record.gold_answer)
    sentence_applicability = "not_applicable" if answer_type == "yes_no" else "applicable"
    return HotpotQAAnswerTypeRecord(
        question_id=record.question_id,
        graph_id=record.graph_id,
        reward_mode=str(record.reward_mode),
        policy_name=record.policy_name,
        answer_selector_name=record.answer_selector_name,
        answer_extractor_name=record.answer_extractor_name,
        answer_mapper_name=record.answer_mapper_name,
        gold_answer=record.gold_answer,
        normalized_gold_answer=normalize_answer(record.gold_answer),
        answer_type=answer_type,
        raw_graph_answer=record.raw_graph_answer,
        selected_graph_answer=record.selected_graph_answer,
        projected_answer=record.projected_answer,
        normalized_projected_answer=record.normalized_projected_answer,
        exact_match=float(record.exact_match),
        f1=float(record.f1),
        projected_eval_score=float(record.projected_eval_score),
        answer_source_type=normalize_answer_source_type(record.answer_source_type),
        projected_answer_type=classify_projected_answer(record.projected_answer),
        touched_sentence=sentence_hit.touched_sentence,
        touched_sentence_count=sentence_hit.touched_sentence_count,
        first_sentence_step_idx=sentence_hit.first_sentence_step_idx,
        gold_in_any_sentence=sentence_hit.gold_in_any_sentence,
        any_touched_sentence_contains_gold=sentence_hit.any_touched_sentence_contains_gold,
        selected_sentence_contains_gold=sentence_hit.selected_sentence_contains_gold,
        sentence_hit_applicability=sentence_applicability,
        sentence_hit_bucket=sentence_hit.diagnostic_bucket,
        base_total_reward=float(record.base_total_reward),
        total_reward=float(record.total_reward),
        step_count=int(record.step_count),
        graph_node_count=int(record.graph_node_count),
        graph_edge_count=int(record.graph_edge_count),
        metadata={
            "source_metadata": record.metadata,
            "sentence_hit_metadata": sentence_hit.metadata,
            "answer_mapping": record.metadata.get("answer_mapping"),
        },
    )


def classify_hotpotqa_answer_type(answer: str | None) -> str:
    """Classify a normalized gold answer into a small fixed bucket set."""

    normalized = normalize_answer(answer)
    if normalized in {"yes", "no"}:
        return "yes_no"
    tokens = normalized.split()
    if _looks_numeric_or_date(tokens):
        return "numeric_or_date"
    if len(tokens) == 1:
        return "single_token_entity_like"
    if 2 <= len(tokens) <= 4:
        return "multi_token_entity_or_title_like"
    return "descriptive_span_or_relation"


def summarize_answer_type_records(records: list[HotpotQAAnswerTypeRecord]) -> dict[str, Any]:
    """Aggregate answer type-aware metrics for one fixed pipeline config."""

    sample_count = len(records)
    grouped: dict[str, list[HotpotQAAnswerTypeRecord]] = defaultdict(list)
    for record in records:
        grouped[record.answer_type].append(record)

    return {
        "sample_count": sample_count,
        "fixed_config": _fixed_config(records),
        "answer_type_order": list(ANSWER_TYPE_BUCKETS),
        "answer_type_buckets": {
            answer_type: _answer_type_bucket_summary(grouped.get(answer_type, []), sample_count, answer_type)
            for answer_type in ANSWER_TYPE_BUCKETS
        },
    }


def save_answer_type_outputs(
    result: HotpotQAAnswerTypeDiagnosticResult,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Save answer type records JSONL and summary JSON."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    records_path = output_path / "hotpotqa_answer_type_records.jsonl"
    summary_path = output_path / "hotpotqa_answer_type_summary.json"
    with records_path.open("w", encoding="utf-8") as handle:
        for record in result.records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    summary_path.write_text(
        json.dumps(result.summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return records_path, summary_path


def _answer_type_bucket_summary(
    records: list[HotpotQAAnswerTypeRecord],
    total_count: int,
    answer_type: str,
) -> dict[str, Any]:
    source_counts = Counter(record.answer_source_type for record in records)
    projected_counts = Counter(record.projected_answer_type for record in records)
    return {
        "count": len(records),
        "ratio": _safe_rate(len(records), total_count),
        "avg_exact_match": _average(record.exact_match for record in records),
        "avg_f1": _average(record.f1 for record in records),
        "avg_projected_eval_score": _average(record.projected_eval_score for record in records),
        "failure_rate": _safe_rate(sum(1 for record in records if record.exact_match < 1.0), len(records)),
        "avg_base_total_reward": _average(record.base_total_reward for record in records),
        "avg_total_reward": _average(record.total_reward for record in records),
        "avg_step_count": _average(record.step_count for record in records),
        "answer_source_type_distribution": {
            source_type: source_counts.get(source_type, 0) for source_type in sorted(ANSWER_SOURCE_TYPES)
        },
        "projected_answer_type_distribution": dict(sorted(projected_counts.items())),
        "sentence_hit": _sentence_hit_summary(records, answer_type),
    }


def _sentence_hit_summary(records: list[HotpotQAAnswerTypeRecord], answer_type: str) -> dict[str, Any]:
    base_summary: dict[str, Any] = {
        "sentence_touch_rate": _safe_rate(sum(1 for record in records if record.touched_sentence), len(records)),
        "avg_touched_sentence_count": _average(record.touched_sentence_count for record in records),
        "sentence_hit_buckets": dict(Counter(record.sentence_hit_bucket for record in records)),
    }
    if answer_type == "yes_no":
        base_summary.update(
            {
                "gold_sentence_hit_rate": "not_applicable",
                "selected_sentence_contains_gold_rate": "not_applicable",
                "gold_in_any_sentence_rate": "not_applicable",
                "not_applicable_reason": "yes_no_answers_do_not_use_substring_sentence_hit",
            }
        )
        return base_summary

    base_summary.update(
        {
            "gold_sentence_hit_rate": _true_rate(record.any_touched_sentence_contains_gold for record in records),
            "selected_sentence_contains_gold_rate": _true_rate(
                record.selected_sentence_contains_gold for record in records
            ),
            "gold_in_any_sentence_rate": _true_rate(record.gold_in_any_sentence for record in records),
        }
    )
    return base_summary


def _looks_numeric_or_date(tokens: list[str]) -> bool:
    if not tokens:
        return False
    if all(_DIGIT_RE.match(token) for token in tokens):
        return True
    allowed = _MONTH_TOKENS | _DATE_CONNECTOR_TOKENS
    has_date_signal = any(token in _MONTH_TOKENS or _DIGIT_RE.match(token) for token in tokens)
    return has_date_signal and all(_DIGIT_RE.match(token) or token in allowed for token in tokens)


def _validate_fixed_config(records: list[HotpotQAGraphEvalRecord]) -> None:
    config = _fixed_config(records)
    mixed_fields = [name for name, values in config.items() if isinstance(values, list) and len(values) > 1]
    if mixed_fields:
        raise ValueError(
            "Answer type diagnostic expects one fixed reward_mode/policy/selector/extractor/mapper config. "
            f"Mixed fields: {', '.join(mixed_fields)}."
        )


def _fixed_config(records: list[HotpotQAAnswerTypeRecord] | list[HotpotQAGraphEvalRecord]) -> dict[str, Any]:
    return {
        "reward_modes": sorted({str(record.reward_mode) for record in records}),
        "policy_names": sorted({record.policy_name for record in records}),
        "answer_selector_names": sorted({record.answer_selector_name for record in records}),
        "answer_extractor_names": sorted({record.answer_extractor_name for record in records}),
        "answer_mapper_names": sorted({record.answer_mapper_name for record in records}),
    }


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
