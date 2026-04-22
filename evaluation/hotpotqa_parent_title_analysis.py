"""Parent-title mapper stability and attribution analysis for HotpotQA.

This module is analysis-only. Gold answers are used only to bucket outcomes
after rollout; they are never fed back into policy, selector, extractor, or
mapper decisions.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from data.benchmarks.answer_normalization import normalize_answer
from evaluation.hotpotqa_answer_type_diagnostic import classify_hotpotqa_answer_type
from evaluation.hotpotqa_metrics import summarize_hotpotqa_records
from evaluation.hotpotqa_sentence_hit_diagnostic import (
    classify_gold_answer,
    contains_gold,
    sentence_hit_record_from_eval_record,
)
from evaluation.hotpotqa_subset_evaluator import HotpotQAGraphEvalRecord

ENTITY_TITLE_ANSWER_TYPES = {
    "single_token_entity_like",
    "multi_token_entity_or_title_like",
}

PARENT_TITLE_ATTRIBUTION_BUCKETS = (
    "parent_title_exact_match_success",
    "parent_title_partial_improvement",
    "identity_already_correct",
    "selected_sentence_parent_title_wrong",
    "selected_sentence_not_relevant",
    "gold_not_title_like_or_mapper_not_applicable",
    "path_touched_wrong_region",
    "needs_span_or_relation_not_title",
)


@dataclass(slots=True)
class HotpotQAParentTitleAttributionRecord:
    """One paired identity-vs-parent-title attribution record."""

    question_id: str
    graph_id: str
    reward_mode: str
    policy_name: str
    answer_selector_name: str
    fixed_extractor_name: str
    gold_answer: str
    normalized_gold_answer: str
    answer_type: str
    selected_graph_answer: str | None
    selected_node_type: str | None
    parent_title_node_id: str | None
    parent_title_text: str
    identity_projected_answer: str
    parent_title_projected_answer: str
    identity_exact_match: float
    parent_title_exact_match: float
    identity_f1: float
    parent_title_f1: float
    identity_projected_eval_score: float
    parent_title_projected_eval_score: float
    delta_exact_match: float
    delta_f1: float
    delta_projected_eval_score: float
    parent_title_matches_gold: bool
    parent_title_fallback_occurred: bool
    parent_title_fallback_target: str | None
    parent_title_fallback_reason: str | None
    touched_sentence: bool
    any_touched_sentence_contains_gold: bool | None
    selected_sentence_contains_gold: bool | None
    gold_in_any_sentence: bool | None
    attribution_bucket: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""

        return asdict(self)


@dataclass(slots=True)
class HotpotQAParentTitleAttributionResult:
    """Parent-title mapper attribution result."""

    records: list[HotpotQAParentTitleAttributionRecord]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""

        return {
            "records": [record.to_dict() for record in self.records],
            "summary": self.summary,
        }


def analyze_parent_title_mapper(
    identity_records: list[HotpotQAGraphEvalRecord],
    parent_title_records: list[HotpotQAGraphEvalRecord],
    *,
    scale_limits: list[int] | None = None,
) -> HotpotQAParentTitleAttributionResult:
    """Compare identity and parent-title records and produce attribution summaries."""

    pairs = _paired_records(identity_records, parent_title_records)
    records = [parent_title_attribution_record(identity, parent) for identity, parent in pairs]
    summary = summarize_parent_title_attribution(
        records,
        identity_records=[identity for identity, _ in pairs],
        parent_title_records=[parent for _, parent in pairs],
        scale_limits=scale_limits,
    )
    return HotpotQAParentTitleAttributionResult(records=records, summary=summary)


def parent_title_attribution_record(
    identity_record: HotpotQAGraphEvalRecord,
    parent_title_record: HotpotQAGraphEvalRecord,
) -> HotpotQAParentTitleAttributionRecord:
    """Build one paired attribution record."""

    if identity_record.question_id != parent_title_record.question_id:
        raise ValueError(
            "Identity and parent_title records must be paired by question_id. "
            f"Got {identity_record.question_id!r} vs {parent_title_record.question_id!r}."
        )

    answer_type = classify_hotpotqa_answer_type(parent_title_record.gold_answer)
    mapping = _answer_mapping(parent_title_record)
    selected_node = _selected_node(mapping, parent_title_record)
    sentence_hit = sentence_hit_record_from_eval_record(parent_title_record)
    selected_contains_gold = _selected_node_contains_gold(selected_node, parent_title_record.gold_answer)
    parent_title_text = str(mapping.get("mapped_answer") or parent_title_record.projected_answer or "")
    normalized_gold = normalize_answer(parent_title_record.gold_answer)
    parent_title_matches_gold = bool(parent_title_text) and normalize_answer(parent_title_text) == normalized_gold
    delta_exact = float(parent_title_record.exact_match) - float(identity_record.exact_match)
    delta_f1 = float(parent_title_record.f1) - float(identity_record.f1)
    delta_score = float(parent_title_record.projected_eval_score) - float(identity_record.projected_eval_score)
    bucket = classify_parent_title_attribution(
        answer_type=answer_type,
        identity_exact_match=float(identity_record.exact_match),
        parent_title_exact_match=float(parent_title_record.exact_match),
        delta_f1=delta_f1,
        parent_title_matches_gold=parent_title_matches_gold,
        parent_title_fallback_occurred=bool(mapping.get("fallback_occurred")),
        selected_node_type=_str_or_none(selected_node.get("node_type")),
        touched_sentence=sentence_hit.touched_sentence,
        any_touched_sentence_contains_gold=sentence_hit.any_touched_sentence_contains_gold,
        selected_sentence_contains_gold=selected_contains_gold,
        gold_in_any_sentence=sentence_hit.gold_in_any_sentence,
    )

    return HotpotQAParentTitleAttributionRecord(
        question_id=parent_title_record.question_id,
        graph_id=parent_title_record.graph_id,
        reward_mode=str(parent_title_record.reward_mode),
        policy_name=parent_title_record.policy_name,
        answer_selector_name=parent_title_record.answer_selector_name,
        fixed_extractor_name=parent_title_record.answer_extractor_name,
        gold_answer=parent_title_record.gold_answer,
        normalized_gold_answer=normalized_gold,
        answer_type=answer_type,
        selected_graph_answer=parent_title_record.selected_graph_answer,
        selected_node_type=_str_or_none(selected_node.get("node_type")),
        parent_title_node_id=_str_or_none(mapping.get("source_node_id")),
        parent_title_text=parent_title_text,
        identity_projected_answer=identity_record.projected_answer,
        parent_title_projected_answer=parent_title_record.projected_answer,
        identity_exact_match=float(identity_record.exact_match),
        parent_title_exact_match=float(parent_title_record.exact_match),
        identity_f1=float(identity_record.f1),
        parent_title_f1=float(parent_title_record.f1),
        identity_projected_eval_score=float(identity_record.projected_eval_score),
        parent_title_projected_eval_score=float(parent_title_record.projected_eval_score),
        delta_exact_match=delta_exact,
        delta_f1=delta_f1,
        delta_projected_eval_score=delta_score,
        parent_title_matches_gold=parent_title_matches_gold,
        parent_title_fallback_occurred=bool(mapping.get("fallback_occurred")),
        parent_title_fallback_target=_str_or_none(mapping.get("fallback_target")),
        parent_title_fallback_reason=_str_or_none(mapping.get("fallback_reason")),
        touched_sentence=sentence_hit.touched_sentence,
        any_touched_sentence_contains_gold=sentence_hit.any_touched_sentence_contains_gold,
        selected_sentence_contains_gold=selected_contains_gold,
        gold_in_any_sentence=sentence_hit.gold_in_any_sentence,
        attribution_bucket=bucket,
        metadata={
            "identity_record": _record_pointer(identity_record),
            "parent_title_record": _record_pointer(parent_title_record),
            "answer_mapping": mapping,
            "sentence_hit_bucket": sentence_hit.diagnostic_bucket,
            "sentence_hit_metadata": sentence_hit.metadata,
        },
    )


def classify_parent_title_attribution(
    *,
    answer_type: str,
    identity_exact_match: float,
    parent_title_exact_match: float,
    delta_f1: float,
    parent_title_matches_gold: bool,
    parent_title_fallback_occurred: bool,
    selected_node_type: str | None,
    touched_sentence: bool,
    any_touched_sentence_contains_gold: bool | None,
    selected_sentence_contains_gold: bool | None,
    gold_in_any_sentence: bool | None,
) -> str:
    """Classify a paired example into a transparent parent-title attribution bucket."""

    if parent_title_exact_match >= 1.0 and identity_exact_match < 1.0:
        return "parent_title_exact_match_success"
    if identity_exact_match >= 1.0 and parent_title_exact_match >= 1.0:
        return "identity_already_correct"
    if delta_f1 > 1e-12:
        return "parent_title_partial_improvement"

    if answer_type in {"yes_no", "numeric_or_date"}:
        return "gold_not_title_like_or_mapper_not_applicable"
    if answer_type == "descriptive_span_or_relation":
        return "needs_span_or_relation_not_title"

    if not touched_sentence:
        return "path_touched_wrong_region"
    if parent_title_fallback_occurred or selected_node_type != "sentence":
        return "gold_not_title_like_or_mapper_not_applicable"
    if gold_in_any_sentence is True and any_touched_sentence_contains_gold is False:
        return "path_touched_wrong_region"
    if selected_sentence_contains_gold is False:
        return "selected_sentence_not_relevant"
    if selected_sentence_contains_gold is True and not parent_title_matches_gold:
        return "selected_sentence_parent_title_wrong"
    return "selected_sentence_parent_title_wrong"


def summarize_parent_title_attribution(
    records: list[HotpotQAParentTitleAttributionRecord],
    *,
    identity_records: list[HotpotQAGraphEvalRecord],
    parent_title_records: list[HotpotQAGraphEvalRecord],
    scale_limits: list[int] | None = None,
) -> dict[str, Any]:
    """Aggregate stability and attribution summaries."""

    scales = _resolve_scale_limits(scale_limits, len(records))
    return {
        "sample_count": len(records),
        "fixed_config": _fixed_config(records),
        "overall": {
            "identity": _metric_summary(identity_records),
            "parent_title": _metric_summary(parent_title_records),
            "delta": _summary_delta(_metric_summary(identity_records), _metric_summary(parent_title_records)),
        },
        "entity_title_like": {
            "identity": _metric_summary(_filter_eval_by_answer_type(identity_records, ENTITY_TITLE_ANSWER_TYPES)),
            "parent_title": _metric_summary(_filter_eval_by_answer_type(parent_title_records, ENTITY_TITLE_ANSWER_TYPES)),
            "delta": _summary_delta(
                _metric_summary(_filter_eval_by_answer_type(identity_records, ENTITY_TITLE_ANSWER_TYPES)),
                _metric_summary(_filter_eval_by_answer_type(parent_title_records, ENTITY_TITLE_ANSWER_TYPES)),
            ),
            "single_token_entity_like": _answer_type_pair_summary(
                identity_records,
                parent_title_records,
                "single_token_entity_like",
            ),
            "multi_token_entity_or_title_like": _answer_type_pair_summary(
                identity_records,
                parent_title_records,
                "multi_token_entity_or_title_like",
            ),
        },
        "attribution_buckets": _bucket_summary(records, lambda record: record.attribution_bucket),
        "success_summary": _success_summary(records),
        "failure_summary": _failure_summary(records),
        "scale_curve": {
            str(limit): _scale_summary(records[:limit], identity_records[:limit], parent_title_records[:limit])
            for limit in scales
        },
    }


def save_parent_title_analysis_outputs(
    result: HotpotQAParentTitleAttributionResult,
    output_dir: str | Path,
) -> tuple[Path, Path, Path]:
    """Save parent-title attribution records and summaries."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    records_path = output_path / "parent_title_attribution_records.jsonl"
    failure_path = output_path / "parent_title_failure_attribution.json"
    stability_path = output_path / "parent_title_stability_summary.json"
    _write_jsonl(records_path, [record.to_dict() for record in result.records])
    failure_path.write_text(
        json.dumps(
            {
                "sample_count": result.summary.get("sample_count", 0),
                "attribution_buckets": result.summary.get("attribution_buckets", {}),
                "success_summary": result.summary.get("success_summary", {}),
                "failure_summary": result.summary.get("failure_summary", {}),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    stability_path.write_text(
        json.dumps(result.summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return records_path, failure_path, stability_path


def _paired_records(
    identity_records: list[HotpotQAGraphEvalRecord],
    parent_title_records: list[HotpotQAGraphEvalRecord],
) -> list[tuple[HotpotQAGraphEvalRecord, HotpotQAGraphEvalRecord]]:
    if len(identity_records) != len(parent_title_records):
        raise ValueError(
            "Identity and parent_title runs must have the same number of records. "
            f"Got {len(identity_records)} vs {len(parent_title_records)}."
        )
    pairs = list(zip(identity_records, parent_title_records, strict=True))
    for identity, parent in pairs:
        if identity.question_id != parent.question_id:
            raise ValueError(
                "Identity and parent_title records must be in the same order. "
                f"Got {identity.question_id!r} vs {parent.question_id!r}."
            )
    return pairs


def _answer_mapping(record: HotpotQAGraphEvalRecord) -> dict[str, Any]:
    mapping = record.metadata.get("answer_mapping")
    return dict(mapping) if isinstance(mapping, dict) else {}


def _selected_node(mapping: dict[str, Any], record: HotpotQAGraphEvalRecord) -> dict[str, Any]:
    metadata = mapping.get("metadata") if isinstance(mapping.get("metadata"), dict) else {}
    selected_node = metadata.get("selected_node") if isinstance(metadata.get("selected_node"), dict) else {}
    if selected_node:
        return dict(selected_node)
    base_mapping = metadata.get("base_mapping") if isinstance(metadata.get("base_mapping"), dict) else {}
    return {
        "node_id": record.selected_graph_answer,
        "node_type": base_mapping.get("source_node_type"),
        "text": base_mapping.get("source_text"),
    }


def _selected_node_contains_gold(selected_node: dict[str, Any], gold_answer: str) -> bool | None:
    if selected_node.get("node_type") != "sentence":
        return False
    return contains_gold(str(selected_node.get("text") or ""), gold_answer, classify_gold_answer(gold_answer))


def _record_pointer(record: HotpotQAGraphEvalRecord) -> dict[str, Any]:
    return {
        "question_id": record.question_id,
        "projected_answer": record.projected_answer,
        "exact_match": record.exact_match,
        "f1": record.f1,
        "answer_mapper_name": record.answer_mapper_name,
    }


def _metric_summary(records: list[HotpotQAGraphEvalRecord]) -> dict[str, Any]:
    summary = summarize_hotpotqa_records(records)
    summary["failure_rate"] = _safe_rate(sum(1 for record in records if record.exact_match < 1.0), len(records))
    return summary


def _summary_delta(identity: dict[str, Any], parent_title: dict[str, Any]) -> dict[str, float]:
    return {
        "avg_exact_match": float(parent_title.get("avg_exact_match", 0.0))
        - float(identity.get("avg_exact_match", 0.0)),
        "avg_f1": float(parent_title.get("avg_f1", 0.0)) - float(identity.get("avg_f1", 0.0)),
        "avg_projected_eval_score": float(parent_title.get("avg_projected_eval_score", 0.0))
        - float(identity.get("avg_projected_eval_score", 0.0)),
        "failure_rate": float(parent_title.get("failure_rate", 0.0)) - float(identity.get("failure_rate", 0.0)),
    }


def _filter_eval_by_answer_type(
    records: list[HotpotQAGraphEvalRecord],
    answer_types: set[str],
) -> list[HotpotQAGraphEvalRecord]:
    return [record for record in records if classify_hotpotqa_answer_type(record.gold_answer) in answer_types]


def _answer_type_pair_summary(
    identity_records: list[HotpotQAGraphEvalRecord],
    parent_title_records: list[HotpotQAGraphEvalRecord],
    answer_type: str,
) -> dict[str, Any]:
    identity_items = _filter_eval_by_answer_type(identity_records, {answer_type})
    parent_items = _filter_eval_by_answer_type(parent_title_records, {answer_type})
    identity_summary = _metric_summary(identity_items)
    parent_summary = _metric_summary(parent_items)
    return {
        "identity": identity_summary,
        "parent_title": parent_summary,
        "delta": _summary_delta(identity_summary, parent_summary),
    }


def _bucket_summary(
    records: list[HotpotQAParentTitleAttributionRecord],
    bucket_fn,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[HotpotQAParentTitleAttributionRecord]] = defaultdict(list)
    for record in records:
        grouped[str(bucket_fn(record))].append(record)
    return {bucket: _attribution_metric_summary(items) for bucket, items in sorted(grouped.items())}


def _attribution_metric_summary(records: list[HotpotQAParentTitleAttributionRecord]) -> dict[str, Any]:
    return {
        "count": len(records),
        "avg_delta_exact_match": _average(record.delta_exact_match for record in records),
        "avg_delta_f1": _average(record.delta_f1 for record in records),
        "avg_delta_projected_eval_score": _average(record.delta_projected_eval_score for record in records),
        "parent_title_exact_match_rate": _average(record.parent_title_exact_match for record in records),
        "identity_exact_match_rate": _average(record.identity_exact_match for record in records),
        "answer_type_distribution": dict(Counter(record.answer_type for record in records)),
    }


def _success_summary(records: list[HotpotQAParentTitleAttributionRecord]) -> dict[str, Any]:
    fixed = [record for record in records if record.parent_title_exact_match >= 1.0 and record.identity_exact_match < 1.0]
    improved = [record for record in records if record.delta_f1 > 1e-12]
    regressed = [record for record in records if record.delta_f1 < -1e-12]
    return {
        "fixed_identity_failure_count": len(fixed),
        "f1_improved_count": len(improved),
        "f1_regressed_count": len(regressed),
        "parent_title_matches_gold_count": sum(1 for record in records if record.parent_title_matches_gold),
        "fixed_answer_type_distribution": dict(Counter(record.answer_type for record in fixed)),
        "improved_answer_type_distribution": dict(Counter(record.answer_type for record in improved)),
    }


def _failure_summary(records: list[HotpotQAParentTitleAttributionRecord]) -> dict[str, Any]:
    failures = [record for record in records if record.parent_title_exact_match < 1.0]
    return {
        "failure_count": len(failures),
        "failure_rate": _safe_rate(len(failures), len(records)),
        "failure_buckets": _bucket_summary(failures, lambda record: record.attribution_bucket),
        "failure_answer_type_distribution": dict(Counter(record.answer_type for record in failures)),
    }


def _scale_summary(
    records: list[HotpotQAParentTitleAttributionRecord],
    identity_records: list[HotpotQAGraphEvalRecord],
    parent_title_records: list[HotpotQAGraphEvalRecord],
) -> dict[str, Any]:
    identity_summary = _metric_summary(identity_records)
    parent_summary = _metric_summary(parent_title_records)
    return {
        "sample_count": len(records),
        "identity": identity_summary,
        "parent_title": parent_summary,
        "delta": _summary_delta(identity_summary, parent_summary),
        "entity_title_like": {
            "identity": _metric_summary(_filter_eval_by_answer_type(identity_records, ENTITY_TITLE_ANSWER_TYPES)),
            "parent_title": _metric_summary(_filter_eval_by_answer_type(parent_title_records, ENTITY_TITLE_ANSWER_TYPES)),
            "delta": _summary_delta(
                _metric_summary(_filter_eval_by_answer_type(identity_records, ENTITY_TITLE_ANSWER_TYPES)),
                _metric_summary(_filter_eval_by_answer_type(parent_title_records, ENTITY_TITLE_ANSWER_TYPES)),
            ),
        },
        "attribution_buckets": _bucket_summary(records, lambda record: record.attribution_bucket),
    }


def _fixed_config(records: list[HotpotQAParentTitleAttributionRecord]) -> dict[str, Any]:
    return {
        "reward_modes": sorted({record.reward_mode for record in records}),
        "policy_names": sorted({record.policy_name for record in records}),
        "answer_selector_names": sorted({record.answer_selector_name for record in records}),
        "fixed_extractor_names": sorted({record.fixed_extractor_name for record in records}),
        "compared_mappers": ["identity", "parent_title"],
    }


def _resolve_scale_limits(scale_limits: list[int] | None, max_count: int) -> list[int]:
    raw_limits = scale_limits or [500, 1000, max_count]
    resolved = sorted({min(int(limit), max_count) for limit in raw_limits if int(limit) > 0 and max_count > 0})
    return resolved or ([max_count] if max_count > 0 else [])


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _average(values: Any) -> float:
    items = [float(value) for value in values]
    if not items:
        return 0.0
    return sum(items) / len(items)


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
