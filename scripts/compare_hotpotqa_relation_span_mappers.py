"""Compare non-privileged HotpotQA relation/span mapper variants."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from answer.hotpotqa_answer_selector import make_answer_selector_factory
from answer.hotpotqa_entity_title_mapper import make_entity_title_mapper_factory
from answer.hotpotqa_relation_span_mapper import (
    RELATION_SPAN_MAPPER_NAMES,
    make_relation_span_mapper_factory,
)
from answer.hotpotqa_yesno_mapper import make_yesno_mapper_factory
from data.benchmarks import parse_indices
from data.benchmarks.common import load_json_records
from evaluation.hotpotqa_answer_type_diagnostic import (
    analyze_hotpotqa_answer_types,
    classify_hotpotqa_answer_type,
    save_answer_type_outputs,
)
from evaluation.hotpotqa_error_analysis import analyze_hotpotqa_error_records, save_error_analysis_outputs
from evaluation.hotpotqa_metrics import summarize_hotpotqa_records
from evaluation.hotpotqa_subset_evaluator import (
    HotpotQAGraphEvalRecord,
    evaluate_hotpotqa_graph_subset,
    save_hotpotqa_eval_outputs,
)
from graph.hotpotqa_policy_variants import make_policy_factory

FIXED_BASE_MAPPER_NAME = "parent_title"
FIXED_YESNO_MAPPER_NAME = "sentence_polarity"
FIXED_EXTRACTOR_NAME = "full_sentence"
FIXED_POLICY_NAME = "sentence_chain"
FIXED_SELECTOR_NAME = "latest_sentence"
FIXED_CANDIDATE_GENERATOR_NAME = "baseline_generator"
RELATION_SPAN_TYPE = "descriptive_span_or_relation"
DEFAULT_RELATION_SPAN_MAPPERS = (
    "identity_relation_span",
    "clause_relation",
    "pattern_span",
    "clause_then_pattern_backoff",
)


def compare_hotpotqa_relation_span_mappers(
    *,
    path: str | Path,
    split: str | None = None,
    limit: int | None = None,
    indices: list[int] | None = None,
    reward_mode: str = "baseline",
    mappers: list[str] | None = None,
    max_steps: int = 4,
    candidate_top_k: int = 5,
    min_expand_steps: int = 1,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run relation/span mapper variants on one fixed HotpotQA sample list."""

    mapper_names = mappers or list(DEFAULT_RELATION_SPAN_MAPPERS)
    resolved_indices = resolve_sample_indices(path=path, limit=limit, indices=indices)
    comparison: dict[str, Any] = {
        "reward_mode": reward_mode,
        "fixed_base_mapper_name": FIXED_BASE_MAPPER_NAME,
        "fixed_yesno_mapper_name": FIXED_YESNO_MAPPER_NAME,
        "fixed_extractor_name": FIXED_EXTRACTOR_NAME,
        "fixed_policy_name": FIXED_POLICY_NAME,
        "fixed_selector_name": FIXED_SELECTOR_NAME,
        "fixed_candidate_generator_name": FIXED_CANDIDATE_GENERATOR_NAME,
        "sample_indices": resolved_indices,
        "sample_count": len(resolved_indices),
        "mapper_order": mapper_names,
        "mappers": {},
    }
    output_path = Path(output_dir) if output_dir else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    for mapper_name in mapper_names:
        _validate_relation_span_mapper(mapper_name)
        eval_result = evaluate_hotpotqa_graph_subset(
            path=path,
            split=split,
            limit=None,
            indices=resolved_indices,
            reward_mode=reward_mode,
            max_steps=max_steps,
            candidate_top_k=candidate_top_k,
            min_expand_steps=min_expand_steps,
            policy_factory=make_policy_factory(FIXED_POLICY_NAME, min_expand_steps=min_expand_steps),
            policy_name=FIXED_POLICY_NAME,
            answer_selector_factory=make_answer_selector_factory(FIXED_SELECTOR_NAME),
            answer_selector_name=FIXED_SELECTOR_NAME,
            answer_extractor_factory=None,
            answer_extractor_name=FIXED_EXTRACTOR_NAME,
            answer_mapper_factory=make_entity_title_mapper_factory(FIXED_BASE_MAPPER_NAME),
            answer_mapper_name=FIXED_BASE_MAPPER_NAME,
            yesno_mapper_factory=make_yesno_mapper_factory(FIXED_YESNO_MAPPER_NAME),
            yesno_mapper_name=FIXED_YESNO_MAPPER_NAME,
            relation_span_mapper_factory=make_relation_span_mapper_factory(mapper_name),
            relation_span_mapper_name=mapper_name,
        )
        error_result = analyze_hotpotqa_error_records(eval_result.records)
        answer_type_result = analyze_hotpotqa_answer_types(eval_result.records)
        relation_summary = _relation_span_subset_summary(eval_result.records, answer_type_result.summary)
        decision_summary = summarize_relation_span_decisions(eval_result.records)
        comparison["mappers"][mapper_name] = {
            "relation_span_mapper_name": mapper_name,
            "base_mapper_name": FIXED_BASE_MAPPER_NAME,
            "yesno_mapper_name": FIXED_YESNO_MAPPER_NAME,
            "extractor_name": FIXED_EXTRACTOR_NAME,
            "eval_summary": eval_result.summary,
            "error_summary": error_result.summary,
            "answer_type_summary": answer_type_result.summary,
            "relation_span_summary": relation_summary,
            "non_relation_span_summary": _non_relation_span_summary(eval_result.records),
            "relation_span_decision_summary": decision_summary,
        }
        if output_path is not None:
            mapper_dir = output_path / mapper_name
            save_hotpotqa_eval_outputs(eval_result, mapper_dir)
            save_error_analysis_outputs(error_result, mapper_dir)
            save_answer_type_outputs(answer_type_result, mapper_dir)
            (mapper_dir / "hotpotqa_relation_span_decision_summary.json").write_text(
                json.dumps(decision_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    comparison["relation_span_gain_summary"] = _relation_span_gain_summary(comparison)
    if output_path is not None:
        (output_path / "hotpotqa_relation_span_mapper_comparison_summary.json").write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare HotpotQA relation/span mapper variants.")
    parser.add_argument("--path", required=True, help="Path to a HotpotQA JSON or JSONL file.")
    parser.add_argument("--split", default=None, help="Optional split label stored in metadata.")
    parser.add_argument("--limit", type=int, default=200, help="Maximum examples to evaluate.")
    parser.add_argument("--indices", default=None, help="Comma-separated record indices to evaluate.")
    parser.add_argument(
        "--reward-mode",
        choices=["baseline", "oracle_counterfactual"],
        default="baseline",
        help="Reward mode used for all mappers.",
    )
    parser.add_argument(
        "--mappers",
        default=",".join(DEFAULT_RELATION_SPAN_MAPPERS),
        help=f"Comma-separated relation/span mappers. Available: {', '.join(RELATION_SPAN_MAPPER_NAMES)}.",
    )
    parser.add_argument("--output-dir", default=None, help="Optional directory for records and summaries.")
    parser.add_argument("--max-steps", type=int, default=4, help="Maximum env steps per example.")
    parser.add_argument("--candidate-top-k", type=int, default=5, help="CandidateGenerator top_k.")
    parser.add_argument("--min-expand-steps", type=int, default=1, help="Fixed sentence_chain min expansion count.")
    args = parser.parse_args()

    comparison = compare_hotpotqa_relation_span_mappers(
        path=args.path,
        split=args.split,
        limit=args.limit,
        indices=parse_indices(args.indices),
        reward_mode=args.reward_mode,
        mappers=parse_mappers(args.mappers),
        max_steps=args.max_steps,
        candidate_top_k=args.candidate_top_k,
        min_expand_steps=args.min_expand_steps,
        output_dir=args.output_dir,
    )
    _print_comparison(comparison)
    if args.output_dir:
        print(
            "comparison_summary_path: "
            f"{Path(args.output_dir) / 'hotpotqa_relation_span_mapper_comparison_summary.json'}"
        )


def resolve_sample_indices(*, path: str | Path, limit: int | None, indices: list[int] | None) -> list[int]:
    """Resolve one explicit sample index list shared by every mapper."""

    record_count = len(load_json_records(path))
    if indices is not None:
        resolved = list(indices)
        if limit is not None:
            resolved = resolved[:limit]
    else:
        resolved_limit = record_count if limit is None else min(limit, record_count)
        resolved = list(range(resolved_limit))
    for index in resolved:
        if index < 0 or index >= record_count:
            raise IndexError(f"Subset index {index} out of range for {record_count} records.")
    return resolved


def parse_mappers(raw: str) -> list[str]:
    """Parse relation/span mapper names."""

    mappers = [item.strip() for item in raw.split(",") if item.strip()]
    for mapper_name in mappers:
        _validate_relation_span_mapper(mapper_name)
    return mappers


def summarize_relation_span_decisions(records: list[HotpotQAGraphEvalRecord]) -> dict[str, Any]:
    """Aggregate relation/span decision metadata from eval records."""

    relation_records = [record for record in records if _answer_type(record) == RELATION_SPAN_TYPE]
    mappings = [_relation_span_mapping(record) for record in relation_records if _relation_span_mapping(record)]
    fallback_counts = Counter(str(mapping.get("fallback_target") or "none") for mapping in mappings)
    reason_counts = Counter(str(mapping.get("selected_span_reason") or "none") for mapping in mappings)
    return {
        "relation_span_sample_count": len(relation_records),
        "mapper_applied_count": len(mappings),
        "avg_candidate_span_count": _average(len(mapping.get("candidate_spans", []) or []) for mapping in mappings),
        "fallback_target_counts": dict(sorted(fallback_counts.items())),
        "selected_span_reason_counts": dict(sorted(reason_counts.items())),
        "fallback_occurred_count": sum(1 for mapping in mappings if bool(mapping.get("fallback_occurred"))),
        "records_with_no_mapping": len(relation_records) - len(mappings),
    }


def _relation_span_subset_summary(records: list[HotpotQAGraphEvalRecord], answer_type_summary: dict[str, Any]) -> dict[str, Any]:
    bucket = answer_type_summary.get("answer_type_buckets", {}).get(RELATION_SPAN_TYPE, {})
    return {
        "count": int(bucket.get("count", 0)),
        "avg_exact_match": float(bucket.get("avg_exact_match", 0.0)),
        "avg_f1": float(bucket.get("avg_f1", 0.0)),
        "avg_projected_eval_score": float(bucket.get("avg_projected_eval_score", 0.0)),
        "failure_rate": float(bucket.get("failure_rate", 0.0)),
        "sentence_hit": bucket.get("sentence_hit", {}),
        "records": [record.question_id for record in records if _answer_type(record) == RELATION_SPAN_TYPE],
    }


def _non_relation_span_summary(records: list[HotpotQAGraphEvalRecord]) -> dict[str, Any]:
    return summarize_hotpotqa_records([record for record in records if _answer_type(record) != RELATION_SPAN_TYPE])


def _relation_span_gain_summary(comparison: dict[str, Any]) -> dict[str, Any]:
    mappers = comparison.get("mappers", {})
    if "identity_relation_span" not in mappers:
        return {"available": False, "reason": "identity_relation_span baseline was not run."}
    identity = mappers["identity_relation_span"]["relation_span_summary"]
    judgments = {}
    improvements: list[str] = []
    for mapper_name, payload in mappers.items():
        if mapper_name == "identity_relation_span":
            continue
        summary = payload["relation_span_summary"]
        delta_em = float(summary.get("avg_exact_match", 0.0)) - float(identity.get("avg_exact_match", 0.0))
        delta_f1 = float(summary.get("avg_f1", 0.0)) - float(identity.get("avg_f1", 0.0))
        delta_failure = float(summary.get("failure_rate", 0.0)) - float(identity.get("failure_rate", 0.0))
        improved = delta_em > 1e-12 or delta_f1 > 1e-12 or delta_failure < -1e-12
        if improved:
            improvements.append(mapper_name)
        judgments[mapper_name] = {
            "delta_relation_span_exact_match": delta_em,
            "delta_relation_span_f1": delta_f1,
            "delta_relation_span_failure_rate": delta_failure,
            "improves_over_identity": improved,
        }
    return {
        "available": True,
        "baseline_mapper": "identity_relation_span",
        "improved_mappers": improvements,
        "has_any_relation_span_gain": bool(improvements),
        "mapper_judgments": judgments,
    }


def _validate_relation_span_mapper(mapper_name: str) -> None:
    if mapper_name not in RELATION_SPAN_MAPPER_NAMES:
        raise ValueError(
            f"Unknown relation/span mapper '{mapper_name}'. "
            f"Available mappers: {', '.join(RELATION_SPAN_MAPPER_NAMES)}"
        )


def _answer_type(record: HotpotQAGraphEvalRecord) -> str:
    label = record.metadata.get("answer_type_label")
    if label in {"yes_no", "numeric_or_date", "single_token_entity_like", "multi_token_entity_or_title_like", RELATION_SPAN_TYPE}:
        return str(label)
    return classify_hotpotqa_answer_type(record.gold_answer)


def _relation_span_mapping(record: HotpotQAGraphEvalRecord) -> dict[str, Any]:
    mapping = record.metadata.get("relation_span_mapping")
    return mapping if isinstance(mapping, dict) else {}


def _average(values: Any) -> float:
    items = list(values)
    return 0.0 if not items else sum(items) / len(items)


def _print_comparison(comparison: dict[str, Any]) -> None:
    print("=== HotpotQA Relation/Span Mapper Comparison ===")
    print(f"reward_mode: {comparison['reward_mode']}")
    print(f"fixed_base_mapper_name: {comparison['fixed_base_mapper_name']}")
    print(f"fixed_yesno_mapper_name: {comparison['fixed_yesno_mapper_name']}")
    print(f"fixed_extractor_name: {comparison['fixed_extractor_name']}")
    print(f"sample_count: {comparison['sample_count']}")
    for mapper_name in comparison["mapper_order"]:
        payload = comparison["mappers"][mapper_name]
        relation = payload["relation_span_summary"]
        decision = payload["relation_span_decision_summary"]
        overall = payload["eval_summary"]
        non_relation = payload["non_relation_span_summary"]
        print(
            f"{mapper_name}: "
            f"overall_EM={float(overall.get('avg_exact_match', 0.0)):.4f} "
            f"overall_F1={float(overall.get('avg_f1', 0.0)):.4f} "
            f"relation_count={int(relation.get('count', 0))} "
            f"relation_EM={float(relation.get('avg_exact_match', 0.0)):.4f} "
            f"relation_F1={float(relation.get('avg_f1', 0.0)):.4f} "
            f"relation_failure={float(relation.get('failure_rate', 0.0)):.4f} "
            f"non_relation_EM={float(non_relation.get('avg_exact_match', 0.0)):.4f} "
            f"avg_candidate_spans={float(decision.get('avg_candidate_span_count', 0.0)):.4f} "
            f"fallbacks={decision.get('fallback_target_counts', {})}"
        )
    gain = comparison.get("relation_span_gain_summary", {})
    print(
        "relation_span_gain: "
        f"{gain.get('has_any_relation_span_gain', False)} improved_mappers={gain.get('improved_mappers', [])}"
    )


if __name__ == "__main__":
    main()
