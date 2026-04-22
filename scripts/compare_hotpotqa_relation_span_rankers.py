"""Compare HotpotQA relation/span candidate span rankers."""

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
from answer.hotpotqa_relation_span_mapper import make_relation_span_mapper_factory
from answer.hotpotqa_relation_span_ranker import (
    RELATION_SPAN_RANKER_NAMES,
    make_relation_span_ranker_factory,
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
from evaluation.hotpotqa_metrics import exact_match, summarize_hotpotqa_records, token_f1
from evaluation.hotpotqa_subset_evaluator import (
    HotpotQAGraphEvalRecord,
    evaluate_hotpotqa_graph_subset,
    save_hotpotqa_eval_outputs,
)
from graph.hotpotqa_policy_variants import make_policy_factory

FIXED_BASE_MAPPER_NAME = "parent_title"
FIXED_YESNO_MAPPER_NAME = "sentence_polarity"
FIXED_RELATION_SPAN_MAPPER_NAME = "pattern_span"
FIXED_EXTRACTOR_NAME = "full_sentence"
FIXED_POLICY_NAME = "sentence_chain"
FIXED_SELECTOR_NAME = "latest_sentence"
FIXED_CANDIDATE_GENERATOR_NAME = "baseline_generator"
RELATION_SPAN_TYPE = "descriptive_span_or_relation"
DEFAULT_RELATION_SPAN_RANKERS = (
    "first_candidate",
    "shortest_nonempty",
    "pattern_priority",
    "shortest_then_pattern_priority",
)


def compare_hotpotqa_relation_span_rankers(
    *,
    path: str | Path,
    split: str | None = None,
    limit: int | None = None,
    indices: list[int] | None = None,
    reward_mode: str = "baseline",
    rankers: list[str] | None = None,
    max_steps: int = 4,
    candidate_top_k: int = 5,
    min_expand_steps: int = 1,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run relation/span ranker variants on one fixed HotpotQA sample list."""

    ranker_names = rankers or list(DEFAULT_RELATION_SPAN_RANKERS)
    resolved_indices = resolve_sample_indices(path=path, limit=limit, indices=indices)
    comparison: dict[str, Any] = {
        "reward_mode": reward_mode,
        "fixed_base_mapper_name": FIXED_BASE_MAPPER_NAME,
        "fixed_yesno_mapper_name": FIXED_YESNO_MAPPER_NAME,
        "fixed_relation_span_mapper_name": FIXED_RELATION_SPAN_MAPPER_NAME,
        "fixed_extractor_name": FIXED_EXTRACTOR_NAME,
        "fixed_policy_name": FIXED_POLICY_NAME,
        "fixed_selector_name": FIXED_SELECTOR_NAME,
        "fixed_candidate_generator_name": FIXED_CANDIDATE_GENERATOR_NAME,
        "sample_indices": resolved_indices,
        "sample_count": len(resolved_indices),
        "ranker_order": ranker_names,
        "rankers": {},
    }
    output_path = Path(output_dir) if output_dir else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    for ranker_name in ranker_names:
        _validate_relation_span_ranker(ranker_name)
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
            relation_span_mapper_factory=make_relation_span_mapper_factory(FIXED_RELATION_SPAN_MAPPER_NAME),
            relation_span_mapper_name=FIXED_RELATION_SPAN_MAPPER_NAME,
            relation_span_ranker_factory=make_relation_span_ranker_factory(ranker_name),
            relation_span_ranker_name=ranker_name,
        )
        error_result = analyze_hotpotqa_error_records(eval_result.records)
        answer_type_result = analyze_hotpotqa_answer_types(eval_result.records)
        relation_summary = _relation_span_subset_summary(eval_result.records, answer_type_result.summary)
        ranking_summary = summarize_relation_span_rankings(eval_result.records)
        comparison["rankers"][ranker_name] = {
            "relation_span_ranker_name": ranker_name,
            "relation_span_mapper_name": FIXED_RELATION_SPAN_MAPPER_NAME,
            "base_mapper_name": FIXED_BASE_MAPPER_NAME,
            "yesno_mapper_name": FIXED_YESNO_MAPPER_NAME,
            "extractor_name": FIXED_EXTRACTOR_NAME,
            "eval_summary": eval_result.summary,
            "error_summary": error_result.summary,
            "answer_type_summary": answer_type_result.summary,
            "relation_span_summary": relation_summary,
            "non_relation_span_summary": _non_relation_span_summary(eval_result.records),
            "relation_span_ranking_summary": ranking_summary,
        }
        if output_path is not None:
            ranker_dir = output_path / ranker_name
            save_hotpotqa_eval_outputs(eval_result, ranker_dir)
            save_error_analysis_outputs(error_result, ranker_dir)
            save_answer_type_outputs(answer_type_result, ranker_dir)
            (ranker_dir / "hotpotqa_relation_span_ranking_summary.json").write_text(
                json.dumps(ranking_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    comparison["relation_span_ranker_gain_summary"] = _relation_span_ranker_gain_summary(comparison)
    if output_path is not None:
        (output_path / "hotpotqa_relation_span_ranker_comparison_summary.json").write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare HotpotQA relation/span candidate span rankers.")
    parser.add_argument("--path", required=True, help="Path to a HotpotQA JSON or JSONL file.")
    parser.add_argument("--split", default=None, help="Optional split label stored in metadata.")
    parser.add_argument("--limit", type=int, default=200, help="Maximum examples to evaluate.")
    parser.add_argument("--indices", default=None, help="Comma-separated record indices to evaluate.")
    parser.add_argument(
        "--reward-mode",
        choices=["baseline", "oracle_counterfactual"],
        default="baseline",
        help="Reward mode used for all rankers.",
    )
    parser.add_argument(
        "--rankers",
        default=",".join(DEFAULT_RELATION_SPAN_RANKERS),
        help=f"Comma-separated relation/span rankers. Available: {', '.join(RELATION_SPAN_RANKER_NAMES)}.",
    )
    parser.add_argument("--output-dir", default=None, help="Optional directory for records and summaries.")
    parser.add_argument("--max-steps", type=int, default=4, help="Maximum env steps per example.")
    parser.add_argument("--candidate-top-k", type=int, default=5, help="CandidateGenerator top_k.")
    parser.add_argument("--min-expand-steps", type=int, default=1, help="Fixed sentence_chain min expansion count.")
    args = parser.parse_args()

    comparison = compare_hotpotqa_relation_span_rankers(
        path=args.path,
        split=args.split,
        limit=args.limit,
        indices=parse_indices(args.indices),
        reward_mode=args.reward_mode,
        rankers=parse_rankers(args.rankers),
        max_steps=args.max_steps,
        candidate_top_k=args.candidate_top_k,
        min_expand_steps=args.min_expand_steps,
        output_dir=args.output_dir,
    )
    _print_comparison(comparison)
    if args.output_dir:
        print(
            "comparison_summary_path: "
            f"{Path(args.output_dir) / 'hotpotqa_relation_span_ranker_comparison_summary.json'}"
        )


def resolve_sample_indices(*, path: str | Path, limit: int | None, indices: list[int] | None) -> list[int]:
    """Resolve one explicit sample index list shared by every ranker."""

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


def parse_rankers(raw: str) -> list[str]:
    """Parse relation/span ranker names."""

    rankers = [item.strip() for item in raw.split(",") if item.strip()]
    for ranker_name in rankers:
        _validate_relation_span_ranker(ranker_name)
    return rankers


def summarize_relation_span_rankings(records: list[HotpotQAGraphEvalRecord]) -> dict[str, Any]:
    """Aggregate relation/span ranking metadata and offline candidate diagnostics."""

    relation_records = [record for record in records if _answer_type(record) == RELATION_SPAN_TYPE]
    mappings = [(_relation_span_mapping(record), record) for record in relation_records]
    mappings = [(mapping, record) for mapping, record in mappings if mapping]
    fallback_counts = Counter(str(mapping.get("fallback_target") or "none") for mapping, _ in mappings)
    reason_counts = Counter(str(mapping.get("selected_span_reason") or "none") for mapping, _ in mappings)
    rank_counts = Counter(str(_ranking_payload(mapping).get("selected_span_rank")) for mapping, _ in mappings)
    failure_buckets = Counter(_ranking_failure_bucket(mapping, record) for mapping, record in mappings)
    exact_candidate_counts = [
        _useful_candidate_count(mapping.get("candidate_spans", []) or [], record.gold_answer)
        for mapping, record in mappings
    ]
    partial_candidate_counts = [
        _partial_candidate_count(mapping.get("candidate_spans", []) or [], record.gold_answer)
        for mapping, record in mappings
    ]
    return {
        "relation_span_sample_count": len(relation_records),
        "ranker_applied_count": sum(1 for mapping, _ in mappings if _ranking_payload(mapping)),
        "avg_candidate_span_count": _average(len(mapping.get("candidate_spans", []) or []) for mapping, _ in mappings),
        "avg_exact_match_candidate_count": _average(exact_candidate_counts),
        "avg_partial_match_candidate_count": _average(partial_candidate_counts),
        "fallback_target_counts": dict(sorted(fallback_counts.items())),
        "selected_span_reason_counts": dict(sorted(reason_counts.items())),
        "selected_span_rank_counts": dict(sorted(rank_counts.items())),
        "fallback_occurred_count": sum(1 for mapping, _ in mappings if bool(mapping.get("fallback_occurred"))),
        "failure_bucket_counts": dict(sorted(failure_buckets.items())),
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


def _relation_span_ranker_gain_summary(comparison: dict[str, Any]) -> dict[str, Any]:
    rankers = comparison.get("rankers", {})
    if "first_candidate" not in rankers:
        return {"available": False, "reason": "first_candidate baseline ranker was not run."}
    baseline = rankers["first_candidate"]["relation_span_summary"]
    judgments = {}
    improvements: list[str] = []
    for ranker_name, payload in rankers.items():
        if ranker_name == "first_candidate":
            continue
        relation = payload["relation_span_summary"]
        delta_em = float(relation["avg_exact_match"]) - float(baseline["avg_exact_match"])
        delta_f1 = float(relation["avg_f1"]) - float(baseline["avg_f1"])
        delta_failure = float(relation["failure_rate"]) - float(baseline["failure_rate"])
        improved = delta_em > 0.0 or delta_f1 > 0.0 or delta_failure < 0.0
        if improved:
            improvements.append(ranker_name)
        judgments[ranker_name] = {
            "delta_relation_span_exact_match": delta_em,
            "delta_relation_span_f1": delta_f1,
            "delta_relation_span_failure_rate": delta_failure,
            "improved_over_first_candidate": improved,
        }
    return {
        "available": True,
        "baseline_ranker": "first_candidate",
        "improved_rankers": improvements,
        "has_any_improvement": bool(improvements),
        "judgments": judgments,
    }


def _ranking_failure_bucket(mapping: dict[str, Any], record: HotpotQAGraphEvalRecord) -> str:
    candidate_spans = mapping.get("candidate_spans", []) or []
    selected_span = str(mapping.get("selected_span") or mapping.get("mapped_answer") or "")
    useful_exists = _useful_candidate_count(candidate_spans, record.gold_answer) > 0
    selected_exact = exact_match(selected_span, record.gold_answer) >= 1.0
    if selected_exact:
        return "useful_candidate_selected"
    if useful_exists:
        return "useful_candidate_exists_but_selected_wrong_span"
    return "no_useful_candidate_span_found"


def _useful_candidate_count(candidate_spans: list[dict[str, Any]], gold_answer: str) -> int:
    return sum(1 for span in candidate_spans if exact_match(str(span.get("text") or ""), gold_answer) >= 1.0)


def _partial_candidate_count(candidate_spans: list[dict[str, Any]], gold_answer: str) -> int:
    return sum(1 for span in candidate_spans if token_f1(str(span.get("text") or ""), gold_answer) > 0.0)


def _answer_type(record: HotpotQAGraphEvalRecord) -> str:
    label = record.metadata.get("answer_type_label")
    return str(label or classify_hotpotqa_answer_type(record.gold_answer))


def _relation_span_mapping(record: HotpotQAGraphEvalRecord) -> dict[str, Any]:
    mapping = record.metadata.get("relation_span_mapping")
    return mapping if isinstance(mapping, dict) else {}


def _ranking_payload(mapping: dict[str, Any]) -> dict[str, Any]:
    metadata = mapping.get("metadata") if isinstance(mapping.get("metadata"), dict) else {}
    ranking = metadata.get("relation_span_ranking")
    return ranking if isinstance(ranking, dict) else {}


def _validate_relation_span_ranker(ranker_name: str) -> None:
    if ranker_name not in RELATION_SPAN_RANKER_NAMES:
        raise ValueError(
            f"Unknown relation/span ranker '{ranker_name}'. "
            f"Available rankers: {', '.join(RELATION_SPAN_RANKER_NAMES)}"
        )


def _average(values: Any) -> float:
    value_list = list(values)
    return float(sum(value_list) / len(value_list)) if value_list else 0.0


def _print_comparison(comparison: dict[str, Any]) -> None:
    print("=== HotpotQA Relation/Span Ranker Comparison ===")
    print(f"reward_mode: {comparison['reward_mode']}")
    print(f"fixed_base_mapper_name: {comparison['fixed_base_mapper_name']}")
    print(f"fixed_yesno_mapper_name: {comparison['fixed_yesno_mapper_name']}")
    print(f"fixed_relation_span_mapper_name: {comparison['fixed_relation_span_mapper_name']}")
    print(f"fixed_extractor_name: {comparison['fixed_extractor_name']}")
    print(f"sample_count: {comparison['sample_count']}")
    for ranker_name in comparison["ranker_order"]:
        payload = comparison["rankers"][ranker_name]
        eval_summary = payload["eval_summary"]
        relation = payload["relation_span_summary"]
        ranking = payload["relation_span_ranking_summary"]
        print(
            f"{ranker_name}: "
            f"overall_EM={eval_summary['avg_exact_match']:.4f} "
            f"overall_F1={eval_summary['avg_f1']:.4f} "
            f"relation_count={relation['count']} "
            f"relation_EM={relation['avg_exact_match']:.4f} "
            f"relation_F1={relation['avg_f1']:.4f} "
            f"relation_failure={relation['failure_rate']:.4f} "
            f"avg_candidates={ranking['avg_candidate_span_count']:.4f} "
            f"failure_buckets={ranking['failure_bucket_counts']}"
        )
    gain = comparison.get("relation_span_ranker_gain_summary", {})
    print(
        "relation_span_ranker_gain: "
        f"{gain.get('has_any_improvement')} improved_rankers={gain.get('improved_rankers')}"
    )


if __name__ == "__main__":
    main()
