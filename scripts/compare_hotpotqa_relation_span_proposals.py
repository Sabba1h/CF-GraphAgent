"""Compare HotpotQA relation/span proposal mechanism variants."""

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
from answer.hotpotqa_relation_span_discovery import make_relation_span_discovery_factory
from answer.hotpotqa_relation_span_mapper import make_relation_span_mapper_factory
from answer.hotpotqa_relation_span_proposal import (
    RELATION_SPAN_PROPOSAL_NAMES,
    make_relation_span_proposal_factory,
)
from answer.hotpotqa_relation_span_ranker import make_relation_span_ranker_factory
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
FIXED_RELATION_SPAN_DISCOVERY_NAME = "baseline_pattern_discovery"
FIXED_RELATION_SPAN_RANKER_NAME = "pattern_priority"
FIXED_EXTRACTOR_NAME = "full_sentence"
FIXED_POLICY_NAME = "sentence_chain"
FIXED_SELECTOR_NAME = "latest_sentence"
FIXED_CANDIDATE_GENERATOR_NAME = "baseline_generator"
RELATION_SPAN_TYPE = "descriptive_span_or_relation"
DEFAULT_RELATION_SPAN_PROPOSALS = (
    "baseline_discovery_proposal",
    "query_conditioned_proposal",
    "constrained_local_proposal",
    "query_conditioned_plus_constrained_proposal",
)


def compare_hotpotqa_relation_span_proposals(
    *,
    path: str | Path,
    split: str | None = None,
    limit: int | None = None,
    indices: list[int] | None = None,
    reward_mode: str = "baseline",
    proposals: list[str] | None = None,
    max_steps: int = 4,
    candidate_top_k: int = 5,
    min_expand_steps: int = 1,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run relation/span proposal variants over one fixed HotpotQA subset."""

    proposal_names = proposals or list(DEFAULT_RELATION_SPAN_PROPOSALS)
    resolved_indices = resolve_sample_indices(path=path, limit=limit, indices=indices)
    comparison: dict[str, Any] = {
        "reward_mode": reward_mode,
        "fixed_base_mapper_name": FIXED_BASE_MAPPER_NAME,
        "fixed_yesno_mapper_name": FIXED_YESNO_MAPPER_NAME,
        "fixed_relation_span_mapper_name": FIXED_RELATION_SPAN_MAPPER_NAME,
        "fixed_relation_span_discovery_name": FIXED_RELATION_SPAN_DISCOVERY_NAME,
        "fixed_relation_span_ranker_name": FIXED_RELATION_SPAN_RANKER_NAME,
        "fixed_extractor_name": FIXED_EXTRACTOR_NAME,
        "fixed_policy_name": FIXED_POLICY_NAME,
        "fixed_selector_name": FIXED_SELECTOR_NAME,
        "fixed_candidate_generator_name": FIXED_CANDIDATE_GENERATOR_NAME,
        "sample_indices": resolved_indices,
        "sample_count": len(resolved_indices),
        "proposal_order": proposal_names,
        "proposals": {},
    }
    output_path = Path(output_dir) if output_dir else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    for proposal_name in proposal_names:
        _validate_relation_span_proposal(proposal_name)
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
            relation_span_discovery_factory=make_relation_span_discovery_factory(FIXED_RELATION_SPAN_DISCOVERY_NAME),
            relation_span_discovery_name=FIXED_RELATION_SPAN_DISCOVERY_NAME,
            relation_span_ranker_factory=make_relation_span_ranker_factory(FIXED_RELATION_SPAN_RANKER_NAME),
            relation_span_ranker_name=FIXED_RELATION_SPAN_RANKER_NAME,
            relation_span_proposal_factory=make_relation_span_proposal_factory(proposal_name),
            relation_span_proposal_name=proposal_name,
        )
        error_result = analyze_hotpotqa_error_records(eval_result.records)
        answer_type_result = analyze_hotpotqa_answer_types(eval_result.records)
        relation_summary = _relation_span_subset_summary(eval_result.records, answer_type_result.summary)
        proposal_summary = summarize_relation_span_proposals(eval_result.records)
        comparison["proposals"][proposal_name] = {
            "relation_span_proposal_name": proposal_name,
            "relation_span_mapper_name": FIXED_RELATION_SPAN_MAPPER_NAME,
            "relation_span_discovery_name": FIXED_RELATION_SPAN_DISCOVERY_NAME,
            "relation_span_ranker_name": FIXED_RELATION_SPAN_RANKER_NAME,
            "base_mapper_name": FIXED_BASE_MAPPER_NAME,
            "yesno_mapper_name": FIXED_YESNO_MAPPER_NAME,
            "extractor_name": FIXED_EXTRACTOR_NAME,
            "eval_summary": eval_result.summary,
            "error_summary": error_result.summary,
            "answer_type_summary": answer_type_result.summary,
            "relation_span_summary": relation_summary,
            "non_relation_span_summary": _non_relation_span_summary(eval_result.records),
            "relation_span_proposal_summary": proposal_summary,
        }
        if output_path is not None:
            proposal_dir = output_path / proposal_name
            save_hotpotqa_eval_outputs(eval_result, proposal_dir)
            save_error_analysis_outputs(error_result, proposal_dir)
            save_answer_type_outputs(answer_type_result, proposal_dir)
            (proposal_dir / "hotpotqa_relation_span_proposal_summary.json").write_text(
                json.dumps(proposal_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    comparison["relation_span_proposal_gain_summary"] = _relation_span_proposal_gain_summary(comparison)
    if output_path is not None:
        (output_path / "hotpotqa_relation_span_proposal_comparison_summary.json").write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare HotpotQA relation/span proposal mechanism variants.")
    parser.add_argument("--path", required=True, help="Path to a HotpotQA JSON or JSONL file.")
    parser.add_argument("--split", default=None, help="Optional split label stored in metadata.")
    parser.add_argument("--limit", type=int, default=200, help="Maximum examples to evaluate.")
    parser.add_argument("--indices", default=None, help="Comma-separated record indices to evaluate.")
    parser.add_argument(
        "--reward-mode",
        choices=["baseline", "oracle_counterfactual"],
        default="baseline",
        help="Reward mode used for all proposal variants.",
    )
    parser.add_argument(
        "--proposals",
        default=",".join(DEFAULT_RELATION_SPAN_PROPOSALS),
        help=f"Comma-separated proposal variants. Available: {', '.join(RELATION_SPAN_PROPOSAL_NAMES)}.",
    )
    parser.add_argument("--output-dir", default=None, help="Optional directory for records and summaries.")
    parser.add_argument("--max-steps", type=int, default=4, help="Maximum env steps per example.")
    parser.add_argument("--candidate-top-k", type=int, default=5, help="CandidateGenerator top_k.")
    parser.add_argument("--min-expand-steps", type=int, default=1, help="Fixed sentence_chain min expansion count.")
    args = parser.parse_args()

    comparison = compare_hotpotqa_relation_span_proposals(
        path=args.path,
        split=args.split,
        limit=args.limit,
        indices=parse_indices(args.indices),
        reward_mode=args.reward_mode,
        proposals=parse_proposals(args.proposals),
        max_steps=args.max_steps,
        candidate_top_k=args.candidate_top_k,
        min_expand_steps=args.min_expand_steps,
        output_dir=args.output_dir,
    )
    _print_comparison(comparison)
    if args.output_dir:
        print(
            "comparison_summary_path: "
            f"{Path(args.output_dir) / 'hotpotqa_relation_span_proposal_comparison_summary.json'}"
        )


def resolve_sample_indices(*, path: str | Path, limit: int | None, indices: list[int] | None) -> list[int]:
    """Resolve one explicit sample index list shared by every proposal variant."""

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


def parse_proposals(raw: str) -> list[str]:
    """Parse relation/span proposal names."""

    proposals = [item.strip() for item in raw.split(",") if item.strip()]
    for proposal_name in proposals:
        _validate_relation_span_proposal(proposal_name)
    return proposals


def summarize_relation_span_proposals(records: list[HotpotQAGraphEvalRecord]) -> dict[str, Any]:
    """Aggregate relation/span proposal metadata and offline useful-span diagnostics."""

    relation_records = [record for record in records if _answer_type(record) == RELATION_SPAN_TYPE]
    mappings = [(_relation_span_mapping(record), record) for record in relation_records]
    mappings = [(mapping, record) for mapping, record in mappings if mapping]
    proposal_payloads = [_proposal_payload(mapping) for mapping, _ in mappings]
    exact_counts = [_useful_candidate_count(mapping.get("candidate_spans", []) or [], record.gold_answer) for mapping, record in mappings]
    partial_counts = [_partial_candidate_count(mapping.get("candidate_spans", []) or [], record.gold_answer) for mapping, record in mappings]
    failure_buckets = Counter(_proposal_failure_bucket(mapping, record) for mapping, record in mappings)
    direct_success_count = sum(
        1
        for mapping, record in mappings
        if not bool(mapping.get("fallback_occurred")) and float(record.projected_eval_score) > 0.0
    )
    fallback_recovery_count = sum(
        1
        for mapping, record in mappings
        if bool(mapping.get("fallback_occurred")) and float(record.projected_eval_score) > 0.0
    )
    relation_count = len(relation_records)
    return {
        "relation_span_sample_count": relation_count,
        "proposal_applied_count": sum(1 for payload in proposal_payloads if payload),
        "avg_candidate_count": _average(len(mapping.get("candidate_spans", []) or []) for mapping, _ in mappings),
        "avg_useful_candidate_count": _average(exact_counts),
        "exact_useful_candidate_count": int(sum(exact_counts)),
        "partial_useful_candidate_count": int(sum(partial_counts)),
        "useful_candidate_coverage": _rate(sum(1 for count in exact_counts if count > 0), relation_count),
        "partial_useful_candidate_coverage": _rate(sum(1 for count in partial_counts if count > 0), relation_count),
        "proposal_direct_success_count": int(direct_success_count),
        "fallback_recovery_count": int(fallback_recovery_count),
        "proposal_direct_success_rate": _rate(direct_success_count, relation_count),
        "fallback_recovery_rate": _rate(fallback_recovery_count, relation_count),
        "no_useful_candidate_span_found_count": int(failure_buckets.get("no_useful_candidate_span_found", 0)),
        "no_useful_candidate_span_found_rate": _rate(
            int(failure_buckets.get("no_useful_candidate_span_found", 0)),
            relation_count,
        ),
        "failure_bucket_counts": dict(sorted(failure_buckets.items())),
        "avg_from_selected_sentence_count": _average(
            int((_proposal_payload(mapping).get("metadata", {}) or {}).get("from_selected_sentence_count", 0))
            for mapping, _ in mappings
        ),
        "avg_from_path_touched_sentence_count": _average(
            int((_proposal_payload(mapping).get("metadata", {}) or {}).get("from_path_touched_sentence_count", 0))
            for mapping, _ in mappings
        ),
        "records_with_no_mapping": relation_count - len(mappings),
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


def _relation_span_proposal_gain_summary(comparison: dict[str, Any]) -> dict[str, Any]:
    proposals = comparison.get("proposals", {})
    if "baseline_discovery_proposal" not in proposals:
        return {"available": False, "reason": "baseline_discovery_proposal was not run."}
    baseline = proposals["baseline_discovery_proposal"]["relation_span_summary"]
    baseline_proposal = proposals["baseline_discovery_proposal"]["relation_span_proposal_summary"]
    judgments = {}
    improvements: list[str] = []
    for proposal_name, payload in proposals.items():
        if proposal_name == "baseline_discovery_proposal":
            continue
        relation = payload["relation_span_summary"]
        proposal_summary = payload["relation_span_proposal_summary"]
        delta_em = float(relation["avg_exact_match"]) - float(baseline["avg_exact_match"])
        delta_f1 = float(relation["avg_f1"]) - float(baseline["avg_f1"])
        delta_failure = float(relation["failure_rate"]) - float(baseline["failure_rate"])
        delta_no_useful = float(proposal_summary["no_useful_candidate_span_found_rate"]) - float(
            baseline_proposal["no_useful_candidate_span_found_rate"]
        )
        delta_useful_coverage = float(proposal_summary["useful_candidate_coverage"]) - float(
            baseline_proposal["useful_candidate_coverage"]
        )
        improved = (
            delta_em > 0.0
            or delta_f1 > 0.0
            or delta_failure < 0.0
            or delta_no_useful < 0.0
            or delta_useful_coverage > 0.0
        )
        if improved:
            improvements.append(proposal_name)
        judgments[proposal_name] = {
            "delta_relation_span_exact_match": delta_em,
            "delta_relation_span_f1": delta_f1,
            "delta_relation_span_failure_rate": delta_failure,
            "delta_no_useful_candidate_span_found_rate": delta_no_useful,
            "delta_useful_candidate_coverage": delta_useful_coverage,
            "improved_over_baseline_discovery_proposal": improved,
        }
    return {
        "available": True,
        "baseline_proposal": "baseline_discovery_proposal",
        "improved_proposals": improvements,
        "has_any_improvement": bool(improvements),
        "judgments": judgments,
    }


def _proposal_failure_bucket(mapping: dict[str, Any], record: HotpotQAGraphEvalRecord) -> str:
    candidate_spans = mapping.get("candidate_spans", []) or []
    useful_exists = _useful_candidate_count(candidate_spans, record.gold_answer) > 0
    recovered = float(record.projected_eval_score) > 0.0
    if not useful_exists:
        return "no_useful_candidate_span_found"
    if recovered and not bool(mapping.get("fallback_occurred")):
        return "proposal_direct_success"
    if recovered and bool(mapping.get("fallback_occurred")):
        return "fallback_recovery_success"
    return "useful_candidate_exists_but_not_recovered"


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


def _proposal_payload(mapping: dict[str, Any]) -> dict[str, Any]:
    metadata = mapping.get("metadata") if isinstance(mapping.get("metadata"), dict) else {}
    proposal = metadata.get("relation_span_proposal")
    return proposal if isinstance(proposal, dict) else {}


def _validate_relation_span_proposal(proposal_name: str) -> None:
    if proposal_name not in RELATION_SPAN_PROPOSAL_NAMES:
        raise ValueError(
            f"Unknown relation/span proposal '{proposal_name}'. "
            f"Available proposals: {', '.join(RELATION_SPAN_PROPOSAL_NAMES)}"
        )


def _average(values: Any) -> float:
    value_list = list(values)
    return float(sum(value_list) / len(value_list)) if value_list else 0.0


def _rate(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def _print_comparison(comparison: dict[str, Any]) -> None:
    print("=== HotpotQA Relation/Span Proposal Comparison ===")
    print(f"reward_mode: {comparison['reward_mode']}")
    print(f"fixed_base_mapper_name: {comparison['fixed_base_mapper_name']}")
    print(f"fixed_yesno_mapper_name: {comparison['fixed_yesno_mapper_name']}")
    print(f"fixed_relation_span_mapper_name: {comparison['fixed_relation_span_mapper_name']}")
    print(f"fixed_relation_span_discovery_name: {comparison['fixed_relation_span_discovery_name']}")
    print(f"fixed_relation_span_ranker_name: {comparison['fixed_relation_span_ranker_name']}")
    print(f"sample_count: {comparison['sample_count']}")
    for proposal_name in comparison["proposal_order"]:
        payload = comparison["proposals"][proposal_name]
        eval_summary = payload["eval_summary"]
        relation = payload["relation_span_summary"]
        proposal = payload["relation_span_proposal_summary"]
        print(
            f"{proposal_name}: "
            f"overall_EM={eval_summary['avg_exact_match']:.4f} "
            f"overall_F1={eval_summary['avg_f1']:.4f} "
            f"relation_count={relation['count']} "
            f"relation_EM={relation['avg_exact_match']:.4f} "
            f"relation_F1={relation['avg_f1']:.4f} "
            f"relation_failure={relation['failure_rate']:.4f} "
            f"no_useful_rate={proposal['no_useful_candidate_span_found_rate']:.4f} "
            f"useful_coverage={proposal['useful_candidate_coverage']:.4f} "
            f"direct_success={proposal['proposal_direct_success_count']} "
            f"fallback_recovery={proposal['fallback_recovery_count']}"
        )
    gain = comparison.get("relation_span_proposal_gain_summary", {})
    print(
        "relation_span_proposal_gain: "
        f"{gain.get('has_any_improvement')} improved_proposals={gain.get('improved_proposals')}"
    )


if __name__ == "__main__":
    main()
