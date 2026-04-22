"""Compare non-privileged HotpotQA yes/no mapper variants."""

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
from answer.hotpotqa_yesno_mapper import YESNO_MAPPER_NAMES, make_yesno_mapper_factory
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
FIXED_EXTRACTOR_NAME = "full_sentence"
FIXED_POLICY_NAME = "sentence_chain"
FIXED_SELECTOR_NAME = "latest_sentence"
FIXED_CANDIDATE_GENERATOR_NAME = "baseline_generator"
DEFAULT_YESNO_MAPPERS = (
    "identity_yesno",
    "sentence_polarity",
    "title_sentence_consistency",
    "abstain_backoff_yesno",
)


def compare_hotpotqa_yesno_mappers(
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
    """Run yes/no mapper variants on one fixed HotpotQA sample list."""

    mapper_names = mappers or list(DEFAULT_YESNO_MAPPERS)
    resolved_indices = resolve_sample_indices(path=path, limit=limit, indices=indices)
    comparison: dict[str, Any] = {
        "reward_mode": reward_mode,
        "fixed_base_mapper_name": FIXED_BASE_MAPPER_NAME,
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
        _validate_yesno_mapper(mapper_name)
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
            yesno_mapper_factory=make_yesno_mapper_factory(mapper_name),
            yesno_mapper_name=mapper_name,
        )
        error_result = analyze_hotpotqa_error_records(eval_result.records)
        answer_type_result = analyze_hotpotqa_answer_types(eval_result.records)
        yesno_decision_summary = summarize_yesno_decisions(eval_result.records)
        comparison["mappers"][mapper_name] = {
            "yesno_mapper_name": mapper_name,
            "base_mapper_name": FIXED_BASE_MAPPER_NAME,
            "extractor_name": FIXED_EXTRACTOR_NAME,
            "eval_summary": eval_result.summary,
            "error_summary": error_result.summary,
            "answer_type_summary": answer_type_result.summary,
            "yesno_summary": _yesno_subset_summary(eval_result.records, answer_type_result.summary),
            "non_yesno_summary": _non_yesno_summary(eval_result.records),
            "yesno_decision_summary": yesno_decision_summary,
        }
        if output_path is not None:
            mapper_dir = output_path / mapper_name
            save_hotpotqa_eval_outputs(eval_result, mapper_dir)
            save_error_analysis_outputs(error_result, mapper_dir)
            save_answer_type_outputs(answer_type_result, mapper_dir)
            (mapper_dir / "hotpotqa_yesno_decision_summary.json").write_text(
                json.dumps(yesno_decision_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    comparison["yesno_gain_summary"] = _yesno_gain_summary(comparison)
    if output_path is not None:
        (output_path / "hotpotqa_yesno_mapper_comparison_summary.json").write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare HotpotQA yes/no mapper variants.")
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
        default=",".join(DEFAULT_YESNO_MAPPERS),
        help=f"Comma-separated yes/no mappers. Available: {', '.join(YESNO_MAPPER_NAMES)}.",
    )
    parser.add_argument("--output-dir", default=None, help="Optional directory for records and summaries.")
    parser.add_argument("--max-steps", type=int, default=4, help="Maximum env steps per example.")
    parser.add_argument("--candidate-top-k", type=int, default=5, help="CandidateGenerator top_k.")
    parser.add_argument("--min-expand-steps", type=int, default=1, help="Fixed sentence_chain min expansion count.")
    args = parser.parse_args()

    comparison = compare_hotpotqa_yesno_mappers(
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
            f"{Path(args.output_dir) / 'hotpotqa_yesno_mapper_comparison_summary.json'}"
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
    """Parse yes/no mapper names."""

    mappers = [item.strip() for item in raw.split(",") if item.strip()]
    for mapper_name in mappers:
        _validate_yesno_mapper(mapper_name)
    return mappers


def summarize_yesno_decisions(records: list[HotpotQAGraphEvalRecord]) -> dict[str, Any]:
    """Aggregate yes/no decision metadata from eval records."""

    yesno_records = [record for record in records if _answer_type(record) == "yes_no"]
    mappings = [_yesno_mapping(record) for record in yesno_records if _yesno_mapping(record)]
    decision_counts = Counter(str(mapping.get("yesno_decision") or "abstain") for mapping in mappings)
    fallback_counts = Counter(str(mapping.get("fallback_target") or "none") for mapping in mappings)
    return {
        "yesno_sample_count": len(yesno_records),
        "mapper_applied_count": len(mappings),
        "decision_counts": dict(sorted(decision_counts.items())),
        "fallback_target_counts": dict(sorted(fallback_counts.items())),
        "avg_evidence_strength": _average(float(mapping.get("evidence_strength", 0.0)) for mapping in mappings),
        "avg_positive_evidence_count": _average(
            float(mapping.get("positive_evidence_count", 0.0)) for mapping in mappings
        ),
        "avg_negative_evidence_count": _average(
            float(mapping.get("negative_evidence_count", 0.0)) for mapping in mappings
        ),
        "conflict_detected_count": sum(1 for mapping in mappings if bool(mapping.get("conflict_detected"))),
        "fallback_occurred_count": sum(1 for mapping in mappings if bool(mapping.get("fallback_occurred"))),
        "records_with_no_mapping": len(yesno_records) - len(mappings),
    }


def _yesno_subset_summary(records: list[HotpotQAGraphEvalRecord], answer_type_summary: dict[str, Any]) -> dict[str, Any]:
    bucket = answer_type_summary.get("answer_type_buckets", {}).get("yes_no", {})
    return {
        "count": int(bucket.get("count", 0)),
        "avg_exact_match": float(bucket.get("avg_exact_match", 0.0)),
        "avg_f1": float(bucket.get("avg_f1", 0.0)),
        "avg_projected_eval_score": float(bucket.get("avg_projected_eval_score", 0.0)),
        "failure_rate": float(bucket.get("failure_rate", 0.0)),
        "sentence_hit": bucket.get("sentence_hit", {}),
        "records": [record.question_id for record in records if _answer_type(record) == "yes_no"],
    }


def _non_yesno_summary(records: list[HotpotQAGraphEvalRecord]) -> dict[str, Any]:
    return summarize_hotpotqa_records([record for record in records if _answer_type(record) != "yes_no"])


def _yesno_gain_summary(comparison: dict[str, Any]) -> dict[str, Any]:
    mappers = comparison.get("mappers", {})
    if "identity_yesno" not in mappers:
        return {"available": False, "reason": "identity_yesno baseline was not run."}
    identity = mappers["identity_yesno"]["yesno_summary"]
    judgments = {}
    improvements: list[str] = []
    for mapper_name, payload in mappers.items():
        if mapper_name == "identity_yesno":
            continue
        yesno = payload["yesno_summary"]
        delta_em = float(yesno.get("avg_exact_match", 0.0)) - float(identity.get("avg_exact_match", 0.0))
        delta_f1 = float(yesno.get("avg_f1", 0.0)) - float(identity.get("avg_f1", 0.0))
        delta_failure = float(yesno.get("failure_rate", 0.0)) - float(identity.get("failure_rate", 0.0))
        improved = delta_em > 1e-12 or delta_f1 > 1e-12 or delta_failure < -1e-12
        if improved:
            improvements.append(mapper_name)
        judgments[mapper_name] = {
            "delta_yesno_exact_match": delta_em,
            "delta_yesno_f1": delta_f1,
            "delta_yesno_failure_rate": delta_failure,
            "improves_over_identity": improved,
        }
    return {
        "available": True,
        "baseline_mapper": "identity_yesno",
        "improved_mappers": improvements,
        "has_any_yesno_gain": bool(improvements),
        "mapper_judgments": judgments,
    }


def _validate_yesno_mapper(mapper_name: str) -> None:
    if mapper_name not in YESNO_MAPPER_NAMES:
        raise ValueError(
            f"Unknown yes/no mapper '{mapper_name}'. Available mappers: {', '.join(YESNO_MAPPER_NAMES)}"
        )


def _answer_type(record: HotpotQAGraphEvalRecord) -> str:
    label = record.metadata.get("answer_type_label")
    if label == "yes_no":
        return "yes_no"
    return classify_hotpotqa_answer_type(record.gold_answer)


def _yesno_mapping(record: HotpotQAGraphEvalRecord) -> dict[str, Any]:
    mapping = record.metadata.get("yesno_mapping")
    return mapping if isinstance(mapping, dict) else {}


def _average(values: Any) -> float:
    items = list(values)
    return 0.0 if not items else sum(items) / len(items)


def _print_comparison(comparison: dict[str, Any]) -> None:
    print("=== HotpotQA Yes/No Mapper Comparison ===")
    print(f"reward_mode: {comparison['reward_mode']}")
    print(f"fixed_base_mapper_name: {comparison['fixed_base_mapper_name']}")
    print(f"fixed_extractor_name: {comparison['fixed_extractor_name']}")
    print(f"fixed_policy_name: {comparison['fixed_policy_name']}")
    print(f"fixed_selector_name: {comparison['fixed_selector_name']}")
    print(f"sample_count: {comparison['sample_count']}")
    for mapper_name in comparison["mapper_order"]:
        payload = comparison["mappers"][mapper_name]
        yesno = payload["yesno_summary"]
        decision = payload["yesno_decision_summary"]
        overall = payload["eval_summary"]
        non_yesno = payload["non_yesno_summary"]
        print(
            f"{mapper_name}: "
            f"overall_EM={float(overall.get('avg_exact_match', 0.0)):.4f} "
            f"overall_F1={float(overall.get('avg_f1', 0.0)):.4f} "
            f"yesno_count={int(yesno.get('count', 0))} "
            f"yesno_EM={float(yesno.get('avg_exact_match', 0.0)):.4f} "
            f"yesno_F1={float(yesno.get('avg_f1', 0.0)):.4f} "
            f"yesno_failure={float(yesno.get('failure_rate', 0.0)):.4f} "
            f"non_yesno_EM={float(non_yesno.get('avg_exact_match', 0.0)):.4f} "
            f"avg_evidence={float(decision.get('avg_evidence_strength', 0.0)):.4f} "
            f"decisions={decision.get('decision_counts', {})}"
        )
    gain = comparison.get("yesno_gain_summary", {})
    print(f"yesno_gain: {gain.get('has_any_yesno_gain', False)} improved_mappers={gain.get('improved_mappers', [])}")


if __name__ == "__main__":
    main()
