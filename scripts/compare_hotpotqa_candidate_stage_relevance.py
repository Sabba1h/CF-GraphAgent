"""Compare HotpotQA candidate-stage question-conditioned relevance variants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from answer.hotpotqa_answer_selector import make_answer_selector_factory
from answer.hotpotqa_entity_title_mapper import make_entity_title_mapper_factory
from candidates.hotpotqa_question_conditioned_pruner import (
    CANDIDATE_STAGE_VARIANTS,
    make_candidate_generator_factory,
    summarize_candidate_stage_from_records,
)
from data.benchmarks import parse_indices
from data.benchmarks.common import load_json_records
from evaluation.hotpotqa_answer_type_diagnostic import analyze_hotpotqa_answer_types, save_answer_type_outputs
from evaluation.hotpotqa_error_analysis import analyze_hotpotqa_error_records, save_error_analysis_outputs
from evaluation.hotpotqa_parent_title_analysis import analyze_parent_title_mapper, save_parent_title_analysis_outputs
from evaluation.hotpotqa_sentence_hit_diagnostic import analyze_hotpotqa_sentence_hits, save_sentence_hit_outputs
from evaluation.hotpotqa_subset_evaluator import (
    HotpotQASubsetEvalResult,
    evaluate_hotpotqa_graph_subset,
    save_hotpotqa_eval_outputs,
)
from graph.hotpotqa_policy_variants import make_policy_factory

FIXED_MAPPER_NAME = "parent_title"
FIXED_EXTRACTOR_NAME = "full_sentence"
FIXED_POLICY_NAME = "sentence_chain"
FIXED_SELECTOR_NAME = "latest_sentence"
DEFAULT_SCORER_NAME = "title_sentence_hybrid"
DEFAULT_VARIANTS = (
    "baseline_generator",
    "overlap_pruned_generator",
    "overlap_ranked_generator",
    "hybrid_prune_then_rank_generator",
)


def compare_hotpotqa_candidate_stage_relevance(
    *,
    path: str | Path,
    split: str | None = None,
    limit: int | None = None,
    indices: list[int] | None = None,
    reward_mode: str = "baseline",
    variants: list[str] | None = None,
    scorer_name: str = DEFAULT_SCORER_NAME,
    max_steps: int = 4,
    candidate_top_k: int = 5,
    candidate_pool_k: int | None = None,
    prune_threshold: float = 0.0,
    min_keep: int = 1,
    min_expand_steps: int = 1,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run candidate-stage variants on one fixed HotpotQA sample list."""

    variant_names = variants or list(DEFAULT_VARIANTS)
    resolved_indices = resolve_sample_indices(path=path, limit=limit, indices=indices)
    comparison: dict[str, Any] = {
        "reward_mode": reward_mode,
        "fixed_mapper_name": FIXED_MAPPER_NAME,
        "fixed_extractor_name": FIXED_EXTRACTOR_NAME,
        "fixed_policy_name": FIXED_POLICY_NAME,
        "fixed_selector_name": FIXED_SELECTOR_NAME,
        "scorer_name": scorer_name,
        "candidate_top_k": candidate_top_k,
        "candidate_pool_k": candidate_pool_k or max(candidate_top_k, 25),
        "sample_indices": resolved_indices,
        "sample_count": len(resolved_indices),
        "variant_order": variant_names,
        "variants": {},
    }
    output_path = Path(output_dir) if output_dir else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    for variant_name in variant_names:
        _validate_variant(variant_name)
        parent_eval = _run_mapper_eval(
            path=path,
            split=split,
            indices=resolved_indices,
            reward_mode=reward_mode,
            mapper_name=FIXED_MAPPER_NAME,
            variant_name=variant_name,
            scorer_name=scorer_name,
            max_steps=max_steps,
            candidate_top_k=candidate_top_k,
            candidate_pool_k=candidate_pool_k,
            prune_threshold=prune_threshold,
            min_keep=min_keep,
            min_expand_steps=min_expand_steps,
        )
        identity_eval = _run_mapper_eval(
            path=path,
            split=split,
            indices=resolved_indices,
            reward_mode=reward_mode,
            mapper_name="identity",
            variant_name=variant_name,
            scorer_name=scorer_name,
            max_steps=max_steps,
            candidate_top_k=candidate_top_k,
            candidate_pool_k=candidate_pool_k,
            prune_threshold=prune_threshold,
            min_keep=min_keep,
            min_expand_steps=min_expand_steps,
        )
        error_result = analyze_hotpotqa_error_records(parent_eval.records)
        sentence_hit_result = analyze_hotpotqa_sentence_hits(parent_eval.records)
        answer_type_result = analyze_hotpotqa_answer_types(parent_eval.records)
        parent_analysis = analyze_parent_title_mapper(identity_eval.records, parent_eval.records)
        candidate_stage_summary = summarize_candidate_stage_from_records(parent_eval.records)
        comparison["variants"][variant_name] = {
            "candidate_generator_name": variant_name,
            "eval_summary": parent_eval.summary,
            "error_summary": error_result.summary,
            "sentence_hit_summary": sentence_hit_result.summary,
            "answer_type_summary": answer_type_result.summary,
            "parent_title_attribution_summary": parent_analysis.summary,
            "candidate_stage_summary": candidate_stage_summary,
            "comparison_metrics": _comparison_metrics(
                parent_eval.summary,
                error_result.summary,
                sentence_hit_result.summary,
                answer_type_result.summary,
                parent_analysis.summary,
                candidate_stage_summary,
            ),
        }
        if output_path is not None:
            variant_dir = output_path / variant_name
            parent_dir = variant_dir / FIXED_MAPPER_NAME
            identity_dir = variant_dir / "identity"
            save_hotpotqa_eval_outputs(parent_eval, parent_dir)
            save_hotpotqa_eval_outputs(identity_eval, identity_dir)
            save_error_analysis_outputs(error_result, parent_dir)
            save_sentence_hit_outputs(sentence_hit_result, parent_dir)
            save_answer_type_outputs(answer_type_result, parent_dir)
            save_parent_title_analysis_outputs(parent_analysis, variant_dir)
            (variant_dir / "candidate_stage_summary.json").write_text(
                json.dumps(candidate_stage_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    if output_path is not None:
        (output_path / "hotpotqa_candidate_stage_relevance_summary.json").write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare HotpotQA candidate-stage relevance variants.")
    parser.add_argument("--path", required=True, help="Path to a HotpotQA JSON or JSONL file.")
    parser.add_argument("--split", default=None, help="Optional split label stored in metadata.")
    parser.add_argument("--limit", type=int, default=200, help="Maximum examples to evaluate.")
    parser.add_argument("--indices", default=None, help="Comma-separated record indices to evaluate.")
    parser.add_argument(
        "--reward-mode",
        choices=["baseline", "oracle_counterfactual"],
        default="baseline",
        help="Reward mode used for all variants.",
    )
    parser.add_argument(
        "--variants",
        default=",".join(DEFAULT_VARIANTS),
        help=f"Comma-separated candidate-stage variants. Available: {', '.join(CANDIDATE_STAGE_VARIANTS)}.",
    )
    parser.add_argument(
        "--scorer",
        choices=["token_overlap", "title_sentence_hybrid"],
        default=DEFAULT_SCORER_NAME,
        help="Question-conditioned scorer used by candidate-stage variants.",
    )
    parser.add_argument("--output-dir", default=None, help="Optional directory for records and summaries.")
    parser.add_argument("--max-steps", type=int, default=4, help="Maximum env steps per example.")
    parser.add_argument("--candidate-top-k", type=int, default=5, help="Final number of EXPAND_EDGE candidates.")
    parser.add_argument(
        "--candidate-pool-k",
        type=int,
        default=None,
        help="Pre-ranking local candidate pool size; default max(candidate_top_k, 25).",
    )
    parser.add_argument("--prune-threshold", type=float, default=0.0, help="Drop candidates at or below this score.")
    parser.add_argument("--min-keep", type=int, default=1, help="Minimum expand candidates to keep after pruning.")
    parser.add_argument("--min-expand-steps", type=int, default=1, help="Fixed sentence_chain min expansion count.")
    args = parser.parse_args()

    comparison = compare_hotpotqa_candidate_stage_relevance(
        path=args.path,
        split=args.split,
        limit=args.limit,
        indices=parse_indices(args.indices),
        reward_mode=args.reward_mode,
        variants=parse_variants(args.variants),
        scorer_name=args.scorer,
        max_steps=args.max_steps,
        candidate_top_k=args.candidate_top_k,
        candidate_pool_k=args.candidate_pool_k,
        prune_threshold=args.prune_threshold,
        min_keep=args.min_keep,
        min_expand_steps=args.min_expand_steps,
        output_dir=args.output_dir,
    )
    _print_comparison(comparison)
    if args.output_dir:
        print(
            "comparison_summary_path: "
            f"{Path(args.output_dir) / 'hotpotqa_candidate_stage_relevance_summary.json'}"
        )


def resolve_sample_indices(*, path: str | Path, limit: int | None, indices: list[int] | None) -> list[int]:
    """Resolve one explicit sample index list shared by every variant."""

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


def parse_variants(raw: str) -> list[str]:
    """Parse candidate-stage variants."""

    variants = [item.strip() for item in raw.split(",") if item.strip()]
    for variant_name in variants:
        _validate_variant(variant_name)
    return variants


def _validate_variant(variant_name: str) -> None:
    if variant_name not in CANDIDATE_STAGE_VARIANTS:
        raise ValueError(
            f"Unknown candidate-stage variant '{variant_name}'. "
            f"Available variants: {', '.join(CANDIDATE_STAGE_VARIANTS)}"
        )


def _run_mapper_eval(
    *,
    path: str | Path,
    split: str | None,
    indices: list[int],
    reward_mode: str,
    mapper_name: str,
    variant_name: str,
    scorer_name: str,
    max_steps: int,
    candidate_top_k: int,
    candidate_pool_k: int | None,
    prune_threshold: float,
    min_keep: int,
    min_expand_steps: int,
) -> HotpotQASubsetEvalResult:
    return evaluate_hotpotqa_graph_subset(
        path=path,
        split=split,
        limit=None,
        indices=indices,
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
        answer_mapper_factory=make_entity_title_mapper_factory(mapper_name),
        answer_mapper_name=mapper_name,
        candidate_generator_factory=make_candidate_generator_factory(
            mode=variant_name,
            top_k=candidate_top_k,
            scorer_name=scorer_name,
            prune_threshold=prune_threshold,
            min_keep=min_keep,
            pool_k=candidate_pool_k,
        ),
        candidate_generator_name=variant_name,
    )


def _comparison_metrics(
    eval_summary: dict[str, Any],
    error_summary: dict[str, Any],
    sentence_hit_summary: dict[str, Any],
    answer_type_summary: dict[str, Any],
    parent_analysis_summary: dict[str, Any],
    candidate_stage_summary: dict[str, Any],
) -> dict[str, Any]:
    attribution_buckets = parent_analysis_summary.get("attribution_buckets", {})
    return {
        "avg_exact_match": eval_summary.get("avg_exact_match", 0.0),
        "avg_f1": eval_summary.get("avg_f1", 0.0),
        "avg_projected_eval_score": eval_summary.get("avg_projected_eval_score", 0.0),
        "failure_rate": error_summary.get("failure_rate", 0.0),
        "entity_title_like": parent_analysis_summary.get("entity_title_like", {}).get(FIXED_MAPPER_NAME, {}),
        "sentence_touch_rate": sentence_hit_summary.get("sentence_touch_rate", 0.0),
        "gold_sentence_touched_rate": sentence_hit_summary.get("gold_sentence_touched_rate", 0.0),
        "selected_sentence_contains_gold_rate": sentence_hit_summary.get("selected_sentence_contains_gold_rate", 0.0),
        "sentence_hit_buckets": sentence_hit_summary.get("diagnostic_buckets", {}),
        "answer_type_buckets": answer_type_summary.get("answer_type_buckets", {}),
        "target_failure_buckets": {
            "path_touched_wrong_region": attribution_buckets.get("path_touched_wrong_region", {}),
            "selected_sentence_not_relevant": attribution_buckets.get("selected_sentence_not_relevant", {}),
        },
        "parent_title_attribution_buckets": attribution_buckets,
        "candidate_stage_summary": candidate_stage_summary,
    }


def _bucket_count(bucket: dict[str, Any]) -> int:
    return int(bucket.get("count", 0)) if isinstance(bucket, dict) else 0


def _print_comparison(comparison: dict[str, Any]) -> None:
    print("=== HotpotQA Candidate-Stage Relevance Comparison ===")
    print(f"reward_mode: {comparison['reward_mode']}")
    print(f"fixed_mapper_name: {comparison['fixed_mapper_name']}")
    print(f"fixed_extractor_name: {comparison['fixed_extractor_name']}")
    print(f"fixed_policy_name: {comparison['fixed_policy_name']}")
    print(f"fixed_selector_name: {comparison['fixed_selector_name']}")
    print(f"sample_count: {comparison['sample_count']}")
    for variant_name in comparison["variant_order"]:
        payload = comparison["variants"][variant_name]
        metrics = payload["comparison_metrics"]
        target = metrics["target_failure_buckets"]
        entity = metrics.get("entity_title_like", {})
        candidate_summary = metrics.get("candidate_stage_summary", {})
        print(
            f"{variant_name}: "
            f"EM={metrics['avg_exact_match']:.4f} "
            f"F1={metrics['avg_f1']:.4f} "
            f"projected_score={metrics['avg_projected_eval_score']:.4f} "
            f"entity_EM={float(entity.get('avg_exact_match', 0.0)):.4f} "
            f"entity_F1={float(entity.get('avg_f1', 0.0)):.4f} "
            f"sentence_touch={metrics['sentence_touch_rate']:.4f} "
            f"gold_sentence_touch={metrics['gold_sentence_touched_rate']:.4f} "
            f"selected_contains_gold={metrics['selected_sentence_contains_gold_rate']:.4f} "
            f"path_wrong={_bucket_count(target['path_touched_wrong_region'])} "
            f"selected_not_relevant={_bucket_count(target['selected_sentence_not_relevant'])} "
            f"avg_pruning_ratio={float(candidate_summary.get('avg_pruning_ratio', 0.0)):.4f}"
        )


if __name__ == "__main__":
    main()

