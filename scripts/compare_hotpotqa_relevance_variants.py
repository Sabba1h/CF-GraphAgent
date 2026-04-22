"""Compare HotpotQA path/selector relevance variants with parent_title fixed."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from answer.hotpotqa_answer_selector import ANSWER_SELECTOR_NAMES, make_answer_selector_factory
from answer.hotpotqa_entity_title_mapper import make_entity_title_mapper_factory
from answer.hotpotqa_relevance_selectors import RELEVANCE_SELECTOR_NAMES, make_relevance_selector_factory
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
from graph.hotpotqa_policy_variants import POLICY_NAMES, make_policy_factory
from graph.hotpotqa_relevance_policies import RELEVANCE_POLICY_NAMES, make_relevance_policy_factory

FIXED_MAPPER_NAME = "parent_title"
FIXED_EXTRACTOR_NAME = "full_sentence"
BASELINE_POLICY_NAME = "sentence_chain"
BASELINE_SELECTOR_NAME = "latest_sentence"

DEFAULT_VARIANTS = (
    "baseline",
    "title_region_commitment_policy",
    "sentence_region_persistence_policy",
    "dominant_title_region_selector",
    "sentence_region_persistence_recent_selector",
)

VARIANT_CONFIGS: dict[str, dict[str, str]] = {
    "baseline": {"policy": BASELINE_POLICY_NAME, "selector": BASELINE_SELECTOR_NAME},
    "title_region_commitment_policy": {
        "policy": "title_region_commitment",
        "selector": BASELINE_SELECTOR_NAME,
    },
    "sentence_region_persistence_policy": {
        "policy": "sentence_region_persistence",
        "selector": BASELINE_SELECTOR_NAME,
    },
    "dominant_title_region_selector": {
        "policy": BASELINE_POLICY_NAME,
        "selector": "dominant_title_region",
    },
    "recent_relevant_region_selector": {
        "policy": BASELINE_POLICY_NAME,
        "selector": "recent_relevant_region",
    },
    "title_region_commitment_dominant_selector": {
        "policy": "title_region_commitment",
        "selector": "dominant_title_region",
    },
    "sentence_region_persistence_recent_selector": {
        "policy": "sentence_region_persistence",
        "selector": "recent_relevant_region",
    },
    "region_commitment_delayed_answer_policy": {
        "policy": "region_commitment_delayed_answer",
        "selector": BASELINE_SELECTOR_NAME,
    },
}


def compare_hotpotqa_relevance_variants(
    *,
    path: str | Path,
    split: str | None = None,
    limit: int | None = None,
    indices: list[int] | None = None,
    reward_mode: str = "baseline",
    variants: list[str] | None = None,
    max_steps: int = 4,
    candidate_top_k: int = 5,
    min_expand_steps: int = 1,
    delayed_min_expand_steps: int = 2,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run relevance variants on one fixed sample index list."""

    variant_names = variants or list(DEFAULT_VARIANTS)
    resolved_indices = resolve_sample_indices(path=path, limit=limit, indices=indices)
    comparison: dict[str, Any] = {
        "reward_mode": reward_mode,
        "fixed_mapper_name": FIXED_MAPPER_NAME,
        "fixed_extractor_name": FIXED_EXTRACTOR_NAME,
        "sample_indices": resolved_indices,
        "sample_count": len(resolved_indices),
        "variant_order": variant_names,
        "variants": {},
    }
    output_path = Path(output_dir) if output_dir else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    for variant_name in variant_names:
        config = variant_config(variant_name)
        policy_name = config["policy"]
        selector_name = config["selector"]
        policy_factory = _policy_factory(
            policy_name,
            min_expand_steps=min_expand_steps,
            delayed_min_expand_steps=delayed_min_expand_steps,
        )
        selector_factory = _selector_factory(selector_name)
        parent_eval = _run_mapper_eval(
            path=path,
            split=split,
            indices=resolved_indices,
            reward_mode=reward_mode,
            policy_name=policy_name,
            policy_factory=policy_factory,
            selector_name=selector_name,
            selector_factory=selector_factory,
            mapper_name=FIXED_MAPPER_NAME,
            max_steps=max_steps,
            candidate_top_k=candidate_top_k,
            min_expand_steps=min_expand_steps,
        )
        identity_eval = _run_mapper_eval(
            path=path,
            split=split,
            indices=resolved_indices,
            reward_mode=reward_mode,
            policy_name=policy_name,
            policy_factory=policy_factory,
            selector_name=selector_name,
            selector_factory=selector_factory,
            mapper_name="identity",
            max_steps=max_steps,
            candidate_top_k=candidate_top_k,
            min_expand_steps=min_expand_steps,
        )

        error_result = analyze_hotpotqa_error_records(parent_eval.records)
        sentence_hit_result = analyze_hotpotqa_sentence_hits(parent_eval.records)
        answer_type_result = analyze_hotpotqa_answer_types(parent_eval.records)
        parent_analysis = analyze_parent_title_mapper(identity_eval.records, parent_eval.records)
        comparison["variants"][variant_name] = {
            "policy_name": policy_name,
            "selector_name": selector_name,
            "eval_summary": parent_eval.summary,
            "error_summary": error_result.summary,
            "sentence_hit_summary": sentence_hit_result.summary,
            "answer_type_summary": answer_type_result.summary,
            "parent_title_attribution_summary": parent_analysis.summary,
            "comparison_metrics": _comparison_metrics(
                parent_eval.summary,
                error_result.summary,
                sentence_hit_result.summary,
                answer_type_result.summary,
                parent_analysis.summary,
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

    if output_path is not None:
        (output_path / "hotpotqa_relevance_comparison_summary.json").write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare HotpotQA path/selector relevance variants.")
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
        help=f"Comma-separated variant names. Available: {', '.join(VARIANT_CONFIGS)}.",
    )
    parser.add_argument("--output-dir", default=None, help="Optional directory for records and summaries.")
    parser.add_argument("--max-steps", type=int, default=4, help="Maximum env steps per example.")
    parser.add_argument("--candidate-top-k", type=int, default=5, help="CandidateGenerator top_k.")
    parser.add_argument("--min-expand-steps", type=int, default=1, help="Baseline/sentence-first min expansion count.")
    parser.add_argument(
        "--delayed-min-expand-steps",
        type=int,
        default=2,
        help="Delayed relevance policies answer only after this many expansions if expansion is available.",
    )
    args = parser.parse_args()

    comparison = compare_hotpotqa_relevance_variants(
        path=args.path,
        split=args.split,
        limit=args.limit,
        indices=parse_indices(args.indices),
        reward_mode=args.reward_mode,
        variants=parse_variants(args.variants),
        max_steps=args.max_steps,
        candidate_top_k=args.candidate_top_k,
        min_expand_steps=args.min_expand_steps,
        delayed_min_expand_steps=args.delayed_min_expand_steps,
        output_dir=args.output_dir,
    )
    _print_comparison(comparison)
    if args.output_dir:
        print(f"comparison_summary_path: {Path(args.output_dir) / 'hotpotqa_relevance_comparison_summary.json'}")


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
    """Parse a comma-separated relevance variant list."""

    variants = [item.strip() for item in raw.split(",") if item.strip()]
    for variant_name in variants:
        variant_config(variant_name)
    return variants


def variant_config(variant_name: str) -> dict[str, str]:
    """Return a copy of a relevance variant config."""

    if variant_name not in VARIANT_CONFIGS:
        raise ValueError(
            f"Unknown relevance variant '{variant_name}'. Available variants: {', '.join(VARIANT_CONFIGS)}"
        )
    return dict(VARIANT_CONFIGS[variant_name])


def _policy_factory(policy_name: str, *, min_expand_steps: int, delayed_min_expand_steps: int):
    if policy_name in POLICY_NAMES:
        return make_policy_factory(
            policy_name,
            min_expand_steps=min_expand_steps,
            delayed_min_expand_steps=delayed_min_expand_steps,
        )
    if policy_name in RELEVANCE_POLICY_NAMES:
        return make_relevance_policy_factory(
            policy_name,
            min_expand_steps=min_expand_steps,
            delayed_min_expand_steps=delayed_min_expand_steps,
        )
    raise ValueError(
        f"Unknown policy '{policy_name}'. Available policies: {', '.join(POLICY_NAMES + RELEVANCE_POLICY_NAMES)}"
    )


def _selector_factory(selector_name: str):
    if selector_name in ANSWER_SELECTOR_NAMES:
        return make_answer_selector_factory(selector_name)
    if selector_name in RELEVANCE_SELECTOR_NAMES:
        return make_relevance_selector_factory(selector_name)
    raise ValueError(
        f"Unknown selector '{selector_name}'. Available selectors: "
        f"{', '.join(ANSWER_SELECTOR_NAMES + RELEVANCE_SELECTOR_NAMES)}"
    )


def _run_mapper_eval(
    *,
    path: str | Path,
    split: str | None,
    indices: list[int],
    reward_mode: str,
    policy_name: str,
    policy_factory,
    selector_name: str,
    selector_factory,
    mapper_name: str,
    max_steps: int,
    candidate_top_k: int,
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
        policy_factory=policy_factory,
        policy_name=policy_name,
        answer_selector_factory=selector_factory,
        answer_selector_name=selector_name,
        answer_extractor_factory=None,
        answer_extractor_name=FIXED_EXTRACTOR_NAME,
        answer_mapper_factory=make_entity_title_mapper_factory(mapper_name),
        answer_mapper_name=mapper_name,
    )


def _comparison_metrics(
    eval_summary: dict[str, Any],
    error_summary: dict[str, Any],
    sentence_hit_summary: dict[str, Any],
    answer_type_summary: dict[str, Any],
    parent_analysis_summary: dict[str, Any],
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
    }


def _bucket_count(bucket: dict[str, Any]) -> int:
    return int(bucket.get("count", 0)) if isinstance(bucket, dict) else 0


def _print_comparison(comparison: dict[str, Any]) -> None:
    print("=== HotpotQA Relevance Variant Comparison ===")
    print(f"reward_mode: {comparison['reward_mode']}")
    print(f"fixed_mapper_name: {comparison['fixed_mapper_name']}")
    print(f"fixed_extractor_name: {comparison['fixed_extractor_name']}")
    print(f"sample_count: {comparison['sample_count']}")
    for variant_name in comparison["variant_order"]:
        payload = comparison["variants"][variant_name]
        metrics = payload["comparison_metrics"]
        target = metrics["target_failure_buckets"]
        entity = metrics.get("entity_title_like", {})
        print(
            f"{variant_name}: "
            f"policy={payload['policy_name']} "
            f"selector={payload['selector_name']} "
            f"EM={metrics['avg_exact_match']:.4f} "
            f"F1={metrics['avg_f1']:.4f} "
            f"projected_score={metrics['avg_projected_eval_score']:.4f} "
            f"entity_EM={float(entity.get('avg_exact_match', 0.0)):.4f} "
            f"entity_F1={float(entity.get('avg_f1', 0.0)):.4f} "
            f"sentence_touch={metrics['sentence_touch_rate']:.4f} "
            f"gold_sentence_touch={metrics['gold_sentence_touched_rate']:.4f} "
            f"selected_contains_gold={metrics['selected_sentence_contains_gold_rate']:.4f} "
            f"path_wrong={_bucket_count(target['path_touched_wrong_region'])} "
            f"selected_not_relevant={_bucket_count(target['selected_sentence_not_relevant'])}"
        )


if __name__ == "__main__":
    main()
