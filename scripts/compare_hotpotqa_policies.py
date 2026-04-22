"""Compare deterministic HotpotQA graph-backed policy variants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.benchmarks import parse_indices
from evaluation.hotpotqa_answer_type_diagnostic import analyze_hotpotqa_answer_types, save_answer_type_outputs
from evaluation.hotpotqa_error_analysis import analyze_hotpotqa_error_records, save_error_analysis_outputs
from evaluation.hotpotqa_sentence_hit_diagnostic import analyze_hotpotqa_sentence_hits, save_sentence_hit_outputs
from evaluation.hotpotqa_subset_evaluator import evaluate_hotpotqa_graph_subset, save_hotpotqa_eval_outputs
from graph.hotpotqa_policy_variants import POLICY_NAMES, make_policy_factory


def compare_hotpotqa_policies(
    *,
    path: str | Path,
    split: str | None = None,
    limit: int | None = None,
    indices: Iterable[int] | None = None,
    reward_mode: str = "baseline",
    policies: list[str] | None = None,
    max_steps: int = 4,
    candidate_top_k: int = 5,
    min_expand_steps: int = 1,
    delayed_min_expand_steps: int = 2,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run eval and error-analysis summaries for each requested policy."""

    policy_names = policies or ["baseline", "sentence_first", "delayed_answer"]
    comparison: dict[str, Any] = {"reward_mode": reward_mode, "policies": {}, "policy_order": policy_names}
    output_path = Path(output_dir) if output_dir else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    for policy_name in policy_names:
        policy_factory = make_policy_factory(
            policy_name,
            min_expand_steps=min_expand_steps,
            delayed_min_expand_steps=delayed_min_expand_steps,
        )
        eval_result = evaluate_hotpotqa_graph_subset(
            path=path,
            split=split,
            limit=limit,
            indices=indices,
            reward_mode=reward_mode,
            max_steps=max_steps,
            candidate_top_k=candidate_top_k,
            min_expand_steps=min_expand_steps,
            policy_factory=policy_factory,
            policy_name=policy_name,
        )
        error_result = analyze_hotpotqa_error_records(eval_result.records)
        sentence_hit_result = analyze_hotpotqa_sentence_hits(eval_result.records)
        answer_type_result = analyze_hotpotqa_answer_types(eval_result.records)
        comparison["policies"][policy_name] = {
            "eval_summary": eval_result.summary,
            "error_summary": error_result.summary,
            "sentence_hit_summary": sentence_hit_result.summary,
            "answer_type_summary": answer_type_result.summary,
            "comparison_metrics": _comparison_metrics(
                eval_result.summary,
                error_result.summary,
                sentence_hit_result.summary,
                answer_type_result.summary,
            ),
        }

        if output_path is not None:
            policy_dir = output_path / policy_name
            save_hotpotqa_eval_outputs(eval_result, policy_dir)
            save_error_analysis_outputs(error_result, policy_dir)
            save_sentence_hit_outputs(sentence_hit_result, policy_dir)
            save_answer_type_outputs(answer_type_result, policy_dir)

    if output_path is not None:
        (output_path / "hotpotqa_policy_comparison_summary.json").write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare deterministic HotpotQA graph-backed policies.")
    parser.add_argument("--path", required=True, help="Path to a HotpotQA JSON or JSONL file.")
    parser.add_argument("--split", default=None, help="Optional split label stored in metadata.")
    parser.add_argument("--limit", type=int, default=3, help="Maximum examples to evaluate.")
    parser.add_argument("--indices", default=None, help="Comma-separated record indices to evaluate.")
    parser.add_argument(
        "--reward-mode",
        choices=["baseline", "oracle_counterfactual"],
        default="baseline",
        help="Reward mode used for all policies.",
    )
    parser.add_argument(
        "--policies",
        default="baseline,sentence_first,delayed_answer",
        help=f"Comma-separated policy names. Available: {', '.join(POLICY_NAMES)}.",
    )
    parser.add_argument("--output-dir", default=None, help="Optional directory for per-policy records and summaries.")
    parser.add_argument("--max-steps", type=int, default=4, help="Maximum env steps per example.")
    parser.add_argument("--candidate-top-k", type=int, default=5, help="CandidateGenerator top_k.")
    parser.add_argument("--min-expand-steps", type=int, default=1, help="Baseline/sentence-first min expansion count.")
    parser.add_argument(
        "--delayed-min-expand-steps",
        type=int,
        default=2,
        help="Delayed/hybrid policies answer only after this many expansions if expansion is available.",
    )
    args = parser.parse_args()

    comparison = compare_hotpotqa_policies(
        path=args.path,
        split=args.split,
        limit=args.limit,
        indices=parse_indices(args.indices),
        reward_mode=args.reward_mode,
        policies=parse_policies(args.policies),
        max_steps=args.max_steps,
        candidate_top_k=args.candidate_top_k,
        min_expand_steps=args.min_expand_steps,
        delayed_min_expand_steps=args.delayed_min_expand_steps,
        output_dir=args.output_dir,
    )
    _print_comparison(comparison)
    if args.output_dir:
        print(f"comparison_summary_path: {Path(args.output_dir) / 'hotpotqa_policy_comparison_summary.json'}")


def parse_policies(raw: str) -> list[str]:
    """Parse a comma-separated policy list."""

    policies = [item.strip() for item in raw.split(",") if item.strip()]
    for policy_name in policies:
        if policy_name not in POLICY_NAMES:
            raise ValueError(f"Unknown policy '{policy_name}'. Available policies: {', '.join(POLICY_NAMES)}")
    return policies


def _comparison_metrics(
    eval_summary: dict[str, Any],
    error_summary: dict[str, Any],
    sentence_hit_summary: dict[str, Any],
    answer_type_summary: dict[str, Any],
) -> dict[str, Any]:
    early_bucket = error_summary.get("path_buckets", {}).get("early_answer", {})
    return {
        "avg_exact_match": eval_summary.get("avg_exact_match", 0.0),
        "avg_f1": eval_summary.get("avg_f1", 0.0),
        "avg_projected_eval_score": eval_summary.get("avg_projected_eval_score", 0.0),
        "failure_rate": error_summary.get("failure_rate", 0.0),
        "avg_step_count": eval_summary.get("avg_step_count", 0.0),
        "answer_source_type_distribution": eval_summary.get("answer_source_type_distribution", {}),
        "path_expand_buckets": error_summary.get("path_buckets", {}).get("expand_count", {}),
        "early_answer_rate": _bucket_rate(early_bucket, "early_answer"),
        "sentence_touch_rate": sentence_hit_summary.get("sentence_touch_rate", 0.0),
        "gold_sentence_touched_rate": sentence_hit_summary.get("gold_sentence_touched_rate", 0.0),
        "selected_sentence_contains_gold_rate": sentence_hit_summary.get(
            "selected_sentence_contains_gold_rate",
            0.0,
        ),
        "sentence_hit_buckets": sentence_hit_summary.get("diagnostic_buckets", {}),
        "answer_type_buckets": answer_type_summary.get("answer_type_buckets", {}),
    }


def _bucket_rate(bucket_summary: dict[str, Any], bucket_name: str) -> float:
    total = sum(int(metrics.get("count", 0)) for metrics in bucket_summary.values())
    if total == 0:
        return 0.0
    return int(bucket_summary.get(bucket_name, {}).get("count", 0)) / total


def _print_comparison(comparison: dict[str, Any]) -> None:
    print("=== HotpotQA Policy Comparison ===")
    print(f"reward_mode: {comparison['reward_mode']}")
    for policy_name in comparison["policy_order"]:
        metrics = comparison["policies"][policy_name]["comparison_metrics"]
        print(
            f"{policy_name}: "
            f"EM={metrics['avg_exact_match']:.4f} "
            f"F1={metrics['avg_f1']:.4f} "
            f"projected_score={metrics['avg_projected_eval_score']:.4f} "
            f"failure_rate={metrics['failure_rate']:.4f} "
            f"avg_steps={metrics['avg_step_count']:.2f} "
            f"early_answer_rate={metrics['early_answer_rate']:.4f} "
            f"sentence_touch_rate={metrics['sentence_touch_rate']:.4f} "
            f"gold_sentence_touched_rate={metrics['gold_sentence_touched_rate']:.4f} "
            f"answer_source={metrics['answer_source_type_distribution']}"
        )


if __name__ == "__main__":
    main()
