"""Analyze HotpotQA graph-backed evaluation errors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.benchmarks import parse_indices
from evaluation.hotpotqa_error_analysis import (
    GraphStructureBucketConfig,
    PathBehaviorConfig,
    analyze_hotpotqa_error_records,
    load_eval_records_jsonl,
    save_error_analysis_outputs,
)
from evaluation.hotpotqa_subset_evaluator import evaluate_hotpotqa_graph_subset


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze HotpotQA graph-backed error buckets.")
    parser.add_argument("--records-path", default=None, help="Existing hotpotqa_graph_eval_records.jsonl file.")
    parser.add_argument("--path", default=None, help="HotpotQA JSON/JSONL path used when records-path is not provided.")
    parser.add_argument("--split", default=None, help="Optional split label stored in metadata.")
    parser.add_argument("--limit", type=int, default=3, help="Maximum examples to run when generating records.")
    parser.add_argument("--indices", default=None, help="Comma-separated record indices when generating records.")
    parser.add_argument(
        "--reward-mode",
        choices=["baseline", "oracle_counterfactual", "both"],
        default="baseline",
        help="Reward mode used when generating records.",
    )
    parser.add_argument("--output-dir", default=None, help="Optional directory for error records and summary JSON.")
    parser.add_argument("--max-steps", type=int, default=4, help="Maximum env steps when generating records.")
    parser.add_argument("--candidate-top-k", type=int, default=5, help="CandidateGenerator top_k when generating records.")
    parser.add_argument(
        "--min-expand-steps",
        type=int,
        default=1,
        help="Deterministic policy expands this many times before ANSWER when generating records.",
    )
    parser.add_argument("--node-small-max", type=int, default=50, help="Max node count for the small graph bucket.")
    parser.add_argument("--node-medium-max", type=int, default=100, help="Max node count for the medium graph bucket.")
    parser.add_argument("--edge-small-max", type=int, default=120, help="Max edge count for the small graph bucket.")
    parser.add_argument("--edge-medium-max", type=int, default=250, help="Max edge count for the medium graph bucket.")
    parser.add_argument(
        "--early-answer-min-expands",
        type=int,
        default=1,
        help="ANSWER is early when fewer than this many EXPAND_EDGE actions occurred before it.",
    )
    args = parser.parse_args()

    records = _load_or_generate_records(args)
    result = analyze_hotpotqa_error_records(
        records,
        graph_bucket_config=GraphStructureBucketConfig(
            node_small_max=args.node_small_max,
            node_medium_max=args.node_medium_max,
            edge_small_max=args.edge_small_max,
            edge_medium_max=args.edge_medium_max,
        ),
        path_behavior_config=PathBehaviorConfig(min_expands_before_answer=args.early_answer_min_expands),
    )
    _print_error_summary(result.summary)
    if args.output_dir:
        records_path, failures_path, summary_path = save_error_analysis_outputs(result, args.output_dir)
        print(f"error_records_path: {records_path}")
        print(f"failure_records_path: {failures_path}")
        print(f"error_summary_path: {summary_path}")


def _load_or_generate_records(args: argparse.Namespace):
    if args.records_path:
        return load_eval_records_jsonl(args.records_path)
    if not args.path:
        raise ValueError("Either --records-path or --path must be provided.")
    subset_result = evaluate_hotpotqa_graph_subset(
        path=args.path,
        split=args.split,
        limit=args.limit,
        indices=parse_indices(args.indices),
        reward_mode=args.reward_mode,
        max_steps=args.max_steps,
        candidate_top_k=args.candidate_top_k,
        min_expand_steps=args.min_expand_steps,
    )
    return subset_result.records


def _print_error_summary(summary: dict) -> None:
    print("=== HotpotQA Graph Error Analysis ===")
    print(f"sample_count: {summary['sample_count']}")
    print(f"failure_count: {summary['failure_count']}")
    print(f"failure_rate: {summary['failure_rate']:.4f}")
    print(f"reward_modes: {summary['reward_modes']}")
    print("answer_source_buckets:")
    _print_bucket(summary["answer_source_buckets"])
    print("projected_answer_type_buckets:")
    _print_bucket(summary["projected_answer_type_buckets"])
    print("graph_structure_buckets.node_count:")
    _print_bucket(summary["graph_structure_buckets"]["node_count"])
    print("graph_structure_buckets.edge_count:")
    _print_bucket(summary["graph_structure_buckets"]["edge_count"])
    print("path_buckets.step_count:")
    _print_bucket(summary["path_buckets"]["step_count"])
    print("path_buckets.expand_count:")
    _print_bucket(summary["path_buckets"]["expand_count"])
    print("path_buckets.early_answer:")
    _print_bucket(summary["path_buckets"]["early_answer"])
    print("path_buckets.fallback_answer:")
    _print_bucket(summary["path_buckets"]["fallback_answer"])


def _print_bucket(bucket_summary: dict) -> None:
    for bucket, metrics in bucket_summary.items():
        print(
            f"  {bucket}: count={metrics['count']} EM={metrics['exact_match']:.4f} "
            f"F1={metrics['avg_f1']:.4f} projected_score={metrics['avg_projected_eval_score']:.4f}"
        )


if __name__ == "__main__":
    main()
