"""Evaluate HotpotQA graph-backed experiments on a small subset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.benchmarks import parse_indices
from evaluation.hotpotqa_subset_evaluator import evaluate_hotpotqa_graph_subset, save_hotpotqa_eval_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HotpotQA graph-backed subset experiments.")
    parser.add_argument("--path", required=True, help="Path to a HotpotQA JSON or JSONL file.")
    parser.add_argument("--split", default=None, help="Optional split label stored in metadata.")
    parser.add_argument("--limit", type=int, default=3, help="Maximum number of examples to evaluate.")
    parser.add_argument("--indices", default=None, help="Comma-separated record indices to evaluate.")
    parser.add_argument(
        "--reward-mode",
        choices=["baseline", "oracle_counterfactual", "both"],
        default="baseline",
        help="Reward mode to evaluate. 'both' runs baseline and oracle_counterfactual.",
    )
    parser.add_argument("--output-dir", default=None, help="Optional directory for JSONL records and summary JSON.")
    parser.add_argument("--max-steps", type=int, default=4, help="Maximum env steps per example.")
    parser.add_argument(
        "--candidate-top-k",
        type=int,
        default=5,
        help="Passed to CFGraphEnv CandidateGenerator top_k.",
    )
    parser.add_argument(
        "--min-expand-steps",
        type=int,
        default=1,
        help="Deterministic policy expands this many times before choosing ANSWER when available.",
    )
    args = parser.parse_args()

    result = evaluate_hotpotqa_graph_subset(
        path=args.path,
        split=args.split,
        limit=args.limit,
        indices=parse_indices(args.indices),
        reward_mode=args.reward_mode,
        max_steps=args.max_steps,
        candidate_top_k=args.candidate_top_k,
        min_expand_steps=args.min_expand_steps,
    )
    _print_summary(result.summary)
    if args.output_dir:
        records_path, summary_path = save_hotpotqa_eval_outputs(result, args.output_dir)
        print(f"records_path: {records_path}")
        print(f"summary_path: {summary_path}")


def _print_summary(summary: dict) -> None:
    print("=== HotpotQA Graph Subset Evaluation ===")
    print(f"sample_count: {summary['sample_count']}")
    print(f"reward_modes: {summary['reward_modes']}")
    print(f"avg_exact_match: {summary['avg_exact_match']:.4f}")
    print(f"avg_f1: {summary['avg_f1']:.4f}")
    print(f"avg_projected_eval_score: {summary['avg_projected_eval_score']:.4f}")
    print(f"avg_base_total_reward: {summary['avg_base_total_reward']:.4f}")
    print(f"avg_total_reward: {summary['avg_total_reward']:.4f}")
    print(f"avg_step_count: {summary['avg_step_count']:.2f}")
    print(f"avg_graph_node_count: {summary['avg_graph_node_count']:.2f}")
    print(f"avg_graph_edge_count: {summary['avg_graph_edge_count']:.2f}")
    print(f"answer_source_type_distribution: {summary['answer_source_type_distribution']}")
    print(f"nonzero_oracle_delta_count: {summary['nonzero_oracle_delta_count']}")
    by_reward_mode = summary.get("by_reward_mode") or {}
    for mode, mode_summary in by_reward_mode.items():
        print(
            f"[{mode}] count={mode_summary['sample_count']} "
            f"EM={mode_summary['avg_exact_match']:.4f} "
            f"F1={mode_summary['avg_f1']:.4f} "
            f"projected_score={mode_summary['avg_projected_eval_score']:.4f} "
            f"avg_total_reward={mode_summary['avg_total_reward']:.4f}"
        )


if __name__ == "__main__":
    main()
