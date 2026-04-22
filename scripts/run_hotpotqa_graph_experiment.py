"""Run HotpotQA graph-backed baseline or oracle rollout experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.experiment_config import RewardMode
from data.benchmarks import parse_indices
from data.benchmarks.hotpotqa import load_hotpotqa
from graph.hotpotqa_graph_runtime import run_hotpotqa_graph_experiment_for_example


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HotpotQA graph-backed rollout experiments.")
    parser.add_argument("--path", required=True, help="Path to a HotpotQA JSON or JSONL file.")
    parser.add_argument("--limit", type=int, default=3, help="Maximum number of examples to run.")
    parser.add_argument("--indices", default=None, help="Comma-separated record indices to run.")
    parser.add_argument("--split", default=None, help="Optional split label stored in metadata.")
    parser.add_argument(
        "--reward-mode",
        choices=["baseline", "oracle_counterfactual", "both"],
        default="baseline",
        help="Experiment reward mode. 'both' runs baseline then oracle_counterfactual.",
    )
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
    parser.add_argument("--print-steps", action="store_true", help="Print per-step reward trace summaries.")
    args = parser.parse_args()

    examples = load_hotpotqa(
        args.path,
        limit=args.limit,
        indices=parse_indices(args.indices),
        split=args.split,
    )
    modes: list[RewardMode] = (
        ["baseline", "oracle_counterfactual"] if args.reward_mode == "both" else [args.reward_mode]
    )

    for mode_index, mode in enumerate(modes):
        if mode_index:
            print()
        print(f"=== HotpotQA Graph Experiment: {mode} ===")
        for example_index, example in enumerate(examples):
            runtime, result = run_hotpotqa_graph_experiment_for_example(
                example,
                reward_mode=mode,
                max_steps=args.max_steps,
                candidate_top_k=args.candidate_top_k,
                min_expand_steps=args.min_expand_steps,
            )
            print(
                f"[{example_index}] question_id={example.question_id} "
                f"graph_id={runtime.local_graph.graph_id} "
                f"nodes={runtime.graph_stats['node_count']} edges={runtime.graph_stats['edge_count']} "
                f"raw_graph_answer={result.metadata.get('raw_graph_answer')} "
                f"projected_answer={result.metadata.get('projected_answer')} "
                f"normalized_projected_answer={result.metadata.get('normalized_projected_answer')} "
                f"gold_answer={result.metadata.get('gold_answer')} "
                f"projected_eval_score={result.metadata.get('projected_eval_score')} "
                f"base_total={result.base_total_reward} "
                f"total={result.total_reward} steps={len(result.step_traces)}"
            )
            if args.print_steps:
                for trace in result.step_traces:
                    print(
                        f"  step={trace.step_idx + 1} action={trace.action} "
                        f"base_reward={trace.base_reward} reward={trace.reward} "
                        f"terminated={trace.terminated} truncated={trace.truncated}"
                    )


if __name__ == "__main__":
    main()
