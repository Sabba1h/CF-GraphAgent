"""Analyze HotpotQA graph-backed sentence hit diagnostics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.benchmarks import parse_indices
from evaluation.hotpotqa_error_analysis import load_eval_records_jsonl
from evaluation.hotpotqa_sentence_hit_diagnostic import (
    HotpotQASentenceHitDiagnosticResult,
    analyze_hotpotqa_sentence_hits,
    save_sentence_hit_outputs,
)
from evaluation.hotpotqa_subset_evaluator import evaluate_hotpotqa_graph_subset
from graph.hotpotqa_policy_variants import POLICY_NAMES, make_policy_factory


def run_hotpotqa_sentence_hit_analysis(
    *,
    records_path: str | Path | None = None,
    path: str | Path | None = None,
    split: str | None = None,
    limit: int | None = None,
    indices: Iterable[int] | None = None,
    reward_mode: str = "baseline",
    policy_name: str = "baseline",
    max_steps: int = 4,
    candidate_top_k: int = 5,
    min_expand_steps: int = 1,
    delayed_min_expand_steps: int = 2,
    output_dir: str | Path | None = None,
) -> HotpotQASentenceHitDiagnosticResult:
    """Run sentence-hit diagnostics from records or a fresh subset evaluation."""

    if records_path is not None:
        records = load_eval_records_jsonl(records_path)
    else:
        if path is None:
            raise ValueError("Either records_path or path must be provided.")
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
        records = eval_result.records

    result = analyze_hotpotqa_sentence_hits(records)
    if output_dir is not None:
        save_sentence_hit_outputs(result, output_dir)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze HotpotQA graph-backed sentence hits.")
    parser.add_argument("--records-path", default=None, help="Existing HotpotQA eval records JSONL.")
    parser.add_argument("--path", default=None, help="Path to a HotpotQA JSON or JSONL file.")
    parser.add_argument("--split", default=None, help="Optional split label stored in metadata.")
    parser.add_argument("--limit", type=int, default=3, help="Maximum examples to evaluate.")
    parser.add_argument("--indices", default=None, help="Comma-separated record indices to evaluate.")
    parser.add_argument(
        "--reward-mode",
        choices=["baseline", "oracle_counterfactual"],
        default="baseline",
        help="Reward mode used when running a fresh subset evaluation.",
    )
    parser.add_argument(
        "--policy",
        choices=POLICY_NAMES,
        default="baseline",
        help="Deterministic policy used when running a fresh subset evaluation.",
    )
    parser.add_argument("--output-dir", default=None, help="Optional directory for diagnostic records and summary.")
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

    result = run_hotpotqa_sentence_hit_analysis(
        records_path=args.records_path,
        path=args.path,
        split=args.split,
        limit=args.limit,
        indices=parse_indices(args.indices),
        reward_mode=args.reward_mode,
        policy_name=args.policy,
        max_steps=args.max_steps,
        candidate_top_k=args.candidate_top_k,
        min_expand_steps=args.min_expand_steps,
        delayed_min_expand_steps=args.delayed_min_expand_steps,
        output_dir=args.output_dir,
    )
    _print_summary(result)
    if args.output_dir:
        print(f"sentence_hit_summary_path: {Path(args.output_dir) / 'hotpotqa_sentence_hit_summary.json'}")


def _print_summary(result: HotpotQASentenceHitDiagnosticResult) -> None:
    summary = result.summary
    print("=== HotpotQA Sentence Hit Diagnostic ===")
    print(f"sample_count: {summary['sample_count']}")
    print(f"sentence_touch_rate: {summary['sentence_touch_rate']:.4f}")
    print(f"avg_touched_sentence_count: {summary['avg_touched_sentence_count']:.4f}")
    print(f"gold_answer_type_distribution: {summary['gold_answer_type_distribution']}")
    print(f"gold_sentence_applicable_count: {summary['gold_sentence_applicable_count']}")
    print(f"gold_in_any_sentence_rate: {summary['gold_in_any_sentence_rate']:.4f}")
    print(f"gold_sentence_touched_rate: {summary['gold_sentence_touched_rate']:.4f}")
    print(f"selected_sentence_contains_gold_rate: {summary['selected_sentence_contains_gold_rate']:.4f}")
    print(f"diagnostic_buckets: {summary['diagnostic_buckets']}")


if __name__ == "__main__":
    main()
