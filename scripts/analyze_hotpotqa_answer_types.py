"""Analyze HotpotQA graph-backed results by gold answer type."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from answer.hotpotqa_answer_extractor import ANSWER_EXTRACTOR_NAMES, make_answer_extractor_factory
from answer.hotpotqa_answer_selector import ANSWER_SELECTOR_NAMES, make_answer_selector_factory
from data.benchmarks import parse_indices
from evaluation.hotpotqa_answer_type_diagnostic import (
    HotpotQAAnswerTypeDiagnosticResult,
    analyze_hotpotqa_answer_types,
    save_answer_type_outputs,
)
from evaluation.hotpotqa_error_analysis import load_eval_records_jsonl
from evaluation.hotpotqa_subset_evaluator import evaluate_hotpotqa_graph_subset
from graph.hotpotqa_policy_variants import POLICY_NAMES, make_policy_factory


def run_hotpotqa_answer_type_analysis(
    *,
    records_path: str | Path | None = None,
    path: str | Path | None = None,
    split: str | None = None,
    limit: int | None = None,
    indices: Iterable[int] | None = None,
    reward_mode: str = "baseline",
    policy_name: str = "sentence_chain",
    selector_name: str = "latest_sentence",
    extractor_name: str = "full_sentence",
    max_steps: int = 4,
    candidate_top_k: int = 5,
    min_expand_steps: int = 1,
    delayed_min_expand_steps: int = 2,
    output_dir: str | Path | None = None,
) -> HotpotQAAnswerTypeDiagnosticResult:
    """Run answer type diagnostics from records or a fixed fresh evaluation."""

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
        selector_factory = make_answer_selector_factory(selector_name)
        extractor_factory = make_answer_extractor_factory(extractor_name)
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
            answer_selector_factory=selector_factory,
            answer_selector_name=selector_name,
            answer_extractor_factory=extractor_factory,
            answer_extractor_name=extractor_name,
        )
        records = eval_result.records

    result = analyze_hotpotqa_answer_types(records, require_fixed_config=True)
    if output_dir is not None:
        save_answer_type_outputs(result, output_dir)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze HotpotQA graph-backed results by answer type.")
    parser.add_argument("--records-path", default=None, help="Existing fixed-config HotpotQA eval records JSONL.")
    parser.add_argument("--path", default=None, help="Path to a HotpotQA JSON or JSONL file.")
    parser.add_argument("--split", default=None, help="Optional split label stored in metadata.")
    parser.add_argument("--limit", type=int, default=3, help="Maximum examples to evaluate.")
    parser.add_argument("--indices", default=None, help="Comma-separated record indices to evaluate.")
    parser.add_argument(
        "--reward-mode",
        choices=["baseline", "oracle_counterfactual"],
        default="baseline",
        help="Reward mode used when running a fresh fixed-config evaluation.",
    )
    parser.add_argument(
        "--policy",
        choices=POLICY_NAMES,
        default="sentence_chain",
        help="Fixed deterministic policy used when running a fresh evaluation.",
    )
    parser.add_argument(
        "--selector",
        choices=ANSWER_SELECTOR_NAMES,
        default="latest_sentence",
        help="Fixed graph answer selector used when running a fresh evaluation.",
    )
    parser.add_argument(
        "--extractor",
        choices=ANSWER_EXTRACTOR_NAMES,
        default="full_sentence",
        help="Fixed answer extractor used when running a fresh evaluation.",
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

    result = run_hotpotqa_answer_type_analysis(
        records_path=args.records_path,
        path=args.path,
        split=args.split,
        limit=args.limit,
        indices=parse_indices(args.indices),
        reward_mode=args.reward_mode,
        policy_name=args.policy,
        selector_name=args.selector,
        extractor_name=args.extractor,
        max_steps=args.max_steps,
        candidate_top_k=args.candidate_top_k,
        min_expand_steps=args.min_expand_steps,
        delayed_min_expand_steps=args.delayed_min_expand_steps,
        output_dir=args.output_dir,
    )
    _print_summary(result)
    if args.output_dir:
        print(f"answer_type_summary_path: {Path(args.output_dir) / 'hotpotqa_answer_type_summary.json'}")


def _print_summary(result: HotpotQAAnswerTypeDiagnosticResult) -> None:
    summary = result.summary
    print("=== HotpotQA Answer Type Diagnostic ===")
    print(f"sample_count: {summary['sample_count']}")
    print(f"fixed_config: {summary['fixed_config']}")
    for answer_type in summary["answer_type_order"]:
        bucket = summary["answer_type_buckets"][answer_type]
        sentence_hit = bucket["sentence_hit"]
        print(
            f"{answer_type}: "
            f"count={bucket['count']} "
            f"ratio={bucket['ratio']:.4f} "
            f"EM={bucket['avg_exact_match']:.4f} "
            f"F1={bucket['avg_f1']:.4f} "
            f"projected_score={bucket['avg_projected_eval_score']:.4f} "
            f"failure_rate={bucket['failure_rate']:.4f} "
            f"sentence_touch_rate={sentence_hit['sentence_touch_rate']:.4f} "
            f"gold_sentence_hit_rate={sentence_hit['gold_sentence_hit_rate']} "
            f"selected_sentence_contains_gold_rate={sentence_hit['selected_sentence_contains_gold_rate']} "
            f"answer_source={bucket['answer_source_type_distribution']}"
        )


if __name__ == "__main__":
    main()
