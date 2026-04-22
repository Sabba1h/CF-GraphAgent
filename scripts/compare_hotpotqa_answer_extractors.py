"""Compare non-privileged HotpotQA answer extractors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from answer.hotpotqa_answer_extractor import ANSWER_EXTRACTOR_NAMES, make_answer_extractor_factory
from answer.hotpotqa_answer_selector import ANSWER_SELECTOR_NAMES, make_answer_selector_factory
from data.benchmarks import parse_indices
from evaluation.hotpotqa_answer_type_diagnostic import analyze_hotpotqa_answer_types, save_answer_type_outputs
from evaluation.hotpotqa_error_analysis import analyze_hotpotqa_error_records, save_error_analysis_outputs
from evaluation.hotpotqa_sentence_hit_diagnostic import analyze_hotpotqa_sentence_hits, save_sentence_hit_outputs
from evaluation.hotpotqa_subset_evaluator import evaluate_hotpotqa_graph_subset, save_hotpotqa_eval_outputs
from graph.hotpotqa_policy_variants import POLICY_NAMES, make_policy_factory


def compare_hotpotqa_answer_extractors(
    *,
    path: str | Path,
    split: str | None = None,
    limit: int | None = None,
    indices: Iterable[int] | None = None,
    reward_mode: str = "baseline",
    policy_name: str = "sentence_chain",
    selector_name: str = "latest_sentence",
    extractors: list[str] | None = None,
    max_steps: int = 4,
    candidate_top_k: int = 5,
    min_expand_steps: int = 1,
    delayed_min_expand_steps: int = 2,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run eval, error, and sentence-hit summaries for each extractor."""

    extractor_names = extractors or ["full_sentence", "clause_trim", "answer_like_span"]
    policy_factory = make_policy_factory(
        policy_name,
        min_expand_steps=min_expand_steps,
        delayed_min_expand_steps=delayed_min_expand_steps,
    )
    selector_factory = make_answer_selector_factory(selector_name)
    comparison: dict[str, Any] = {
        "reward_mode": reward_mode,
        "policy_name": policy_name,
        "selector_name": selector_name,
        "extractors": {},
        "extractor_order": extractor_names,
    }
    output_path = Path(output_dir) if output_dir else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    for extractor_name in extractor_names:
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
        error_result = analyze_hotpotqa_error_records(eval_result.records)
        sentence_hit_result = analyze_hotpotqa_sentence_hits(eval_result.records)
        answer_type_result = analyze_hotpotqa_answer_types(eval_result.records)
        comparison["extractors"][extractor_name] = {
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
            "sample_answers": _sample_answers(eval_result.records),
        }

        if output_path is not None:
            extractor_dir = output_path / extractor_name
            save_hotpotqa_eval_outputs(eval_result, extractor_dir)
            save_error_analysis_outputs(error_result, extractor_dir)
            save_sentence_hit_outputs(sentence_hit_result, extractor_dir)
            save_answer_type_outputs(answer_type_result, extractor_dir)

    if output_path is not None:
        (output_path / "hotpotqa_answer_extractor_comparison_summary.json").write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare HotpotQA graph answer extractors.")
    parser.add_argument("--path", required=True, help="Path to a HotpotQA JSON or JSONL file.")
    parser.add_argument("--split", default=None, help="Optional split label stored in metadata.")
    parser.add_argument("--limit", type=int, default=3, help="Maximum examples to evaluate.")
    parser.add_argument("--indices", default=None, help="Comma-separated record indices to evaluate.")
    parser.add_argument(
        "--reward-mode",
        choices=["baseline", "oracle_counterfactual"],
        default="baseline",
        help="Reward mode used for all extractors.",
    )
    parser.add_argument(
        "--policy",
        choices=POLICY_NAMES,
        default="sentence_chain",
        help="Fixed deterministic policy used for all extractors.",
    )
    parser.add_argument(
        "--selector",
        choices=ANSWER_SELECTOR_NAMES,
        default="latest_sentence",
        help="Fixed graph answer selector used for all extractors.",
    )
    parser.add_argument(
        "--extractors",
        default="full_sentence,clause_trim,answer_like_span",
        help=f"Comma-separated extractor names. Available: {', '.join(ANSWER_EXTRACTOR_NAMES)}.",
    )
    parser.add_argument("--output-dir", default=None, help="Optional directory for per-extractor records and summaries.")
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

    comparison = compare_hotpotqa_answer_extractors(
        path=args.path,
        split=args.split,
        limit=args.limit,
        indices=parse_indices(args.indices),
        reward_mode=args.reward_mode,
        policy_name=args.policy,
        selector_name=args.selector,
        extractors=parse_extractors(args.extractors),
        max_steps=args.max_steps,
        candidate_top_k=args.candidate_top_k,
        min_expand_steps=args.min_expand_steps,
        delayed_min_expand_steps=args.delayed_min_expand_steps,
        output_dir=args.output_dir,
    )
    _print_comparison(comparison)
    if args.output_dir:
        print(f"comparison_summary_path: {Path(args.output_dir) / 'hotpotqa_answer_extractor_comparison_summary.json'}")


def parse_extractors(raw: str) -> list[str]:
    """Parse a comma-separated extractor list."""

    extractors = [item.strip() for item in raw.split(",") if item.strip()]
    for extractor_name in extractors:
        if extractor_name not in ANSWER_EXTRACTOR_NAMES:
            raise ValueError(
                f"Unknown extractor '{extractor_name}'. Available extractors: {', '.join(ANSWER_EXTRACTOR_NAMES)}"
            )
    return extractors


def _comparison_metrics(
    eval_summary: dict[str, Any],
    error_summary: dict[str, Any],
    sentence_hit_summary: dict[str, Any],
    answer_type_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "avg_exact_match": eval_summary.get("avg_exact_match", 0.0),
        "avg_f1": eval_summary.get("avg_f1", 0.0),
        "avg_projected_eval_score": eval_summary.get("avg_projected_eval_score", 0.0),
        "failure_rate": error_summary.get("failure_rate", 0.0),
        "answer_source_type_distribution": eval_summary.get("answer_source_type_distribution", {}),
        "projected_answer_type_buckets": error_summary.get("projected_answer_type_buckets", {}),
        "sentence_touch_rate": sentence_hit_summary.get("sentence_touch_rate", 0.0),
        "gold_sentence_touched_rate": sentence_hit_summary.get("gold_sentence_touched_rate", 0.0),
        "selected_sentence_contains_gold_rate": sentence_hit_summary.get(
            "selected_sentence_contains_gold_rate",
            0.0,
        ),
        "sentence_hit_buckets": sentence_hit_summary.get("diagnostic_buckets", {}),
        "answer_type_buckets": answer_type_summary.get("answer_type_buckets", {}),
    }


def _sample_answers(records: list[Any]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for record in records:
        extraction = record.metadata.get("answer_extraction") or {}
        samples.append(
            {
                "question_id": record.question_id,
                "selected_graph_answer": record.selected_graph_answer,
                "projected_answer": record.projected_answer,
                "gold_answer": record.gold_answer,
                "exact_match": record.exact_match,
                "f1": record.f1,
                "answer_extractor_name": record.answer_extractor_name,
                "fallback_occurred": extraction.get("fallback_occurred"),
                "fallback_target": extraction.get("fallback_target"),
                "fallback_reason": extraction.get("fallback_reason"),
            }
        )
    return samples


def _print_comparison(comparison: dict[str, Any]) -> None:
    print("=== HotpotQA Answer Extractor Comparison ===")
    print(f"reward_mode: {comparison['reward_mode']}")
    print(f"policy_name: {comparison['policy_name']}")
    print(f"selector_name: {comparison['selector_name']}")
    for extractor_name in comparison["extractor_order"]:
        payload = comparison["extractors"][extractor_name]
        metrics = payload["comparison_metrics"]
        print(
            f"{extractor_name}: "
            f"EM={metrics['avg_exact_match']:.4f} "
            f"F1={metrics['avg_f1']:.4f} "
            f"projected_score={metrics['avg_projected_eval_score']:.4f} "
            f"failure_rate={metrics['failure_rate']:.4f} "
            f"sentence_touch_rate={metrics['sentence_touch_rate']:.4f} "
            f"answer_source={metrics['answer_source_type_distribution']} "
            f"projected_type={metrics['projected_answer_type_buckets']}"
        )
        for sample in payload["sample_answers"][:3]:
            print(
                f"  [{sample['question_id']}] selected={sample['selected_graph_answer']} "
                f"projected={sample['projected_answer']} gold={sample['gold_answer']} "
                f"fallback={sample['fallback_occurred']}:{sample['fallback_target']}:{sample['fallback_reason']}"
            )


if __name__ == "__main__":
    main()
