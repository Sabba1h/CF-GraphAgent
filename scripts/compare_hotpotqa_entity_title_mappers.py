"""Compare non-privileged HotpotQA entity/title-like answer mappers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from answer.hotpotqa_answer_selector import ANSWER_SELECTOR_NAMES, make_answer_selector_factory
from answer.hotpotqa_entity_title_mapper import ENTITY_TITLE_MAPPER_NAMES, make_entity_title_mapper_factory
from data.benchmarks import parse_indices
from evaluation.hotpotqa_answer_type_diagnostic import analyze_hotpotqa_answer_types, save_answer_type_outputs
from evaluation.hotpotqa_error_analysis import analyze_hotpotqa_error_records, save_error_analysis_outputs
from evaluation.hotpotqa_subset_evaluator import evaluate_hotpotqa_graph_subset, save_hotpotqa_eval_outputs
from graph.hotpotqa_policy_variants import POLICY_NAMES, make_policy_factory

FIXED_EXTRACTOR_NAME = "full_sentence"


def compare_hotpotqa_entity_title_mappers(
    *,
    path: str | Path,
    split: str | None = None,
    limit: int | None = None,
    indices: Iterable[int] | None = None,
    reward_mode: str = "baseline",
    policy_name: str = "sentence_chain",
    selector_name: str = "latest_sentence",
    mappers: list[str] | None = None,
    max_steps: int = 4,
    candidate_top_k: int = 5,
    min_expand_steps: int = 1,
    delayed_min_expand_steps: int = 2,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run eval, error, and answer-type summaries for each mapper."""

    mapper_names = mappers or ["identity", "parent_title", "capitalized_span"]
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
        "fixed_extractor_name": FIXED_EXTRACTOR_NAME,
        "mappers": {},
        "mapper_order": mapper_names,
    }
    output_path = Path(output_dir) if output_dir else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    for mapper_name in mapper_names:
        mapper_factory = make_entity_title_mapper_factory(mapper_name)
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
            answer_extractor_factory=None,
            answer_extractor_name=FIXED_EXTRACTOR_NAME,
            answer_mapper_factory=mapper_factory,
            answer_mapper_name=mapper_name,
        )
        error_result = analyze_hotpotqa_error_records(eval_result.records)
        answer_type_result = analyze_hotpotqa_answer_types(eval_result.records)
        comparison["mappers"][mapper_name] = {
            "eval_summary": eval_result.summary,
            "error_summary": error_result.summary,
            "answer_type_summary": answer_type_result.summary,
            "comparison_metrics": _comparison_metrics(
                eval_result.summary,
                error_result.summary,
                answer_type_result.summary,
                eval_result.records,
            ),
            "sample_answers": _sample_answers(eval_result.records),
        }

        if output_path is not None:
            mapper_dir = output_path / mapper_name
            save_hotpotqa_eval_outputs(eval_result, mapper_dir)
            save_error_analysis_outputs(error_result, mapper_dir)
            save_answer_type_outputs(answer_type_result, mapper_dir)

    if output_path is not None:
        (output_path / "hotpotqa_entity_title_mapper_comparison_summary.json").write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare HotpotQA entity/title-like answer mappers.")
    parser.add_argument("--path", required=True, help="Path to a HotpotQA JSON or JSONL file.")
    parser.add_argument("--split", default=None, help="Optional split label stored in metadata.")
    parser.add_argument("--limit", type=int, default=3, help="Maximum examples to evaluate.")
    parser.add_argument("--indices", default=None, help="Comma-separated record indices to evaluate.")
    parser.add_argument(
        "--reward-mode",
        choices=["baseline", "oracle_counterfactual"],
        default="baseline",
        help="Reward mode used for all mappers.",
    )
    parser.add_argument(
        "--policy",
        choices=POLICY_NAMES,
        default="sentence_chain",
        help="Fixed deterministic policy used for all mappers.",
    )
    parser.add_argument(
        "--selector",
        choices=ANSWER_SELECTOR_NAMES,
        default="latest_sentence",
        help="Fixed graph answer selector used before mapper comparison.",
    )
    parser.add_argument(
        "--mappers",
        default="identity,parent_title,capitalized_span",
        help=f"Comma-separated mapper names. Available: {', '.join(ENTITY_TITLE_MAPPER_NAMES)}.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory for per-mapper records and summaries.",
    )
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

    comparison = compare_hotpotqa_entity_title_mappers(
        path=args.path,
        split=args.split,
        limit=args.limit,
        indices=parse_indices(args.indices),
        reward_mode=args.reward_mode,
        policy_name=args.policy,
        selector_name=args.selector,
        mappers=parse_mappers(args.mappers),
        max_steps=args.max_steps,
        candidate_top_k=args.candidate_top_k,
        min_expand_steps=args.min_expand_steps,
        delayed_min_expand_steps=args.delayed_min_expand_steps,
        output_dir=args.output_dir,
    )
    _print_comparison(comparison)
    if args.output_dir:
        print(f"comparison_summary_path: {Path(args.output_dir) / 'hotpotqa_entity_title_mapper_comparison_summary.json'}")


def parse_mappers(raw: str) -> list[str]:
    """Parse a comma-separated mapper list."""

    mappers = [item.strip() for item in raw.split(",") if item.strip()]
    for mapper_name in mappers:
        if mapper_name not in ENTITY_TITLE_MAPPER_NAMES:
            raise ValueError(
                f"Unknown mapper '{mapper_name}'. Available mappers: {', '.join(ENTITY_TITLE_MAPPER_NAMES)}"
            )
    return mappers


def _comparison_metrics(
    eval_summary: dict[str, Any],
    error_summary: dict[str, Any],
    answer_type_summary: dict[str, Any],
    records: list[Any],
) -> dict[str, Any]:
    return {
        "avg_exact_match": eval_summary.get("avg_exact_match", 0.0),
        "avg_f1": eval_summary.get("avg_f1", 0.0),
        "avg_projected_eval_score": eval_summary.get("avg_projected_eval_score", 0.0),
        "failure_rate": error_summary.get("failure_rate", 0.0),
        "answer_source_type_distribution": eval_summary.get("answer_source_type_distribution", {}),
        "entity_title_like": _entity_title_like_metrics(answer_type_summary),
        "mapper_behavior": _mapper_behavior(records),
    }


def _entity_title_like_metrics(answer_type_summary: dict[str, Any]) -> dict[str, Any]:
    buckets = answer_type_summary.get("answer_type_buckets", {})
    relevant = [
        buckets.get("single_token_entity_like", {}),
        buckets.get("multi_token_entity_or_title_like", {}),
    ]
    total_count = sum(int(bucket.get("count", 0)) for bucket in relevant)
    return {
        "count": total_count,
        "avg_exact_match": _weighted_average(relevant, "avg_exact_match", total_count),
        "avg_f1": _weighted_average(relevant, "avg_f1", total_count),
        "avg_projected_eval_score": _weighted_average(relevant, "avg_projected_eval_score", total_count),
        "failure_rate": _weighted_average(relevant, "failure_rate", total_count),
        "single_token_entity_like": buckets.get("single_token_entity_like", {}),
        "multi_token_entity_or_title_like": buckets.get("multi_token_entity_or_title_like", {}),
    }


def _weighted_average(buckets: list[dict[str, Any]], key: str, total_count: int) -> float:
    if total_count <= 0:
        return 0.0
    return sum(float(bucket.get(key, 0.0)) * int(bucket.get("count", 0)) for bucket in buckets) / total_count


def _mapper_behavior(records: list[Any]) -> dict[str, Any]:
    fallback_count = 0
    mapped_changed_count = 0
    fallback_reasons: dict[str, int] = {}
    for record in records:
        mapping = record.metadata.get("answer_mapping") or {}
        if not isinstance(mapping, dict):
            continue
        if mapping.get("fallback_occurred"):
            fallback_count += 1
            reason = str(mapping.get("fallback_reason") or "unknown")
            fallback_reasons[reason] = fallback_reasons.get(reason, 0) + 1
        mapped_answer = str(mapping.get("mapped_answer") or "")
        original_selected_text = _original_selected_text(mapping)
        if mapped_answer and mapped_answer != original_selected_text:
            mapped_changed_count += 1
    return {
        "fallback_count": fallback_count,
        "mapped_changed_count": mapped_changed_count,
        "fallback_reasons": dict(sorted(fallback_reasons.items())),
    }


def _original_selected_text(mapping: dict[str, Any]) -> str:
    metadata = mapping.get("metadata") if isinstance(mapping.get("metadata"), dict) else {}
    selected_node = metadata.get("selected_node") if isinstance(metadata.get("selected_node"), dict) else {}
    if selected_node.get("text") is not None:
        return str(selected_node.get("text") or "")
    base_mapping = metadata.get("base_mapping") if isinstance(metadata.get("base_mapping"), dict) else {}
    if base_mapping.get("source_text") is not None:
        return str(base_mapping.get("source_text") or "")
    return str(mapping.get("source_text") or "")


def _sample_answers(records: list[Any]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for record in records:
        mapping = record.metadata.get("answer_mapping") or {}
        samples.append(
            {
                "question_id": record.question_id,
                "raw_graph_answer": record.raw_graph_answer,
                "selected_graph_answer": record.selected_graph_answer,
                "mapped_answer": mapping.get("mapped_answer") if isinstance(mapping, dict) else None,
                "projected_answer": record.projected_answer,
                "gold_answer": record.gold_answer,
                "exact_match": record.exact_match,
                "f1": record.f1,
                "answer_mapper_name": record.answer_mapper_name,
                "fallback_occurred": mapping.get("fallback_occurred") if isinstance(mapping, dict) else None,
                "fallback_target": mapping.get("fallback_target") if isinstance(mapping, dict) else None,
                "fallback_reason": mapping.get("fallback_reason") if isinstance(mapping, dict) else None,
            }
        )
    return samples


def _print_comparison(comparison: dict[str, Any]) -> None:
    print("=== HotpotQA Entity/Title Mapper Comparison ===")
    print(f"reward_mode: {comparison['reward_mode']}")
    print(f"policy_name: {comparison['policy_name']}")
    print(f"selector_name: {comparison['selector_name']}")
    print(f"fixed_extractor_name: {comparison['fixed_extractor_name']}")
    for mapper_name in comparison["mapper_order"]:
        payload = comparison["mappers"][mapper_name]
        metrics = payload["comparison_metrics"]
        entity_metrics = metrics["entity_title_like"]
        print(
            f"{mapper_name}: "
            f"EM={metrics['avg_exact_match']:.4f} "
            f"F1={metrics['avg_f1']:.4f} "
            f"projected_score={metrics['avg_projected_eval_score']:.4f} "
            f"failure_rate={metrics['failure_rate']:.4f} "
            f"entity_title_EM={entity_metrics['avg_exact_match']:.4f} "
            f"entity_title_F1={entity_metrics['avg_f1']:.4f} "
            f"entity_title_failure={entity_metrics['failure_rate']:.4f} "
            f"mapper_behavior={metrics['mapper_behavior']} "
            f"answer_source={metrics['answer_source_type_distribution']}"
        )
        for sample in payload["sample_answers"][:3]:
            print(
                f"  [{sample['question_id']}] selected={sample['selected_graph_answer']} "
                f"mapped={sample['mapped_answer']} projected={sample['projected_answer']} "
                f"gold={sample['gold_answer']} "
                f"fallback={sample['fallback_occurred']}:{sample['fallback_target']}:{sample['fallback_reason']}"
            )


if __name__ == "__main__":
    main()
