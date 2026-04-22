"""Analyze parent_title mapper stability and failure attribution."""

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
from answer.hotpotqa_entity_title_mapper import make_entity_title_mapper_factory
from data.benchmarks import parse_indices
from evaluation.hotpotqa_parent_title_analysis import (
    HotpotQAParentTitleAttributionResult,
    analyze_parent_title_mapper,
    save_parent_title_analysis_outputs,
)
from evaluation.hotpotqa_subset_evaluator import (
    HotpotQASubsetEvalResult,
    evaluate_hotpotqa_graph_subset,
)
from graph.hotpotqa_policy_variants import POLICY_NAMES, make_policy_factory

FIXED_EXTRACTOR_NAME = "full_sentence"
DEFAULT_SCALE_LIMITS = (500, 1000)


def run_parent_title_mapper_analysis(
    *,
    path: str | Path,
    split: str | None = None,
    limit: int = 2000,
    indices: Iterable[int] | None = None,
    reward_mode: str = "baseline",
    policy_name: str = "sentence_chain",
    selector_name: str = "latest_sentence",
    max_steps: int = 4,
    candidate_top_k: int = 5,
    min_expand_steps: int = 1,
    delayed_min_expand_steps: int = 2,
    scale_limits: list[int] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run identity vs parent_title and return stability plus attribution outputs."""

    resolved_scales = _scale_limits(scale_limits, limit)
    policy_factory = make_policy_factory(
        policy_name,
        min_expand_steps=min_expand_steps,
        delayed_min_expand_steps=delayed_min_expand_steps,
    )
    selector_factory = make_answer_selector_factory(selector_name)

    identity_eval = _run_mapper_eval(
        path=path,
        split=split,
        limit=limit,
        indices=indices,
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
    parent_title_eval = _run_mapper_eval(
        path=path,
        split=split,
        limit=limit,
        indices=indices,
        reward_mode=reward_mode,
        policy_name=policy_name,
        policy_factory=policy_factory,
        selector_name=selector_name,
        selector_factory=selector_factory,
        mapper_name="parent_title",
        max_steps=max_steps,
        candidate_top_k=candidate_top_k,
        min_expand_steps=min_expand_steps,
    )
    attribution = analyze_parent_title_mapper(
        identity_eval.records,
        parent_title_eval.records,
        scale_limits=resolved_scales,
    )
    result = {
        "fixed_extractor_name": FIXED_EXTRACTOR_NAME,
        "scale_limits": resolved_scales,
        "identity_eval_summary": identity_eval.summary,
        "parent_title_eval_summary": parent_title_eval.summary,
        "parent_title_analysis": attribution.summary,
    }

    if output_dir is not None:
        _save_outputs(
            identity_eval=identity_eval,
            parent_title_eval=parent_title_eval,
            attribution=attribution,
            result=result,
            output_dir=Path(output_dir),
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze HotpotQA parent_title mapper stability.")
    parser.add_argument("--path", required=True, help="Path to a HotpotQA JSON or JSONL file.")
    parser.add_argument("--split", default=None, help="Optional split label stored in metadata.")
    parser.add_argument("--limit", type=int, default=2000, help="Maximum examples to evaluate.")
    parser.add_argument("--indices", default=None, help="Comma-separated record indices to evaluate.")
    parser.add_argument(
        "--reward-mode",
        choices=["baseline", "oracle_counterfactual"],
        default="baseline",
        help="Reward mode used for both mappers.",
    )
    parser.add_argument(
        "--policy",
        choices=POLICY_NAMES,
        default="sentence_chain",
        help="Fixed deterministic policy used for both mappers.",
    )
    parser.add_argument(
        "--selector",
        choices=ANSWER_SELECTOR_NAMES,
        default="latest_sentence",
        help="Fixed graph answer selector used before mapper comparison.",
    )
    parser.add_argument(
        "--scale-limits",
        default=None,
        help="Comma-separated stability scales. Defaults to 500,1000 plus --limit.",
    )
    parser.add_argument("--output-dir", default=None, help="Optional directory for JSONL/JSON outputs.")
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

    result = run_parent_title_mapper_analysis(
        path=args.path,
        split=args.split,
        limit=args.limit,
        indices=parse_indices(args.indices),
        reward_mode=args.reward_mode,
        policy_name=args.policy,
        selector_name=args.selector,
        max_steps=args.max_steps,
        candidate_top_k=args.candidate_top_k,
        min_expand_steps=args.min_expand_steps,
        delayed_min_expand_steps=args.delayed_min_expand_steps,
        scale_limits=parse_scale_limits(args.scale_limits),
        output_dir=args.output_dir,
    )
    _print_result(result)
    if args.output_dir:
        print(f"output_dir: {Path(args.output_dir)}")


def parse_scale_limits(raw: str | None) -> list[int] | None:
    """Parse comma-separated stability scale limits."""

    if raw is None:
        return None
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _scale_limits(scale_limits: list[int] | None, limit: int) -> list[int]:
    base_limits = list(scale_limits) if scale_limits is not None else list(DEFAULT_SCALE_LIMITS)
    base_limits.append(limit)
    return sorted({value for value in base_limits if value > 0})


def _run_mapper_eval(
    *,
    path: str | Path,
    split: str | None,
    limit: int,
    indices: Iterable[int] | None,
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
        answer_mapper_factory=make_entity_title_mapper_factory(mapper_name),
        answer_mapper_name=mapper_name,
    )


def _save_outputs(
    *,
    identity_eval: HotpotQASubsetEvalResult,
    parent_title_eval: HotpotQASubsetEvalResult,
    attribution: HotpotQAParentTitleAttributionResult,
    result: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_eval_records(output_dir / "identity_eval_records.jsonl", identity_eval)
    _write_eval_records(output_dir / "parent_title_eval_records.jsonl", parent_title_eval)
    (output_dir / "identity_eval_summary.json").write_text(
        json.dumps(identity_eval.summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "parent_title_eval_summary.json").write_text(
        json.dumps(parent_title_eval.summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    save_parent_title_analysis_outputs(attribution, output_dir)
    (output_dir / "parent_title_analysis_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_eval_records(path: Path, result: HotpotQASubsetEvalResult) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in result.records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def _print_result(result: dict[str, Any]) -> None:
    analysis = result["parent_title_analysis"]
    print("=== HotpotQA Parent Title Mapper Analysis ===")
    print(f"fixed_extractor_name: {result['fixed_extractor_name']}")
    print(f"sample_count: {analysis['sample_count']}")
    print("Scale curve:")
    for scale, payload in analysis["scale_curve"].items():
        delta = payload["delta"]
        entity_delta = payload["entity_title_like"]["delta"]
        print(
            f"  limit={scale}: "
            f"overall_delta_EM={delta['avg_exact_match']:.4f} "
            f"overall_delta_F1={delta['avg_f1']:.4f} "
            f"overall_delta_projected={delta['avg_projected_eval_score']:.4f} "
            f"entity_delta_EM={entity_delta['avg_exact_match']:.4f} "
            f"entity_delta_F1={entity_delta['avg_f1']:.4f}"
        )
    overall = analysis["overall"]
    entity = analysis["entity_title_like"]
    print(
        "Overall: "
        f"identity_EM={overall['identity']['avg_exact_match']:.4f} "
        f"parent_title_EM={overall['parent_title']['avg_exact_match']:.4f} "
        f"delta_EM={overall['delta']['avg_exact_match']:.4f} "
        f"identity_F1={overall['identity']['avg_f1']:.4f} "
        f"parent_title_F1={overall['parent_title']['avg_f1']:.4f} "
        f"delta_F1={overall['delta']['avg_f1']:.4f}"
    )
    print(
        "Entity/title-like: "
        f"count={entity['identity']['sample_count']} "
        f"identity_EM={entity['identity']['avg_exact_match']:.4f} "
        f"parent_title_EM={entity['parent_title']['avg_exact_match']:.4f} "
        f"delta_EM={entity['delta']['avg_exact_match']:.4f} "
        f"identity_F1={entity['identity']['avg_f1']:.4f} "
        f"parent_title_F1={entity['parent_title']['avg_f1']:.4f} "
        f"delta_F1={entity['delta']['avg_f1']:.4f}"
    )
    print(f"success_summary: {analysis['success_summary']}")
    print(f"failure_buckets: {analysis['failure_summary']['failure_buckets']}")


if __name__ == "__main__":
    main()
