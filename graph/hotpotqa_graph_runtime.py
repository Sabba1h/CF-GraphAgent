"""HotpotQA graph-backed runtime helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agent.graph_rollout_manager import GraphRolloutManager
from answer.hotpotqa_answer_adapter import HotpotQAAnswerAdapter
from answer.hotpotqa_answer_extractor import AnswerExtractor, HotpotQAAnswerExtraction, make_full_sentence_extractor
from answer.hotpotqa_entity_title_mapper import EntityTitleMapper, HotpotQAEntityTitleMapping
from answer.hotpotqa_answer_selector import AnswerSelector, make_raw_final_node_selector
from answer.hotpotqa_yesno_mapper import HotpotQAYesNoMapping, YesNoMapper
from answer.hotpotqa_relation_span_mapper import HotpotQARelationSpanMapping, RelationSpanMapper
from answer.hotpotqa_relation_span_discovery import RelationSpanDiscovery
from answer.hotpotqa_relation_span_ranker import RelationSpanRanker
from answer.hotpotqa_relation_span_proposal import RelationSpanProposal
from candidates.generator import CandidateGenerator
from core.experiment_config import ExperimentConfig, RewardMode
from core.experiment_result import ExperimentResult
from data.benchmarks.answer_normalization import normalize_answer
from core.task import TaskSample
from data.benchmarks.common import BenchmarkExample
from envs.cf_graph_env import CFGraphEnv
from graph.benchmark_graph_builder import build_from_benchmark_example
from graph.benchmark_graph_loader import compute_graph_stats, local_graph_to_graph_store
from graph.benchmark_graph_types import BenchmarkLocalGraph
from graph.graph_store import GraphStore
from graph.hotpotqa_policy_variants import make_baseline_policy

PolicyFn = Callable[[dict[str, Any]], Any]


@dataclass(slots=True)
class HotpotQAGraphRuntime:
    """Runtime bundle for one HotpotQA graph-backed environment."""

    example: BenchmarkExample
    task_sample: TaskSample
    local_graph: BenchmarkLocalGraph
    graph_store: GraphStore
    graph_stats: dict[str, Any]
    env: CFGraphEnv


def build_hotpotqa_graph_runtime(
    example: BenchmarkExample,
    *,
    max_steps: int = 4,
    candidate_top_k: int = 5,
    candidate_generator: CandidateGenerator | None = None,
) -> HotpotQAGraphRuntime:
    """Build local graph, GraphStore, TaskSample, and env for one HotpotQA example."""

    local_graph = build_from_benchmark_example(example)
    graph_store = local_graph_to_graph_store(local_graph)
    graph_stats = compute_graph_stats(local_graph)
    task_sample = example.to_task_sample()
    task_sample.metadata.update(
        {
            "graph_id": local_graph.graph_id,
            "graph_stats": graph_stats,
            "graph_backed": True,
        }
    )
    env = CFGraphEnv(
        graph_store=graph_store,
        query=task_sample.query,
        ground_truth=task_sample.ground_truth,
        max_steps=max_steps,
        candidate_top_k=candidate_top_k,
    )
    if candidate_generator is not None:
        env.candidate_generator = candidate_generator
        env.transition_engine.candidate_generator = candidate_generator
    return HotpotQAGraphRuntime(
        example=example,
        task_sample=task_sample,
        local_graph=local_graph,
        graph_store=graph_store,
        graph_stats=graph_stats,
        env=env,
    )


def make_hotpotqa_graph_policy(*, min_expand_steps: int = 1) -> PolicyFn:
    """Return the default deterministic policy kept for compatibility."""

    return make_baseline_policy(min_expand_steps=min_expand_steps)


def run_hotpotqa_graph_experiment_for_example(
    example: BenchmarkExample,
    *,
    reward_mode: RewardMode = "baseline",
    max_steps: int = 4,
    candidate_top_k: int = 5,
    min_expand_steps: int = 1,
    policy: PolicyFn | None = None,
    policy_name: str = "baseline",
    answer_selector: AnswerSelector | None = None,
    answer_selector_name: str = "raw_final_node",
    answer_extractor: AnswerExtractor | None = None,
    answer_extractor_name: str = "full_sentence",
    answer_mapper: EntityTitleMapper | None = None,
    answer_mapper_name: str = "identity",
    yesno_mapper: YesNoMapper | None = None,
    yesno_mapper_name: str | None = None,
    relation_span_mapper: RelationSpanMapper | None = None,
    relation_span_mapper_name: str | None = None,
    relation_span_discovery: RelationSpanDiscovery | None = None,
    relation_span_discovery_name: str | None = None,
    relation_span_ranker: RelationSpanRanker | None = None,
    relation_span_ranker_name: str | None = None,
    relation_span_proposal: RelationSpanProposal | None = None,
    relation_span_proposal_name: str | None = None,
    candidate_generator: CandidateGenerator | None = None,
    candidate_generator_name: str = "baseline_generator",
    use_counterfactual_merge: bool | None = None,
) -> tuple[HotpotQAGraphRuntime, ExperimentResult]:
    """Run one HotpotQA graph-backed rollout experiment."""

    runtime = build_hotpotqa_graph_runtime(
        example,
        max_steps=max_steps,
        candidate_top_k=candidate_top_k,
        candidate_generator=candidate_generator,
    )
    merge_counterfactual = reward_mode == "oracle_counterfactual" if use_counterfactual_merge is None else use_counterfactual_merge
    config = ExperimentConfig(
        reward_mode=reward_mode,
        counterfactual_mode="replace",
        use_counterfactual_merge=merge_counterfactual,
        max_steps=max_steps,
        record_step_traces=True,
        record_counterfactual_metadata=True,
        metadata={
            "dataset_name": example.dataset_name,
            "question_id": example.question_id,
            "graph_id": runtime.local_graph.graph_id,
            "graph_backed": True,
            "policy_name": policy_name,
            "answer_selector_name": answer_selector_name,
            "answer_extractor_name": answer_extractor_name,
            "answer_mapper_name": _effective_answer_mapper_name(
                answer_mapper_name,
                yesno_mapper_name=yesno_mapper_name,
                relation_span_mapper_name=relation_span_mapper_name,
                relation_span_discovery_name=relation_span_discovery_name,
                relation_span_ranker_name=relation_span_ranker_name,
                relation_span_proposal_name=relation_span_proposal_name,
            ),
            "base_answer_mapper_name": answer_mapper_name,
            "yesno_mapper_name": yesno_mapper_name,
            "relation_span_mapper_name": relation_span_mapper_name,
            "relation_span_discovery_name": relation_span_discovery_name,
            "relation_span_ranker_name": relation_span_ranker_name,
            "relation_span_proposal_name": relation_span_proposal_name,
            "candidate_generator_name": candidate_generator_name,
        },
    )
    result = GraphRolloutManager().run_experiment(
        env=runtime.env,
        policy=policy or make_hotpotqa_graph_policy(min_expand_steps=min_expand_steps),
        config=config,
    )
    selector = answer_selector or make_raw_final_node_selector()
    answer_selection = selector(result, runtime.graph_store)
    answer_mapping: HotpotQAEntityTitleMapping | None = None
    yesno_mapping: HotpotQAYesNoMapping | None = None
    relation_span_mapping: HotpotQARelationSpanMapping | None = None
    answer_type_label = _answer_type_label(example)
    if answer_mapper is None:
        extractor = answer_extractor or make_full_sentence_extractor()
        answer_extraction = extractor(
            answer_selection.selected_graph_answer,
            runtime.graph_store,
            answer_selection.to_dict(),
        )
        alignment_graph_answer = answer_selection.selected_graph_answer
        base_mapping_payload = _base_mapping_from_extraction(answer_extraction, answer_selection.selected_graph_answer)
    else:
        answer_mapping = answer_mapper(
            answer_selection.selected_graph_answer,
            runtime.graph_store,
            answer_selection.to_dict(),
        )
        answer_extraction = _extraction_from_mapping(
            answer_mapping,
            extractor_name=answer_extractor_name,
        )
        alignment_graph_answer = answer_mapping.source_node_id
        base_mapping_payload = answer_mapping.to_dict()
    if yesno_mapper is not None and answer_type_label == "yes_no":
        yesno_mapping = yesno_mapper(
            answer_selection.selected_graph_answer,
            runtime.graph_store,
            {
                "query_text": example.question,
                "answer_type_label": answer_type_label,
                "answer_selection": answer_selection.to_dict(),
                "base_mapping": base_mapping_payload,
            },
        )
        answer_extraction = _extraction_from_yesno_mapping(
            yesno_mapping,
            extractor_name=answer_extractor_name,
        )
        alignment_graph_answer = yesno_mapping.source_node_id or answer_selection.selected_graph_answer
    if relation_span_mapper is not None and answer_type_label == "descriptive_span_or_relation":
        relation_span_mapping = relation_span_mapper(
            answer_selection.selected_graph_answer,
            runtime.graph_store,
            {
                "query_text": example.question,
                "answer_type_label": answer_type_label,
                "answer_selection": answer_selection.to_dict(),
                "base_mapping": base_mapping_payload,
                "relation_span_discovery": relation_span_discovery,
                "relation_span_discovery_name": relation_span_discovery_name,
                "relation_span_ranker": relation_span_ranker,
                "relation_span_ranker_name": relation_span_ranker_name,
                "relation_span_proposal": relation_span_proposal,
                "relation_span_proposal_name": relation_span_proposal_name,
            },
        )
        answer_extraction = _extraction_from_relation_span_mapping(
            relation_span_mapping,
            extractor_name=answer_extractor_name,
        )
        alignment_graph_answer = (
            relation_span_mapping.source_node_ids[0]
            if relation_span_mapping.source_node_ids
            else answer_selection.selected_graph_answer
        )
    answer_alignment = HotpotQAAnswerAdapter().align(
        raw_graph_answer=alignment_graph_answer,
        graph_store=runtime.graph_store,
        gold_answer=example.answer,
        extraction=answer_extraction,
    )
    result.metadata.update(
        {
            "dataset_name": example.dataset_name,
            "question_id": example.question_id,
            "graph_id": runtime.local_graph.graph_id,
            "graph_stats": runtime.graph_stats,
            "graph_sentence_nodes": _sentence_nodes(runtime.graph_store),
            "graph_backed": True,
            "policy_name": policy_name,
            "answer_selector_name": answer_selector_name,
            "answer_extractor_name": answer_extractor_name,
            "answer_mapper_name": _effective_answer_mapper_name(
                answer_mapper_name,
                yesno_mapper_name=yesno_mapper_name,
                relation_span_mapper_name=relation_span_mapper_name,
                relation_span_discovery_name=relation_span_discovery_name,
                relation_span_ranker_name=relation_span_ranker_name,
                relation_span_proposal_name=relation_span_proposal_name,
            ),
            "base_answer_mapper_name": answer_mapper_name,
            "yesno_mapper_name": yesno_mapper_name,
            "relation_span_mapper_name": relation_span_mapper_name,
            "relation_span_discovery_name": relation_span_discovery_name,
            "relation_span_ranker_name": relation_span_ranker_name,
            "relation_span_proposal_name": relation_span_proposal_name,
            "answer_type_label": answer_type_label,
            "yesno_mapper_applied": yesno_mapping is not None,
            "relation_span_mapper_applied": relation_span_mapping is not None,
            "relation_span_discovery_applied": (
                relation_span_mapping is not None
                and isinstance(relation_span_mapping.metadata.get("relation_span_discovery"), dict)
            ),
            "relation_span_ranker_applied": (
                relation_span_mapping is not None
                and isinstance(relation_span_mapping.metadata.get("relation_span_ranking"), dict)
            ),
            "relation_span_proposal_applied": (
                relation_span_mapping is not None
                and isinstance(relation_span_mapping.metadata.get("relation_span_proposal"), dict)
            ),
            "candidate_generator_name": candidate_generator_name,
            "raw_graph_answer": answer_selection.raw_graph_answer,
            "selected_graph_answer": answer_selection.selected_graph_answer,
            "mapped_answer": (
                yesno_mapping.mapped_answer
                if yesno_mapping is not None
                else relation_span_mapping.mapped_answer
                if relation_span_mapping is not None
                else answer_mapping.mapped_answer
                if answer_mapping is not None
                else None
            ),
            "answer_selection": answer_selection.to_dict(),
            "answer_mapping": answer_mapping.to_dict() if answer_mapping is not None else None,
            "yesno_mapping": yesno_mapping.to_dict() if yesno_mapping is not None else None,
            "relation_span_mapping": relation_span_mapping.to_dict() if relation_span_mapping is not None else None,
            "answer_extraction": answer_extraction.to_dict(),
            "extracted_answer": answer_extraction.extracted_answer,
            "projected_answer": answer_alignment.projected_answer,
            "normalized_projected_answer": answer_alignment.normalized_projected_answer,
            "gold_answer": answer_alignment.gold_answer,
            "normalized_gold_answer": answer_alignment.normalized_gold_answer,
            "projected_eval_score": answer_alignment.projected_eval_score,
            "projected_is_correct": answer_alignment.is_correct,
            "answer_projection": answer_alignment.to_dict(),
        }
    )
    return runtime, result


def _extraction_from_mapping(
    answer_mapping: HotpotQAEntityTitleMapping,
    *,
    extractor_name: str,
) -> HotpotQAAnswerExtraction:
    """Wrap mapper output as a fixed extractor passthrough for evaluation."""

    return HotpotQAAnswerExtraction(
        extractor_name=extractor_name,
        source_node_id=answer_mapping.source_node_id,
        source_node_type=answer_mapping.source_node_type,
        source_text=answer_mapping.source_text,
        extracted_answer=answer_mapping.mapped_answer,
        fallback_occurred=answer_mapping.fallback_occurred,
        fallback_target=answer_mapping.fallback_target,
        fallback_reason=answer_mapping.fallback_reason,
        metadata={
            "strategy": "entity_title_mapper_passthrough",
            "fixed_extractor": extractor_name,
            "answer_mapping": answer_mapping.to_dict(),
        },
    )


def _extraction_from_yesno_mapping(
    yesno_mapping: HotpotQAYesNoMapping,
    *,
    extractor_name: str,
) -> HotpotQAAnswerExtraction:
    """Wrap yes/no mapper output as a fixed extractor passthrough."""

    return HotpotQAAnswerExtraction(
        extractor_name=extractor_name,
        source_node_id=yesno_mapping.source_node_id,
        source_node_type=yesno_mapping.source_node_type,
        source_text=yesno_mapping.source_text,
        extracted_answer=yesno_mapping.mapped_answer,
        fallback_occurred=yesno_mapping.fallback_occurred,
        fallback_target=yesno_mapping.fallback_target,
        fallback_reason=yesno_mapping.fallback_reason,
        metadata={
            "strategy": "yesno_mapper_passthrough",
            "fixed_extractor": extractor_name,
            "yesno_mapping": yesno_mapping.to_dict(),
        },
    )


def _extraction_from_relation_span_mapping(
    relation_span_mapping: HotpotQARelationSpanMapping,
    *,
    extractor_name: str,
) -> HotpotQAAnswerExtraction:
    """Wrap relation/span mapper output as a fixed extractor passthrough."""

    return HotpotQAAnswerExtraction(
        extractor_name=extractor_name,
        source_node_id=relation_span_mapping.source_node_ids[0] if relation_span_mapping.source_node_ids else None,
        source_node_type=relation_span_mapping.source_node_type,
        source_text=relation_span_mapping.source_text,
        extracted_answer=relation_span_mapping.mapped_answer,
        fallback_occurred=relation_span_mapping.fallback_occurred,
        fallback_target=relation_span_mapping.fallback_target,
        fallback_reason=relation_span_mapping.fallback_reason,
        metadata={
            "strategy": "relation_span_mapper_passthrough",
            "fixed_extractor": extractor_name,
            "relation_span_mapping": relation_span_mapping.to_dict(),
        },
    )


def _base_mapping_from_extraction(
    answer_extraction: HotpotQAAnswerExtraction,
    selected_graph_answer: str | None,
) -> dict[str, Any]:
    """Return a mapper-like payload for no-mapper base paths."""

    return {
        "mapper_name": "identity_extraction",
        "selected_graph_answer": selected_graph_answer,
        "mapped_answer": answer_extraction.extracted_answer,
        "source_node_id": answer_extraction.source_node_id,
        "source_node_type": answer_extraction.source_node_type,
        "source_text": answer_extraction.source_text,
        "fallback_occurred": answer_extraction.fallback_occurred,
        "fallback_target": answer_extraction.fallback_target,
        "fallback_reason": answer_extraction.fallback_reason,
        "metadata": {"answer_extraction": answer_extraction.to_dict()},
    }


def _answer_type_label(example: BenchmarkExample) -> str:
    """Return the routing label; do not pass the gold string to mappers."""

    normalized = normalize_answer(example.normalized_answer or example.answer)
    if normalized in {"yes", "no"}:
        return "yes_no"
    tokens = normalized.split()
    if _looks_numeric_or_date(tokens):
        return "numeric_or_date"
    if len(tokens) == 1:
        return "single_token_entity_like"
    if 2 <= len(tokens) <= 4:
        return "multi_token_entity_or_title_like"
    return "descriptive_span_or_relation"


def _effective_answer_mapper_name(
    answer_mapper_name: str,
    *,
    yesno_mapper_name: str | None,
    relation_span_mapper_name: str | None = None,
    relation_span_discovery_name: str | None = None,
    relation_span_ranker_name: str | None = None,
    relation_span_proposal_name: str | None = None,
) -> str:
    """Return a stable config label for eval records."""

    suffixes = [
        name
        for name in (
            yesno_mapper_name,
            relation_span_mapper_name,
            relation_span_discovery_name,
            relation_span_ranker_name,
            relation_span_proposal_name,
        )
        if name
    ]
    return "+".join([answer_mapper_name] + suffixes) if suffixes else answer_mapper_name


def _looks_numeric_or_date(tokens: list[str]) -> bool:
    if not tokens:
        return False
    month_tokens = {
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    }
    date_connectors = {"ad", "bc", "ce", "bce"}
    has_date_signal = any(token.isdigit() or token in month_tokens for token in tokens)
    return has_date_signal and all(token.isdigit() or token in month_tokens or token in date_connectors for token in tokens)


def _sentence_nodes(graph_store: GraphStore) -> list[dict[str, Any]]:
    """Return sentence nodes for diagnostics only, not for policy decisions."""

    sentence_nodes: list[dict[str, Any]] = []
    for node_id in graph_store.iter_node_ids():
        attrs = graph_store.get_node_attributes(node_id)
        if attrs.get("node_type") != "sentence":
            continue
        metadata = attrs.get("metadata") if isinstance(attrs.get("metadata"), dict) else {}
        sentence_nodes.append(
            {
                "node_id": node_id,
                "node_type": "sentence",
                "text": str(attrs.get("text") or attrs.get("name") or ""),
                "metadata": dict(metadata),
            }
        )
    return sentence_nodes
