"""Tests for HotpotQA candidate-stage question-conditioned relevance."""

from __future__ import annotations

import json
from pathlib import Path

from candidates.generator import CandidateGenerator
from candidates.hotpotqa_question_conditioned_pruner import (
    HotpotQAQuestionConditionedCandidateGenerator,
    apply_candidate_stage_filter,
    summarize_candidate_stage_from_records,
)
from core.actions import ActionType, CandidateAction
from graph.graph_store import EdgeRecord, GraphStore
from memory.working_memory import WorkingMemory
from scripts.compare_hotpotqa_candidate_stage_relevance import compare_hotpotqa_candidate_stage_relevance


def _candidate(
    candidate_id: int,
    dst_text: str,
    *,
    relation: str = "title_to_sentence",
    src_text: str = "First Title",
) -> CandidateAction:
    return CandidateAction(
        candidate_id=candidate_id,
        action_type=ActionType.EXPAND_EDGE,
        description=f"Expand edge e{candidate_id}: title -[{relation}]-> sentence",
        edge_id=f"e{candidate_id}",
        score=1.0,
        metadata={
            "src": "title",
            "dst": f"sentence-{candidate_id}",
            "relation": relation,
            "src_text": src_text,
            "dst_text": dst_text,
            "src_node_type": "title",
            "dst_node_type": "sentence",
            "forbidden_metadata": {"gold_answer": "must-not-be-used"},
        },
    )


def _graph_store() -> GraphStore:
    graph_store = GraphStore()
    graph_store.add_node("q", node_type="question", text="Which page discusses First Title?")
    graph_store.add_node("title-1", node_type="title", text="First Title")
    graph_store.add_node("title-2", node_type="title", text="Second Page")
    graph_store.add_node("sentence-1", node_type="sentence", text="This sentence discusses First Title.")
    graph_store.add_node("sentence-2", node_type="sentence", text="A different unrelated sentence.")
    graph_store.add_edge(EdgeRecord(edge_id="e1", src="q", dst="title-1", relation="question_to_title"))
    graph_store.add_edge(EdgeRecord(edge_id="e2", src="q", dst="title-2", relation="question_to_title"))
    graph_store.add_edge(EdgeRecord(edge_id="e3", src="title-1", dst="sentence-1", relation="title_to_sentence"))
    graph_store.add_edge(EdgeRecord(edge_id="e4", src="title-2", dst="sentence-2", relation="title_to_sentence"))
    return graph_store


def _working_memory() -> WorkingMemory:
    memory = WorkingMemory(max_steps=4)
    memory.frontier_nodes = {"q"}
    return memory


def _fixture_record(answer: str = "First Title") -> dict:
    return {
        "id": "hp-candidate-stage-1",
        "question": "Which page discusses First Title?",
        "answer": answer,
        "context": {
            "title": ["First Title", "Second Page"],
            "sentences": [
                ["This sentence discusses First Title."],
                ["A different unrelated sentence."],
            ],
        },
        "supporting_facts": {"title": ["First Title"], "sent_id": [0]},
    }


def _write_fixture(path: Path) -> Path:
    data_path = path / "hotpotqa_candidate_stage_fixture.json"
    data_path.write_text(json.dumps([_fixture_record()], ensure_ascii=False), encoding="utf-8")
    return data_path


def test_candidate_stage_ranker_is_deterministic_with_stable_tie_break() -> None:
    first = _candidate(2, "First Title")
    second = _candidate(1, "First Title")

    kept, summary = apply_candidate_stage_filter(
        query_text="Which page discusses First Title?",
        expand_actions=[first, second],
        mode="overlap_ranked_generator",
        top_k=2,
    )

    assert [action.edge_id for action in kept] == ["e2", "e1"]
    assert summary.original_expand_count == 2
    assert summary.kept_expand_count == 2
    assert summary.pruning_ratio == 0.0
    assert summary.top_candidate_score > 0.0


def test_candidate_stage_pruner_uses_whitelisted_candidate_fields_and_records_reason() -> None:
    relevant = _candidate(0, "This sentence discusses First Title.")
    irrelevant = _candidate(1, "Completely unrelated.", src_text="Other")

    kept, summary = apply_candidate_stage_filter(
        query_text="Which page discusses First Title?",
        expand_actions=[relevant, irrelevant],
        mode="hybrid_prune_then_rank_generator",
        top_k=2,
        prune_threshold=0.0,
    )

    assert [action.edge_id for action in kept] == ["e0"]
    kept_metadata = kept[0].metadata
    assert "candidate_stage_score_components" in kept_metadata
    assert "forbidden_metadata" not in kept_metadata["candidate_stage_score_components"]
    assert any(decision["pruning_reason"] == "score_lte_threshold:0.0" for decision in summary.decisions)
    json.dumps(summary.to_dict(), ensure_ascii=False)


def test_baseline_generator_factory_keeps_default_candidate_order() -> None:
    graph_store = _graph_store()
    memory = _working_memory()

    baseline = CandidateGenerator(top_k=3)
    explicit_baseline = HotpotQAQuestionConditionedCandidateGenerator(top_k=3, mode="baseline_generator")

    first = baseline.generate(query="Which page discusses First Title?", graph_store=graph_store, working_memory=memory)
    second = explicit_baseline.generate(query="Which page discusses First Title?", graph_store=graph_store, working_memory=memory)

    assert [action.to_dict() for action in first] == [action.to_dict() for action in second]


def test_question_conditioned_generator_adds_candidate_stage_summary() -> None:
    graph_store = _graph_store()
    generator = HotpotQAQuestionConditionedCandidateGenerator(
        top_k=2,
        mode="overlap_ranked_generator",
        pool_k=4,
    )

    actions = generator.generate(
        query="Which page discusses First Title?",
        graph_store=graph_store,
        working_memory=_working_memory(),
    )

    expand_actions = [action for action in actions if action.action_type == ActionType.EXPAND_EDGE]
    assert expand_actions
    summary = expand_actions[0].metadata["candidate_stage_summary"]
    assert summary["original_expand_count"] >= summary["kept_expand_count"]
    assert "top_k_scores" in summary
    assert "kept_relation_counts" in summary


def test_candidate_stage_summary_from_records_uses_explicit_metadata() -> None:
    class Record:
        metadata = {
            "candidate_stage_summaries": [
                {
                    "original_expand_count": 4,
                    "kept_expand_count": 2,
                    "pruning_ratio": 0.5,
                    "top_candidate_score": 0.7,
                    "kept_relation_counts": {"title_to_sentence": 2},
                    "kept_node_type_counts": {"title": 1, "sentence": 1},
                    "component_description": {"formula": "test"},
                }
            ]
        }

    summary = summarize_candidate_stage_from_records([Record()])

    assert summary["available"] is True
    assert summary["avg_original_expand_count"] == 4
    assert summary["avg_pruning_ratio"] == 0.5
    assert summary["kept_relation_counts"] == {"title_to_sentence": 2}


def test_candidate_stage_comparison_script_smoke(tmp_path: Path) -> None:
    data_path = _write_fixture(tmp_path)
    output_dir = tmp_path / "candidate-stage-output"

    comparison = compare_hotpotqa_candidate_stage_relevance(
        path=data_path,
        split="validation",
        limit=1,
        reward_mode="baseline",
        variants=["baseline_generator", "overlap_ranked_generator"],
        output_dir=output_dir,
    )

    assert comparison["fixed_mapper_name"] == "parent_title"
    assert comparison["fixed_extractor_name"] == "full_sentence"
    assert comparison["sample_indices"] == [0]
    assert comparison["variant_order"] == ["baseline_generator", "overlap_ranked_generator"]
    ranked_payload = comparison["variants"]["overlap_ranked_generator"]
    assert ranked_payload["candidate_stage_summary"]["available"] is True
    assert "target_failure_buckets" in ranked_payload["comparison_metrics"]
    assert (output_dir / "overlap_ranked_generator" / "parent_title" / "hotpotqa_graph_eval_records.jsonl").exists()
    assert (output_dir / "hotpotqa_candidate_stage_relevance_summary.json").exists()
