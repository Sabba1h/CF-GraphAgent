"""Tests for HotpotQA path/selector relevance variants."""

from __future__ import annotations

import json
from pathlib import Path

from answer.hotpotqa_relevance_selectors import (
    make_dominant_title_region_selector,
    make_recent_relevant_region_selector,
)
from core.experiment_config import ExperimentConfig
from core.experiment_result import ExperimentResult, ExperimentStepTrace
from graph.graph_store import EdgeRecord, GraphStore
from graph.hotpotqa_relevance_policies import (
    make_sentence_region_persistence_policy,
    make_title_region_commitment_policy,
)
from scripts.compare_hotpotqa_relevance_variants import compare_hotpotqa_relevance_variants


def _candidate(
    candidate_id: int,
    relation: str,
    src: str,
    dst: str,
    *,
    repeated: bool = False,
) -> dict:
    suffix = " (already in working subgraph)" if repeated else ""
    return {
        "candidate_id": candidate_id,
        "action_type": "EXPAND_EDGE",
        "description": f"Expand {src} -[{relation}]-> {dst}{suffix}",
        "metadata": {"relation": relation, "src": src, "dst": dst},
    }


def _observation(candidates: list[dict]) -> dict:
    return {
        "candidate_actions": candidates
        + [
            {"candidate_id": 90, "action_type": "ANSWER", "description": "Answer", "metadata": {}},
            {"candidate_id": 91, "action_type": "STOP", "description": "Stop", "metadata": {}},
        ]
    }


def _graph_store() -> GraphStore:
    graph_store = GraphStore()
    graph_store.add_node("title-1", node_type="title", text="First Title", metadata={"title": "First Title"})
    graph_store.add_node("title-2", node_type="title", text="Second Title", metadata={"title": "Second Title"})
    graph_store.add_node("global-title", node_type="title", text="Forbidden Global Title")
    graph_store.add_node(
        "sentence-1",
        node_type="sentence",
        text="This sentence mentions First Title.",
        metadata={"title": "First Title"},
    )
    graph_store.add_node(
        "sentence-2",
        node_type="sentence",
        text="Second sentence.",
        metadata={"title": "First Title"},
    )
    graph_store.add_edge(EdgeRecord(edge_id="e-title-s1", src="title-1", dst="sentence-1", relation="title_to_sentence"))
    graph_store.add_edge(EdgeRecord(edge_id="e-s1-title", src="sentence-1", dst="title-1", relation="sentence_to_title"))
    graph_store.add_edge(EdgeRecord(edge_id="e-s1-s2", src="sentence-1", dst="sentence-2", relation="next_sentence"))
    graph_store.add_edge(EdgeRecord(edge_id="e-title2-s2", src="title-2", dst="sentence-2", relation="title_to_sentence"))
    return graph_store


def _experiment_result() -> ExperimentResult:
    return ExperimentResult(
        config=ExperimentConfig(metadata={"query": "Question"}),
        final_answer="global-title",
        step_traces=[
            ExperimentStepTrace(
                step_idx=0,
                action=0,
                reward_mode="baseline",
                base_reward=0.1,
                reward=0.1,
                reward_breakdown=None,
                metadata={"info": {"expanded_edge": {"src": "title-1", "dst": "sentence-1"}}},
            ),
            ExperimentStepTrace(
                step_idx=1,
                action=1,
                reward_mode="baseline",
                base_reward=0.1,
                reward=0.1,
                reward_breakdown=None,
                metadata={"info": {"expanded_edge": {"src": "sentence-1", "dst": "sentence-2"}}},
            ),
        ],
    )


def _fixture_record(answer: str = "First Title") -> dict:
    return {
        "id": "hp-relevance-fixture-1",
        "question": "Which page is first?",
        "answer": answer,
        "context": {
            "title": ["First Title", "Second Title"],
            "sentences": [
                ["This sentence mentions First Title.", "Another sentence."],
                ["The second page sentence."],
            ],
        },
        "supporting_facts": {"title": ["First Title"], "sent_id": [0]},
    }


def _write_fixture(path: Path) -> Path:
    data_path = path / "hotpotqa_relevance_fixture.json"
    data_path.write_text(json.dumps([_fixture_record()], ensure_ascii=False), encoding="utf-8")
    return data_path


def test_relevance_policies_are_deterministic_and_allow_answer_escape() -> None:
    first_observation = _observation([
        _candidate(5, "title_to_sentence", "title-2", "sentence-3"),
        _candidate(1, "title_to_sentence", "title-1", "sentence-1"),
    ])
    second_observation = _observation([
        _candidate(3, "title_to_sentence", "title-2", "sentence-3"),
        _candidate(2, "next_sentence", "sentence-1", "sentence-2"),
    ])
    answer_only = _observation([])

    policy_a = make_title_region_commitment_policy()
    policy_b = make_title_region_commitment_policy()
    assert policy_a(first_observation) == policy_b(first_observation) == 1
    assert policy_a(second_observation) == policy_b(second_observation) == 2
    assert make_sentence_region_persistence_policy()(answer_only) == 90


def test_relevance_selectors_choose_only_path_touched_nodes() -> None:
    graph_store = _graph_store()
    result = _experiment_result()

    dominant = make_dominant_title_region_selector()(result, graph_store)
    recent = make_recent_relevant_region_selector()(result, graph_store)

    assert dominant.selected_graph_answer in {"sentence-1", "sentence-2", "title-1"}
    assert recent.selected_graph_answer in {"sentence-1", "sentence-2", "title-1"}
    assert dominant.selected_graph_answer != "global-title"
    assert recent.selected_graph_answer != "global-title"
    assert all(node["node_id"] != "global-title" or node["source"] == "raw_final_node" for node in dominant.candidate_nodes)


def test_relevance_comparison_script_smoke_uses_fixed_samples_and_mapper(tmp_path: Path) -> None:
    data_path = _write_fixture(tmp_path)
    output_dir = tmp_path / "relevance-output"

    comparison = compare_hotpotqa_relevance_variants(
        path=data_path,
        split="validation",
        limit=1,
        reward_mode="baseline",
        variants=["baseline", "dominant_title_region_selector"],
        output_dir=output_dir,
    )

    assert comparison["fixed_mapper_name"] == "parent_title"
    assert comparison["fixed_extractor_name"] == "full_sentence"
    assert comparison["sample_indices"] == [0]
    assert comparison["variant_order"] == ["baseline", "dominant_title_region_selector"]
    for variant_name in comparison["variant_order"]:
        payload = comparison["variants"][variant_name]
        assert "target_failure_buckets" in payload["comparison_metrics"]
        assert payload["parent_title_attribution_summary"]["fixed_config"]["compared_mappers"] == [
            "identity",
            "parent_title",
        ]
        assert (output_dir / variant_name / "parent_title" / "hotpotqa_graph_eval_records.jsonl").exists()
        assert (output_dir / variant_name / "identity" / "hotpotqa_graph_eval_records.jsonl").exists()
    assert (output_dir / "hotpotqa_relevance_comparison_summary.json").exists()
