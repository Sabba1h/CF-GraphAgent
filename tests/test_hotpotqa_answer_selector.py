"""Tests for non-privileged HotpotQA answer selectors."""

from __future__ import annotations

import json
from pathlib import Path

from core.experiment_config import ExperimentConfig
from core.experiment_result import ExperimentResult, ExperimentStepTrace
from graph.graph_store import GraphStore
from answer.hotpotqa_answer_selector import (
    collect_path_candidate_nodes,
    make_answer_selector_factory,
    make_latest_sentence_selector,
    make_prefer_sentence_over_title_selector,
    make_raw_final_node_selector,
)
from scripts.compare_hotpotqa_answer_selectors import compare_hotpotqa_answer_selectors


def _graph_store() -> GraphStore:
    graph_store = GraphStore()
    graph_store.add_node("q", node_type="question", text="Question text")
    graph_store.add_node("t1", node_type="title", text="First Title")
    graph_store.add_node("s1", node_type="sentence", text="The first sentence.")
    graph_store.add_node("t2", node_type="title", text="Second Title")
    return graph_store


def _trace(step_idx: int, action: int, src: str, dst: str) -> ExperimentStepTrace:
    return ExperimentStepTrace(
        step_idx=step_idx,
        action=action,
        reward_mode="baseline",
        base_reward=0.1,
        reward=0.1,
        reward_breakdown=None,
        metadata={
            "info": {
                "expanded_edge": {
                    "edge_id": f"e{step_idx}",
                    "src": src,
                    "relation": "title_to_sentence",
                    "dst": dst,
                }
            },
            "observation": {
                "candidate_actions": [
                    {
                        "candidate_id": action,
                        "action_type": "EXPAND_EDGE",
                        "metadata": {"src": src, "dst": dst, "relation": "title_to_sentence"},
                    }
                ]
            },
        },
    )


def _experiment_result() -> ExperimentResult:
    return ExperimentResult(
        config=ExperimentConfig(),
        final_answer="t1",
        total_reward=0.2,
        base_total_reward=0.2,
        step_traces=[
            _trace(0, 0, "q", "t1"),
            _trace(1, 1, "t1", "s1"),
        ],
        metadata={
            "raw_graph_answer": "t1",
            "gold_answer": "First Title",
            "supporting_facts": [{"title": "should-not-be-read"}],
            "normalized_gold_answer": "should-not-be-read",
        },
    )


def _record(answer: str = "First Title") -> dict:
    return {
        "id": "hp-selector-1",
        "question": "Which page is first?",
        "answer": answer,
        "context": {
            "title": ["First Title", "Second Title"],
            "sentences": [
                ["The first sentence."],
                ["The second sentence."],
            ],
        },
        "supporting_facts": {"title": ["First Title"], "sent_id": [0]},
    }


def _write_fixture(path: Path) -> Path:
    data_path = path / "hotpotqa_selector_fixture.json"
    data_path.write_text(json.dumps([_record()], ensure_ascii=False), encoding="utf-8")
    return data_path


def test_raw_selector_keeps_raw_final_node() -> None:
    graph_store = _graph_store()
    result = _experiment_result()

    selection = make_raw_final_node_selector()(result, graph_store)

    assert selection.selector_name == "raw_final_node"
    assert selection.raw_graph_answer == "t1"
    assert selection.selected_graph_answer == "t1"


def test_collect_path_candidate_nodes_uses_only_touched_nodes_and_raw_answer() -> None:
    graph_store = _graph_store()
    graph_store.add_node("global_sentence", node_type="sentence", text="Must not be scanned")

    nodes = collect_path_candidate_nodes(
        result=_experiment_result(),
        graph_store=graph_store,
        raw_graph_answer="t2",
    )

    node_ids = [node["node_id"] for node in nodes]
    assert node_ids == ["q", "t1", "s1", "t2"]
    assert "global_sentence" not in node_ids


def test_latest_sentence_selector_is_deterministic_and_prefers_reached_sentence() -> None:
    graph_store = _graph_store()
    result = _experiment_result()
    selector = make_latest_sentence_selector()

    first = selector(result, graph_store)
    second = selector(result, graph_store)

    assert first.selected_graph_answer == "s1"
    assert second.selected_graph_answer == "s1"
    assert first.to_dict() == second.to_dict()
    assert first.selection_source == "sentence"


def test_prefer_sentence_selector_falls_back_to_reached_title_without_sentence() -> None:
    graph_store = _graph_store()
    result = ExperimentResult(
        config=ExperimentConfig(),
        final_answer="t1",
        step_traces=[_trace(0, 0, "q", "t1")],
        metadata={"raw_graph_answer": "t1"},
    )

    selection = make_prefer_sentence_over_title_selector()(result, graph_store)

    assert selection.selected_graph_answer == "t1"
    assert selection.selection_source == "title"


def test_answer_selector_factory_builds_supported_selectors() -> None:
    for selector_name in ("raw_final_node", "latest_sentence", "prefer_sentence_over_title", "latest_non_question"):
        selector = make_answer_selector_factory(selector_name)()
        assert callable(selector)


def test_answer_selector_comparison_smoke_keeps_raw_selected_projected_outputs(tmp_path: Path) -> None:
    data_path = _write_fixture(tmp_path)
    output_dir = tmp_path / "selector-comparison"

    comparison = compare_hotpotqa_answer_selectors(
        path=data_path,
        limit=1,
        split="validation",
        reward_mode="baseline",
        policy_name="baseline",
        selectors=["raw_final_node", "latest_sentence"],
        max_steps=4,
        candidate_top_k=4,
        output_dir=output_dir,
    )

    assert comparison["selector_order"] == ["raw_final_node", "latest_sentence"]
    for selector_name in comparison["selector_order"]:
        payload = comparison["selectors"][selector_name]
        assert "eval_summary" in payload
        assert "error_summary" in payload
        assert payload["sample_answers"]
        sample = payload["sample_answers"][0]
        assert "raw_graph_answer" in sample
        assert "selected_graph_answer" in sample
        assert "projected_answer" in sample
        assert (output_dir / selector_name / "hotpotqa_graph_eval_records.jsonl").exists()
        assert (output_dir / selector_name / "hotpotqa_error_summary.json").exists()
    assert (output_dir / "hotpotqa_answer_selector_comparison_summary.json").exists()
