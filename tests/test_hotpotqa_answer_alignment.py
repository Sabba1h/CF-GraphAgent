"""Tests for HotpotQA graph answer projection."""

from __future__ import annotations

from answer.graph_answer_projector import GraphAnswerProjector
from answer.hotpotqa_answer_adapter import HotpotQAAnswerAdapter
from data.benchmarks.hotpotqa import record_to_benchmark_example
from graph.hotpotqa_graph_runtime import build_hotpotqa_graph_runtime, run_hotpotqa_graph_experiment_for_example


def _record() -> dict:
    return {
        "id": "hp-answer-1",
        "question": "Which page is first?",
        "answer": "First Title",
        "context": {
            "title": ["First Title", "Second Title"],
            "sentences": [
                ["The first sentence."],
                ["The second sentence."],
            ],
        },
        "supporting_facts": {"title": ["First Title"], "sent_id": [0]},
    }


def test_title_node_projects_to_title_text() -> None:
    runtime = build_hotpotqa_graph_runtime(record_to_benchmark_example(_record()))
    projection = GraphAnswerProjector().project(
        raw_graph_answer="hotpotqa::hp-answer-1::title::0",
        graph_store=runtime.graph_store,
    )

    assert projection.projected_answer_text == "First Title"
    assert projection.node_type == "title"
    assert projection.projection_source == "title"


def test_sentence_node_projects_to_sentence_text() -> None:
    runtime = build_hotpotqa_graph_runtime(record_to_benchmark_example(_record()))
    projection = GraphAnswerProjector().project(
        raw_graph_answer="hotpotqa::hp-answer-1::sentence::0::0",
        graph_store=runtime.graph_store,
    )

    assert projection.projected_answer_text == "The first sentence."
    assert projection.node_type == "sentence"
    assert projection.projection_source == "sentence"


def test_question_and_unknown_nodes_fallback_to_empty_string() -> None:
    runtime = build_hotpotqa_graph_runtime(record_to_benchmark_example(_record()))
    projector = GraphAnswerProjector()

    question_projection = projector.project(
        raw_graph_answer="hotpotqa::hp-answer-1::question",
        graph_store=runtime.graph_store,
    )
    unknown_projection = projector.project(
        raw_graph_answer="hotpotqa::hp-answer-1::missing",
        graph_store=runtime.graph_store,
    )

    assert question_projection.projected_answer_text == ""
    assert question_projection.metadata["projection_fallback_reason"] == "question_node"
    assert unknown_projection.projected_answer_text == ""
    assert unknown_projection.metadata["projection_fallback_reason"] == "unknown_node"


def test_hotpotqa_adapter_normalizes_and_evaluates_projected_answer() -> None:
    runtime = build_hotpotqa_graph_runtime(record_to_benchmark_example(_record()))
    alignment = HotpotQAAnswerAdapter().align(
        raw_graph_answer="hotpotqa::hp-answer-1::title::0",
        graph_store=runtime.graph_store,
        gold_answer="first title",
    )

    assert alignment.projected_answer == "First Title"
    assert alignment.normalized_projected_answer == "first title"
    assert alignment.normalized_gold_answer == "first title"
    assert alignment.projected_eval_score == 1.0
    assert alignment.is_correct is True


def test_hotpotqa_graph_experiment_records_projected_answer_without_overwriting_final_answer() -> None:
    example = record_to_benchmark_example(_record())
    runtime, result = run_hotpotqa_graph_experiment_for_example(
        example,
        reward_mode="baseline",
        max_steps=3,
        candidate_top_k=4,
        min_expand_steps=1,
    )

    assert result.final_answer == result.metadata["raw_graph_answer"]
    assert result.final_answer.startswith(runtime.local_graph.graph_id)
    assert result.metadata["projected_answer"] != result.final_answer
    assert result.metadata["normalized_projected_answer"] == "first title"
    assert result.metadata["gold_answer"] == "First Title"
    assert "projected_eval_score" in result.metadata
