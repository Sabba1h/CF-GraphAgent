"""Tests for HotpotQA relation/span candidate span rankers."""

from __future__ import annotations

import json
from pathlib import Path

from answer.hotpotqa_answer_selector import make_answer_selector_factory
from answer.hotpotqa_entity_title_mapper import make_entity_title_mapper_factory
from answer.hotpotqa_relation_span_mapper import make_pattern_span_mapper, make_relation_span_mapper_factory
from answer.hotpotqa_relation_span_ranker import (
    make_first_candidate_ranker,
    make_pattern_priority_ranker,
    make_relation_span_ranker_factory,
    make_shortest_nonempty_ranker,
)
from answer.hotpotqa_yesno_mapper import make_yesno_mapper_factory
from data.benchmarks.hotpotqa import record_to_benchmark_example
from graph.graph_store import EdgeRecord, GraphStore
from graph.hotpotqa_graph_runtime import run_hotpotqa_graph_experiment_for_example
from graph.hotpotqa_policy_variants import make_policy_factory
from scripts.compare_hotpotqa_relation_span_rankers import compare_hotpotqa_relation_span_rankers


def _graph_store() -> GraphStore:
    graph_store = GraphStore()
    graph_store.add_node("title-1", node_type="title", text="First Title")
    graph_store.add_node(
        "sentence-1",
        node_type="sentence",
        text="He served as the Chief of Protocol, and was the acting director.",
    )
    graph_store.add_edge(EdgeRecord(edge_id="e1", src="title-1", dst="sentence-1", relation="title_to_sentence"))
    graph_store.add_edge(EdgeRecord(edge_id="e2", src="sentence-1", dst="title-1", relation="sentence_to_title"))
    return graph_store


def _context(ranker_name: str | None = None) -> dict:
    context = {
        "query_text": "What role did he serve as?",
        "answer_type_label": "descriptive_span_or_relation",
        "answer_selection": {
            "candidate_nodes": [
                {
                    "node_id": "sentence-1",
                    "node_type": "sentence",
                    "text": "He served as the Chief of Protocol, and was the acting director.",
                    "source": "path",
                    "step_idx": 1,
                }
            ]
        },
        "base_mapping": {"mapped_answer": "He served as the Chief of Protocol, and was the acting director."},
    }
    if ranker_name is not None:
        context["relation_span_ranker"] = make_relation_span_ranker_factory(ranker_name)()
        context["relation_span_ranker_name"] = ranker_name
    return context


def _fixture_record(answer: str = "complex descriptive relation answer phrase example") -> dict:
    return {
        "id": f"hp-ranker-{answer[:8].replace(' ', '-')}",
        "question": "What role did the person serve as?",
        "answer": answer,
        "context": {
            "title": ["First Title", "Other Page"],
            "sentences": [
                ["He served as the Chief of Protocol, and was the acting director."],
                ["Other Page served as the deputy director before leaving."],
            ],
        },
        "supporting_facts": {"title": ["First Title"], "sent_id": [0]},
    }


def _write_fixture(path: Path) -> Path:
    data_path = path / "hotpotqa_relation_span_ranker_fixture.json"
    data_path.write_text(
        json.dumps([_fixture_record(), _fixture_record("First Title")], ensure_ascii=False),
        encoding="utf-8",
    )
    return data_path


def test_first_candidate_ranker_reproduces_batch28_pattern_selection() -> None:
    graph_store = _graph_store()
    default_mapping = make_pattern_span_mapper()("sentence-1", graph_store, _context())
    ranked_mapping = make_pattern_span_mapper()("sentence-1", graph_store, _context("first_candidate"))

    assert ranked_mapping.mapped_answer == default_mapping.mapped_answer
    assert ranked_mapping.selected_span == default_mapping.selected_span
    ranking = ranked_mapping.metadata["relation_span_ranking"]
    assert ranking["ranker_name"] == "first_candidate"
    assert ranking["selected_span_reason"] == "batch28_default_selection_reproduced"
    json.dumps(ranking, ensure_ascii=False)


def test_rankers_are_deterministic_and_use_candidate_metadata_only() -> None:
    candidates = make_pattern_span_mapper()("sentence-1", _graph_store(), _context()).candidate_spans

    for factory in (
        make_first_candidate_ranker,
        make_shortest_nonempty_ranker,
        make_pattern_priority_ranker,
    ):
        ranker = factory()
        first = ranker(candidates, {"gold_answer": "should be ignored", "supporting_facts": ["ignored"]})
        second = ranker(candidates, {"gold_answer": "different ignored gold"})
        assert first.to_dict() == second.to_dict()
        assert first.selected_span
        serialized = json.dumps(first.to_dict(), ensure_ascii=False)
        assert "should be ignored" not in serialized
        assert "supporting_facts" not in serialized


def test_pattern_priority_can_change_span_selection_without_new_discovery() -> None:
    default_mapping = make_pattern_span_mapper()("sentence-1", _graph_store(), _context("first_candidate"))
    priority_mapping = make_pattern_span_mapper()("sentence-1", _graph_store(), _context("pattern_priority"))

    assert default_mapping.candidate_spans == priority_mapping.candidate_spans
    assert priority_mapping.selected_span == "Chief of Protocol"
    assert priority_mapping.metadata["relation_span_ranking"]["ranker_name"] == "pattern_priority"


def test_runtime_routes_ranker_only_for_relation_span() -> None:
    relation_example = record_to_benchmark_example(_fixture_record(), split="validation")
    entity_example = record_to_benchmark_example(_fixture_record("First Title"), split="validation")

    _, relation_result = run_hotpotqa_graph_experiment_for_example(
        relation_example,
        answer_selector=make_answer_selector_factory("latest_sentence")(),
        answer_selector_name="latest_sentence",
        answer_mapper=make_entity_title_mapper_factory("parent_title")(),
        answer_mapper_name="parent_title",
        yesno_mapper=make_yesno_mapper_factory("sentence_polarity")(),
        yesno_mapper_name="sentence_polarity",
        relation_span_mapper=make_relation_span_mapper_factory("pattern_span")(),
        relation_span_mapper_name="pattern_span",
        relation_span_ranker=make_relation_span_ranker_factory("pattern_priority")(),
        relation_span_ranker_name="pattern_priority",
        policy=make_policy_factory("sentence_chain", min_expand_steps=2)(),
        policy_name="sentence_chain",
        min_expand_steps=2,
    )
    _, entity_result = run_hotpotqa_graph_experiment_for_example(
        entity_example,
        answer_selector=make_answer_selector_factory("latest_sentence")(),
        answer_selector_name="latest_sentence",
        answer_mapper=make_entity_title_mapper_factory("parent_title")(),
        answer_mapper_name="parent_title",
        yesno_mapper=make_yesno_mapper_factory("sentence_polarity")(),
        yesno_mapper_name="sentence_polarity",
        relation_span_mapper=make_relation_span_mapper_factory("pattern_span")(),
        relation_span_mapper_name="pattern_span",
        relation_span_ranker=make_relation_span_ranker_factory("pattern_priority")(),
        relation_span_ranker_name="pattern_priority",
        policy=make_policy_factory("sentence_chain", min_expand_steps=2)(),
        policy_name="sentence_chain",
        min_expand_steps=2,
    )

    assert relation_result.metadata["relation_span_ranker_applied"] is True
    assert relation_result.metadata["relation_span_mapping"]["metadata"]["relation_span_ranking"]
    assert entity_result.metadata["relation_span_mapper_applied"] is False
    assert entity_result.metadata["relation_span_ranker_applied"] is False


def test_compare_relation_span_rankers_script_smoke(tmp_path: Path) -> None:
    data_path = _write_fixture(tmp_path)
    output_dir = tmp_path / "relation-span-ranker-output"

    comparison = compare_hotpotqa_relation_span_rankers(
        path=data_path,
        split="validation",
        limit=2,
        reward_mode="baseline",
        rankers=["first_candidate", "pattern_priority"],
        min_expand_steps=2,
        output_dir=output_dir,
    )

    assert comparison["fixed_relation_span_mapper_name"] == "pattern_span"
    assert comparison["sample_indices"] == [0, 1]
    first_summary = comparison["rankers"]["first_candidate"]["relation_span_ranking_summary"]
    assert first_summary["relation_span_sample_count"] == 1
    assert first_summary["ranker_applied_count"] == 1
    assert "no_useful_candidate_span_found" in first_summary["failure_bucket_counts"]
    assert (output_dir / "pattern_priority" / "hotpotqa_graph_eval_records.jsonl").exists()
    assert (output_dir / "hotpotqa_relation_span_ranker_comparison_summary.json").exists()
