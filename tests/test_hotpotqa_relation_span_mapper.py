"""Tests for HotpotQA relation/span mapper branch."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from answer.hotpotqa_answer_selector import make_answer_selector_factory
from answer.hotpotqa_entity_title_mapper import make_entity_title_mapper_factory
from answer.hotpotqa_relation_span_mapper import (
    PATTERN_RULES,
    make_clause_relation_mapper,
    make_pattern_span_mapper,
    make_relation_span_mapper_factory,
)
from answer.hotpotqa_yesno_mapper import make_yesno_mapper_factory
from data.benchmarks.hotpotqa import record_to_benchmark_example
from graph.graph_store import EdgeRecord, GraphStore
from graph.hotpotqa_graph_runtime import run_hotpotqa_graph_experiment_for_example
from scripts.compare_hotpotqa_relation_span_mappers import compare_hotpotqa_relation_span_mappers


def _graph_store() -> GraphStore:
    graph_store = GraphStore()
    graph_store.add_node("title-1", node_type="title", text="First Title")
    graph_store.add_node("sentence-1", node_type="sentence", text="He served as the Chief of Protocol, before leaving.")
    graph_store.add_edge(EdgeRecord(edge_id="e1", src="title-1", dst="sentence-1", relation="title_to_sentence"))
    graph_store.add_edge(EdgeRecord(edge_id="e2", src="sentence-1", dst="title-1", relation="sentence_to_title"))
    return graph_store


def _context() -> dict:
    return {
        "query_text": "What role did he serve as?",
        "answer_type_label": "descriptive_span_or_relation",
        "answer_selection": {
            "candidate_nodes": [
                {
                    "node_id": "sentence-1",
                    "node_type": "sentence",
                    "text": "He served as the Chief of Protocol, before leaving.",
                    "source": "path",
                    "step_idx": 1,
                }
            ]
        },
        "base_mapping": {"mapped_answer": "He served as the Chief of Protocol, before leaving."},
    }


def _fixture_record(answer: str = "complex descriptive relation answer phrase example") -> dict:
    return {
        "id": f"hp-relation-{answer[:8].replace(' ', '-')}",
        "question": "What role did the person serve as?",
        "answer": answer,
        "context": {
            "title": ["First Title", "Other Page"],
            "sentences": [
                ["He served as the Chief of Protocol, before leaving."],
                ["Other Page is different from First Title."],
            ],
        },
        "supporting_facts": {"title": ["First Title"], "sent_id": [0]},
    }


def _write_fixture(path: Path) -> Path:
    data_path = path / "hotpotqa_relation_span_fixture.json"
    data_path.write_text(
        json.dumps([_fixture_record(), _fixture_record("First Title")], ensure_ascii=False),
        encoding="utf-8",
    )
    return data_path


def test_pattern_list_is_fixed_and_pattern_mapper_records_candidate_spans() -> None:
    pattern_names = [name for name, _ in PATTERN_RULES]
    assert pattern_names == [
        "served_as",
        "was_the",
        "born_in",
        "part_of",
        "known_for",
        "located_in",
        "member_of",
        "based_in",
    ]

    mapping = make_pattern_span_mapper()("sentence-1", _graph_store(), _context())

    assert mapping.mapped_answer == "Chief of Protocol"
    assert mapping.selected_span == "Chief of Protocol"
    assert mapping.selected_span_reason == "served_as"
    assert mapping.fallback_occurred is False
    assert mapping.candidate_spans
    assert mapping.metadata["candidate_spans"]
    json.dumps(mapping.to_dict(), ensure_ascii=False)


def test_clause_mapper_records_candidate_spans_and_is_deterministic() -> None:
    graph_store = _graph_store()
    mapper = make_clause_relation_mapper()

    first = mapper("sentence-1", graph_store, _context())
    second = mapper("sentence-1", graph_store, _context())

    assert first.to_dict() == second.to_dict()
    assert first.candidate_spans
    assert first.selected_span
    assert first.selected_span_reason == "shortest_non_empty_clause"


def test_relation_span_mapper_rejects_context_without_relation_label() -> None:
    with pytest.raises(ValueError):
        make_relation_span_mapper_factory("pattern_span")()(
            "sentence-1",
            _graph_store(),
            {"answer_type_label": "multi_token_entity_or_title_like"},
        )


def test_runtime_routes_relation_span_only_without_passing_gold_to_mapper() -> None:
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
    )

    relation_mapping = relation_result.metadata["relation_span_mapping"]
    assert relation_result.metadata["relation_span_mapper_applied"] is True
    assert relation_mapping["metadata"]["whitelist_fields"]
    serialized = json.dumps(relation_mapping["metadata"], ensure_ascii=False)
    assert "gold_answer" not in serialized
    assert "normalized_gold_answer" not in serialized
    assert entity_result.metadata["relation_span_mapper_applied"] is False
    assert entity_result.metadata["relation_span_mapping"] is None


def test_compare_relation_span_mappers_script_smoke(tmp_path: Path) -> None:
    data_path = _write_fixture(tmp_path)
    output_dir = tmp_path / "relation-span-output"

    comparison = compare_hotpotqa_relation_span_mappers(
        path=data_path,
        split="validation",
        limit=2,
        reward_mode="baseline",
        mappers=["identity_relation_span", "pattern_span"],
        output_dir=output_dir,
    )

    assert comparison["fixed_base_mapper_name"] == "parent_title"
    assert comparison["fixed_yesno_mapper_name"] == "sentence_polarity"
    assert comparison["fixed_extractor_name"] == "full_sentence"
    assert comparison["sample_indices"] == [0, 1]
    assert comparison["mappers"]["pattern_span"]["relation_span_decision_summary"]["relation_span_sample_count"] == 1
    assert comparison["mappers"]["pattern_span"]["relation_span_decision_summary"]["mapper_applied_count"] == 1
    assert comparison["mappers"]["identity_relation_span"]["non_relation_span_summary"]["sample_count"] == 1
    assert (output_dir / "pattern_span" / "hotpotqa_graph_eval_records.jsonl").exists()
    assert (output_dir / "hotpotqa_relation_span_mapper_comparison_summary.json").exists()
