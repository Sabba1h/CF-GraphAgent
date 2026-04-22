"""Tests for HotpotQA yes/no mapper branch."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from answer.hotpotqa_yesno_mapper import (
    make_identity_yesno_mapper,
    make_sentence_polarity_mapper,
    make_title_sentence_consistency_yesno_mapper,
)
from graph.graph_store import EdgeRecord, GraphStore
from graph.hotpotqa_graph_runtime import run_hotpotqa_graph_experiment_for_example
from answer.hotpotqa_answer_selector import make_answer_selector_factory
from answer.hotpotqa_entity_title_mapper import make_entity_title_mapper_factory
from data.benchmarks.hotpotqa import record_to_benchmark_example
from scripts.compare_hotpotqa_yesno_mappers import compare_hotpotqa_yesno_mappers


def _graph_store() -> GraphStore:
    graph_store = GraphStore()
    graph_store.add_node("title-1", node_type="title", text="First Title")
    graph_store.add_node("sentence-1", node_type="sentence", text="Both pages discuss First Title.")
    graph_store.add_edge(EdgeRecord(edge_id="e1", src="title-1", dst="sentence-1", relation="title_to_sentence"))
    graph_store.add_edge(EdgeRecord(edge_id="e2", src="sentence-1", dst="title-1", relation="sentence_to_title"))
    return graph_store


def _context() -> dict:
    return {
        "query_text": "Are both pages about First Title?",
        "answer_type_label": "yes_no",
        "answer_selection": {
            "candidate_nodes": [
                {
                    "node_id": "title-1",
                    "node_type": "title",
                    "text": "First Title",
                    "source": "path",
                    "step_idx": 0,
                },
                {
                    "node_id": "sentence-1",
                    "node_type": "sentence",
                    "text": "Both pages discuss First Title.",
                    "source": "path",
                    "step_idx": 1,
                },
            ]
        },
        "base_mapping": {"mapped_answer": "First Title", "source_node_id": "title-1"},
    }


def _fixture_record(answer: str = "yes") -> dict:
    return {
        "id": f"hp-yesno-{answer}",
        "question": "Are both pages about First Title?",
        "answer": answer,
        "context": {
            "title": ["First Title", "Other Page"],
            "sentences": [
                ["Both pages discuss First Title."],
                ["Other Page is different from First Title."],
            ],
        },
        "supporting_facts": {"title": ["First Title"], "sent_id": [0]},
    }


def _write_fixture(path: Path) -> Path:
    data_path = path / "hotpotqa_yesno_fixture.json"
    data_path.write_text(
        json.dumps([_fixture_record("yes"), _fixture_record("First Title")], ensure_ascii=False),
        encoding="utf-8",
    )
    return data_path


def test_yesno_mappers_are_deterministic_and_serialize_metadata() -> None:
    graph_store = _graph_store()
    mapper = make_sentence_polarity_mapper()

    first = mapper("sentence-1", graph_store, _context())
    second = mapper("sentence-1", graph_store, _context())

    assert first.to_dict() == second.to_dict()
    assert first.mapped_answer in {"yes", "no"}
    assert first.evidence_strength > 0
    assert first.positive_evidence_count >= 1
    assert first.negative_evidence_count >= 0
    assert first.metadata["whitelist_fields"]
    json.dumps(first.to_dict(), ensure_ascii=False)


def test_yesno_mapper_rejects_context_without_yesno_label() -> None:
    with pytest.raises(ValueError):
        make_identity_yesno_mapper()("sentence-1", _graph_store(), {"answer_type_label": "multi_token_entity_like"})


def test_title_sentence_consistency_uses_local_neighborhood() -> None:
    graph_store = _graph_store()
    mapping = make_title_sentence_consistency_yesno_mapper()("sentence-1", graph_store, _context())

    assert mapping.mapped_answer == "yes"
    assert "sentence-1" in mapping.evidence_node_ids
    assert mapping.metadata["evidence_nodes"]


def test_runtime_routes_yesno_only_without_passing_gold_to_mapper() -> None:
    yes_example = record_to_benchmark_example(_fixture_record("yes"), split="validation")
    non_yes_example = record_to_benchmark_example(_fixture_record("First Title"), split="validation")

    _, yes_result = run_hotpotqa_graph_experiment_for_example(
        yes_example,
        answer_selector=make_answer_selector_factory("latest_sentence")(),
        answer_selector_name="latest_sentence",
        answer_mapper=make_entity_title_mapper_factory("parent_title")(),
        answer_mapper_name="parent_title",
        yesno_mapper=make_sentence_polarity_mapper(),
        yesno_mapper_name="sentence_polarity",
    )
    _, non_yes_result = run_hotpotqa_graph_experiment_for_example(
        non_yes_example,
        answer_selector=make_answer_selector_factory("latest_sentence")(),
        answer_selector_name="latest_sentence",
        answer_mapper=make_entity_title_mapper_factory("parent_title")(),
        answer_mapper_name="parent_title",
        yesno_mapper=make_sentence_polarity_mapper(),
        yesno_mapper_name="sentence_polarity",
    )

    yesno_mapping = yes_result.metadata["yesno_mapping"]
    assert yes_result.metadata["yesno_mapper_applied"] is True
    assert yesno_mapping["metadata"]["answer_type_label"] == "yes_no"
    serialized = json.dumps(yesno_mapping["metadata"], ensure_ascii=False)
    assert "gold_answer" not in serialized
    assert "normalized_gold_answer" not in serialized
    assert non_yes_result.metadata["yesno_mapper_applied"] is False
    assert non_yes_result.metadata["yesno_mapping"] is None


def test_compare_yesno_mappers_script_smoke(tmp_path: Path) -> None:
    data_path = _write_fixture(tmp_path)
    output_dir = tmp_path / "yesno-output"

    comparison = compare_hotpotqa_yesno_mappers(
        path=data_path,
        split="validation",
        limit=2,
        reward_mode="baseline",
        mappers=["identity_yesno", "sentence_polarity"],
        output_dir=output_dir,
    )

    assert comparison["fixed_base_mapper_name"] == "parent_title"
    assert comparison["fixed_extractor_name"] == "full_sentence"
    assert comparison["sample_indices"] == [0, 1]
    assert comparison["mappers"]["sentence_polarity"]["yesno_decision_summary"]["yesno_sample_count"] == 1
    assert comparison["mappers"]["sentence_polarity"]["yesno_decision_summary"]["mapper_applied_count"] == 1
    assert comparison["mappers"]["identity_yesno"]["non_yesno_summary"]["sample_count"] == 1
    assert (output_dir / "sentence_polarity" / "hotpotqa_graph_eval_records.jsonl").exists()
    assert (output_dir / "hotpotqa_yesno_mapper_comparison_summary.json").exists()
