"""Tests for non-privileged HotpotQA entity/title-like mappers."""

from __future__ import annotations

import json
from pathlib import Path

from answer.hotpotqa_entity_title_mapper import (
    ENTITY_TITLE_MAPPER_NAMES,
    make_capitalized_span_mapper,
    make_entity_title_mapper_factory,
    make_identity_mapper,
    make_parent_title_mapper,
)
from graph.graph_store import EdgeRecord, GraphStore
from scripts.compare_hotpotqa_entity_title_mappers import compare_hotpotqa_entity_title_mappers


def _graph_store() -> GraphStore:
    graph_store = GraphStore()
    graph_store.add_node("title-1", node_type="title", text="First Title", metadata={"title": "First Title"})
    graph_store.add_node("title-2", node_type="title", text="Unrelated Title", metadata={"title": "Unrelated Title"})
    graph_store.add_node(
        "sentence-1",
        node_type="sentence",
        text='The film stars Barack Obama in "Love in the Ruins".',
        metadata={"title": "First Title", "sentence_index": 0},
    )
    graph_store.add_node(
        "sentence-2",
        node_type="sentence",
        text="No obvious entity phrase here.",
        metadata={"title": "First Title", "sentence_index": 1},
    )
    graph_store.add_node("global", node_type="sentence", text="Global Forbidden Entity")
    graph_store.add_edge(EdgeRecord(edge_id="e-title-sentence", src="title-1", dst="sentence-1", relation="title_to_sentence"))
    graph_store.add_edge(EdgeRecord(edge_id="e-sentence-title", src="sentence-1", dst="title-1", relation="sentence_to_title"))
    return graph_store


def _record(answer: str = "First Title") -> dict:
    return {
        "id": "hp-mapper-1",
        "question": "Which page is first?",
        "answer": answer,
        "context": {
            "title": ["First Title", "Second Title"],
            "sentences": [
                ['The film stars Barack Obama in "Love in the Ruins".'],
                ["The second sentence."],
            ],
        },
        "supporting_facts": {"title": ["First Title"], "sent_id": [0]},
    }


def _write_fixture(path: Path) -> Path:
    data_path = path / "hotpotqa_mapper_fixture.json"
    data_path.write_text(json.dumps([_record()], ensure_ascii=False), encoding="utf-8")
    return data_path


def test_mapper_factories_build_supported_mappers() -> None:
    for mapper_name in ENTITY_TITLE_MAPPER_NAMES:
        mapper = make_entity_title_mapper_factory(mapper_name)()
        assert callable(mapper)


def test_identity_mapper_matches_selected_node_text_and_records_metadata() -> None:
    mapping = make_identity_mapper()("sentence-1", _graph_store(), {"gold_answer": "ignored"})

    assert mapping.mapper_name == "identity"
    assert mapping.mapped_answer == 'The film stars Barack Obama in "Love in the Ruins".'
    assert mapping.source_node_id == "sentence-1"
    assert mapping.source_node_type == "sentence"
    assert mapping.fallback_occurred is False
    assert mapping.fallback_target is None
    assert mapping.fallback_reason is None


def test_parent_title_mapper_uses_only_selected_node_local_neighborhood() -> None:
    graph_store = _graph_store()

    first = make_parent_title_mapper()("sentence-1", graph_store, {"supporting_facts": ["ignored"]})
    second = make_parent_title_mapper()("sentence-1", graph_store, {"gold_answer": "ignored"})

    assert first.to_dict() == second.to_dict()
    assert first.mapped_answer == "First Title"
    assert first.source_node_id == "title-1"
    assert first.source_node_type == "title"
    assert "Unrelated Title" not in first.mapped_answer
    assert first.fallback_occurred is False


def test_parent_title_mapper_falls_back_with_reason_when_no_local_title_exists() -> None:
    mapping = make_parent_title_mapper()("sentence-2", _graph_store(), None)

    assert mapping.fallback_occurred is True
    assert mapping.fallback_target == "identity"
    assert mapping.fallback_reason == "no_parent_title_found"
    assert mapping.mapped_answer == "No obvious entity phrase here."


def test_capitalized_span_mapper_uses_current_sentence_text_only() -> None:
    graph_store = _graph_store()

    mapping = make_capitalized_span_mapper()("sentence-1", graph_store, {"gold_answer": "ignored"})

    assert mapping.mapped_answer == "Love in the Ruins"
    assert mapping.source_node_id == "sentence-1"
    assert mapping.source_node_type == "sentence"
    assert "Global Forbidden Entity" not in mapping.mapped_answer
    assert mapping.fallback_occurred is False


def test_capitalized_span_mapper_fallback_records_reason_and_target() -> None:
    mapping = make_capitalized_span_mapper()("sentence-2", _graph_store(), None)

    assert mapping.fallback_occurred is True
    assert mapping.fallback_target == "identity"
    assert mapping.fallback_reason == "no_capitalized_span_found"
    assert mapping.mapped_answer == "No obvious entity phrase here."


def test_entity_title_mapper_comparison_smoke_uses_fixed_extractor(tmp_path: Path) -> None:
    data_path = _write_fixture(tmp_path)
    output_dir = tmp_path / "mapper-comparison"

    comparison = compare_hotpotqa_entity_title_mappers(
        path=data_path,
        split="validation",
        limit=1,
        reward_mode="baseline",
        policy_name="sentence_chain",
        selector_name="latest_sentence",
        mappers=["identity", "parent_title", "capitalized_span"],
        output_dir=output_dir,
    )

    assert comparison["fixed_extractor_name"] == "full_sentence"
    assert comparison["mapper_order"] == ["identity", "parent_title", "capitalized_span"]
    for mapper_name in comparison["mapper_order"]:
        payload = comparison["mappers"][mapper_name]
        assert "eval_summary" in payload
        assert "error_summary" in payload
        assert "answer_type_summary" in payload
        assert "entity_title_like" in payload["comparison_metrics"]
        assert payload["sample_answers"][0]["answer_mapper_name"] == mapper_name
        assert "fallback_reason" in payload["sample_answers"][0]
        assert (output_dir / mapper_name / "hotpotqa_graph_eval_records.jsonl").exists()
        assert (output_dir / mapper_name / "hotpotqa_error_summary.json").exists()
        assert (output_dir / mapper_name / "hotpotqa_answer_type_summary.json").exists()
    assert (output_dir / "hotpotqa_entity_title_mapper_comparison_summary.json").exists()
