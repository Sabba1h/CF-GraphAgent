"""Tests for non-privileged HotpotQA answer extractors."""

from __future__ import annotations

import json
from pathlib import Path

from answer.hotpotqa_answer_extractor import (
    ANSWER_EXTRACTOR_NAMES,
    make_answer_extractor_factory,
    make_answer_like_span_extractor,
    make_clause_trim_extractor,
    make_full_sentence_extractor,
)
from graph.graph_store import GraphStore
from scripts.compare_hotpotqa_answer_extractors import compare_hotpotqa_answer_extractors


def _graph_store() -> GraphStore:
    graph_store = GraphStore()
    graph_store.add_node("title", node_type="title", text="First Title", metadata={"title": "First Title"})
    graph_store.add_node(
        "sentence",
        node_type="sentence",
        text='Barack Obama was born in 1961, in Honolulu, Hawaii.',
        metadata={"title": "First Title", "sentence_index": 0},
    )
    graph_store.add_node(
        "quoted_sentence",
        node_type="sentence",
        text='The film was retitled "Almost a Bride" before release.',
        metadata={"title": "Film Title", "sentence_index": 1},
    )
    graph_store.add_node(
        "plain_sentence",
        node_type="sentence",
        text="This sentence has no obvious compact span",
        metadata={"title": "Plain Title", "sentence_index": 2},
    )
    graph_store.add_node("global", node_type="sentence", text="Do not scan this global node")
    return graph_store


def _record(answer: str = "First Title") -> dict:
    return {
        "id": "hp-extractor-1",
        "question": "Which page is first?",
        "answer": answer,
        "context": {
            "title": ["First Title", "Second Title"],
            "sentences": [
                ["The first sentence mentions First Title."],
                ["The second sentence."],
            ],
        },
        "supporting_facts": {"title": ["First Title"], "sent_id": [0]},
    }


def _write_fixture(path: Path) -> Path:
    data_path = path / "hotpotqa_extractor_fixture.json"
    data_path.write_text(json.dumps([_record()], ensure_ascii=False), encoding="utf-8")
    return data_path


def test_extractor_factory_builds_supported_extractors() -> None:
    for extractor_name in ANSWER_EXTRACTOR_NAMES:
        extractor = make_answer_extractor_factory(extractor_name)()
        assert callable(extractor)


def test_full_sentence_extractor_matches_selected_node_text() -> None:
    graph_store = _graph_store()
    extraction = make_full_sentence_extractor()("sentence", graph_store, {"gold_answer": "ignored"})

    assert extraction.extracted_answer == 'Barack Obama was born in 1961, in Honolulu, Hawaii.'
    assert extraction.fallback_occurred is False
    assert extraction.fallback_target is None
    assert extraction.fallback_reason is None


def test_clause_trim_extractor_is_deterministic_and_shorter_than_full_sentence() -> None:
    graph_store = _graph_store()
    extractor = make_clause_trim_extractor()

    first = extractor("sentence", graph_store, {"supporting_facts": ["ignored"]})
    second = extractor("sentence", graph_store, {"supporting_facts": ["ignored"]})

    assert first.to_dict() == second.to_dict()
    assert first.extracted_answer
    assert len(first.extracted_answer) < len(first.source_text)
    assert first.fallback_occurred is False


def test_answer_like_span_extractor_uses_local_string_rules_only() -> None:
    graph_store = _graph_store()

    numeric = make_answer_like_span_extractor()("sentence", graph_store, {"gold_answer": "ignored"})
    quoted = make_answer_like_span_extractor()("quoted_sentence", graph_store, None)

    assert numeric.extracted_answer == "1961"
    assert quoted.extracted_answer == "Almost a Bride"
    assert "Do not scan this global node" not in numeric.extracted_answer
    assert numeric.fallback_occurred is False
    assert quoted.fallback_occurred is False


def test_answer_like_span_fallback_records_reason_and_target() -> None:
    graph_store = _graph_store()

    extraction = make_answer_like_span_extractor()("plain_sentence", graph_store, None)

    assert extraction.fallback_occurred is True
    assert extraction.fallback_target in {"clause_trim", "full_sentence"}
    assert extraction.fallback_reason == "no_answer_like_span_found"
    assert extraction.extracted_answer


def test_unknown_or_question_like_node_falls_back_to_empty() -> None:
    graph_store = _graph_store()

    extraction = make_clause_trim_extractor()(None, graph_store, None)

    assert extraction.fallback_occurred is True
    assert extraction.fallback_target == "empty"
    assert extraction.fallback_reason == "empty_or_unknown_answer"
    assert extraction.extracted_answer == ""


def test_answer_extractor_comparison_smoke_uses_fixed_policy_and_selector(tmp_path: Path) -> None:
    data_path = _write_fixture(tmp_path)
    output_dir = tmp_path / "extractor-comparison"

    comparison = compare_hotpotqa_answer_extractors(
        path=data_path,
        split="validation",
        limit=1,
        reward_mode="baseline",
        policy_name="sentence_chain",
        selector_name="latest_sentence",
        extractors=["full_sentence", "clause_trim", "answer_like_span"],
        output_dir=output_dir,
    )

    assert comparison["policy_name"] == "sentence_chain"
    assert comparison["selector_name"] == "latest_sentence"
    assert comparison["extractor_order"] == ["full_sentence", "clause_trim", "answer_like_span"]
    for extractor_name in comparison["extractor_order"]:
        payload = comparison["extractors"][extractor_name]
        assert "eval_summary" in payload
        assert "error_summary" in payload
        assert "sentence_hit_summary" in payload
        assert payload["sample_answers"][0]["answer_extractor_name"] == extractor_name
        assert "fallback_reason" in payload["sample_answers"][0]
        assert (output_dir / extractor_name / "hotpotqa_graph_eval_records.jsonl").exists()
        assert (output_dir / extractor_name / "hotpotqa_error_summary.json").exists()
        assert (output_dir / extractor_name / "hotpotqa_sentence_hit_summary.json").exists()
    assert (output_dir / "hotpotqa_answer_extractor_comparison_summary.json").exists()
