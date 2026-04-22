"""Tests for HotpotQA answer type-aware diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.hotpotqa_answer_type_diagnostic import (
    analyze_hotpotqa_answer_types,
    answer_type_record_from_eval_record,
    classify_hotpotqa_answer_type,
    save_answer_type_outputs,
)
from evaluation.hotpotqa_subset_evaluator import HotpotQAGraphEvalRecord
from scripts.analyze_hotpotqa_answer_types import run_hotpotqa_answer_type_analysis


def _eval_record(
    *,
    question_id: str = "hp-type-1",
    gold_answer: str = "Chicago",
    projected_answer: str = "Chicago is a city.",
    policy_name: str = "sentence_chain",
    answer_selector_name: str = "latest_sentence",
    answer_extractor_name: str = "full_sentence",
    reward_mode: str = "baseline",
    answer_source_type: str = "sentence",
    selected_graph_answer: str | None = "s1",
    exact_match: float = 0.0,
    f1: float = 0.5,
) -> HotpotQAGraphEvalRecord:
    return HotpotQAGraphEvalRecord(
        question_id=question_id,
        graph_id=f"hotpotqa::{question_id}",
        reward_mode=reward_mode,
        gold_answer=gold_answer,
        raw_graph_answer="t1",
        selected_graph_answer=selected_graph_answer,
        projected_answer=projected_answer,
        normalized_projected_answer=projected_answer.lower(),
        projected_eval_score=exact_match,
        exact_match=exact_match,
        f1=f1,
        answer_source_type=answer_source_type,
        base_total_reward=0.1,
        total_reward=0.1,
        step_count=2,
        graph_node_count=5,
        graph_edge_count=8,
        policy_name=policy_name,
        answer_selector_name=answer_selector_name,
        answer_extractor_name=answer_extractor_name,
        metadata={
            "answer_selection": {
                "candidate_nodes": [
                    {"node_id": "q", "node_type": "question", "text": "Question", "source": "path", "step_idx": 0},
                    {
                        "node_id": "s1",
                        "node_type": "sentence",
                        "text": "Chicago is a city in Illinois.",
                        "source": "path",
                        "step_idx": 1,
                    },
                ]
            },
            "graph_sentence_nodes": [
                {"node_id": "s1", "node_type": "sentence", "text": "Chicago is a city in Illinois."},
                {"node_id": "s2", "node_type": "sentence", "text": "Another sentence."},
            ],
        },
    )


def _fixture_record(answer: str = "First Title") -> dict:
    return {
        "id": "hp-type-fixture-1",
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
    data_path = path / "hotpotqa_answer_type_fixture.json"
    data_path.write_text(json.dumps([_fixture_record()], ensure_ascii=False), encoding="utf-8")
    return data_path


def test_answer_type_classification_is_transparent() -> None:
    assert classify_hotpotqa_answer_type("yes") == "yes_no"
    assert classify_hotpotqa_answer_type("No") == "yes_no"
    assert classify_hotpotqa_answer_type("1961") == "numeric_or_date"
    assert classify_hotpotqa_answer_type("January 1 1961") == "numeric_or_date"
    assert classify_hotpotqa_answer_type("Chicago") == "single_token_entity_like"
    assert classify_hotpotqa_answer_type("Chief of Protocol") == "multi_token_entity_or_title_like"
    assert classify_hotpotqa_answer_type("the office responsible for diplomatic protocol") == (
        "descriptive_span_or_relation"
    )


def test_answer_type_record_is_json_serializable() -> None:
    record = answer_type_record_from_eval_record(_eval_record())
    payload = record.to_dict()

    assert record.answer_type == "single_token_entity_like"
    assert record.sentence_hit_applicability == "applicable"
    assert record.any_touched_sentence_contains_gold is True
    json.dumps(payload)


def test_yes_no_sentence_hit_metrics_are_not_applicable() -> None:
    result = analyze_hotpotqa_answer_types([
        _eval_record(question_id="yes-no", gold_answer="yes", projected_answer="A sentence with yes.")
    ])
    bucket = result.summary["answer_type_buckets"]["yes_no"]
    sentence_hit = bucket["sentence_hit"]

    assert bucket["count"] == 1
    assert sentence_hit["gold_sentence_hit_rate"] == "not_applicable"
    assert sentence_hit["selected_sentence_contains_gold_rate"] == "not_applicable"
    assert sentence_hit["gold_in_any_sentence_rate"] == "not_applicable"


def test_answer_type_summary_and_save_outputs(tmp_path: Path) -> None:
    result = analyze_hotpotqa_answer_types([
        _eval_record(question_id="r1", gold_answer="Chicago"),
        _eval_record(question_id="r2", gold_answer="Chief of Protocol", projected_answer="Chief of Protocol"),
    ])
    records_path, summary_path = save_answer_type_outputs(result, tmp_path / "answer-types")

    assert result.summary["sample_count"] == 2
    assert result.summary["answer_type_buckets"]["single_token_entity_like"]["count"] == 1
    assert result.summary["answer_type_buckets"]["multi_token_entity_or_title_like"]["count"] == 1
    assert records_path.exists()
    assert summary_path.exists()


def test_answer_type_diagnostic_rejects_mixed_pipeline_configs() -> None:
    with pytest.raises(ValueError, match="fixed reward_mode/policy/selector/extractor"):
        analyze_hotpotqa_answer_types([
            _eval_record(question_id="r1", policy_name="sentence_chain"),
            _eval_record(question_id="r2", policy_name="baseline"),
        ])


def test_answer_type_script_smoke(tmp_path: Path) -> None:
    data_path = _write_fixture(tmp_path)
    output_dir = tmp_path / "answer-type-output"

    result = run_hotpotqa_answer_type_analysis(
        path=data_path,
        split="validation",
        limit=1,
        reward_mode="baseline",
        policy_name="sentence_chain",
        selector_name="latest_sentence",
        extractor_name="full_sentence",
        output_dir=output_dir,
    )

    assert result.summary["sample_count"] == 1
    assert result.summary["fixed_config"]["policy_names"] == ["sentence_chain"]
    assert result.summary["fixed_config"]["answer_selector_names"] == ["latest_sentence"]
    assert result.summary["fixed_config"]["answer_extractor_names"] == ["full_sentence"]
    assert (output_dir / "hotpotqa_answer_type_records.jsonl").exists()
    assert (output_dir / "hotpotqa_answer_type_summary.json").exists()
