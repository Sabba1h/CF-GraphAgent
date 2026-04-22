"""Tests for parent-title mapper stability and attribution analysis."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.hotpotqa_parent_title_analysis import (
    analyze_parent_title_mapper,
    classify_parent_title_attribution,
    parent_title_attribution_record,
    save_parent_title_analysis_outputs,
)
from evaluation.hotpotqa_subset_evaluator import HotpotQAGraphEvalRecord
from scripts.analyze_parent_title_mapper import run_parent_title_mapper_analysis


def _eval_record(
    *,
    mapper_name: str,
    question_id: str = "hp-parent-1",
    gold_answer: str = "First Title",
    projected_answer: str = "First Title",
    exact_match: float = 1.0,
    f1: float = 1.0,
    selected_graph_answer: str | None = "sentence-1",
    answer_source_type: str = "title",
    mapping: dict | None = None,
) -> HotpotQAGraphEvalRecord:
    return HotpotQAGraphEvalRecord(
        question_id=question_id,
        graph_id=f"hotpotqa::{question_id}",
        reward_mode="baseline",
        gold_answer=gold_answer,
        raw_graph_answer="sentence-1",
        selected_graph_answer=selected_graph_answer,
        projected_answer=projected_answer,
        normalized_projected_answer=projected_answer.lower(),
        projected_eval_score=exact_match,
        exact_match=exact_match,
        f1=f1,
        answer_source_type=answer_source_type,
        base_total_reward=0.1,
        total_reward=0.1,
        step_count=3,
        graph_node_count=5,
        graph_edge_count=8,
        policy_name="sentence_chain",
        answer_selector_name="latest_sentence",
        answer_extractor_name="full_sentence",
        answer_mapper_name=mapper_name,
        metadata={
            "answer_mapping": mapping,
            "answer_selection": {
                "candidate_nodes": [
                    {"node_id": "title-1", "node_type": "title", "text": "First Title", "source": "path", "step_idx": 0},
                    {
                        "node_id": "sentence-1",
                        "node_type": "sentence",
                        "text": "This sentence mentions First Title.",
                        "source": "path",
                        "step_idx": 1,
                    },
                ]
            },
            "graph_sentence_nodes": [
                {"node_id": "sentence-1", "node_type": "sentence", "text": "This sentence mentions First Title."}
            ],
        },
    )


def _parent_mapping(mapped_answer: str = "First Title", fallback: bool = False) -> dict:
    return {
        "mapper_name": "parent_title",
        "selected_graph_answer": "sentence-1",
        "mapped_answer": mapped_answer,
        "source_node_id": "title-1",
        "source_node_type": "title",
        "source_text": mapped_answer,
        "fallback_occurred": fallback,
        "fallback_target": "identity" if fallback else None,
        "fallback_reason": "no_parent_title_found" if fallback else None,
        "metadata": {
            "strategy": "adjacent_parent_title",
            "selected_node": {
                "node_id": "sentence-1",
                "node_type": "sentence",
                "text": "This sentence mentions First Title.",
            },
            "local_neighborhood_only": True,
        },
    }


def _fixture_record(answer: str = "First Title") -> dict:
    return {
        "id": "hp-parent-fixture-1",
        "question": "Which page is first?",
        "answer": answer,
        "context": {
            "title": ["First Title", "Second Title"],
            "sentences": [
                ["This sentence mentions First Title."],
                ["The second sentence."],
            ],
        },
        "supporting_facts": {"title": ["First Title"], "sent_id": [0]},
    }


def _write_fixture(path: Path) -> Path:
    data_path = path / "hotpotqa_parent_title_fixture.json"
    data_path.write_text(json.dumps([_fixture_record()], ensure_ascii=False), encoding="utf-8")
    return data_path


def test_parent_title_attribution_record_is_serializable() -> None:
    identity = _eval_record(mapper_name="identity", projected_answer="This sentence mentions First Title.", exact_match=0.0, f1=0.5, answer_source_type="sentence")
    parent = _eval_record(mapper_name="parent_title", mapping=_parent_mapping())

    record = parent_title_attribution_record(identity, parent)
    payload = record.to_dict()

    assert record.attribution_bucket == "parent_title_exact_match_success"
    assert record.parent_title_matches_gold is True
    assert record.selected_node_type == "sentence"
    json.dumps(payload)


def test_parent_title_failure_bucket_classification_is_transparent() -> None:
    assert (
        classify_parent_title_attribution(
            answer_type="multi_token_entity_or_title_like",
            identity_exact_match=0.0,
            parent_title_exact_match=0.0,
            delta_f1=0.0,
            parent_title_matches_gold=False,
            parent_title_fallback_occurred=False,
            selected_node_type="sentence",
            touched_sentence=True,
            any_touched_sentence_contains_gold=True,
            selected_sentence_contains_gold=True,
            gold_in_any_sentence=True,
        )
        == "selected_sentence_parent_title_wrong"
    )
    assert (
        classify_parent_title_attribution(
            answer_type="descriptive_span_or_relation",
            identity_exact_match=0.0,
            parent_title_exact_match=0.0,
            delta_f1=0.0,
            parent_title_matches_gold=False,
            parent_title_fallback_occurred=False,
            selected_node_type="sentence",
            touched_sentence=True,
            any_touched_sentence_contains_gold=True,
            selected_sentence_contains_gold=True,
            gold_in_any_sentence=True,
        )
        == "needs_span_or_relation_not_title"
    )


def test_parent_title_summary_and_save_outputs(tmp_path: Path) -> None:
    identity = _eval_record(
        mapper_name="identity",
        projected_answer="This sentence mentions First Title.",
        exact_match=0.0,
        f1=0.5,
        answer_source_type="sentence",
    )
    parent = _eval_record(mapper_name="parent_title", mapping=_parent_mapping())
    result = analyze_parent_title_mapper([identity], [parent], scale_limits=[1, 500])
    records_path, failure_path, stability_path = save_parent_title_analysis_outputs(result, tmp_path / "parent-title")

    assert result.summary["sample_count"] == 1
    assert result.summary["scale_curve"]["1"]["delta"]["avg_exact_match"] == 1.0
    assert result.summary["entity_title_like"]["delta"]["avg_exact_match"] == 1.0
    assert records_path.exists()
    assert failure_path.exists()
    assert stability_path.exists()


def test_parent_title_analysis_script_smoke_uses_fixed_extractor(tmp_path: Path) -> None:
    data_path = _write_fixture(tmp_path)
    output_dir = tmp_path / "parent-title-output"

    result = run_parent_title_mapper_analysis(
        path=data_path,
        split="validation",
        limit=1,
        reward_mode="baseline",
        policy_name="sentence_chain",
        selector_name="latest_sentence",
        scale_limits=[1],
        output_dir=output_dir,
    )

    assert result["fixed_extractor_name"] == "full_sentence"
    assert result["parent_title_analysis"]["fixed_config"]["fixed_extractor_names"] == ["full_sentence"]
    assert (output_dir / "identity_eval_records.jsonl").exists()
    assert (output_dir / "parent_title_eval_records.jsonl").exists()
    assert (output_dir / "parent_title_failure_attribution.json").exists()
    assert (output_dir / "parent_title_stability_summary.json").exists()
