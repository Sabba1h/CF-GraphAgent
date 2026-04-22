"""Tests for HotpotQA graph-backed error analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.hotpotqa_error_analysis import (
    GraphStructureBucketConfig,
    PathBehaviorConfig,
    analyze_hotpotqa_error_records,
    classify_projected_answer,
    error_record_from_eval_record,
    load_eval_records_jsonl,
    save_error_analysis_outputs,
)
from evaluation.hotpotqa_subset_evaluator import HotpotQAGraphEvalRecord


def _eval_record(
    *,
    question_id: str = "hp-error-1",
    projected_answer: str = "First Title",
    gold_answer: str = "Second Title",
    answer_source_type: str = "title",
    exact_match: float = 0.0,
    f1: float = 0.5,
    projected_eval_score: float = 0.0,
    graph_node_count: int = 50,
    graph_edge_count: int = 117,
    expand_count: int = 1,
    answer_step_idx: int | None = 1,
) -> HotpotQAGraphEvalRecord:
    return HotpotQAGraphEvalRecord(
        question_id=question_id,
        graph_id=f"hotpotqa::{question_id}",
        reward_mode="baseline",
        gold_answer=gold_answer,
        raw_graph_answer=f"hotpotqa::{question_id}::title::0",
        selected_graph_answer=f"hotpotqa::{question_id}::title::0",
        projected_answer=projected_answer,
        normalized_projected_answer=projected_answer.lower(),
        projected_eval_score=projected_eval_score,
        exact_match=exact_match,
        f1=f1,
        answer_source_type=answer_source_type,
        base_total_reward=0.1,
        total_reward=0.1,
        step_count=2,
        graph_node_count=graph_node_count,
        graph_edge_count=graph_edge_count,
        metadata={
            "path_summary": {
                "step_count": 2,
                "selected_action_types": ["EXPAND_EDGE", "ANSWER"] if expand_count else ["ANSWER"],
                "expand_count": expand_count,
                "answer_step_idx": answer_step_idx,
                "stop_step_idx": None,
                "expand_count_before_answer": expand_count,
            }
        },
    )


def test_error_record_is_json_serializable_and_keeps_required_fields() -> None:
    record = error_record_from_eval_record(_eval_record())
    payload = record.to_dict()

    assert record.question_id == "hp-error-1"
    assert record.projected_answer_type == "multi_token"
    assert record.answer_source_type == "title"
    assert record.path_summary["early_answer"] is False
    assert record.metadata["graph_node_bucket"] == "small"
    assert record.metadata["graph_edge_bucket"] == "small"
    json.dumps(payload)


def test_projected_answer_type_classification_uses_fixed_enum() -> None:
    assert classify_projected_answer("") == "empty"
    assert classify_projected_answer("yes") == "yes_no"
    assert classify_projected_answer("No") == "yes_no"
    assert classify_projected_answer("Chicago") == "single_token"
    assert classify_projected_answer("First Title") == "multi_token"


def test_bucket_summaries_cover_answer_source_projected_type_graph_and_path() -> None:
    records = [
        _eval_record(question_id="r1", projected_answer="First Title", answer_source_type="title", graph_node_count=50),
        _eval_record(
            question_id="r2",
            projected_answer="",
            answer_source_type="fallback",
            graph_node_count=120,
            graph_edge_count=300,
            expand_count=0,
            answer_step_idx=0,
        ),
    ]

    result = analyze_hotpotqa_error_records(
        records,
        graph_bucket_config=GraphStructureBucketConfig(node_small_max=50, node_medium_max=100),
        path_behavior_config=PathBehaviorConfig(min_expands_before_answer=1),
    )
    summary = result.summary

    assert summary["sample_count"] == 2
    assert summary["answer_source_buckets"]["title"]["count"] == 1
    assert summary["answer_source_buckets"]["fallback"]["count"] == 1
    assert summary["projected_answer_type_buckets"]["multi_token"]["count"] == 1
    assert summary["projected_answer_type_buckets"]["empty"]["count"] == 1
    assert summary["graph_structure_buckets"]["node_count"]["small"]["count"] == 1
    assert summary["graph_structure_buckets"]["node_count"]["large"]["count"] == 1
    assert summary["path_buckets"]["expand_count"]["expand_0"]["count"] == 1
    assert summary["path_buckets"]["early_answer"]["early_answer"]["count"] == 1


def test_analyzer_can_load_existing_eval_records_jsonl(tmp_path: Path) -> None:
    records_path = tmp_path / "records.jsonl"
    record = _eval_record().to_dict()
    records_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    loaded = load_eval_records_jsonl(records_path)
    result = analyze_hotpotqa_error_records(loaded)

    assert len(loaded) == 1
    assert result.summary["sample_count"] == 1
    assert result.records[0].question_id == "hp-error-1"


def test_error_analysis_outputs_are_saved(tmp_path: Path) -> None:
    result = analyze_hotpotqa_error_records([_eval_record()])

    records_path, failures_path, summary_path = save_error_analysis_outputs(result, tmp_path / "analysis")

    assert records_path.exists()
    assert failures_path.exists()
    assert summary_path.exists()
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    failures = [json.loads(line) for line in failures_path.read_text(encoding="utf-8").splitlines()]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert records[0]["question_id"] == "hp-error-1"
    assert failures[0]["question_id"] == "hp-error-1"
    assert summary["failure_count"] == 1
