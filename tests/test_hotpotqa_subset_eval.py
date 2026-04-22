"""Tests for HotpotQA graph-backed subset evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.benchmarks.hotpotqa import record_to_benchmark_example
from evaluation.hotpotqa_metrics import exact_match, normalize_answer_source_type, token_f1
from evaluation.hotpotqa_subset_evaluator import (
    HotpotQAGraphEvalRecord,
    evaluate_hotpotqa_graph_subset,
    record_from_experiment_result,
    save_hotpotqa_eval_outputs,
)
from graph.hotpotqa_graph_runtime import run_hotpotqa_graph_experiment_for_example


def _record(answer: str = "First Title") -> dict:
    return {
        "id": "hp-eval-1",
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
    data_path = path / "hotpotqa_eval_fixture.json"
    data_path.write_text(json.dumps([_record()], ensure_ascii=False), encoding="utf-8")
    return data_path


def test_hotpotqa_eval_record_is_json_serializable() -> None:
    example = record_to_benchmark_example(_record())
    _, result = run_hotpotqa_graph_experiment_for_example(
        example,
        reward_mode="baseline",
        max_steps=3,
        candidate_top_k=4,
        min_expand_steps=1,
    )

    record = record_from_experiment_result(result)
    payload = record.to_dict()

    assert isinstance(record, HotpotQAGraphEvalRecord)
    assert record.question_id == "hp-eval-1"
    assert record.answer_source_type == "title"
    assert record.projected_eval_score == pytest.approx(1.0)
    assert record.exact_match == pytest.approx(1.0)
    assert record.f1 == pytest.approx(1.0)
    json.dumps(payload)


def test_hotpotqa_metrics_keep_projected_score_em_and_f1_separate() -> None:
    assert exact_match("First Title", "first title") == pytest.approx(1.0)
    assert exact_match("First Title", "Second Title") == pytest.approx(0.0)
    assert token_f1("First Second", "First Third") == pytest.approx(0.5)
    assert normalize_answer_source_type("title") == "title"
    assert normalize_answer_source_type("sentence") == "sentence"
    assert normalize_answer_source_type("fallback") == "fallback"
    assert normalize_answer_source_type("node_text") == "unknown"


def test_hotpotqa_subset_evaluator_baseline_smoke(tmp_path: Path) -> None:
    data_path = _write_fixture(tmp_path)

    result = evaluate_hotpotqa_graph_subset(
        path=data_path,
        limit=1,
        split="validation",
        reward_mode="baseline",
        max_steps=3,
        candidate_top_k=4,
        min_expand_steps=1,
    )

    assert len(result.records) == 1
    assert result.records[0].reward_mode == "baseline"
    assert result.records[0].base_total_reward == result.records[0].total_reward
    assert result.summary["sample_count"] == 1
    assert result.summary["avg_exact_match"] == pytest.approx(1.0)
    assert result.summary["avg_f1"] == pytest.approx(1.0)
    assert result.summary["avg_projected_eval_score"] == pytest.approx(1.0)
    assert result.summary["answer_source_type_distribution"]["title"] == 1


def test_hotpotqa_subset_evaluator_oracle_smoke(tmp_path: Path) -> None:
    data_path = _write_fixture(tmp_path)

    result = evaluate_hotpotqa_graph_subset(
        path=data_path,
        limit=1,
        split="validation",
        reward_mode="oracle_counterfactual",
        max_steps=3,
        candidate_top_k=4,
        min_expand_steps=1,
    )

    assert len(result.records) == 1
    record = result.records[0]
    assert record.reward_mode == "oracle_counterfactual"
    assert record.metadata["counterfactual_summaries"]
    assert "oracle_counterfactual" in result.summary["by_reward_mode"]


def test_hotpotqa_subset_evaluator_can_save_outputs(tmp_path: Path) -> None:
    data_path = _write_fixture(tmp_path)
    output_dir = tmp_path / "eval-output"
    result = evaluate_hotpotqa_graph_subset(path=data_path, limit=1, reward_mode="baseline", max_steps=3)

    records_path, summary_path = save_hotpotqa_eval_outputs(result, output_dir)

    assert records_path.exists()
    assert summary_path.exists()
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert records[0]["question_id"] == "hp-eval-1"
    assert summary["sample_count"] == 1
