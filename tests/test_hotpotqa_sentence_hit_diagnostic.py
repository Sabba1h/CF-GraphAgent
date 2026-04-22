"""Tests for HotpotQA sentence-hit diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.hotpotqa_sentence_hit_diagnostic import (
    analyze_hotpotqa_sentence_hits,
    classify_sentence_hit_bucket,
    contains_gold,
    save_sentence_hit_outputs,
    sentence_hit_record_from_eval_record,
)
from evaluation.hotpotqa_subset_evaluator import HotpotQAGraphEvalRecord
from graph.hotpotqa_policy_variants import make_policy_factory
from scripts.analyze_hotpotqa_sentence_hits import run_hotpotqa_sentence_hit_analysis
from scripts.compare_hotpotqa_policies import compare_hotpotqa_policies


def _eval_record(
    *,
    question_id: str = "hp-sentence-1",
    gold_answer: str = "Chicago",
    projected_answer: str = "Chicago is a city.",
    answer_source_type: str = "sentence",
    selected_graph_answer: str | None = "s1",
    exact_match: float = 0.0,
    f1: float = 0.5,
) -> HotpotQAGraphEvalRecord:
    return HotpotQAGraphEvalRecord(
        question_id=question_id,
        graph_id=f"hotpotqa::{question_id}",
        reward_mode="baseline",
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
        policy_name="sentence_chain",
        answer_selector_name="latest_sentence",
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
                {
                    "node_id": "s1",
                    "node_type": "sentence",
                    "text": "Chicago is a city in Illinois.",
                },
                {
                    "node_id": "s2",
                    "node_type": "sentence",
                    "text": "Another sentence.",
                },
            ],
        },
    )


def _fixture_record(answer: str = "First Title") -> dict:
    return {
        "id": "hp-sentence-fixture-1",
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
    data_path = path / "hotpotqa_sentence_fixture.json"
    data_path.write_text(json.dumps([_fixture_record()], ensure_ascii=False), encoding="utf-8")
    return data_path


def test_contains_gold_handles_span_and_yes_no_separately() -> None:
    assert contains_gold("Chicago is a city.", "Chicago") is True
    assert contains_gold("Boston is a city.", "Chicago") is False
    assert contains_gold("yes appears as a short token", "yes") is None
    assert contains_gold("no appears as a short token", "no") is None


def test_sentence_hit_record_is_json_serializable() -> None:
    record = sentence_hit_record_from_eval_record(_eval_record())
    payload = record.to_dict()

    assert record.touched_sentence is True
    assert record.touched_sentence_count == 1
    assert record.first_sentence_step_idx == 1
    assert record.any_touched_sentence_contains_gold is True
    assert record.selected_sentence_contains_gold is True
    assert record.gold_in_any_sentence is True
    assert record.diagnostic_bucket == "selected_sentence_contains_gold_but_eval_still_zero"
    json.dumps(payload)


def test_yes_no_samples_use_not_applicable_bucket() -> None:
    record = sentence_hit_record_from_eval_record(
        _eval_record(
            question_id="yes-no",
            gold_answer="yes",
            projected_answer="A sentence with yes.",
            selected_graph_answer="s1",
        )
    )

    assert record.gold_answer_type == "yes_no"
    assert record.any_touched_sentence_contains_gold is None
    assert record.selected_sentence_contains_gold is None
    assert record.gold_in_any_sentence is None
    assert record.diagnostic_bucket == "yes_no_not_applicable"


def test_sentence_hit_summary_and_save_outputs(tmp_path: Path) -> None:
    result = analyze_hotpotqa_sentence_hits([
        _eval_record(question_id="r1"),
        _eval_record(question_id="r2", selected_graph_answer="t1", answer_source_type="title"),
    ])
    records_path, summary_path = save_sentence_hit_outputs(result, tmp_path / "sentence-hits")

    assert result.summary["sample_count"] == 2
    assert result.summary["sentence_touch_rate"] == 1.0
    assert "selected_sentence_contains_gold_but_eval_still_zero" in result.summary["diagnostic_buckets"]
    assert records_path.exists()
    assert summary_path.exists()


def test_sentence_hit_bucket_rules() -> None:
    assert (
        classify_sentence_hit_bucket(
            touched_sentence=False,
            gold_answer_type="span",
            any_touched_sentence_contains_gold=False,
            selected_sentence_contains_gold=False,
            projected_eval_score=0.0,
            gold_in_any_sentence=True,
        )
        == "no_sentence_touched"
    )
    assert (
        classify_sentence_hit_bucket(
            touched_sentence=True,
            gold_answer_type="span",
            any_touched_sentence_contains_gold=False,
            selected_sentence_contains_gold=False,
            projected_eval_score=0.0,
            gold_in_any_sentence=True,
        )
        == "sentence_touched_but_no_gold_sentence"
    )


def test_refined_policy_factories_are_deterministic_without_privileged_fields() -> None:
    observation = {
        "candidate_actions": [
            {"candidate_id": 3, "action_type": "ANSWER", "metadata": {}},
            {"candidate_id": 2, "action_type": "EXPAND_EDGE", "metadata": {"relation": "question_to_title"}},
            {"candidate_id": 1, "action_type": "EXPAND_EDGE", "metadata": {"relation": "title_to_sentence"}},
        ],
        "supporting_facts": [{"title": "ignored"}],
        "gold_answer": "ignored",
    }

    for policy_name in ("sentence_chain", "require_sentence_before_answer", "hybrid_sentence_chain_delayed"):
        first_policy = make_policy_factory(policy_name)()
        second_policy = make_policy_factory(policy_name)()
        assert first_policy(observation) == second_policy(observation) == 1


def test_sentence_hit_script_and_policy_compare_smoke(tmp_path: Path) -> None:
    data_path = _write_fixture(tmp_path)
    diagnostic_output = tmp_path / "diagnostic-output"
    comparison_output = tmp_path / "comparison-output"

    diagnostic = run_hotpotqa_sentence_hit_analysis(
        path=data_path,
        split="validation",
        limit=1,
        reward_mode="baseline",
        policy_name="sentence_chain",
        output_dir=diagnostic_output,
    )
    comparison = compare_hotpotqa_policies(
        path=data_path,
        split="validation",
        limit=1,
        reward_mode="baseline",
        policies=["baseline", "sentence_chain", "require_sentence_before_answer"],
        output_dir=comparison_output,
    )

    assert diagnostic.summary["sample_count"] == 1
    assert (diagnostic_output / "hotpotqa_sentence_hit_summary.json").exists()
    for policy_name in comparison["policy_order"]:
        assert "sentence_hit_summary" in comparison["policies"][policy_name]
        assert (comparison_output / policy_name / "hotpotqa_sentence_hit_summary.json").exists()
