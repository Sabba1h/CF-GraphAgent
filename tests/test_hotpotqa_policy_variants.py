"""Tests for deterministic HotpotQA policy variants and comparison flow."""

from __future__ import annotations

import json
from pathlib import Path

from graph.hotpotqa_policy_variants import (
    POLICY_NAMES,
    make_baseline_policy,
    make_delayed_answer_policy,
    make_policy_factory,
    make_require_sentence_before_answer_policy,
    make_sentence_chain_policy,
    make_sentence_first_policy,
)
from scripts.compare_hotpotqa_policies import compare_hotpotqa_policies


def _candidate(candidate_id: int, action_type: str, relation: str = "", description: str = "") -> dict:
    return {
        "candidate_id": candidate_id,
        "action_type": action_type,
        "description": description,
        "metadata": {"relation": relation} if relation else {},
    }


def _record() -> dict:
    return {
        "id": "hp-policy-1",
        "question": "Which page is first?",
        "answer": "First Title",
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
    data_path = path / "hotpotqa_policy_fixture.json"
    data_path.write_text(json.dumps([_record()], ensure_ascii=False), encoding="utf-8")
    return data_path


def test_baseline_policy_is_deterministic_and_uses_stable_candidate_order() -> None:
    observation = {
        "candidate_actions": [
            _candidate(4, "ANSWER"),
            _candidate(2, "EXPAND_EDGE", "question_to_title"),
            _candidate(1, "EXPAND_EDGE", "question_to_title"),
        ]
    }

    assert make_baseline_policy(min_expand_steps=1)(observation) == 1
    assert make_baseline_policy(min_expand_steps=1)(observation) == 1


def test_sentence_first_prefers_sentence_relations_without_privileged_fields() -> None:
    observation = {
        "candidate_actions": [
            _candidate(1, "EXPAND_EDGE", "question_to_title"),
            _candidate(5, "ANSWER"),
            _candidate(3, "EXPAND_EDGE", "title_to_sentence"),
            _candidate(2, "EXPAND_EDGE", "next_sentence"),
        ],
        "supporting_facts": [{"title": "ignored"}],
        "gold_answer": "ignored",
    }

    policy = make_sentence_first_policy(min_expand_steps=1)

    assert policy(observation) == 2


def test_delayed_answer_delays_only_when_expand_is_available() -> None:
    delayed = make_delayed_answer_policy(min_expand_steps=2)
    with_expand = {
        "candidate_actions": [
            _candidate(0, "ANSWER"),
            _candidate(1, "STOP"),
            _candidate(2, "EXPAND_EDGE", "title_to_sentence"),
        ]
    }
    no_expand = {
        "candidate_actions": [
            _candidate(3, "STOP"),
            _candidate(2, "ANSWER"),
        ]
    }

    assert delayed(with_expand) == 2
    assert make_delayed_answer_policy(min_expand_steps=2)(no_expand) == 2


def test_policy_factory_builds_supported_policy_names() -> None:
    for policy_name in POLICY_NAMES:
        policy = make_policy_factory(policy_name)()
        assert callable(policy)


def test_sentence_chain_policy_enters_sentence_then_prefers_next_sentence() -> None:
    first_observation = {
        "candidate_actions": [
            _candidate(3, "ANSWER"),
            _candidate(2, "EXPAND_EDGE", "question_to_title"),
            _candidate(1, "EXPAND_EDGE", "title_to_sentence"),
        ],
        "supporting_facts": [{"title": "ignored"}],
        "gold_answer": "ignored",
    }
    second_observation = {
        "candidate_actions": [
            _candidate(3, "ANSWER"),
            _candidate(4, "EXPAND_EDGE", "sentence_to_title"),
            _candidate(2, "EXPAND_EDGE", "next_sentence"),
        ],
        "supporting_facts": [{"title": "ignored"}],
        "gold_answer": "ignored",
    }
    policy = make_sentence_chain_policy()

    assert policy(first_observation) == 1
    assert policy(second_observation) == 2


def test_require_sentence_before_answer_keeps_escape_when_no_useful_expand() -> None:
    policy = make_require_sentence_before_answer_policy()
    with_sentence_expand = {
        "candidate_actions": [
            _candidate(0, "ANSWER"),
            _candidate(1, "STOP"),
            _candidate(2, "EXPAND_EDGE", "title_to_sentence"),
        ]
    }
    no_useful_expand = {
        "candidate_actions": [
            {
                **_candidate(3, "EXPAND_EDGE", "title_to_sentence", "already in working subgraph"),
            },
            _candidate(0, "ANSWER"),
            _candidate(1, "STOP"),
        ]
    }

    assert policy(with_sentence_expand) == 2
    assert policy(no_useful_expand) == 0


def test_policy_comparison_smoke_uses_unified_eval_and_error_analysis(tmp_path: Path) -> None:
    data_path = _write_fixture(tmp_path)
    output_dir = tmp_path / "comparison"

    comparison = compare_hotpotqa_policies(
        path=data_path,
        limit=1,
        split="validation",
        reward_mode="baseline",
        policies=["baseline", "sentence_first", "delayed_answer"],
        max_steps=4,
        candidate_top_k=4,
        output_dir=output_dir,
    )

    assert comparison["policy_order"] == ["baseline", "sentence_first", "delayed_answer"]
    for policy_name in comparison["policy_order"]:
        assert policy_name in comparison["policies"]
        assert "eval_summary" in comparison["policies"][policy_name]
        assert "error_summary" in comparison["policies"][policy_name]
        assert (output_dir / policy_name / "hotpotqa_graph_eval_records.jsonl").exists()
        assert (output_dir / policy_name / "hotpotqa_error_summary.json").exists()
    assert (output_dir / "hotpotqa_policy_comparison_summary.json").exists()
