"""Tests for counterfactual dataset export."""

from __future__ import annotations

import json

from core.counterfactual_dataset import examples_from_experiment_result, write_jsonl
from scripts.build_counterfactual_dataset import build_examples
from scripts.run_rollout_experiment import run_single_experiment


def test_dataset_examples_from_experiment_result_are_json_serializable() -> None:
    result = run_single_experiment("oracle_counterfactual")
    examples = examples_from_experiment_result(
        result=result,
        task_id="toy_task",
        trajectory_id="toy_traj",
    )

    assert len(examples) == 3
    first = examples[0].to_dict()
    assert first["schema_version"] == "v1"
    assert first["task_id"] == "toy_task"
    assert first["trajectory_id"] == "toy_traj"
    assert first["query"] == "Which city was the director of Forrest Gump born in?"
    assert first["action"] == 0
    assert first["action_type"] == "EXPAND_EDGE"
    assert "working_subgraph_summary" in first
    assert "candidate_actions_summary" in first
    assert "history_summary" in first
    assert first["score_delta"] == 1.0
    assert first["counterfactual_reward"] == 1.0
    json.dumps(first)


def test_build_examples_and_write_jsonl(tmp_path) -> None:
    examples = build_examples(reward_mode="oracle_counterfactual", counterfactual_mode="replace")
    output_path = write_jsonl(examples, tmp_path / "dataset.jsonl")

    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    payload = json.loads(lines[0])
    assert payload["schema_version"] == "v1"
    assert payload["action"] == 0
    assert "original_score" in payload
    assert "counterfactual_score" in payload
    assert "score_delta" in payload
    assert "counterfactual_reward" in payload
