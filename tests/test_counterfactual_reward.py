"""Tests for counterfactual comparison and reward delta scaffolding."""

import pytest

from core import CounterfactualComparisonResult
from data.toy_data import build_toy_graph, get_demo_task
from envs.cf_graph_env import CFGraphEnv
from graph.graph_backend import GraphBackend
from replay import CounterfactualRunner, ReplayEngine, SnapshotManager
from reward.counterfactual import CounterfactualRewardEngine
from reward.reward_engine import RewardEngine


def _build_env(max_steps: int = 5) -> CFGraphEnv:
    task = get_demo_task()
    return CFGraphEnv(
        graph_store=build_toy_graph(),
        query=task["query"],
        ground_truth=task["ground_truth"],
        max_steps=max_steps,
    )


def _candidate_id(observation: dict, action_type: str, edge_id: str | None = None) -> int:
    for candidate in observation["candidate_actions"]:
        if candidate["action_type"] != action_type:
            continue
        if edge_id is None or candidate.get("edge_id") == edge_id:
            return candidate["candidate_id"]
    raise AssertionError(f"Candidate not found: {action_type} {edge_id}")


def _runner_after_evidence_ready() -> tuple[CounterfactualRunner, int, int]:
    env = _build_env()
    observation, _ = env.reset()
    observation, _, _, _, _ = env.step(_candidate_id(observation, "EXPAND_EDGE", "e1"))
    observation, _, _, _, _ = env.step(_candidate_id(observation, "EXPAND_EDGE", "e2"))
    snapshot = SnapshotManager().create_snapshot(env.state)
    answer_id = _candidate_id(observation, "ANSWER")
    stop_id = _candidate_id(observation, "STOP")
    runner = CounterfactualRunner(
        replay_engine=ReplayEngine(graph_backend=GraphBackend(env.graph_store)),
        base_snapshot=snapshot,
        original_actions=[answer_id],
    )
    return runner, answer_id, stop_id


def test_counterfactual_runner_returns_comparison_results_for_all_modes() -> None:
    runner, answer_id, stop_id = _runner_after_evidence_ready()

    removed = runner.remove_action(0)
    replaced = runner.replace_action(0, stop_id)
    nulled = runner.null_action(0)

    for comparison, mode in [(removed, "remove"), (replaced, "replace"), (nulled, "null")]:
        assert isinstance(comparison, CounterfactualComparisonResult)
        assert comparison.mode == mode
        assert comparison.step_idx == 0
        assert comparison.original_action == answer_id
        assert comparison.original_score == 1.0
        assert comparison.counterfactual_score == 0.0
        assert comparison.score_delta == 1.0

    assert replaced.counterfactual_action == stop_id
    assert removed.counterfactual_action is None
    assert nulled.counterfactual_action is None


def test_counterfactual_reward_engine_uses_score_delta() -> None:
    runner, _, _ = _runner_after_evidence_ready()
    comparison = runner.remove_action(0)

    reward_result = CounterfactualRewardEngine().compute(comparison)

    assert reward_result.counterfactual_reward == 1.0
    assert reward_result.metrics["score_delta"] == 1.0
    assert reward_result.metrics["counterfactual_reward"] == 1.0


def test_reward_engine_default_behavior_and_explicit_counterfactual_merge() -> None:
    runner, _, _ = _runner_after_evidence_ready()
    comparison = runner.remove_action(0)
    reward_engine = RewardEngine()

    base_reward = reward_engine.reward_for_expand(is_valid=True, is_repeated=False)
    assert base_reward.reward == 0.1
    assert base_reward.breakdown.counterfactual_reward == 0.0
    assert base_reward.breakdown.total_reward == 0.1
    assert reward_engine.compute_counterfactual_bonus() == 0.0

    merged = reward_engine.merge_counterfactual_reward(base_reward=base_reward, comparison=comparison)
    assert merged.counterfactual_bonus == 1.0
    assert merged.breakdown.process_reward == 0.1
    assert merged.breakdown.counterfactual_reward == 1.0
    assert merged.breakdown.total_reward == pytest.approx(1.1)
    assert merged.reward == pytest.approx(1.1)
