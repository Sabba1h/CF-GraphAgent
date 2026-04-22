"""Tests for opt-in oracle counterfactual reward in rollout and adapter paths."""

import pytest

from adapters.verl import BatchedGraphBackend, VerlRewardBridge, VerlRolloutAdapter
from agent.graph_rollout_manager import GraphRolloutManager
from core import RewardBreakdown
from data.toy_data import build_toy_graph, get_demo_task
from envs.cf_graph_env import CFGraphEnv


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


def _demo_path_policy(observation: dict) -> int:
    for edge_id in ("e1", "e2"):
        for candidate in observation["candidate_actions"]:
            if candidate["action_type"] == "EXPAND_EDGE" and candidate.get("edge_id") == edge_id:
                if "already in working subgraph" not in candidate["description"]:
                    return candidate["candidate_id"]
    return _candidate_id(observation, "ANSWER")


def _batched_demo_path_policy(prompts: list[str], observations: list[dict]) -> list[int]:
    assert len(prompts) == len(observations)
    return [_demo_path_policy(observation) for observation in observations]


def test_graph_rollout_manager_baseline_mode_keeps_old_reward_semantics() -> None:
    result = GraphRolloutManager().run_episode(
        env=_build_env(),
        policy=_demo_path_policy,
        reward_mode="baseline",
    )

    assert result.reward_mode == "baseline"
    assert len(result.steps) == 3
    assert [step.base_reward for step in result.steps] == [0.1, 0.1, 1.0]
    assert result.total_reward == pytest.approx(1.2)
    assert result.base_total_reward == pytest.approx(1.2)
    assert all(step.counterfactual_reward == 0.0 for step in result.steps)
    assert all(step.counterfactual_comparison is None for step in result.steps)


def test_graph_rollout_manager_oracle_mode_records_stepwise_counterfactual_reward() -> None:
    result = GraphRolloutManager().run_episode(
        env=_build_env(),
        policy=_demo_path_policy,
        reward_mode="oracle_counterfactual",
    )

    assert result.reward_mode == "oracle_counterfactual"
    assert len(result.steps) == 3
    assert result.base_total_reward == pytest.approx(1.2)
    assert result.total_reward > result.base_total_reward

    for step in result.steps:
        assert step.reward_mode == "oracle_counterfactual"
        assert step.reward_breakdown is not None
        assert step.counterfactual_comparison is not None
        assert step.counterfactual_reward > 0.0
        assert step.reward_breakdown.counterfactual_reward == step.counterfactual_reward
        assert step.counterfactual_comparison.metadata["original_final_answer"] == "Chicago"
        assert step.counterfactual_comparison.metadata["counterfactual_final_answer"] is None
        assert "all_comparisons" in step.counterfactual_comparison.metadata


def test_verl_reward_bridge_supports_baseline_and_oracle_modes() -> None:
    bridge = VerlRewardBridge()
    baseline = bridge.bridge(RewardBreakdown(process_reward=0.1, total_reward=0.1))

    assert baseline.reward == 0.1
    assert baseline.metrics["reward_mode"] == "baseline"
    assert baseline.metrics["counterfactual_reward"] == 0.0

    oracle_result = GraphRolloutManager().run_episode(
        env=_build_env(),
        policy=_demo_path_policy,
        reward_mode="oracle_counterfactual",
    )
    oracle_step = oracle_result.steps[-1]
    oracle = bridge.bridge(
        oracle_step.reward_breakdown,
        reward_mode="oracle_counterfactual",
        comparison=oracle_step.counterfactual_comparison,
    )

    assert oracle.reward == oracle_step.reward_breakdown.total_reward
    assert oracle.metrics["reward_mode"] == "oracle_counterfactual"
    assert oracle.metrics["counterfactual_reward"] == oracle_step.counterfactual_reward
    assert oracle.metrics["score_delta"] == oracle_step.counterfactual_comparison.score_delta


def test_verl_rollout_adapter_supports_baseline_and_oracle_mock_rollouts() -> None:
    baseline_adapter = VerlRolloutAdapter(batch_backend=BatchedGraphBackend(envs=[_build_env()]))
    baseline = baseline_adapter.run_rollout(
        policy=_batched_demo_path_policy,
        max_steps=5,
        reward_mode="baseline",
    )

    assert baseline.reward_mode == "baseline"
    assert baseline.total_rewards == pytest.approx([1.2])
    assert all(step.reward_mode == "baseline" for step in baseline.steps)
    assert all(step.reward_outputs[0].metrics["reward_mode"] == "baseline" for step in baseline.steps)

    oracle_adapter = VerlRolloutAdapter(batch_backend=BatchedGraphBackend(envs=[_build_env()]))
    oracle = oracle_adapter.run_rollout(
        policy=_batched_demo_path_policy,
        max_steps=5,
        reward_mode="oracle_counterfactual",
    )

    assert oracle.reward_mode == "oracle_counterfactual"
    assert oracle.total_rewards[0] > 1.2
    assert len(oracle.steps) == 3
    for step in oracle.steps:
        assert step.reward_mode == "oracle_counterfactual"
        assert step.base_rewards[0] in {0.1, 1.0}
        assert step.reward_breakdowns[0].counterfactual_reward > 0.0
        assert step.reward_outputs[0].metrics["reward_mode"] == "oracle_counterfactual"
        assert step.reward_outputs[0].metrics["score_delta"] > 0.0
