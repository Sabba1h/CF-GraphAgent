"""Tests for unified experiment config and result flow."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from adapters.verl import BatchedGraphBackend, VerlRewardBridge, VerlRolloutAdapter
from agent.graph_rollout_manager import GraphRolloutManager
from core import ExperimentConfig, ExperimentResult, RewardBreakdown
from data.toy_data import build_toy_graph, get_demo_task
from envs.cf_graph_env import CFGraphEnv
from scripts.run_rollout_experiment import demo_policy, run_single_experiment


ROOT = Path(__file__).resolve().parents[1]


def _build_env(max_steps: int = 5) -> CFGraphEnv:
    task = get_demo_task()
    return CFGraphEnv(
        graph_store=build_toy_graph(),
        query=task["query"],
        ground_truth=task["ground_truth"],
        max_steps=max_steps,
    )


def _batched_demo_policy(prompts: list[str], observations: list[dict]) -> list[int]:
    assert len(prompts) == len(observations)
    return [demo_policy(observation) for observation in observations]


def test_experiment_config_defaults_are_baseline_compatible() -> None:
    config = ExperimentConfig()

    assert config.reward_mode == "baseline"
    assert config.counterfactual_mode == "mixed"
    assert config.resolved_counterfactual_mode == "remove"
    assert config.use_counterfactual_merge is False
    assert config.max_steps is None


def test_graph_rollout_manager_outputs_unified_experiment_result() -> None:
    manager = GraphRolloutManager()

    baseline = manager.run_experiment(env=_build_env(), policy=demo_policy, config=ExperimentConfig(max_steps=5))
    assert isinstance(baseline, ExperimentResult)
    assert baseline.config.reward_mode == "baseline"
    assert baseline.final_answer == "Chicago"
    assert baseline.base_total_reward == pytest.approx(1.2)
    assert baseline.total_reward == pytest.approx(1.2)
    assert all(trace.counterfactual_reward == 0.0 for trace in baseline.step_traces)

    oracle_config = ExperimentConfig(
        reward_mode="oracle_counterfactual",
        counterfactual_mode="replace",
        use_counterfactual_merge=True,
        max_steps=5,
    )
    oracle = manager.run_experiment(env=_build_env(), policy=demo_policy, config=oracle_config)

    assert isinstance(oracle, ExperimentResult)
    assert oracle.config.reward_mode == "oracle_counterfactual"
    assert oracle.base_total_reward == pytest.approx(1.2)
    assert oracle.total_reward > oracle.base_total_reward
    assert len(oracle.step_traces) == 3
    assert all(trace.counterfactual_comparison is not None for trace in oracle.step_traces)
    assert all(trace.counterfactual_reward > 0.0 for trace in oracle.step_traces)


def test_verl_reward_bridge_uses_unified_config() -> None:
    bridge = VerlRewardBridge()
    baseline = bridge.bridge(RewardBreakdown(process_reward=0.1, total_reward=0.1), config=ExperimentConfig())
    assert baseline.reward == 0.1
    assert baseline.metrics["reward_mode"] == "baseline"
    assert baseline.metrics["use_counterfactual_merge"] is False

    oracle = run_single_experiment("oracle_counterfactual")
    trace = oracle.step_traces[-1]
    output = bridge.bridge(
        trace.reward_breakdown,
        comparison=trace.counterfactual_comparison,
        config=oracle.config,
    )
    assert output.metrics["reward_mode"] == "oracle_counterfactual"
    assert output.metrics["counterfactual_mode"] == "replace"
    assert output.metrics["counterfactual_reward"] == trace.counterfactual_reward
    assert output.metrics["score_delta"] == trace.counterfactual_comparison.score_delta


def test_verl_rollout_adapter_outputs_unified_experiment_result_for_single_env() -> None:
    adapter = VerlRolloutAdapter(batch_backend=BatchedGraphBackend(envs=[_build_env()]))
    baseline = adapter.run_experiment(policy=_batched_demo_policy, config=ExperimentConfig(max_steps=5))
    assert isinstance(baseline, ExperimentResult)
    assert baseline.base_total_reward == pytest.approx(1.2)
    assert baseline.total_reward == pytest.approx(1.2)

    oracle_config = ExperimentConfig(
        reward_mode="oracle_counterfactual",
        counterfactual_mode="replace",
        use_counterfactual_merge=True,
        max_steps=5,
    )
    oracle_adapter = VerlRolloutAdapter(batch_backend=BatchedGraphBackend(envs=[_build_env()]))
    oracle = oracle_adapter.run_experiment(policy=_batched_demo_policy, config=oracle_config)

    assert isinstance(oracle, ExperimentResult)
    assert oracle.base_total_reward == pytest.approx(1.2)
    assert oracle.total_reward > oracle.base_total_reward
    assert all(trace.counterfactual_reward > 0.0 for trace in oracle.step_traces)


def test_rollout_experiment_script_smoke() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/run_rollout_experiment.py", "--mode", "baseline"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "Rollout Experiment: baseline" in completed.stdout
    assert "Final Answer: Chicago" in completed.stdout
    assert "Total Reward: 1.2" in completed.stdout
