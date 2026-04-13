"""Smoke tests for the minimal verl adapter skeleton."""

from adapters.verl import (
    BatchedEpisodeState,
    BatchedGraphBackend,
    VerlActionBridge,
    VerlPromptBuilder,
    VerlRewardBridge,
    VerlRolloutAdapter,
    VerlTrainerHooks,
)
from core import GraphEpisodeState, RewardBreakdown
from data.toy_data import build_toy_graph, get_demo_task
from envs.cf_graph_env import CFGraphEnv


def _build_env(max_steps: int = 4) -> CFGraphEnv:
    task = get_demo_task()
    return CFGraphEnv(
        graph_store=build_toy_graph(),
        query=task["query"],
        ground_truth=task["ground_truth"],
        max_steps=max_steps,
    )


def _stop_candidate_id(observation: dict) -> int:
    for candidate in observation["candidate_actions"]:
        if candidate["action_type"] == "STOP":
            return candidate["candidate_id"]
    raise AssertionError("STOP candidate was not generated.")


def test_verl_adapter_imports_are_available() -> None:
    assert BatchedEpisodeState is not None
    assert BatchedGraphBackend is not None
    assert VerlActionBridge is not None
    assert VerlPromptBuilder is not None
    assert VerlRewardBridge is not None
    assert VerlRolloutAdapter is not None
    assert VerlTrainerHooks is not None


def test_batched_graph_backend_reset_step_and_prompt_smoke() -> None:
    backend = BatchedGraphBackend(envs=[_build_env(), _build_env()])

    observations, batched_state, infos = backend.batch_reset()
    assert batched_state.batch_size == 2
    assert all(batched_state.active_mask)
    assert all(isinstance(state, GraphEpisodeState) for state in batched_state.states)
    assert len(observations) == 2
    assert len(infos) == 2

    prompts = backend.batch_render_prompts()
    assert len(prompts) == 2
    assert "Query:" in prompts[0]

    actions = [_stop_candidate_id(observation) for observation in observations]
    _, rewards, terminated, truncated, _, batched_state = backend.batch_step(actions)
    assert rewards == [-0.5, -0.5]
    assert terminated == [True, True]
    assert truncated == [False, False]
    assert batched_state.active_mask == [False, False]


def test_verl_prompt_action_and_reward_bridges() -> None:
    assert VerlPromptBuilder().build_prompt("text observation") == "text observation"
    assert VerlActionBridge().to_candidate_id({"candidate_id": 3}) == 3

    breakdown = RewardBreakdown(
        task_reward=1.0,
        process_reward=0.1,
        constraint_penalty=0.0,
        counterfactual_reward=0.0,
        total_reward=1.1,
    )
    output = VerlRewardBridge().bridge(breakdown)
    assert output.reward == 1.1
    assert output.metrics["total_reward"] == 1.1
    assert output.metrics["task_reward"] == 1.0


def test_verl_rollout_adapter_runs_mock_policy_smoke() -> None:
    backend = BatchedGraphBackend(envs=[_build_env(), _build_env()])
    adapter = VerlRolloutAdapter(batch_backend=backend)

    def stop_policy(prompts: list[str], observations: list[dict]):
        assert len(prompts) == len(observations)
        return [_stop_candidate_id(observation) for observation in observations]

    result = adapter.run_rollout(policy=stop_policy, max_steps=3)

    assert len(result.steps) == 1
    assert result.total_rewards == [-0.5, -0.5]
    assert result.final_active_mask == [False, False]
