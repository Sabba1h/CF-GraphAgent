"""Tests for minimal agent rollout skeleton."""

import pytest

from agent.graph_action_parser import GraphActionParser
from agent.graph_rollout_manager import GraphRolloutManager
from data.toy_data import build_toy_graph, get_demo_task
from envs.cf_graph_env import CFGraphEnv


def test_graph_action_parser_accepts_candidate_id_forms() -> None:
    parser = GraphActionParser()

    assert parser.parse(1).candidate_id == 1
    assert parser.parse({"candidate_id": 2}).candidate_id == 2
    assert parser.parse("3").candidate_id == 3


@pytest.mark.parametrize("bad_action", [True, {"candidate_id": True}, {"candidate_id": 1, "x": 2}, "abc"])
def test_graph_action_parser_rejects_invalid_actions(bad_action) -> None:
    with pytest.raises(ValueError):
        GraphActionParser().parse(bad_action)


def test_rollout_manager_runs_single_env_sync_loop() -> None:
    task = get_demo_task()
    env = CFGraphEnv(graph_store=build_toy_graph(), query=task["query"], ground_truth=task["ground_truth"], max_steps=5)

    def policy(observation: dict):
        for candidate in observation["candidate_actions"]:
            if candidate["action_type"] == "STOP":
                return candidate["candidate_id"]
        return 0

    result = GraphRolloutManager().run_episode(env=env, policy=policy)

    assert len(result.steps) == 1
    assert result.steps[0].terminated is True
    assert result.total_reward == -0.5
