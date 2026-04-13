"""Tests for the minimal CFGraphEnv loop."""

import pytest

from core.state import GraphEpisodeState
from data.toy_data import build_toy_graph, get_demo_task
from envs.cf_graph_env import CFGraphEnv
from graph.graph_backend import GraphBackend


def _find_candidate_id(observation: dict, *, action_type: str | None = None, edge_id: str | None = None) -> int:
    for candidate in observation["candidate_actions"]:
        if action_type is not None and candidate["action_type"] != action_type:
            continue
        if edge_id is not None and candidate.get("edge_id") != edge_id:
            continue
        return candidate["candidate_id"]
    raise AssertionError(f"Candidate not found for action_type={action_type}, edge_id={edge_id}")


def test_env_reset_and_answer_loop() -> None:
    task = get_demo_task()
    env = CFGraphEnv(graph_store=build_toy_graph(), query=task["query"], ground_truth=task["ground_truth"], max_steps=5)

    observation, info = env.reset()
    assert observation["query"] == task["query"]
    assert len(observation["candidate_actions"]) >= 3
    assert "text_observation" in info
    assert isinstance(env.state, GraphEpisodeState)
    assert isinstance(env.graph_backend, GraphBackend)
    assert env.transition_engine is not None

    first_expand = _find_candidate_id(observation, edge_id="e1")
    observation, reward, terminated, truncated, info = env.step(first_expand)
    assert reward == 0.1
    assert terminated is False
    assert truncated is False
    assert "e1" in env.state.working_edge_ids
    assert "e1" in env.working_memory.working_edge_ids

    second_expand = _find_candidate_id(observation, edge_id="e2")
    observation, reward, terminated, truncated, info = env.step({"candidate_id": second_expand})
    assert reward == 0.1
    assert "e2" in env.working_memory.working_edge_ids

    answer_action = _find_candidate_id(observation, action_type="ANSWER")
    observation, reward, terminated, truncated, info = env.step(answer_action)
    assert terminated is True
    assert truncated is False
    assert reward == 1.0
    assert info["answer"] == "Chicago"


def test_env_invalid_candidate_id_gets_penalty() -> None:
    task = get_demo_task()
    env = CFGraphEnv(graph_store=build_toy_graph(), query=task["query"], ground_truth=task["ground_truth"], max_steps=5)

    env.reset()
    observation, reward, terminated, truncated, info = env.step(999)

    assert reward == -1.0
    assert terminated is False
    assert truncated is False
    assert "error" in info
    assert observation["query"] == task["query"]


@pytest.mark.parametrize(
    "bad_action",
    [
        True,
        False,
        {"candidate_id": 1, "foo": "bar"},
        {"candidate_id": True},
        "1",
    ],
)
def test_env_rejects_invalid_action_input_shapes(bad_action) -> None:
    task = get_demo_task()
    env = CFGraphEnv(graph_store=build_toy_graph(), query=task["query"], ground_truth=task["ground_truth"], max_steps=5)

    env.reset()
    with pytest.raises(ValueError):
        env.step(bad_action)


def test_env_legal_dict_action_still_works() -> None:
    task = get_demo_task()
    env = CFGraphEnv(graph_store=build_toy_graph(), query=task["query"], ground_truth=task["ground_truth"], max_steps=5)

    observation, _ = env.reset()
    first_expand = _find_candidate_id(observation, edge_id="e1")
    _, reward, terminated, truncated, _ = env.step({"candidate_id": first_expand})

    assert reward == 0.1
    assert terminated is False
    assert truncated is False


def test_env_stop_terminates_episode() -> None:
    task = get_demo_task()
    env = CFGraphEnv(graph_store=build_toy_graph(), query=task["query"], ground_truth=task["ground_truth"], max_steps=5)

    observation, _ = env.reset()
    stop_action = _find_candidate_id(observation, action_type="STOP")
    _, reward, terminated, truncated, info = env.step(stop_action)

    assert reward == -0.5
    assert terminated is True
    assert truncated is False
    assert info["trajectory"]["termination_reason"] == "stop"
