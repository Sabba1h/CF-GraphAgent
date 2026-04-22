"""Smoke tests for snapshot and replay scaffolding."""

from core import ActionType, GraphEpisodeState
from data.toy_data import build_toy_graph, get_demo_task
from envs.cf_graph_env import CFGraphEnv
from graph.graph_backend import GraphBackend
from replay import CounterfactualRunner, ReplayEngine, SnapshotManager


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


def test_snapshot_is_not_polluted_by_later_state_mutation() -> None:
    env = _build_env()
    observation, _ = env.reset()
    env.step(_candidate_id(observation, "EXPAND_EDGE", "e1"))

    snapshot = SnapshotManager().create_snapshot(env.state)
    env.state.working_edge_ids.add("polluting-edge")
    env.state.latest_candidate_actions.clear()

    assert "polluting-edge" not in snapshot.working_edge_ids
    assert snapshot.candidate_actions
    assert snapshot.ground_truth == "Chicago"


def test_replay_engine_restores_snapshot_and_answers_with_evaluator() -> None:
    env = _build_env()
    observation, _ = env.reset()
    observation, _, _, _, _ = env.step(_candidate_id(observation, "EXPAND_EDGE", "e1"))
    observation, _, _, _, _ = env.step(_candidate_id(observation, "EXPAND_EDGE", "e2"))
    snapshot = SnapshotManager().create_snapshot(env.state)

    replay_engine = ReplayEngine(graph_backend=GraphBackend(env.graph_store))
    restored_state = replay_engine.restore_state(snapshot)
    answer_id = replay_engine.find_candidate_id(snapshot, ActionType.ANSWER)
    replay_result = replay_engine.replay_one_step(snapshot=snapshot, action={"candidate_id": answer_id})

    assert isinstance(restored_state, GraphEpisodeState)
    assert restored_state.working_edge_ids == {"e1", "e2"}
    assert len(replay_result.steps) == 1
    assert replay_result.steps[0].transition.terminated is True
    assert replay_result.final_answer == "Chicago"
    assert replay_result.total_reward == 1.0
    assert replay_result.steps[0].transition.info["eval_score"] == 1.0


def test_counterfactual_runner_scaffold_methods_have_smoke_paths() -> None:
    env = _build_env()
    observation, _ = env.reset()
    base_snapshot = SnapshotManager().create_snapshot(env.state)
    stop_id = _candidate_id(observation, "STOP")
    answer_id = _candidate_id(observation, "ANSWER")
    replay_engine = ReplayEngine(graph_backend=GraphBackend(env.graph_store))
    runner = CounterfactualRunner(
        replay_engine=replay_engine,
        base_snapshot=base_snapshot,
        original_actions=[stop_id],
    )

    removed = runner.remove_action(0)
    replaced = runner.replace_action(0, answer_id)
    nulled = runner.null_action(0)

    assert removed.mode == "remove"
    assert removed.metadata["counterfactual_actions"] == []
    assert replaced.mode == "replace"
    assert replaced.counterfactual_action == answer_id
    assert replaced.counterfactual_eval.score == 0.0
    assert nulled.mode == "null"
    assert nulled.metadata["counterfactual_actions"] == []
