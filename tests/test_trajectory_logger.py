"""Tests for trajectory logger snapshot semantics."""

from memory.trajectory_logger import TrajectoryLogger


def test_log_step_stores_defensive_snapshots() -> None:
    logger = TrajectoryLogger()
    logger.start_episode(query="q", ground_truth="a")

    candidate_actions = [{"candidate_id": 0, "metadata": {"edge_id": "e1"}}]
    selected_action = {"candidate_id": 0, "metadata": {"edge_id": "e1"}}
    working_summary = {"edges": [{"edge_id": "e1"}]}
    info = {"nested": {"value": 1}}

    logger.log_step(
        step_index=1,
        candidate_actions=candidate_actions,
        selected_action=selected_action,
        reward=0.1,
        reward_reason="valid_expand",
        working_subgraph_summary=working_summary,
        terminated=False,
        truncated=False,
        info=info,
    )

    candidate_actions[0]["metadata"]["edge_id"] = "mutated"
    selected_action["metadata"]["edge_id"] = "mutated"
    working_summary["edges"][0]["edge_id"] = "mutated"
    info["nested"]["value"] = 999

    step = logger.current_summary().steps[0]
    assert step.candidate_actions[0]["metadata"]["edge_id"] == "e1"
    assert step.selected_action["metadata"]["edge_id"] == "e1"
    assert step.working_subgraph_summary["edges"][0]["edge_id"] == "e1"
    assert step.info["nested"]["value"] == 1
