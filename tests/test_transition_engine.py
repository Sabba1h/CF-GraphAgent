"""Tests for GraphBackend and TransitionEngine integration."""

from answer.answer_engine import AnswerEngine
from answer.evaluator import AnswerEvaluator
from candidates.generator import CandidateGenerator
from core.task import TaskSample
from core.transition import TransitionEngine
from data.toy_data import build_toy_graph, get_demo_task
from graph.graph_backend import GraphBackend
from reward.reward_engine import RewardEngine


def test_transition_engine_reset_and_expand() -> None:
    task = get_demo_task()
    backend = GraphBackend(build_toy_graph())
    engine = TransitionEngine(
        graph_backend=backend,
        candidate_generator=CandidateGenerator(top_k=5),
        reward_engine=RewardEngine(),
        answer_engine=AnswerEngine(),
        answer_evaluator=AnswerEvaluator(),
    )

    state, seed_nodes = engine.reset(
        task=TaskSample(query=task["query"], ground_truth=task["ground_truth"]),
        max_steps=5,
    )
    assert seed_nodes
    assert state.latest_candidate_actions

    expand_action = next(candidate for candidate in state.latest_candidate_actions if candidate.edge_id == "e1")
    result = engine.step(state=state, candidate_id=expand_action.candidate_id)

    assert result.reward_result.reward == 0.1
    assert result.terminated is False
    assert "e1" in state.working_edge_ids
