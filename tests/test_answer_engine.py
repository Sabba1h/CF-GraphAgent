"""Tests for answer generation and evaluation separation."""

from answer.answer_engine import AnswerEngine
from answer.evaluator import AnswerEvaluator
from data.toy_data import build_toy_graph, get_demo_task
from memory.working_memory import WorkingMemory


def test_answer_and_evaluator_can_be_called_independently() -> None:
    graph_store = build_toy_graph()
    task = get_demo_task()
    memory = WorkingMemory(max_steps=5)
    memory.initialize_frontier(["Forrest Gump"])
    memory.accept_edge(graph_store.get_edge_by_id("e1"))
    memory.accept_edge(graph_store.get_edge_by_id("e2"))

    answer_result = AnswerEngine().answer(query=task["query"], graph_store=graph_store, working_memory=memory)
    eval_result = AnswerEvaluator().evaluate(
        predicted_answer=answer_result.answer,
        ground_truth=task["ground_truth"],
    )

    assert answer_result.answer == "Chicago"
    assert answer_result.reward == 0.0
    assert eval_result.score == 1.0
    assert eval_result.is_correct is True


def test_generate_answer_compatibility_wrapper_keeps_stage1_score_semantics() -> None:
    graph_store = build_toy_graph()
    task = get_demo_task()
    memory = WorkingMemory(max_steps=5)
    memory.accept_edge(graph_store.get_edge_by_id("e2"))

    answer_result = AnswerEngine().generate_answer(
        query=task["query"],
        graph_store=graph_store,
        working_memory=memory,
        ground_truth=task["ground_truth"],
    )

    assert answer_result.answer == "Chicago"
    assert answer_result.reward == 1.0
    assert answer_result.is_correct is True
