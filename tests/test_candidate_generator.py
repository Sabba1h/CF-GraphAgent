"""Tests for CandidateGenerator."""

from candidates.generator import CandidateGenerator
from data.toy_data import build_toy_graph, get_demo_task
from memory.working_memory import WorkingMemory


def test_candidate_generator_finds_reasonable_seeds() -> None:
    graph_store = build_toy_graph()
    query = get_demo_task()["query"]
    generator = CandidateGenerator(top_k=4)

    seeds = generator.find_seed_nodes(query, graph_store)

    assert "Forrest Gump" in seeds
    assert "Robert Zemeckis" in seeds


def test_candidate_generator_always_includes_answer_and_stop() -> None:
    graph_store = build_toy_graph()
    query = get_demo_task()["query"]
    memory = WorkingMemory(max_steps=5)
    memory.initialize_frontier(["Forrest Gump"])

    generator = CandidateGenerator(top_k=4)
    candidates = generator.generate(query=query, graph_store=graph_store, working_memory=memory)

    action_types = [candidate.action_type.value for candidate in candidates]
    candidate_ids = [candidate.candidate_id for candidate in candidates]

    assert "EXPAND_EDGE" in action_types
    assert "ANSWER" in action_types
    assert "STOP" in action_types
    assert candidate_ids == list(range(len(candidate_ids)))
