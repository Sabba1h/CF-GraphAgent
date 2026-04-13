"""Tests for split observation layers and compatibility facade."""

from candidates.generator import CandidateGenerator
from data.toy_data import build_toy_graph, get_demo_task
from memory.working_memory import WorkingMemory
from observation.prompt_renderer import PromptRenderer
from observation.renderer import ObservationRenderer
from observation.structured_view import StructuredObservationView
from observation.text_renderer import TextObservationRenderer


def _memory_with_candidates() -> tuple[dict[str, str], object, WorkingMemory]:
    graph_store = build_toy_graph()
    task = get_demo_task()
    memory = WorkingMemory(max_steps=5)
    memory.initialize_frontier(["Forrest Gump"])
    candidates = CandidateGenerator(top_k=4).generate(
        query=task["query"],
        graph_store=graph_store,
        working_memory=memory,
    )
    memory.set_latest_candidates(candidates)
    return task, graph_store, memory


def test_split_observation_layers_match_facade() -> None:
    task, graph_store, memory = _memory_with_candidates()

    facade = ObservationRenderer()
    structured = StructuredObservationView().render(
        query=task["query"],
        graph_store=graph_store,
        working_memory=memory,
    )
    facade_structured = facade.render_structured_observation(
        query=task["query"],
        graph_store=graph_store,
        working_memory=memory,
    )
    text = TextObservationRenderer().render(
        query=task["query"],
        graph_store=graph_store,
        working_memory=memory,
    )
    facade_text = facade.render_text_observation(
        query=task["query"],
        graph_store=graph_store,
        working_memory=memory,
    )

    assert structured == facade_structured
    assert text == facade_text
    assert "Candidate Actions:" in text


def test_prompt_renderer_is_thin_wrapper() -> None:
    prompt = PromptRenderer().render_prompt("Query: example")

    assert prompt == "Query: example"
