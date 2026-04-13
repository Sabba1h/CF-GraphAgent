"""Observation rendering for structured and text views."""

from __future__ import annotations

from graph.graph_store import GraphStore
from memory.working_memory import WorkingMemory
from observation.structured_view import StructuredObservationView
from observation.text_renderer import TextObservationRenderer
from schemas import ObservationDict


class ObservationRenderer:
    """Render structured and text observations from structured state."""

    def __init__(self, *, history_window: int = 5) -> None:
        self.history_window = history_window
        self.structured_view = StructuredObservationView(history_window=history_window)
        self.text_renderer = TextObservationRenderer(history_window=history_window)

    def render_structured_observation(
        self,
        *,
        query: str,
        graph_store: GraphStore,
        working_memory: WorkingMemory,
    ) -> ObservationDict:
        """Render a structured observation dictionary."""

        return self.structured_view.render(query=query, graph_store=graph_store, working_memory=working_memory)

    def render_text_observation(
        self,
        *,
        query: str,
        graph_store: GraphStore,
        working_memory: WorkingMemory,
    ) -> str:
        """Render a compact text observation for an LLM-facing policy."""

        return self.text_renderer.render(query=query, graph_store=graph_store, working_memory=working_memory)
