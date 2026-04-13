"""Structured observation view."""

from __future__ import annotations

from graph.graph_store import GraphStore
from schemas import ObservationDict


class StructuredObservationView:
    """Render structured observations from episode state."""

    def __init__(self, *, history_window: int = 5) -> None:
        self.history_window = history_window

    def render(
        self,
        *,
        query: str,
        graph_store: GraphStore,
        working_memory,
    ) -> ObservationDict:
        """Render a structured observation dictionary."""

        summary = graph_store.export_subgraph_summary(working_memory.working_edge_ids)
        return {
            "query": query,
            "current_working_nodes": summary["nodes"],
            "current_working_edges": summary["edges"],
            "working_subgraph_summary": summary,
            "frontier_nodes": sorted(working_memory.frontier_nodes),
            "candidate_actions": [candidate.to_dict() for candidate in working_memory.latest_candidate_actions],
            "steps_left": working_memory.steps_left,
            "history_summary": working_memory.recent_action_history(self.history_window),
        }
