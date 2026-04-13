"""Text observation renderer."""

from __future__ import annotations

from typing import Any

from core.actions import CandidateAction
from graph.graph_store import GraphStore


class TextObservationRenderer:
    """Render compact text observations for LLM-facing policies."""

    def __init__(self, *, history_window: int = 5) -> None:
        self.history_window = history_window

    def render(
        self,
        *,
        query: str,
        graph_store: GraphStore,
        working_memory,
    ) -> str:
        """Render the existing stage-1 text observation format."""

        summary = graph_store.export_subgraph_summary(working_memory.working_edge_ids)
        history_lines = self._format_history(working_memory.recent_action_history(self.history_window))
        candidate_lines = self._format_candidates(working_memory.latest_candidate_actions)
        return "\n".join(
            [
                f"Query: {query}",
                f"Working Subgraph: {summary['text_summary']}",
                f"Frontier Nodes: {', '.join(sorted(working_memory.frontier_nodes)) or 'None'}",
                "Recent Actions:",
                history_lines,
                "Candidate Actions:",
                candidate_lines,
                f"Steps Left: {working_memory.steps_left}",
            ]
        )

    def _format_history(self, history: list[dict[str, Any]]) -> str:
        if not history:
            return "- No previous actions."
        return "\n".join(
            f"- Step {item.get('step_index', '?')}: {item.get('action_type', 'UNKNOWN')} | {item.get('description', '')}"
            for item in history
        )

    def _format_candidates(self, candidate_actions: list[CandidateAction]) -> str:
        return "\n".join(
            f"- [{candidate.candidate_id}] {candidate.action_type.value}: {candidate.description}"
            for candidate in candidate_actions
        )
