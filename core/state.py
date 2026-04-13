"""Episode state schema for graph reasoning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.actions import CandidateAction
from core.task import TaskSample
from graph.graph_store import EdgeRecord


@dataclass
class GraphEpisodeState:
    """Single source of truth for an active graph episode."""

    task: TaskSample
    max_steps: int
    working_edge_ids: set[str] = field(default_factory=set)
    visited_edge_ids: set[str] = field(default_factory=set)
    observed_edge_ids: set[str] = field(default_factory=set)
    visited_nodes: set[str] = field(default_factory=set)
    frontier_nodes: set[str] = field(default_factory=set)
    action_history: list[dict[str, Any]] = field(default_factory=list)
    steps_used: int = 0
    latest_candidate_actions: list[CandidateAction] = field(default_factory=list)
    final_answer: str | None = None
    termination_reason: str | None = None
    final_score: float = 0.0

    @property
    def query(self) -> str:
        """Return the active query."""

        return self.task.query

    @property
    def ground_truth(self) -> str | None:
        """Return the optional ground truth answer."""

        return self.task.ground_truth

    @property
    def steps_left(self) -> int:
        """Return the remaining step budget."""

        return max(self.max_steps - self.steps_used, 0)

    def initialize_frontier(self, seed_nodes: list[str]) -> None:
        """Initialize frontier and visited nodes from reset seeds."""

        self.frontier_nodes = set(seed_nodes)
        self.visited_nodes.update(seed_nodes)

    def accept_edge(self, edge: EdgeRecord) -> bool:
        """Add an edge into the working subgraph and refresh the frontier."""

        is_new_edge = edge.edge_id not in self.working_edge_ids
        self.working_edge_ids.add(edge.edge_id)
        self.visited_edge_ids.add(edge.edge_id)
        self.visited_nodes.update({edge.src, edge.dst})
        self.frontier_nodes = {edge.dst}
        return is_new_edge

    def set_latest_candidates(self, candidate_actions: list[CandidateAction]) -> None:
        """Cache current step candidates and mark observed edge candidates."""

        self.latest_candidate_actions = list(candidate_actions)
        for candidate in candidate_actions:
            if candidate.action_type.value == "EXPAND_EDGE" and candidate.edge_id:
                self.observed_edge_ids.add(candidate.edge_id)

    def increment_step(self) -> None:
        """Consume one step of budget."""

        self.steps_used += 1

    def add_action_record(self, record: dict[str, Any]) -> None:
        """Store a compact action summary."""

        self.action_history.append(record)

    def recent_action_history(self, limit: int = 5) -> list[dict[str, Any]]:
        """Return the most recent action history window."""

        if limit <= 0:
            return []
        return self.action_history[-limit:]
