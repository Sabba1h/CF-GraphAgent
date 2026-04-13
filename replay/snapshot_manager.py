"""Snapshot creation for replay-oriented single-episode state."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from core.snapshots import StateSnapshot
from core.state import GraphEpisodeState


class SnapshotManager:
    """Create defensive step-time snapshots from GraphEpisodeState."""

    def create_snapshot(
        self,
        state: GraphEpisodeState,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> StateSnapshot:
        """Return the minimal state closure needed for single-episode replay.

        The snapshot stores copied values, not live references to the active
        episode state.
        """

        return StateSnapshot(
            query=state.query,
            ground_truth=state.ground_truth,
            step_index=state.steps_used,
            max_steps=state.max_steps,
            working_edge_ids=sorted(state.working_edge_ids),
            visited_edge_ids=sorted(state.visited_edge_ids),
            observed_edge_ids=sorted(state.observed_edge_ids),
            visited_nodes=sorted(state.visited_nodes),
            frontier_nodes=sorted(state.frontier_nodes),
            candidate_actions=deepcopy([candidate.to_dict() for candidate in state.latest_candidate_actions]),
            steps_left=state.steps_left,
            metadata=deepcopy(metadata or {}),
        )
