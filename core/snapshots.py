"""State snapshot schemas for replay-oriented code."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class StateSnapshot:
    """Frozen view of an episode state at a step boundary."""

    query: str
    step_index: int
    ground_truth: str | None = None
    max_steps: int = 0
    working_edge_ids: list[str] = field(default_factory=list)
    visited_edge_ids: list[str] = field(default_factory=list)
    observed_edge_ids: list[str] = field(default_factory=list)
    visited_nodes: list[str] = field(default_factory=list)
    frontier_nodes: list[str] = field(default_factory=list)
    candidate_actions: list[dict[str, Any]] = field(default_factory=list)
    steps_left: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary copy."""

        return asdict(self)
