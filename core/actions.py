"""Action schemas shared by env, rollout, and adapter layers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ActionType(str, Enum):
    """Supported high-level graph action types."""

    EXPAND_EDGE = "EXPAND_EDGE"
    ANSWER = "ANSWER"
    STOP = "STOP"
    ADD_EDGE = "ADD_EDGE"
    DELETE_EDGE = "DELETE_EDGE"
    UPDATE_EDGE_WEIGHT = "UPDATE_EDGE_WEIGHT"
    CHANGE_FRONTIER = "CHANGE_FRONTIER"


@dataclass(slots=True)
class CandidateAction:
    """Stable candidate action schema exposed to the agent."""

    candidate_id: int
    action_type: ActionType
    description: str
    edge_id: str | None = None
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary view."""

        payload = asdict(self)
        payload["action_type"] = self.action_type.value
        return payload
