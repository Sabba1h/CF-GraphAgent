"""Batched episode state wrapper for future verl integration."""

from __future__ import annotations

from dataclasses import dataclass

from core.state import GraphEpisodeState


@dataclass(slots=True)
class BatchedEpisodeState:
    """Minimal batch wrapper around single-episode graph states."""

    states: list[GraphEpisodeState]
    active_mask: list[bool] | None = None

    def __post_init__(self) -> None:
        self.states = list(self.states)
        if self.active_mask is None:
            self.active_mask = [True] * len(self.states)
        else:
            self.active_mask = list(self.active_mask)
        if len(self.active_mask) != len(self.states):
            raise ValueError("active_mask length must match states length.")

    @property
    def batch_size(self) -> int:
        """Return the number of episode states in the batch."""

        return len(self.states)

    def active_indices(self) -> list[int]:
        """Return indices for currently active episodes."""

        return [idx for idx, is_active in enumerate(self.active_mask or []) if is_active]
