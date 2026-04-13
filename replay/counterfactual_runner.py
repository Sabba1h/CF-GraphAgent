"""Counterfactual replay scaffold without reward integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.snapshots import StateSnapshot
from replay.replay_engine import ReplayEngine, ReplayResult


@dataclass(slots=True)
class CounterfactualReplayResult:
    """Result payload for a scaffold counterfactual replay operation."""

    mode: str
    step_idx: int
    actions: list[Any]
    replay_result: ReplayResult
    note: str = ""


class CounterfactualRunner:
    """Minimal single-episode counterfactual replay helper."""

    def __init__(
        self,
        *,
        replay_engine: ReplayEngine,
        base_snapshot: StateSnapshot,
        original_actions: list[Any],
    ) -> None:
        self.replay_engine = replay_engine
        self.base_snapshot = base_snapshot
        self.original_actions = list(original_actions)

    def remove_action(self, step_idx: int) -> CounterfactualReplayResult:
        """Replay with one recorded action removed."""

        self._validate_step_idx(step_idx)
        actions = self.original_actions[:step_idx] + self.original_actions[step_idx + 1 :]
        replay_result = self.replay_engine.replay(snapshot=self.base_snapshot, actions=actions)
        return CounterfactualReplayResult(
            mode="remove_action",
            step_idx=step_idx,
            actions=actions,
            replay_result=replay_result,
            note="Scaffold only; no counterfactual reward is computed.",
        )

    def replace_action(self, step_idx: int, new_action: Any) -> CounterfactualReplayResult:
        """Replay with one recorded action replaced by a caller-provided action."""

        self._validate_step_idx(step_idx)
        actions = list(self.original_actions)
        actions[step_idx] = new_action
        replay_result = self.replay_engine.replay(snapshot=self.base_snapshot, actions=actions)
        return CounterfactualReplayResult(
            mode="replace_action",
            step_idx=step_idx,
            actions=actions,
            replay_result=replay_result,
            note="Scaffold only; no counterfactual reward is computed.",
        )

    def null_action(self, step_idx: int) -> CounterfactualReplayResult:
        """Replay only the prefix before a target action.

        The omitted suffix keeps this method as a structural placeholder rather
        than a semantic no-op action implementation.
        """

        self._validate_step_idx(step_idx)
        actions = self.original_actions[:step_idx]
        replay_result = self.replay_engine.replay(snapshot=self.base_snapshot, actions=actions)
        return CounterfactualReplayResult(
            mode="null_action",
            step_idx=step_idx,
            actions=actions,
            replay_result=replay_result,
            note="Scaffold only; target and subsequent actions are omitted.",
        )

    def _validate_step_idx(self, step_idx: int) -> None:
        if type(step_idx) is not int or step_idx < 0 or step_idx >= len(self.original_actions):
            raise ValueError("step_idx must refer to an existing original action.")
