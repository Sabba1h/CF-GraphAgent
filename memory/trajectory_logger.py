"""Trajectory logging utilities for episode replay and debugging."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from schemas import EpisodeSummary, StepRecord


class TrajectoryLogger:
    """Collect step-wise and episode-wise environment traces."""

    def __init__(self) -> None:
        self._summary: EpisodeSummary | None = None

    def start_episode(self, query: str, ground_truth: str | None = None) -> None:
        """Initialize a fresh episode summary."""

        self._summary = EpisodeSummary(query=query, ground_truth=ground_truth)

    def log_step(
        self,
        *,
        step_index: int,
        candidate_actions: list[dict[str, Any]],
        selected_action: dict[str, Any] | None,
        reward: float,
        reward_reason: str,
        working_subgraph_summary: dict[str, Any],
        terminated: bool,
        truncated: bool,
        info: dict[str, Any] | None = None,
    ) -> None:
        """Append a step-time snapshot into the active episode.

        The logger owns defensive copies of mutable payloads. Historical
        records must not depend on live references held by env.step().
        """

        if self._summary is None:
            raise RuntimeError("TrajectoryLogger.start_episode() must be called before logging steps.")
        self._summary.steps.append(
            StepRecord(
                step_index=step_index,
                candidate_actions=deepcopy(candidate_actions),
                selected_action=deepcopy(selected_action),
                reward=reward,
                reward_reason=reward_reason,
                working_subgraph_summary=deepcopy(working_subgraph_summary),
                terminated=terminated,
                truncated=truncated,
                info=deepcopy(info or {}),
            )
        )

    def finalize(self, *, termination_reason: str, final_answer: str | None, final_score: float) -> None:
        """Finalize the current episode summary."""

        if self._summary is None:
            raise RuntimeError("TrajectoryLogger.start_episode() must be called before finalizing.")
        self._summary.termination_reason = termination_reason
        self._summary.final_answer = final_answer
        self._summary.final_score = final_score

    def current_summary(self) -> EpisodeSummary:
        """Return the active episode summary object."""

        if self._summary is None:
            raise RuntimeError("TrajectoryLogger.start_episode() must be called before reading summary.")
        return self._summary

    def as_dict(self) -> dict[str, Any]:
        """Return the active episode summary as a plain dictionary."""

        return self.current_summary().to_dict()

    def reset(self) -> None:
        """Clear the active episode summary."""

        self._summary = None
