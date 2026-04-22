"""Counterfactual replay comparison runner."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from core.episode_result import CounterfactualComparisonResult, EvalResult
from core.snapshots import StateSnapshot
from core.state import GraphEpisodeState
from replay.replay_engine import ReplayEngine, ReplayResult


CounterfactualReplayResult = CounterfactualComparisonResult
SuffixPolicy = Callable[[GraphEpisodeState], Any | None]


class CounterfactualRunner:
    """Run single-episode counterfactual comparisons through ReplayEngine."""

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

    def remove_action(
        self,
        step_idx: int,
        *,
        suffix_policy: SuffixPolicy | None = None,
        max_suffix_steps: int | None = None,
    ) -> CounterfactualComparisonResult:
        """Replay with one recorded action removed and compare outcomes."""

        self._validate_step_idx(step_idx)
        actions = self.original_actions[:step_idx] + self.original_actions[step_idx + 1 :]
        return self._compare(
            mode="remove",
            step_idx=step_idx,
            counterfactual_actions=actions,
            counterfactual_action=None,
            suffix_policy=suffix_policy,
            max_suffix_steps=max_suffix_steps,
        )

    def replace_action(
        self,
        step_idx: int,
        new_action: Any,
        *,
        suffix_policy: SuffixPolicy | None = None,
        max_suffix_steps: int | None = None,
    ) -> CounterfactualComparisonResult:
        """Replay with one recorded action replaced and compare outcomes."""

        self._validate_step_idx(step_idx)
        actions = list(self.original_actions)
        actions[step_idx] = new_action
        return self._compare(
            mode="replace",
            step_idx=step_idx,
            counterfactual_actions=actions,
            counterfactual_action=new_action,
            suffix_policy=suffix_policy,
            max_suffix_steps=max_suffix_steps,
        )

    def null_action(
        self,
        step_idx: int,
        *,
        suffix_policy: SuffixPolicy | None = None,
        max_suffix_steps: int | None = None,
    ) -> CounterfactualComparisonResult:
        """Replay only the prefix before a target action and compare outcomes.

        The omitted suffix keeps this method as a structural placeholder rather
        than a semantic no-op action implementation.
        """

        self._validate_step_idx(step_idx)
        actions = self.original_actions[:step_idx]
        return self._compare(
            mode="null",
            step_idx=step_idx,
            counterfactual_actions=actions,
            counterfactual_action=None,
            suffix_policy=suffix_policy,
            max_suffix_steps=max_suffix_steps,
        )

    def _compare(
        self,
        *,
        mode: str,
        step_idx: int,
        counterfactual_actions: list[Any],
        counterfactual_action: Any | None,
        suffix_policy: SuffixPolicy | None,
        max_suffix_steps: int | None,
    ) -> CounterfactualComparisonResult:
        original_actions = self._complete_with_suffix(
            prefix_actions=self.original_actions,
            suffix_policy=suffix_policy,
            max_suffix_steps=max_suffix_steps,
        )
        completed_counterfactual_actions = self._complete_with_suffix(
            prefix_actions=counterfactual_actions,
            suffix_policy=suffix_policy,
            max_suffix_steps=max_suffix_steps,
        )
        original_replay = self.replay_engine.replay(snapshot=self.base_snapshot, actions=original_actions)
        counterfactual_replay = self.replay_engine.replay(
            snapshot=self.base_snapshot,
            actions=completed_counterfactual_actions,
        )
        original_eval = self._eval_from_replay(original_replay)
        counterfactual_eval = self._eval_from_replay(counterfactual_replay)
        original_score = original_eval.score
        counterfactual_score = counterfactual_eval.score
        return CounterfactualComparisonResult(
            mode=mode,
            step_idx=step_idx,
            original_action=self.original_actions[step_idx],
            counterfactual_action=counterfactual_action,
            original_eval=original_eval,
            counterfactual_eval=counterfactual_eval,
            original_score=original_score,
            counterfactual_score=counterfactual_score,
            score_delta=original_score - counterfactual_score,
            metadata={
                "original_actions": list(original_actions),
                "counterfactual_actions": list(completed_counterfactual_actions),
                "original_total_reward": original_replay.total_reward,
                "counterfactual_total_reward": counterfactual_replay.total_reward,
                "original_final_answer": original_replay.final_answer,
                "counterfactual_final_answer": counterfactual_replay.final_answer,
            },
        )

    def _complete_with_suffix(
        self,
        *,
        prefix_actions: list[Any],
        suffix_policy: SuffixPolicy | None,
        max_suffix_steps: int | None,
    ) -> list[Any]:
        actions = list(prefix_actions)
        if suffix_policy is None:
            return actions

        suffix_budget = max_suffix_steps if max_suffix_steps is not None else max(self.base_snapshot.steps_left, 0)
        for _ in range(suffix_budget):
            replay_result = self.replay_engine.replay(snapshot=self.base_snapshot, actions=actions)
            if replay_result.steps:
                last_transition = replay_result.steps[-1].transition
                if last_transition.terminated or last_transition.truncated:
                    break
            state = replay_result.final_state
            if state is None or state.steps_left <= 0 or not state.latest_candidate_actions:
                break
            next_action = suffix_policy(state)
            if next_action is None:
                break
            actions.append(next_action)
        return actions

    def _eval_from_replay(self, replay_result: ReplayResult) -> EvalResult:
        if not replay_result.steps:
            return EvalResult(
                score=0.0,
                is_correct=False,
                reason="no_replayed_answer",
                details={"steps": 0},
            )

        last_transition = replay_result.steps[-1].transition
        info = last_transition.info
        if "eval_score" in info:
            return EvalResult(
                score=float(info["eval_score"]),
                is_correct=info.get("is_correct"),
                reason=str(info.get("eval_reason", "answer_eval")),
                details={
                    "final_answer": last_transition.final_answer,
                    "termination_reason": last_transition.termination_reason,
                    "steps": len(replay_result.steps),
                },
            )

        return EvalResult(
            score=0.0,
            is_correct=False,
            reason=f"no_answer_{last_transition.termination_reason or 'unfinished'}",
            details={
                "termination_reason": last_transition.termination_reason,
                "terminated": last_transition.terminated,
                "truncated": last_transition.truncated,
                "steps": len(replay_result.steps),
            },
        )

    def _validate_step_idx(self, step_idx: int) -> None:
        if type(step_idx) is not int or step_idx < 0 or step_idx >= len(self.original_actions):
            raise ValueError("step_idx must refer to an existing original action.")
