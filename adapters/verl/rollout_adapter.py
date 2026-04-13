"""Minimal verl rollout adapter shell."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from adapters.verl.action_bridge import VerlActionBridge
from adapters.verl.batch_backend import BatchedGraphBackend
from adapters.verl.prompt_builder import VerlPromptBuilder
from adapters.verl.reward_bridge import VerlRewardBridge


VerlPolicyFn = Callable[[list[str], list[dict[str, Any]]], Sequence[Any]]


@dataclass(slots=True)
class VerlBatchStep:
    """One synchronous batched adapter step."""

    prompts: list[str]
    actions: list[Any]
    rewards: list[float]
    terminated: list[bool]
    truncated: list[bool]
    infos: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class VerlRolloutResult:
    """Result returned by the minimal mock rollout adapter."""

    steps: list[VerlBatchStep] = field(default_factory=list)
    final_active_mask: list[bool] = field(default_factory=list)

    @property
    def total_rewards(self) -> list[float]:
        """Return accumulated rewards per batch element."""

        if not self.steps:
            return []
        totals = [0.0] * len(self.steps[0].rewards)
        for step in self.steps:
            for idx, reward in enumerate(step.rewards):
                totals[idx] += reward
        return totals


class VerlRolloutAdapter:
    """Organize batch, prompt, action, and reward bridges for smoke rollouts."""

    def __init__(
        self,
        *,
        batch_backend: BatchedGraphBackend,
        prompt_builder: VerlPromptBuilder | None = None,
        action_bridge: VerlActionBridge | None = None,
        reward_bridge: VerlRewardBridge | None = None,
    ) -> None:
        self.batch_backend = batch_backend
        self.prompt_builder = prompt_builder or batch_backend.prompt_builder
        self.action_bridge = action_bridge or VerlActionBridge()
        self.reward_bridge = reward_bridge or VerlRewardBridge()

    def run_rollout(self, *, policy: VerlPolicyFn, max_steps: int | None = None) -> VerlRolloutResult:
        """Run a synchronous mock rollout with a caller-provided policy."""

        observations, batched_state, _ = self.batch_backend.batch_reset()
        result = VerlRolloutResult()
        steps_used = 0

        while any(batched_state.active_mask):
            if max_steps is not None and steps_used >= max_steps:
                break
            prompts = self.batch_backend.batch_render_prompts()
            raw_actions = list(policy(prompts, observations))
            if len(raw_actions) != batched_state.batch_size:
                raise ValueError("policy must return one action per batch element.")
            candidate_ids = self.action_bridge.batch_to_candidate_ids(raw_actions)
            observations, rewards, terminated, truncated, infos, batched_state = self.batch_backend.batch_step(
                candidate_ids
            )
            result.steps.append(
                VerlBatchStep(
                    prompts=prompts,
                    actions=raw_actions,
                    rewards=rewards,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                )
            )
            steps_used += 1

        result.final_active_mask = list(batched_state.active_mask)
        return result
