"""Synchronous batched graph backend shell for future verl integration."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from adapters.verl.batch_state import BatchedEpisodeState
from adapters.verl.prompt_builder import VerlPromptBuilder


class BatchedGraphBackend:
    """Dispatch a small batch by calling existing single-env APIs sequentially."""

    def __init__(self, *, envs: Sequence[Any], prompt_builder: VerlPromptBuilder | None = None) -> None:
        self.envs = list(envs)
        self.prompt_builder = prompt_builder or VerlPromptBuilder()
        self.batched_state: BatchedEpisodeState | None = None
        self.last_observations: list[dict[str, Any]] = []
        self.last_infos: list[dict[str, Any]] = []

    def batch_reset(
        self,
        *,
        reset_options: Sequence[dict[str, Any] | None] | None = None,
    ) -> tuple[list[dict[str, Any]], BatchedEpisodeState, list[dict[str, Any]]]:
        """Reset every env sequentially and collect GraphEpisodeState objects."""

        if reset_options is not None and len(reset_options) != len(self.envs):
            raise ValueError("reset_options length must match env batch size.")

        observations: list[dict[str, Any]] = []
        infos: list[dict[str, Any]] = []
        states = []
        for idx, env in enumerate(self.envs):
            options = None if reset_options is None else reset_options[idx]
            if options is None:
                observation, info = env.reset()
            else:
                observation, info = env.reset(options=options)
            state = getattr(env, "state", None)
            if state is None:
                raise RuntimeError("BatchedGraphBackend requires env.state after reset().")
            observations.append(observation)
            infos.append(info)
            states.append(state)

        self.last_observations = observations
        self.last_infos = infos
        self.batched_state = BatchedEpisodeState(states=states)
        return observations, self.batched_state, infos

    def batch_step(
        self,
        actions: Sequence[Any],
    ) -> tuple[list[dict[str, Any]], list[float], list[bool], list[bool], list[dict[str, Any]], BatchedEpisodeState]:
        """Step active envs sequentially with already parsed env actions."""

        if self.batched_state is None:
            raise RuntimeError("batch_reset() must be called before batch_step().")
        if len(actions) != len(self.envs):
            raise ValueError("actions length must match env batch size.")

        observations: list[dict[str, Any]] = []
        rewards: list[float] = []
        terminateds: list[bool] = []
        truncateds: list[bool] = []
        infos: list[dict[str, Any]] = []

        for idx, (env, action) in enumerate(zip(self.envs, actions, strict=True)):
            if not self.batched_state.active_mask[idx]:
                observations.append(self.last_observations[idx] if idx < len(self.last_observations) else {})
                rewards.append(0.0)
                terminateds.append(True)
                truncateds.append(False)
                info = dict(self.last_infos[idx]) if idx < len(self.last_infos) else {}
                info["inactive"] = True
                infos.append(info)
                continue

            observation, reward, terminated, truncated, info = env.step(action)
            observations.append(observation)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info)
            self.batched_state.active_mask[idx] = not (terminated or truncated)
            state = getattr(env, "state", None)
            if state is not None:
                self.batched_state.states[idx] = state

        self.last_observations = observations
        self.last_infos = infos
        return observations, rewards, terminateds, truncateds, infos, self.batched_state

    def batch_render_prompts(self, text_observations: Sequence[str] | None = None) -> list[str]:
        """Render prompts from existing text observations without new templates."""

        if text_observations is None:
            text_observations = [str(getattr(env, "last_text_observation", "")) for env in self.envs]
        return self.prompt_builder.build_prompts(text_observations)
