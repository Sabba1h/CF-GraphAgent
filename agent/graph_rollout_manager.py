"""Minimal synchronous rollout manager for one graph environment."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agent.graph_action_parser import GraphActionParser
from agent.rollout_types import RolloutResult, RolloutStep


PolicyFn = Callable[[dict[str, Any]], Any]


class GraphRolloutManager:
    """Run a single-env, single-episode synchronous rollout."""

    def __init__(self, *, action_parser: GraphActionParser | None = None) -> None:
        self.action_parser = action_parser or GraphActionParser()

    def run_episode(self, *, env, policy: PolicyFn, max_steps: int | None = None) -> RolloutResult:
        """Run one episode using a caller-provided synchronous policy."""

        observation, _ = env.reset()
        result = RolloutResult()
        terminated = False
        truncated = False
        steps_used = 0

        while not terminated and not truncated:
            if max_steps is not None and steps_used >= max_steps:
                break
            raw_action = policy(observation)
            parsed_action = self.action_parser.parse(raw_action)
            next_observation, reward, terminated, truncated, info = env.step(parsed_action.candidate_id)
            result.steps.append(
                RolloutStep(
                    observation=observation,
                    action=raw_action,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                )
            )
            observation = next_observation
            steps_used += 1

        return result
