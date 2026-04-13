"""Prompt bridge for verl-facing rollout code."""

from __future__ import annotations

from collections.abc import Sequence

from observation.prompt_renderer import PromptRenderer


class VerlPromptBuilder:
    """Build prompts by reusing the existing thin PromptRenderer."""

    def __init__(self, *, prompt_renderer: PromptRenderer | None = None) -> None:
        self.prompt_renderer = prompt_renderer or PromptRenderer()

    def build_prompt(self, text_observation: str) -> str:
        """Return a model-facing prompt without changing text semantics."""

        return self.prompt_renderer.render_prompt(text_observation)

    def build_prompts(self, text_observations: Sequence[str]) -> list[str]:
        """Build prompts for a small synchronous batch."""

        return [self.build_prompt(text_observation) for text_observation in text_observations]
