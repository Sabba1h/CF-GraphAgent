"""Thin prompt renderer for future model-facing use."""

from __future__ import annotations


class PromptRenderer:
    """Wrap an existing text observation without changing its semantics."""

    def render_prompt(self, text_observation: str) -> str:
        """Return the current text observation as the prompt body."""

        return text_observation
