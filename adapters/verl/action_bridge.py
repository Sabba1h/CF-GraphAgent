"""Action bridge from model outputs to candidate ids."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from agent.graph_action_parser import GraphActionParser
from agent.rollout_types import ParsedGraphAction


class VerlActionBridge:
    """Convert verl-side model outputs into environment candidate ids."""

    def __init__(self, *, parser: GraphActionParser | None = None) -> None:
        self.parser = parser or GraphActionParser()

    def parse(self, model_output: Any) -> ParsedGraphAction:
        """Parse one model output using the existing graph action parser."""

        return self.parser.parse(model_output)

    def to_candidate_id(self, model_output: Any) -> int:
        """Return the candidate_id selected by one model output."""

        return self.parse(model_output).candidate_id

    def batch_to_candidate_ids(self, model_outputs: Sequence[Any]) -> list[int]:
        """Parse a small synchronous batch of model outputs."""

        return [self.to_candidate_id(model_output) for model_output in model_outputs]
