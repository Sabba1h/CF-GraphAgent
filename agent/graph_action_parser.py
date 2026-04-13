"""Candidate-id action parser for graph rollouts."""

from __future__ import annotations

from typing import Any

from agent.rollout_types import ParsedGraphAction


class GraphActionParser:
    """Parse model/policy outputs into candidate_id actions."""

    def parse(self, action: Any) -> ParsedGraphAction:
        """Parse a minimal candidate_id action.

        This parser is intentionally separate from CFGraphEnv.step(). It may
        accept simple text outputs for rollout experiments while the env keeps
        its strict action API.
        """

        if type(action) is int:
            return ParsedGraphAction(candidate_id=action, raw_action=action)
        if isinstance(action, dict) and set(action.keys()) == {"candidate_id"} and type(action["candidate_id"]) is int:
            return ParsedGraphAction(candidate_id=action["candidate_id"], raw_action=action)
        if isinstance(action, str) and action.strip().isdigit():
            return ParsedGraphAction(candidate_id=int(action.strip()), raw_action=action)
        raise ValueError("GraphActionParser expects int, {'candidate_id': int}, or a digit string.")
