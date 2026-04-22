"""Deterministic HotpotQA path relevance policies.

These policies are non-privileged: they only use current candidate actions,
relation types, local src/dst ids, and the policy's own visited path state.
They do not read gold answers, supporting facts, lexical overlap, or any
external retrieval signal.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from graph.hotpotqa_policy_variants import (
    candidate_id,
    expand_candidates_sorted,
    first_action,
    non_repeated_expand_candidates,
    relation,
    sorted_candidates,
)

PolicyFn = Callable[[dict[str, Any]], int]
PolicyFactory = Callable[[], PolicyFn]

RELEVANCE_POLICY_NAMES = (
    "title_region_commitment",
    "sentence_region_persistence",
    "region_commitment_delayed_answer",
)


def make_title_region_commitment_policy(
    *,
    min_expand_steps: int = 1,
    region_expand_limit: int = 2,
) -> PolicyFn:
    """Commit briefly to one title region before answering or moving elsewhere."""

    expand_steps_used = 0
    current_title_id: str | None = None
    last_sentence_id: str | None = None
    region_expands = 0

    def policy(observation: dict[str, Any]) -> int:
        nonlocal expand_steps_used, current_title_id, last_sentence_id, region_expands
        candidates = sorted_candidates(observation)
        useful_expands = non_repeated_expand_candidates(candidates)
        expand_candidates = useful_expands or expand_candidates_sorted(candidates)

        if useful_expands:
            ranked = rank_title_region_commitment(
                useful_expands,
                current_title_id=current_title_id,
                last_sentence_id=last_sentence_id,
            )
            if current_title_id and region_expands < region_expand_limit:
                region_candidates = [
                    candidate for candidate in ranked if _candidate_in_region(candidate, current_title_id, last_sentence_id)
                ]
                if region_candidates:
                    chosen = region_candidates[0]
                    expand_steps_used += 1
                    current_title_id, last_sentence_id = _update_region_state(
                        chosen,
                        current_title_id=current_title_id,
                        last_sentence_id=last_sentence_id,
                    )
                    region_expands += 1
                    return candidate_id(chosen)

            if current_title_id is None:
                title_entry = [candidate for candidate in ranked if relation(candidate) == "title_to_sentence"]
                if title_entry:
                    chosen = title_entry[0]
                    expand_steps_used += 1
                    current_title_id, last_sentence_id = _update_region_state(
                        chosen,
                        current_title_id=current_title_id,
                        last_sentence_id=last_sentence_id,
                    )
                    region_expands = 1
                    return candidate_id(chosen)

            if expand_steps_used < min_expand_steps:
                chosen = ranked[0]
                expand_steps_used += 1
                current_title_id, last_sentence_id = _update_region_state(
                    chosen,
                    current_title_id=current_title_id,
                    last_sentence_id=last_sentence_id,
                )
                region_expands += 1 if current_title_id else 0
                return candidate_id(chosen)

        terminal = _answer_or_stop(candidates)
        if terminal is not None:
            return candidate_id(terminal)
        if expand_candidates:
            chosen = rank_title_region_commitment(
                expand_candidates,
                current_title_id=current_title_id,
                last_sentence_id=last_sentence_id,
            )[0]
            expand_steps_used += 1
            current_title_id, last_sentence_id = _update_region_state(
                chosen,
                current_title_id=current_title_id,
                last_sentence_id=last_sentence_id,
            )
            return candidate_id(chosen)
        raise ValueError("Policy received an observation with no candidate actions.")

    return policy


def make_sentence_region_persistence_policy(
    *,
    min_expand_steps: int = 1,
    sentence_chain_limit: int = 2,
) -> PolicyFn:
    """Persist locally inside a reached sentence chain before answering."""

    expand_steps_used = 0
    touched_sentence = False
    last_sentence_id: str | None = None
    sentence_chain_expands = 0

    def policy(observation: dict[str, Any]) -> int:
        nonlocal expand_steps_used, touched_sentence, last_sentence_id, sentence_chain_expands
        candidates = sorted_candidates(observation)
        useful_expands = non_repeated_expand_candidates(candidates)
        expand_candidates = useful_expands or expand_candidates_sorted(candidates)

        if useful_expands:
            ranked = rank_sentence_region_persistence(useful_expands, last_sentence_id=last_sentence_id)
            if touched_sentence and sentence_chain_expands < sentence_chain_limit:
                local_sentence = [
                    candidate
                    for candidate in ranked
                    if relation(candidate) == "next_sentence" and (last_sentence_id is None or src(candidate) == last_sentence_id)
                ]
                if local_sentence:
                    chosen = local_sentence[0]
                    expand_steps_used += 1
                    touched_sentence = True
                    last_sentence_id = dst(chosen) or last_sentence_id
                    sentence_chain_expands += 1
                    return candidate_id(chosen)

            if not touched_sentence:
                sentence_entry = [candidate for candidate in ranked if relation(candidate) == "title_to_sentence"]
                if sentence_entry:
                    chosen = sentence_entry[0]
                    expand_steps_used += 1
                    touched_sentence = True
                    last_sentence_id = dst(chosen) or last_sentence_id
                    sentence_chain_expands = 1
                    return candidate_id(chosen)

            if expand_steps_used < min_expand_steps:
                chosen = ranked[0]
                expand_steps_used += 1
                if _moves_to_sentence(chosen):
                    touched_sentence = True
                    last_sentence_id = dst(chosen) or last_sentence_id
                    sentence_chain_expands += 1
                return candidate_id(chosen)

        terminal = _answer_or_stop(candidates)
        if terminal is not None:
            return candidate_id(terminal)
        if expand_candidates:
            chosen = rank_sentence_region_persistence(expand_candidates, last_sentence_id=last_sentence_id)[0]
            expand_steps_used += 1
            if _moves_to_sentence(chosen):
                touched_sentence = True
                last_sentence_id = dst(chosen) or last_sentence_id
            return candidate_id(chosen)
        raise ValueError("Policy received an observation with no candidate actions.")

    return policy


def make_region_commitment_delayed_answer_policy(
    *,
    min_expand_steps: int = 2,
    region_expand_limit: int = 3,
) -> PolicyFn:
    """Title-region commitment with a slightly longer local exploration budget."""

    return make_title_region_commitment_policy(
        min_expand_steps=min_expand_steps,
        region_expand_limit=region_expand_limit,
    )


def make_relevance_policy_factory(
    policy_name: str,
    *,
    min_expand_steps: int = 1,
    delayed_min_expand_steps: int = 2,
) -> PolicyFactory:
    """Return a fresh relevance-policy factory."""

    if policy_name == "title_region_commitment":
        return lambda: make_title_region_commitment_policy(min_expand_steps=min_expand_steps)
    if policy_name == "sentence_region_persistence":
        return lambda: make_sentence_region_persistence_policy(min_expand_steps=min_expand_steps)
    if policy_name == "region_commitment_delayed_answer":
        return lambda: make_region_commitment_delayed_answer_policy(min_expand_steps=delayed_min_expand_steps)
    raise ValueError(
        f"Unknown HotpotQA relevance policy '{policy_name}'. "
        f"Available policies: {', '.join(RELEVANCE_POLICY_NAMES)}"
    )


def rank_title_region_commitment(
    candidates: list[dict[str, Any]],
    *,
    current_title_id: str | None,
    last_sentence_id: str | None,
) -> list[dict[str, Any]]:
    """Rank candidates by current title region before stable candidate id."""

    def priority(candidate: dict[str, Any]) -> tuple[int, str, int]:
        candidate_relation = relation(candidate)
        if current_title_id and _candidate_in_region(candidate, current_title_id, last_sentence_id):
            if candidate_relation == "next_sentence":
                return (0, candidate_relation, candidate_id(candidate))
            if candidate_relation == "title_to_sentence":
                return (1, candidate_relation, candidate_id(candidate))
            if candidate_relation == "sentence_to_title":
                return (3, candidate_relation, candidate_id(candidate))
            return (4, candidate_relation, candidate_id(candidate))
        if candidate_relation == "title_to_sentence":
            return (2, candidate_relation, candidate_id(candidate))
        if candidate_relation == "next_sentence":
            return (5, candidate_relation, candidate_id(candidate))
        return (6, candidate_relation, candidate_id(candidate))

    return sorted(candidates, key=priority)


def rank_sentence_region_persistence(
    candidates: list[dict[str, Any]],
    *,
    last_sentence_id: str | None,
) -> list[dict[str, Any]]:
    """Rank candidates to persist inside local sentence adjacency."""

    def priority(candidate: dict[str, Any]) -> tuple[int, str, int]:
        candidate_relation = relation(candidate)
        if candidate_relation == "next_sentence" and (last_sentence_id is None or src(candidate) == last_sentence_id):
            return (0, candidate_relation, candidate_id(candidate))
        if candidate_relation == "next_sentence":
            return (1, candidate_relation, candidate_id(candidate))
        if candidate_relation == "title_to_sentence":
            return (2, candidate_relation, candidate_id(candidate))
        if candidate_relation == "sentence_to_title":
            return (4, candidate_relation, candidate_id(candidate))
        return (5, candidate_relation, candidate_id(candidate))

    return sorted(candidates, key=priority)


def src(candidate: dict[str, Any]) -> str | None:
    metadata = candidate.get("metadata") or {}
    value = metadata.get("src")
    return None if value is None else str(value)


def dst(candidate: dict[str, Any]) -> str | None:
    metadata = candidate.get("metadata") or {}
    value = metadata.get("dst")
    return None if value is None else str(value)


def _candidate_in_region(candidate: dict[str, Any], current_title_id: str, last_sentence_id: str | None) -> bool:
    candidate_relation = relation(candidate)
    if candidate_relation == "title_to_sentence" and src(candidate) == current_title_id:
        return True
    if candidate_relation == "sentence_to_title" and dst(candidate) == current_title_id:
        return True
    if candidate_relation == "next_sentence" and (last_sentence_id is None or src(candidate) == last_sentence_id):
        return True
    return False


def _update_region_state(
    candidate: dict[str, Any],
    *,
    current_title_id: str | None,
    last_sentence_id: str | None,
) -> tuple[str | None, str | None]:
    candidate_relation = relation(candidate)
    if candidate_relation == "title_to_sentence":
        return src(candidate) or current_title_id, dst(candidate) or last_sentence_id
    if candidate_relation == "next_sentence":
        return current_title_id, dst(candidate) or last_sentence_id
    if candidate_relation == "sentence_to_title":
        return dst(candidate) or current_title_id, last_sentence_id
    return current_title_id, last_sentence_id


def _moves_to_sentence(candidate: dict[str, Any]) -> bool:
    return relation(candidate) in {"title_to_sentence", "next_sentence"}


def _answer_or_stop(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    answer = first_action(candidates, "ANSWER")
    if answer is not None:
        return answer
    return first_action(candidates, "STOP")
