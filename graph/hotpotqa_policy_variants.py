"""Deterministic non-privileged HotpotQA graph policy variants."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

PolicyFn = Callable[[dict[str, Any]], int]
PolicyFactory = Callable[[], PolicyFn]

POLICY_NAMES = (
    "baseline",
    "sentence_first",
    "delayed_answer",
    "hybrid_sentence_delayed",
    "sentence_chain",
    "require_sentence_before_answer",
    "hybrid_sentence_chain_delayed",
)
SENTENCE_RELATIONS = {"next_sentence", "sentence_to_title", "title_to_sentence"}


def make_baseline_policy(*, min_expand_steps: int = 1) -> PolicyFn:
    """Return the current baseline policy with explicit candidate sorting."""

    expand_steps_used = 0

    def policy(observation: dict[str, Any]) -> int:
        nonlocal expand_steps_used
        candidates = sorted_candidates(observation)
        expand_candidates = non_repeated_expand_candidates(candidates) or expand_candidates_sorted(candidates)
        if expand_steps_used < min_expand_steps and expand_candidates:
            expand_steps_used += 1
            return candidate_id(expand_candidates[0])
        answer = first_action(candidates, "ANSWER")
        if answer is not None:
            return candidate_id(answer)
        stop = first_action(candidates, "STOP")
        if stop is not None:
            return candidate_id(stop)
        if expand_candidates:
            expand_steps_used += 1
            return candidate_id(expand_candidates[0])
        raise ValueError("Policy received an observation with no candidate actions.")

    return policy


def make_sentence_first_policy(*, min_expand_steps: int = 1) -> PolicyFn:
    """Prefer sentence-related graph expansions before answering."""

    expand_steps_used = 0

    def policy(observation: dict[str, Any]) -> int:
        nonlocal expand_steps_used
        candidates = sorted_candidates(observation)
        expand_candidates = non_repeated_expand_candidates(candidates) or expand_candidates_sorted(candidates)
        if expand_steps_used < min_expand_steps and expand_candidates:
            expand_steps_used += 1
            return candidate_id(rank_sentence_first(expand_candidates)[0])

        sentence_candidates = [candidate for candidate in expand_candidates if relation(candidate) in SENTENCE_RELATIONS]
        if sentence_candidates:
            expand_steps_used += 1
            return candidate_id(rank_sentence_first(sentence_candidates)[0])

        answer = first_action(candidates, "ANSWER")
        if answer is not None:
            return candidate_id(answer)
        stop = first_action(candidates, "STOP")
        if stop is not None:
            return candidate_id(stop)
        if expand_candidates:
            expand_steps_used += 1
            return candidate_id(rank_sentence_first(expand_candidates)[0])
        raise ValueError("Policy received an observation with no candidate actions.")

    return policy


def make_delayed_answer_policy(*, min_expand_steps: int = 2) -> PolicyFn:
    """Delay ANSWER while useful EXPAND_EDGE candidates still exist."""

    expand_steps_used = 0

    def policy(observation: dict[str, Any]) -> int:
        nonlocal expand_steps_used
        candidates = sorted_candidates(observation)
        expand_candidates = non_repeated_expand_candidates(candidates) or expand_candidates_sorted(candidates)
        if expand_candidates and expand_steps_used < min_expand_steps:
            expand_steps_used += 1
            return candidate_id(expand_candidates[0])

        answer = first_action(candidates, "ANSWER")
        if answer is not None:
            return candidate_id(answer)
        stop = first_action(candidates, "STOP")
        if stop is not None:
            return candidate_id(stop)
        if expand_candidates:
            expand_steps_used += 1
            return candidate_id(expand_candidates[0])
        raise ValueError("Policy received an observation with no candidate actions.")

    return policy


def make_hybrid_sentence_delayed_policy(*, min_expand_steps: int = 2) -> PolicyFn:
    """Combine sentence-first expansion with delayed-answer timing."""

    expand_steps_used = 0

    def policy(observation: dict[str, Any]) -> int:
        nonlocal expand_steps_used
        candidates = sorted_candidates(observation)
        expand_candidates = non_repeated_expand_candidates(candidates) or expand_candidates_sorted(candidates)
        if expand_candidates and expand_steps_used < min_expand_steps:
            expand_steps_used += 1
            return candidate_id(rank_sentence_first(expand_candidates)[0])

        sentence_candidates = [candidate for candidate in expand_candidates if relation(candidate) in SENTENCE_RELATIONS]
        if sentence_candidates:
            expand_steps_used += 1
            return candidate_id(rank_sentence_first(sentence_candidates)[0])

        answer = first_action(candidates, "ANSWER")
        if answer is not None:
            return candidate_id(answer)
        stop = first_action(candidates, "STOP")
        if stop is not None:
            return candidate_id(stop)
        if expand_candidates:
            expand_steps_used += 1
            return candidate_id(rank_sentence_first(expand_candidates)[0])
        raise ValueError("Policy received an observation with no candidate actions.")

    return policy


def make_sentence_chain_policy(*, min_expand_steps: int = 1) -> PolicyFn:
    """Prefer entering sentence layer, then continue along next_sentence."""

    expand_steps_used = 0
    touched_sentence = False

    def policy(observation: dict[str, Any]) -> int:
        nonlocal expand_steps_used, touched_sentence
        candidates = sorted_candidates(observation)
        useful_expand_candidates = non_repeated_expand_candidates(candidates)
        expand_candidates = useful_expand_candidates or expand_candidates_sorted(candidates)

        if useful_expand_candidates:
            ranked = rank_sentence_chain(useful_expand_candidates, touched_sentence=touched_sentence)
            if touched_sentence:
                next_sentence = [candidate for candidate in ranked if relation(candidate) == "next_sentence"]
                if next_sentence:
                    chosen = next_sentence[0]
                    expand_steps_used += 1
                    touched_sentence = touched_sentence or expands_to_sentence(chosen)
                    return candidate_id(chosen)

            if not touched_sentence:
                sentence_entry = [candidate for candidate in ranked if relation(candidate) == "title_to_sentence"]
                if sentence_entry:
                    chosen = sentence_entry[0]
                    expand_steps_used += 1
                    touched_sentence = True
                    return candidate_id(chosen)

            if expand_steps_used < min_expand_steps:
                chosen = ranked[0]
                expand_steps_used += 1
                touched_sentence = touched_sentence or expands_to_sentence(chosen)
                return candidate_id(chosen)

        answer = first_action(candidates, "ANSWER")
        if answer is not None:
            return candidate_id(answer)
        stop = first_action(candidates, "STOP")
        if stop is not None:
            return candidate_id(stop)
        if expand_candidates:
            chosen = rank_sentence_chain(expand_candidates, touched_sentence=touched_sentence)[0]
            expand_steps_used += 1
            touched_sentence = touched_sentence or expands_to_sentence(chosen)
            return candidate_id(chosen)
        raise ValueError("Policy received an observation with no candidate actions.")

    return policy


def make_require_sentence_before_answer_policy(*, min_expand_steps: int = 1) -> PolicyFn:
    """Require a reachable sentence before ANSWER, with escape when no expand exists."""

    expand_steps_used = 0
    touched_sentence = False

    def policy(observation: dict[str, Any]) -> int:
        nonlocal expand_steps_used, touched_sentence
        candidates = sorted_candidates(observation)
        useful_expand_candidates = non_repeated_expand_candidates(candidates)
        expand_candidates = useful_expand_candidates or expand_candidates_sorted(candidates)

        if useful_expand_candidates and (not touched_sentence or expand_steps_used < min_expand_steps):
            ranked = rank_sentence_chain(useful_expand_candidates, touched_sentence=touched_sentence)
            chosen = ranked[0]
            expand_steps_used += 1
            touched_sentence = touched_sentence or expands_to_sentence(chosen)
            return candidate_id(chosen)

        answer = first_action(candidates, "ANSWER")
        if answer is not None:
            return candidate_id(answer)
        stop = first_action(candidates, "STOP")
        if stop is not None:
            return candidate_id(stop)
        if expand_candidates:
            chosen = rank_sentence_chain(expand_candidates, touched_sentence=touched_sentence)[0]
            expand_steps_used += 1
            touched_sentence = touched_sentence or expands_to_sentence(chosen)
            return candidate_id(chosen)
        raise ValueError("Policy received an observation with no candidate actions.")

    return policy


def make_hybrid_sentence_chain_delayed_policy(*, min_expand_steps: int = 2) -> PolicyFn:
    """Combine sentence-chain preference with delayed answer timing."""

    expand_steps_used = 0
    touched_sentence = False

    def policy(observation: dict[str, Any]) -> int:
        nonlocal expand_steps_used, touched_sentence
        candidates = sorted_candidates(observation)
        useful_expand_candidates = non_repeated_expand_candidates(candidates)
        expand_candidates = useful_expand_candidates or expand_candidates_sorted(candidates)

        if useful_expand_candidates:
            ranked = rank_sentence_chain(useful_expand_candidates, touched_sentence=touched_sentence)
            if not touched_sentence or expand_steps_used < min_expand_steps:
                chosen = ranked[0]
                expand_steps_used += 1
                touched_sentence = touched_sentence or expands_to_sentence(chosen)
                return candidate_id(chosen)
            next_sentence = [candidate for candidate in ranked if relation(candidate) == "next_sentence"]
            if next_sentence:
                chosen = next_sentence[0]
                expand_steps_used += 1
                touched_sentence = True
                return candidate_id(chosen)

        answer = first_action(candidates, "ANSWER")
        if answer is not None:
            return candidate_id(answer)
        stop = first_action(candidates, "STOP")
        if stop is not None:
            return candidate_id(stop)
        if expand_candidates:
            chosen = rank_sentence_chain(expand_candidates, touched_sentence=touched_sentence)[0]
            expand_steps_used += 1
            touched_sentence = touched_sentence or expands_to_sentence(chosen)
            return candidate_id(chosen)
        raise ValueError("Policy received an observation with no candidate actions.")

    return policy


def make_policy_factory(
    policy_name: str,
    *,
    min_expand_steps: int = 1,
    delayed_min_expand_steps: int = 2,
) -> PolicyFactory:
    """Return a fresh-policy factory for the requested variant."""

    if policy_name == "baseline":
        return lambda: make_baseline_policy(min_expand_steps=min_expand_steps)
    if policy_name == "sentence_first":
        return lambda: make_sentence_first_policy(min_expand_steps=min_expand_steps)
    if policy_name == "delayed_answer":
        return lambda: make_delayed_answer_policy(min_expand_steps=delayed_min_expand_steps)
    if policy_name == "hybrid_sentence_delayed":
        return lambda: make_hybrid_sentence_delayed_policy(min_expand_steps=delayed_min_expand_steps)
    if policy_name == "sentence_chain":
        return lambda: make_sentence_chain_policy(min_expand_steps=min_expand_steps)
    if policy_name == "require_sentence_before_answer":
        return lambda: make_require_sentence_before_answer_policy(min_expand_steps=min_expand_steps)
    if policy_name == "hybrid_sentence_chain_delayed":
        return lambda: make_hybrid_sentence_chain_delayed_policy(min_expand_steps=delayed_min_expand_steps)
    raise ValueError(f"Unknown HotpotQA policy '{policy_name}'. Available policies: {', '.join(POLICY_NAMES)}")


def sorted_candidates(observation: dict[str, Any]) -> list[dict[str, Any]]:
    """Return candidates in a stable order before applying policy rules."""

    candidates = observation.get("candidate_actions", [])
    return sorted(candidates, key=lambda candidate: (candidate_id(candidate), action_type(candidate), relation(candidate)))


def expand_candidates_sorted(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return EXPAND_EDGE candidates in stable candidate_id order."""

    return [candidate for candidate in candidates if action_type(candidate) == "EXPAND_EDGE"]


def non_repeated_expand_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prefer candidates that are not explicitly marked as already expanded."""

    return [candidate for candidate in expand_candidates_sorted(candidates) if "already in working subgraph" not in description(candidate)]


def rank_sentence_first(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank candidates by sentence relation priority and stable tie-breakers."""

    return sorted(
        candidates,
        key=lambda candidate: (
            0 if relation(candidate) in SENTENCE_RELATIONS else 1,
            relation(candidate),
            candidate_id(candidate),
        ),
    )


def rank_sentence_chain(candidates: list[dict[str, Any]], *, touched_sentence: bool) -> list[dict[str, Any]]:
    """Rank candidates to enter sentence layer, then follow sentence chains."""

    def priority(candidate: dict[str, Any]) -> tuple[int, str, int]:
        candidate_relation = relation(candidate)
        if touched_sentence and candidate_relation == "next_sentence":
            return (0, candidate_relation, candidate_id(candidate))
        if candidate_relation == "title_to_sentence":
            return (1, candidate_relation, candidate_id(candidate))
        if candidate_relation == "next_sentence":
            return (2, candidate_relation, candidate_id(candidate))
        if candidate_relation == "sentence_to_title":
            return (3, candidate_relation, candidate_id(candidate))
        return (4, candidate_relation, candidate_id(candidate))

    return sorted(candidates, key=priority)


def expands_to_sentence(candidate: dict[str, Any]) -> bool:
    """Infer whether a non-privileged candidate moves into/within sentence layer."""

    return relation(candidate) in {"title_to_sentence", "next_sentence"}


def first_action(candidates: list[dict[str, Any]], expected_action_type: str) -> dict[str, Any] | None:
    """Return the first action of a type after stable sorting."""

    for candidate in candidates:
        if action_type(candidate) == expected_action_type:
            return candidate
    return None


def candidate_id(candidate: dict[str, Any]) -> int:
    return int(candidate.get("candidate_id", -1))


def action_type(candidate: dict[str, Any]) -> str:
    return str(candidate.get("action_type", ""))


def relation(candidate: dict[str, Any]) -> str:
    metadata = candidate.get("metadata") or {}
    return str(metadata.get("relation", ""))


def description(candidate: dict[str, Any]) -> str:
    return str(candidate.get("description", ""))
