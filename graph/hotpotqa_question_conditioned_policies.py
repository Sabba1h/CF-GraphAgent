"""Question-conditioned HotpotQA path policies.

Policies in this module use only observation query text and whitelisted
candidate metadata fields: src/dst ids, node types, node texts, and relation.
They do not read gold answers, supporting facts, or benchmark metadata.
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
from relevance.hotpotqa_question_conditioned_scorer import RelevanceScore, score_candidate_text

PolicyFn = Callable[[dict[str, Any]], int]
PolicyFactory = Callable[[], PolicyFn]

QUESTION_CONDITIONED_POLICY_NAMES = ("overlap_guided_region",)


def make_overlap_guided_region_policy(
    *,
    min_expand_steps: int = 1,
    scorer_name: str = "title_sentence_hybrid",
) -> PolicyFn:
    """Rank structurally useful candidates by question-conditioned overlap."""

    expand_steps_used = 0
    current_title_id: str | None = None
    last_sentence_id: str | None = None

    def policy(observation: dict[str, Any]) -> int:
        nonlocal expand_steps_used, current_title_id, last_sentence_id
        candidates = sorted_candidates(observation)
        useful_expands = non_repeated_expand_candidates(candidates)
        expand_candidates = useful_expands or expand_candidates_sorted(candidates)
        query_text = str(observation.get("query") or "")

        if useful_expands:
            ranked = rank_overlap_guided_candidates(
                useful_expands,
                query_text=query_text,
                current_title_id=current_title_id,
                last_sentence_id=last_sentence_id,
                scorer_name=scorer_name,
            )
            best_score = score_candidate(
                ranked[0],
                query_text=query_text,
                region_continuity=_region_continuity(ranked[0], current_title_id, last_sentence_id),
                scorer_name=scorer_name,
            )
            if expand_steps_used < min_expand_steps or best_score.total_score > 0.0:
                chosen = ranked[0]
                expand_steps_used += 1
                current_title_id, last_sentence_id = _update_region_state(
                    chosen,
                    current_title_id=current_title_id,
                    last_sentence_id=last_sentence_id,
                )
                return candidate_id(chosen)

        terminal = _answer_or_stop(candidates)
        if terminal is not None:
            return candidate_id(terminal)
        if expand_candidates:
            chosen = rank_overlap_guided_candidates(
                expand_candidates,
                query_text=query_text,
                current_title_id=current_title_id,
                last_sentence_id=last_sentence_id,
                scorer_name=scorer_name,
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


def make_question_conditioned_policy_factory(
    policy_name: str,
    *,
    min_expand_steps: int = 1,
    scorer_name: str = "title_sentence_hybrid",
) -> PolicyFactory:
    """Return a fresh question-conditioned policy factory."""

    if policy_name == "overlap_guided_region":
        return lambda: make_overlap_guided_region_policy(
            min_expand_steps=min_expand_steps,
            scorer_name=scorer_name,
        )
    raise ValueError(
        f"Unknown HotpotQA question-conditioned policy '{policy_name}'. "
        f"Available policies: {', '.join(QUESTION_CONDITIONED_POLICY_NAMES)}"
    )


def rank_overlap_guided_candidates(
    candidates: list[dict[str, Any]],
    *,
    query_text: str,
    current_title_id: str | None,
    last_sentence_id: str | None,
    scorer_name: str = "title_sentence_hybrid",
) -> list[dict[str, Any]]:
    """Rank EXPAND_EDGE candidates with structure priority plus overlap score."""

    def priority(candidate: dict[str, Any]) -> tuple[int, float, str, int]:
        candidate_relation = relation(candidate)
        region_continuity = _region_continuity(candidate, current_title_id, last_sentence_id)
        score = score_candidate(candidate, query_text=query_text, region_continuity=region_continuity, scorer_name=scorer_name)
        if candidate_relation == "title_to_sentence":
            structure_rank = 0
        elif candidate_relation == "next_sentence":
            structure_rank = 1
        elif candidate_relation == "sentence_to_title":
            structure_rank = 3
        else:
            structure_rank = 4
        return (structure_rank, -score.total_score, candidate_relation, candidate_id(candidate))

    return sorted(candidates, key=priority)


def score_candidate(
    candidate: dict[str, Any],
    *,
    query_text: str,
    region_continuity: float = 0.0,
    scorer_name: str = "title_sentence_hybrid",
) -> RelevanceScore:
    """Score a candidate using only whitelisted candidate metadata fields."""

    title_text, sentence_text = _candidate_title_sentence_text(candidate)
    return score_candidate_text(
        query_text=query_text,
        title_text=title_text,
        sentence_text=sentence_text,
        path_stats={"region_continuity": region_continuity, "recent_region": region_continuity},
        scorer_name=scorer_name,
    )


def _candidate_title_sentence_text(candidate: dict[str, Any]) -> tuple[str, str]:
    metadata = candidate.get("metadata") or {}
    src_type = str(metadata.get("src_node_type") or "")
    dst_type = str(metadata.get("dst_node_type") or "")
    src_text = str(metadata.get("src_text") or "")
    dst_text = str(metadata.get("dst_text") or "")
    title_text = ""
    sentence_text = ""
    if src_type == "title":
        title_text = src_text
    if dst_type == "title":
        title_text = dst_text
    if src_type == "sentence":
        sentence_text = src_text
    if dst_type == "sentence":
        sentence_text = dst_text
    return title_text, sentence_text


def _region_continuity(candidate: dict[str, Any], current_title_id: str | None, last_sentence_id: str | None) -> float:
    if current_title_id is None:
        return 0.0
    candidate_relation = relation(candidate)
    src_id = src(candidate)
    dst_id = dst(candidate)
    if candidate_relation == "title_to_sentence" and src_id == current_title_id:
        return 1.0
    if candidate_relation == "sentence_to_title" and dst_id == current_title_id:
        return 1.0
    if candidate_relation == "next_sentence" and (last_sentence_id is None or src_id == last_sentence_id):
        return 1.0
    return 0.0


def src(candidate: dict[str, Any]) -> str | None:
    metadata = candidate.get("metadata") or {}
    value = metadata.get("src")
    return None if value is None else str(value)


def dst(candidate: dict[str, Any]) -> str | None:
    metadata = candidate.get("metadata") or {}
    value = metadata.get("dst")
    return None if value is None else str(value)


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


def _answer_or_stop(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    answer = first_action(candidates, "ANSWER")
    if answer is not None:
        return answer
    return first_action(candidates, "STOP")
