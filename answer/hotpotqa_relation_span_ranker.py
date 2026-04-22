"""Candidate span rankers for HotpotQA relation/span answers.

Rankers operate only on candidate span metadata already produced by the
relation/span mapper. They do not discover new spans, inspect graph metadata,
read gold answers, or use supporting facts.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class HotpotQARelationSpanRankingResult:
    """A deterministic selection result over relation/span candidate spans."""

    ranker_name: str
    candidate_spans: list[dict[str, Any]]
    ranked_spans: list[dict[str, Any]]
    selected_span_payload: dict[str, Any] | None
    selected_span: str
    selected_span_rank: int | None
    selected_span_reason: str
    fallback_occurred: bool
    fallback_reason: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""

        return asdict(self)


RelationSpanRanker = Callable[
    [list[dict[str, Any]], dict[str, Any] | None],
    HotpotQARelationSpanRankingResult,
]
RelationSpanRankerFactory = Callable[[], RelationSpanRanker]

RELATION_SPAN_RANKER_NAMES = (
    "first_candidate",
    "shortest_nonempty",
    "pattern_priority",
    "shortest_then_pattern_priority",
)

PATTERN_PRIORITY: dict[str, int] = {
    "served_as": 0,
    "was_the": 1,
    "known_for": 2,
    "part_of": 3,
    "member_of": 4,
    "born_in": 5,
    "located_in": 6,
    "based_in": 7,
    "clause_split": 90,
}
LOW_INFORMATION_TOKENS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "before",
    "by",
    "for",
    "from",
    "he",
    "in",
    "it",
    "of",
    "on",
    "or",
    "she",
    "the",
    "to",
    "was",
    "were",
}
TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)


def make_first_candidate_ranker() -> RelationSpanRanker:
    """Return the batch-28 default selector.

    Despite the name, this intentionally selects the first span after the same
    default ordering used by batch 28 ``pattern_span``. This makes it a strict
    old-selection baseline for batch 29.
    """

    def ranker(
        candidate_spans: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
    ) -> HotpotQARelationSpanRankingResult:
        del context
        ranked = _rank_batch28_default(candidate_spans)
        return _result(
            ranker_name="first_candidate",
            candidate_spans=candidate_spans,
            ranked_spans=ranked,
            selected_span_reason="batch28_default_selection_reproduced",
            metadata={
                "strategy": "batch28_default_order",
                "sort_key": ["token_count", "char_count", "node_index", "span_index", "text"],
            },
        )

    return ranker


def make_shortest_nonempty_ranker() -> RelationSpanRanker:
    """Return a ranker preferring short informative spans."""

    def ranker(
        candidate_spans: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
    ) -> HotpotQARelationSpanRankingResult:
        del context
        usable = [span for span in candidate_spans if not _is_low_information_span(str(span.get("text") or ""))]
        ranked = sorted(usable or candidate_spans, key=_shortest_key)
        return _result(
            ranker_name="shortest_nonempty",
            candidate_spans=candidate_spans,
            ranked_spans=ranked,
            selected_span_reason="shortest_informative_span",
            metadata={
                "strategy": "shortest_informative_span",
                "filtered_low_information_count": len(candidate_spans) - len(usable),
            },
        )

    return ranker


def make_pattern_priority_ranker() -> RelationSpanRanker:
    """Return a ranker using fixed batch-28 pattern strategy priorities."""

    def ranker(
        candidate_spans: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
    ) -> HotpotQARelationSpanRankingResult:
        del context
        ranked = sorted(candidate_spans, key=_pattern_priority_key)
        return _result(
            ranker_name="pattern_priority",
            candidate_spans=candidate_spans,
            ranked_spans=ranked,
            selected_span_reason="fixed_pattern_priority",
            metadata={
                "strategy": "fixed_pattern_priority",
                "pattern_priority": dict(PATTERN_PRIORITY),
            },
        )

    return ranker


def make_shortest_then_pattern_priority_ranker() -> RelationSpanRanker:
    """Return a conservative hybrid: shortest bucket, then pattern priority."""

    def ranker(
        candidate_spans: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
    ) -> HotpotQARelationSpanRankingResult:
        del context
        if not candidate_spans:
            ranked: list[dict[str, Any]] = []
            min_tokens = None
            shortlisted: list[dict[str, Any]] = []
        else:
            min_tokens = min(_token_count(span) for span in candidate_spans)
            shortlisted = [span for span in candidate_spans if _token_count(span) <= min_tokens + 2]
            ranked = sorted(shortlisted, key=_pattern_priority_key)
        return _result(
            ranker_name="shortest_then_pattern_priority",
            candidate_spans=candidate_spans,
            ranked_spans=ranked,
            selected_span_reason="shortest_bucket_then_fixed_pattern_priority",
            metadata={
                "strategy": "shortest_then_fixed_pattern_priority",
                "min_token_count": min_tokens,
                "shortlisted_count": len(shortlisted),
                "pattern_priority": dict(PATTERN_PRIORITY),
            },
        )

    return ranker


def make_relation_span_ranker_factory(ranker_name: str) -> RelationSpanRankerFactory:
    """Return a fresh relation/span ranker factory."""

    if ranker_name == "first_candidate":
        return make_first_candidate_ranker
    if ranker_name == "shortest_nonempty":
        return make_shortest_nonempty_ranker
    if ranker_name == "pattern_priority":
        return make_pattern_priority_ranker
    if ranker_name == "shortest_then_pattern_priority":
        return make_shortest_then_pattern_priority_ranker
    raise ValueError(
        f"Unknown HotpotQA relation/span ranker '{ranker_name}'. "
        f"Available rankers: {', '.join(RELATION_SPAN_RANKER_NAMES)}"
    )


def rank_spans_batch28_default(candidate_spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expose the exact batch-28 default relation/span span ordering."""

    return _rank_batch28_default(candidate_spans)


def _result(
    *,
    ranker_name: str,
    candidate_spans: list[dict[str, Any]],
    ranked_spans: list[dict[str, Any]],
    selected_span_reason: str,
    metadata: dict[str, Any],
) -> HotpotQARelationSpanRankingResult:
    if not ranked_spans:
        return HotpotQARelationSpanRankingResult(
            ranker_name=ranker_name,
            candidate_spans=list(candidate_spans),
            ranked_spans=[],
            selected_span_payload=None,
            selected_span="",
            selected_span_rank=None,
            selected_span_reason="no_candidate_span_available",
            fallback_occurred=True,
            fallback_reason="no_candidate_span_available",
            metadata=metadata,
        )
    selected = dict(ranked_spans[0])
    selected_rank = _selected_rank(candidate_spans, selected)
    return HotpotQARelationSpanRankingResult(
        ranker_name=ranker_name,
        candidate_spans=list(candidate_spans),
        ranked_spans=[dict(span) for span in ranked_spans],
        selected_span_payload=selected,
        selected_span=str(selected.get("text") or ""),
        selected_span_rank=selected_rank,
        selected_span_reason=selected_span_reason,
        fallback_occurred=False,
        fallback_reason=None,
        metadata=metadata,
    )


def _rank_batch28_default(candidate_spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(candidate_spans, key=_batch28_default_key)


def _batch28_default_key(span: dict[str, Any]) -> tuple[int, int, int, int, str]:
    return (
        _token_count(span),
        _char_count(span),
        int(span.get("node_index", 0)),
        int(span.get("span_index", 0)),
        str(span.get("text", "")),
    )


def _shortest_key(span: dict[str, Any]) -> tuple[int, int, int, int, str]:
    return _batch28_default_key(span)


def _pattern_priority_key(span: dict[str, Any]) -> tuple[int, int, int, int, int, str]:
    return (
        PATTERN_PRIORITY.get(str(span.get("strategy") or ""), 100),
        _token_count(span),
        _char_count(span),
        int(span.get("node_index", 0)),
        int(span.get("span_index", 0)),
        str(span.get("text", "")),
    )


def _selected_rank(candidate_spans: list[dict[str, Any]], selected: dict[str, Any]) -> int | None:
    selected_key = _identity_key(selected)
    for index, span in enumerate(candidate_spans):
        if _identity_key(span) == selected_key:
            return index
    return None


def _identity_key(span: dict[str, Any]) -> tuple[Any, ...]:
    return (
        span.get("text"),
        span.get("strategy"),
        span.get("source_node_id"),
        span.get("node_index"),
        span.get("span_index"),
    )


def _token_count(span: dict[str, Any]) -> int:
    if span.get("token_count") is not None:
        return int(span.get("token_count") or 0)
    return len(TOKEN_RE.findall(str(span.get("text") or "")))


def _char_count(span: dict[str, Any]) -> int:
    if span.get("char_count") is not None:
        return int(span.get("char_count") or 0)
    return len(str(span.get("text") or ""))


def _is_low_information_span(text: str) -> bool:
    tokens = [token.lower() for token in TOKEN_RE.findall(text)]
    return not tokens or all(token in LOW_INFORMATION_TOKENS for token in tokens)
