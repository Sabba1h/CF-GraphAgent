"""Transparent question-conditioned relevance scorers for HotpotQA.

The scorer API is deliberately whitelist-based. Callers pass only query text,
title text, sentence text, and small path-local statistics. The scorer does not
accept graph nodes, benchmark records, gold answers, supporting facts, or raw
metadata dictionaries.
"""

from __future__ import annotations

import re
import string
from dataclasses import asdict, dataclass, field
from typing import Any

STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "his",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "she",
        "that",
        "the",
        "their",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "with",
    }
)
TOKENIZATION_DESCRIPTION = {
    "lowercasing": True,
    "punctuation_stripping": "ASCII punctuation translated to spaces before whitespace tokenization",
    "stopword_handling": "fixed HotpotQA relevance STOPWORDS are removed",
    "tokenization": "whitespace split after punctuation stripping",
}
_PUNCT_TRANSLATION = str.maketrans({char: " " for char in string.punctuation})
_SPACE_RE = re.compile(r"\s+")


@dataclass(slots=True)
class RelevanceScore:
    """One transparent question-conditioned relevance score."""

    scorer_name: str
    total_score: float
    component_scores: dict[str, float]
    component_description: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""

        return asdict(self)


def normalize_relevance_tokens(text: str | None) -> list[str]:
    """Normalize text with fixed lower/punctuation/stopword rules."""

    lowered = (text or "").lower().translate(_PUNCT_TRANSLATION)
    compact = _SPACE_RE.sub(" ", lowered).strip()
    if not compact:
        return []
    return [token for token in compact.split(" ") if token and token not in STOPWORDS]


def compute_token_overlap_score(
    *,
    query_text: str,
    title_text: str = "",
    sentence_text: str = "",
    path_stats: dict[str, float] | None = None,
) -> RelevanceScore:
    """Compute a simple max token overlap score against title/sentence text."""

    stats = _whitelisted_path_stats(path_stats)
    query_tokens = normalize_relevance_tokens(query_text)
    title_tokens = normalize_relevance_tokens(title_text)
    sentence_tokens = normalize_relevance_tokens(sentence_text)
    title_overlap = _overlap_score(query_tokens, title_tokens)
    sentence_overlap = _overlap_score(query_tokens, sentence_tokens)
    max_overlap = max(title_overlap, sentence_overlap)
    continuity_bonus = 0.05 * stats["region_continuity"]
    recent_bonus = 0.03 * stats["recent_region"]
    total = max_overlap + continuity_bonus + recent_bonus
    return RelevanceScore(
        scorer_name="token_overlap",
        total_score=total,
        component_scores={
            "title_overlap": title_overlap,
            "sentence_overlap": sentence_overlap,
            "max_overlap": max_overlap,
            "region_continuity_bonus": continuity_bonus,
            "recent_region_bonus": recent_bonus,
        },
        component_description={
            **TOKENIZATION_DESCRIPTION,
            "formula": "max(title_overlap, sentence_overlap) + 0.05*region_continuity + 0.03*recent_region",
            "overlap_score": "intersection(query_tokens, text_tokens) / max(1, len(query_tokens))",
        },
        metadata={
            "query_token_count": len(query_tokens),
            "title_token_count": len(title_tokens),
            "sentence_token_count": len(sentence_tokens),
        },
    )


def compute_title_sentence_hybrid_score(
    *,
    query_text: str,
    title_text: str = "",
    sentence_text: str = "",
    path_stats: dict[str, float] | None = None,
) -> RelevanceScore:
    """Compute a fixed title/sentence/local-structure hybrid score."""

    stats = _whitelisted_path_stats(path_stats)
    token_score = compute_token_overlap_score(
        query_text=query_text,
        title_text=title_text,
        sentence_text=sentence_text,
        path_stats=stats,
    )
    title_overlap = token_score.component_scores["title_overlap"]
    sentence_overlap = token_score.component_scores["sentence_overlap"]
    continuity = stats["region_continuity"]
    recent = stats["recent_region"]
    total = 0.4 * title_overlap + 0.5 * sentence_overlap + 0.07 * continuity + 0.03 * recent
    return RelevanceScore(
        scorer_name="title_sentence_hybrid",
        total_score=total,
        component_scores={
            "title_overlap": title_overlap,
            "sentence_overlap": sentence_overlap,
            "region_continuity_bonus": 0.07 * continuity,
            "recent_region_bonus": 0.03 * recent,
            "weighted_total": total,
        },
        component_description={
            **TOKENIZATION_DESCRIPTION,
            "formula": "0.4*title_overlap + 0.5*sentence_overlap + 0.07*region_continuity + 0.03*recent_region",
            "overlap_score": "intersection(query_tokens, text_tokens) / max(1, len(query_tokens))",
        },
        metadata=token_score.metadata,
    )


def score_candidate_text(
    *,
    query_text: str,
    title_text: str = "",
    sentence_text: str = "",
    path_stats: dict[str, float] | None = None,
    scorer_name: str = "title_sentence_hybrid",
) -> RelevanceScore:
    """Dispatch a whitelisted text payload to a supported scorer."""

    if scorer_name == "token_overlap":
        return compute_token_overlap_score(
            query_text=query_text,
            title_text=title_text,
            sentence_text=sentence_text,
            path_stats=path_stats,
        )
    if scorer_name == "title_sentence_hybrid":
        return compute_title_sentence_hybrid_score(
            query_text=query_text,
            title_text=title_text,
            sentence_text=sentence_text,
            path_stats=path_stats,
        )
    raise ValueError("scorer_name must be one of {'token_overlap', 'title_sentence_hybrid'}.")


def _overlap_score(query_tokens: list[str], text_tokens: list[str]) -> float:
    if not query_tokens or not text_tokens:
        return 0.0
    return len(set(query_tokens) & set(text_tokens)) / max(1, len(set(query_tokens)))


def _whitelisted_path_stats(path_stats: dict[str, float] | None) -> dict[str, float]:
    raw = path_stats or {}
    return {
        "region_continuity": _clamped_float(raw.get("region_continuity", 0.0)),
        "recent_region": _clamped_float(raw.get("recent_region", 0.0)),
    }


def _clamped_float(value: Any) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, resolved))
