"""Proposal upgrades for HotpotQA relation/span candidate spans.

Proposal variants sit between relation/span discovery and ranking. They do not
change answer-type routing, do not inspect gold answers or supporting facts,
and only use whitelisted local text fields plus existing discovery candidates.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from relevance.hotpotqa_question_conditioned_scorer import (
    STOPWORDS,
    TOKENIZATION_DESCRIPTION,
    normalize_relevance_tokens,
    score_candidate_text,
)


@dataclass(slots=True)
class HotpotQARelationSpanProposalResult:
    """Candidate spans proposed for one relation/span decision."""

    proposal_variant_name: str
    candidate_spans: list[dict[str, Any]]
    proposal_reason: str
    useful_candidate_proxy_stats: dict[str, Any]
    fallback_occurred: bool
    fallback_reason: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""

        return asdict(self)


RelationSpanProposal = Callable[[dict[str, Any]], HotpotQARelationSpanProposalResult]
RelationSpanProposalFactory = Callable[[], RelationSpanProposal]

RELATION_SPAN_PROPOSAL_NAMES = (
    "baseline_discovery_proposal",
    "query_conditioned_proposal",
    "constrained_local_proposal",
    "query_conditioned_plus_constrained_proposal",
)

CLAUSE_SPLIT_RE = re.compile(r"[,;:\-\u2013\u2014()]")
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
CONNECTOR_TOKENS = {"of", "in", "for", "with", "from", "to", "at", "on"}
WHITELIST_FIELDS = [
    "query_text",
    "selected_sentence_text",
    "path_touched_local_sentence_text",
    "title_text",
    "parent_title_text",
    "local_structure_statistics",
    "discovery_candidate_spans",
]


def make_baseline_discovery_proposal() -> RelationSpanProposal:
    """Return the batch-30 discovery passthrough baseline."""

    def proposal(context: dict[str, Any]) -> HotpotQARelationSpanProposalResult:
        candidates = _baseline_candidates(context)
        return _result(
            proposal_variant_name="baseline_discovery_proposal",
            candidate_spans=candidates,
            proposal_reason="discovery_candidate_passthrough",
            fallback_reason="no_discovery_candidate_available" if not candidates else None,
            metadata={
                "strategy": "baseline_discovery_passthrough",
                "scorer_name": None,
                "scorer_normalization": None,
            },
        )

    return proposal


def make_query_conditioned_proposal(
    *,
    scorer_name: str = "title_sentence_hybrid",
    per_candidate_limit: int = 3,
    sentence_backfill_limit: int = 2,
) -> RelationSpanProposal:
    """Return a query-conditioned proposal reusing the current scorer recipe."""

    def proposal(context: dict[str, Any]) -> HotpotQARelationSpanProposalResult:
        base_candidates = _baseline_candidates(context)
        source_by_id = _source_node_index(context)
        query_text = str(context.get("query_text") or "")
        proposed: list[dict[str, Any]] = []
        scored_sentence_count = 0

        for candidate in base_candidates:
            source_node = source_by_id.get(str(candidate.get("source_node_id") or ""))
            if source_node is None:
                continue
            scored_sentence_count += 1
            variants = _query_conditioned_variants(
                candidate=candidate,
                source_node=source_node,
                query_text=query_text,
                scorer_name=scorer_name,
                per_candidate_limit=per_candidate_limit,
            )
            proposed.extend(variants)

        if not proposed:
            proposed.extend(
                _query_conditioned_sentence_backfill(
                    context=context,
                    scorer_name=scorer_name,
                    limit=sentence_backfill_limit,
                )
            )

        proposed = _dedupe_candidates(proposed)
        return _result(
            proposal_variant_name="query_conditioned_proposal",
            candidate_spans=proposed,
            proposal_reason="query_conditioned_local_subspan_proposal",
            fallback_reason="no_query_conditioned_candidate_available" if not proposed else None,
            metadata={
                "strategy": "query_conditioned_local_subspans",
                "scorer_name": scorer_name,
                "scorer_normalization": {**TOKENIZATION_DESCRIPTION},
                "scored_sentence_count": scored_sentence_count,
            },
        )

    return proposal


def make_constrained_local_proposal() -> RelationSpanProposal:
    """Return a constraint-driven local proposal over discovery candidates."""

    def proposal(context: dict[str, Any]) -> HotpotQARelationSpanProposalResult:
        base_candidates = _baseline_candidates(context)
        source_by_id = _source_node_index(context)
        proposed: list[dict[str, Any]] = []

        for candidate in base_candidates:
            source_node = source_by_id.get(str(candidate.get("source_node_id") or ""))
            if source_node is None:
                continue
            proposed.extend(_constrained_variants(candidate=candidate, source_node=source_node))

        proposed = _dedupe_candidates(proposed)
        return _result(
            proposal_variant_name="constrained_local_proposal",
            candidate_spans=proposed,
            proposal_reason="constrained_local_subspan_proposal",
            fallback_reason="no_constrained_local_candidate_available" if not proposed else None,
            metadata={
                "strategy": "constrained_local_subspans",
                "scorer_name": None,
                "scorer_normalization": None,
            },
        )

    return proposal


def make_query_conditioned_plus_constrained_proposal(
    *,
    scorer_name: str = "title_sentence_hybrid",
) -> RelationSpanProposal:
    """Combine query-conditioned sentence choice with constrained local variants."""

    query_conditioned = make_query_conditioned_proposal(scorer_name=scorer_name)

    def proposal(context: dict[str, Any]) -> HotpotQARelationSpanProposalResult:
        query_result = query_conditioned(context)
        constrained = make_constrained_local_proposal()(context)
        combined = _dedupe_candidates(query_result.candidate_spans + constrained.candidate_spans)
        return _result(
            proposal_variant_name="query_conditioned_plus_constrained_proposal",
            candidate_spans=combined,
            proposal_reason="query_conditioned_candidates_plus_constrained_local_subspans",
            fallback_reason="no_combined_proposal_candidate_available" if not combined else None,
            metadata={
                "strategy": "query_conditioned_plus_constrained",
                "scorer_name": scorer_name,
                "scorer_normalization": {**TOKENIZATION_DESCRIPTION},
                "query_conditioned_result": query_result.to_dict(),
                "constrained_result": constrained.to_dict(),
            },
        )

    return proposal


def make_relation_span_proposal_factory(proposal_name: str) -> RelationSpanProposalFactory:
    """Return a fresh relation/span proposal factory."""

    if proposal_name == "baseline_discovery_proposal":
        return make_baseline_discovery_proposal
    if proposal_name == "query_conditioned_proposal":
        return make_query_conditioned_proposal
    if proposal_name == "constrained_local_proposal":
        return make_constrained_local_proposal
    if proposal_name == "query_conditioned_plus_constrained_proposal":
        return make_query_conditioned_plus_constrained_proposal
    raise ValueError(
        f"Unknown HotpotQA relation/span proposal '{proposal_name}'. "
        f"Available proposals: {', '.join(RELATION_SPAN_PROPOSAL_NAMES)}"
    )


def _baseline_candidates(context: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = context.get("base_candidate_spans")
    if not isinstance(candidates, list):
        return []
    return [dict(candidate) for candidate in candidates if isinstance(candidate, dict)]


def _source_node_index(context: dict[str, Any]) -> dict[str, dict[str, Any]]:
    source_nodes = context.get("source_nodes")
    if not isinstance(source_nodes, list):
        return {}
    resolved: dict[str, dict[str, Any]] = {}
    for node in source_nodes:
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("node_id") or "")
        if not node_id:
            continue
        resolved[node_id] = dict(node)
    return resolved


def _query_conditioned_variants(
    *,
    candidate: dict[str, Any],
    source_node: dict[str, Any],
    query_text: str,
    scorer_name: str,
    per_candidate_limit: int,
) -> list[dict[str, Any]]:
    base_text = str(candidate.get("text") or "")
    base_tokens = _raw_tokens(base_text)
    variants: list[dict[str, Any]] = []
    for variant_index, variant in enumerate(_content_subspan_variants(base_text)):
        score = _candidate_score(
            query_text=query_text,
            source_node=source_node,
            candidate_text=variant["text"],
            scorer_name=scorer_name,
        )
        variants.append(
            _proposal_payload_from_candidate(
                candidate=candidate,
                source_node=source_node,
                text=variant["text"],
                span_index_offset=100 + variant_index,
                proposal_origin="query_conditioned_subspan",
                proposal_reason=variant["reason"],
                local_score=score.total_score,
                local_score_components=score.component_scores,
                local_component_description=score.component_description,
            )
        )
    ranked = sorted(
        variants,
        key=lambda item: (
            -float(item.get("proposal_local_score") or 0.0),
            int(item.get("token_count") or 0),
            int(item.get("char_count") or 0),
            str(item.get("text") or ""),
        ),
    )
    if base_tokens and len(base_tokens) < 5:
        return ranked[:1] or [dict(candidate)]
    return ranked[: max(1, per_candidate_limit)]


def _query_conditioned_sentence_backfill(
    *,
    context: dict[str, Any],
    scorer_name: str,
    limit: int,
) -> list[dict[str, Any]]:
    query_text = str(context.get("query_text") or "")
    source_nodes = list(_source_node_index(context).values())
    scored_clauses: list[dict[str, Any]] = []
    for node_index, source_node in enumerate(source_nodes):
        for clause_index, clause in enumerate(_clause_candidates(str(source_node.get("text") or ""))):
            score = _candidate_score(
                query_text=query_text,
                source_node=source_node,
                candidate_text=clause,
                scorer_name=scorer_name,
            )
            scored_clauses.append(
                _proposal_payload(
                    text=clause,
                    strategy="clause_split",
                    source_node=source_node,
                    node_index=node_index,
                    span_index=8000 + clause_index,
                    proposal_origin="query_conditioned_clause_backfill",
                    proposal_reason="query_conditioned_clause_backfill",
                    local_score=score.total_score,
                    local_score_components=score.component_scores,
                    local_component_description=score.component_description,
                )
            )
    ranked = sorted(
        scored_clauses,
        key=lambda item: (
            -float(item.get("proposal_local_score") or 0.0),
            int(item.get("token_count") or 0),
            int(item.get("char_count") or 0),
            str(item.get("text") or ""),
        ),
    )
    return ranked[:limit]


def _constrained_variants(*, candidate: dict[str, Any], source_node: dict[str, Any]) -> list[dict[str, Any]]:
    base_text = str(candidate.get("text") or "")
    variants = _content_subspan_variants(base_text)
    payloads: list[dict[str, Any]] = []
    for variant_index, variant in enumerate(variants):
        payloads.append(
            _proposal_payload_from_candidate(
                candidate=candidate,
                source_node=source_node,
                text=variant["text"],
                span_index_offset=300 + variant_index,
                proposal_origin="constrained_local_subspan",
                proposal_reason=variant["reason"],
                local_score=None,
                local_score_components=None,
                local_component_description=None,
            )
        )
    return payloads


def _content_subspan_variants(text: str) -> list[dict[str, str]]:
    tokens = _raw_tokens(text)
    if not tokens:
        return []
    variants: list[dict[str, str]] = [{"text": " ".join(tokens), "reason": "full_candidate_span"}]
    stripped = _strip_leading_stopwords(tokens)
    if stripped != tokens:
        variants.append({"text": " ".join(stripped), "reason": "strip_leading_stopwords"})
    if len(stripped) >= 5:
        for width in range(2, min(4, len(stripped)) + 1):
            variants.append(
                {
                    "text": " ".join(stripped[:width]),
                    "reason": f"prefix_{width}_tokens",
                }
            )
        connector_index = _first_connector_index(stripped)
        if connector_index is not None and connector_index >= 2:
            variants.append(
                {
                    "text": " ".join(stripped[:connector_index]),
                    "reason": "prefix_before_connector",
                }
            )
    return _dedupe_variant_texts(variants)


def _candidate_score(
    *,
    query_text: str,
    source_node: dict[str, Any],
    candidate_text: str,
    scorer_name: str,
) -> Any:
    source_level = str(source_node.get("source_level") or "")
    step_idx = int(source_node.get("step_idx") or 0)
    path_stats = {
        "region_continuity": 1.0 if source_level == "selected_sentence" else 0.4,
        "recent_region": 1.0 / max(1, step_idx),
    }
    return score_candidate_text(
        query_text=query_text,
        title_text=str(source_node.get("parent_title_text") or source_node.get("title_text") or ""),
        sentence_text=candidate_text,
        path_stats=path_stats,
        scorer_name=scorer_name,
    )


def _proposal_payload_from_candidate(
    *,
    candidate: dict[str, Any],
    source_node: dict[str, Any],
    text: str,
    span_index_offset: int,
    proposal_origin: str,
    proposal_reason: str,
    local_score: float | None,
    local_score_components: dict[str, float] | None,
    local_component_description: dict[str, Any] | None,
) -> dict[str, Any]:
    return _proposal_payload(
        text=text,
        strategy=str(candidate.get("strategy") or ""),
        source_node=source_node,
        node_index=int(candidate.get("node_index", 0)),
        span_index=int(candidate.get("span_index", 0)) * 100 + span_index_offset,
        proposal_origin=proposal_origin,
        proposal_reason=proposal_reason,
        local_score=local_score,
        local_score_components=local_score_components,
        local_component_description=local_component_description,
    )


def _proposal_payload(
    *,
    text: str,
    strategy: str,
    source_node: dict[str, Any],
    node_index: int,
    span_index: int,
    proposal_origin: str,
    proposal_reason: str,
    local_score: float | None,
    local_score_components: dict[str, float] | None,
    local_component_description: dict[str, Any] | None,
) -> dict[str, Any]:
    clean = _clean_span(text)
    tokens = _raw_tokens(clean)
    return {
        "text": clean,
        "strategy": strategy,
        "source_node_id": source_node.get("node_id"),
        "source_node_type": source_node.get("node_type"),
        "source_text": source_node.get("text"),
        "source_level": source_node.get("source_level"),
        "title_text": source_node.get("title_text") or source_node.get("parent_title_text") or "",
        "token_count": len(tokens),
        "char_count": len(clean),
        "node_index": node_index,
        "span_index": span_index,
        "proposal_origin": proposal_origin,
        "proposal_reason": proposal_reason,
        "proposal_local_score": local_score,
        "proposal_local_score_components": dict(local_score_components or {}),
        "proposal_component_description": dict(local_component_description or {}),
    }


def _result(
    *,
    proposal_variant_name: str,
    candidate_spans: list[dict[str, Any]],
    proposal_reason: str,
    fallback_reason: str | None,
    metadata: dict[str, Any],
) -> HotpotQARelationSpanProposalResult:
    candidates = [candidate for candidate in candidate_spans if _is_usable_candidate(candidate)]
    proxy_stats = _proxy_stats(candidates)
    return HotpotQARelationSpanProposalResult(
        proposal_variant_name=proposal_variant_name,
        candidate_spans=candidates,
        proposal_reason=proposal_reason,
        useful_candidate_proxy_stats=proxy_stats,
        fallback_occurred=not candidates,
        fallback_reason=fallback_reason if not candidates else None,
        metadata={
            **metadata,
            **_source_level_summary(candidates),
            "candidate_count": len(candidates),
            "whitelist_fields": list(WHITELIST_FIELDS),
        },
    )


def _proxy_stats(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(candidate.get("proposal_local_score") or 0.0) for candidate in candidates]
    return {
        "candidate_count": len(candidates),
        "positive_local_score_count": sum(1 for score in scores if score > 0.0),
        "avg_local_score": _average(scores),
        "max_local_score": max(scores) if scores else 0.0,
        "avg_candidate_token_count": _average(int(candidate.get("token_count") or 0) for candidate in candidates),
        "query_overlap_candidate_count": sum(
            1
            for candidate in candidates
            if float((candidate.get("proposal_local_score_components") or {}).get("max_overlap", 0.0)) > 0.0
        ),
    }


def _source_level_summary(candidates: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "from_selected_sentence_count": sum(
            1 for candidate in candidates if str(candidate.get("source_level") or "") == "selected_sentence"
        ),
        "from_path_touched_sentence_count": sum(
            1 for candidate in candidates if str(candidate.get("source_level") or "") == "path_touched_sentence"
        ),
    }


def _dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for candidate in candidates:
        key = (
            candidate.get("text"),
            candidate.get("strategy"),
            candidate.get("source_node_id"),
            candidate.get("proposal_origin"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _dedupe_variant_texts(variants: list[dict[str, str]]) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    seen: set[str] = set()
    for variant in variants:
        clean = _clean_span(variant["text"])
        if not clean or clean in seen:
            continue
        seen.add(clean)
        deduped.append({"text": clean, "reason": variant["reason"]})
    return deduped


def _raw_tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def _clean_span(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" \t\n\r\"'`.,;:()[]{}")


def _strip_leading_stopwords(tokens: list[str]) -> list[str]:
    index = 0
    lowered = [token.lower() for token in tokens]
    while index < len(tokens) and lowered[index] in STOPWORDS:
        index += 1
    return tokens[index:] or tokens


def _first_connector_index(tokens: list[str]) -> int | None:
    lowered = [token.lower() for token in tokens]
    for index, token in enumerate(lowered):
        if token in CONNECTOR_TOKENS:
            return index
    return None


def _clause_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for raw in CLAUSE_SPLIT_RE.split(text):
        clean = _clean_span(raw)
        if not clean:
            continue
        tokens = _raw_tokens(clean)
        if 2 <= len(tokens) <= 8:
            candidates.append(clean)
    return candidates


def _is_usable_candidate(candidate: dict[str, Any]) -> bool:
    text = str(candidate.get("text") or "")
    tokens = _raw_tokens(text)
    if not text or not tokens:
        return False
    if len(tokens) > 12:
        return False
    normalized = normalize_relevance_tokens(text)
    return bool(normalized or len(tokens) <= 4)


def _average(values: Any) -> float:
    value_list = list(values)
    return float(sum(value_list) / len(value_list)) if value_list else 0.0
