"""Non-privileged relation/span mappers for HotpotQA.

Relation/span mappers are only used after answer-type routing has produced the
``descriptive_span_or_relation`` label. They do not read gold answers,
supporting facts, benchmark metadata, or scan the full graph. The pattern list
is intentionally fixed for this batch to avoid result-driven rule expansion.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from answer.hotpotqa_relation_span_discovery import (
    HotpotQARelationSpanDiscoveryResult,
    RelationSpanDiscovery,
)
from answer.hotpotqa_relation_span_ranker import (
    HotpotQARelationSpanRankingResult,
    RelationSpanRanker,
)
from answer.hotpotqa_relation_span_proposal import (
    HotpotQARelationSpanProposalResult,
    RelationSpanProposal,
)
from graph.graph_store import GraphStore


@dataclass(slots=True)
class HotpotQARelationSpanMapping:
    """A relation/span mapping decision with explicit candidate span metadata."""

    relation_span_mapper_name: str
    selected_graph_answer: str | None
    mapped_answer: str
    decision_reason: str
    source_node_ids: list[str]
    source_node_type: str | None
    source_text: str
    candidate_spans: list[dict[str, Any]]
    selected_span: str
    selected_span_reason: str
    fallback_occurred: bool
    fallback_target: str | None
    fallback_reason: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""

        return asdict(self)


RelationSpanMapper = Callable[[str | None, GraphStore, dict[str, Any] | None], HotpotQARelationSpanMapping]
RelationSpanMapperFactory = Callable[[], RelationSpanMapper]

RELATION_SPAN_MAPPER_NAMES = (
    "identity_relation_span",
    "clause_relation",
    "pattern_span",
    "clause_then_pattern_backoff",
)

# Fixed batch-28 pattern set. Do not tune from run outputs.
PATTERN_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("served_as", re.compile(r"\bserved as\s+(?:an?\s+|the\s+)?([^.;,:()\-\u2013\u2014]{2,90})", re.I)),
    ("was_the", re.compile(r"\b(?:was|is)\s+(?:an?\s+|the\s+)([^.;,:()\-\u2013\u2014]{2,90})", re.I)),
    ("born_in", re.compile(r"\bborn in\s+([^.;,:()\-\u2013\u2014]{2,70})", re.I)),
    ("part_of", re.compile(r"\bpart of\s+([^.;,:()\-\u2013\u2014]{2,90})", re.I)),
    ("known_for", re.compile(r"\bknown for\s+([^.;,:()\-\u2013\u2014]{2,90})", re.I)),
    ("located_in", re.compile(r"\blocated in\s+([^.;,:()\-\u2013\u2014]{2,70})", re.I)),
    ("member_of", re.compile(r"\bmember of\s+([^.;,:()\-\u2013\u2014]{2,90})", re.I)),
    ("based_in", re.compile(r"\bbased in\s+([^.;,:()\-\u2013\u2014]{2,70})", re.I)),
)

CLAUSE_SPLIT_RE = re.compile(r"[,;:\-\u2013\u2014()]")
TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)


def make_identity_relation_span_mapper() -> RelationSpanMapper:
    """Return baseline relation/span mapper: preserve existing base mapping."""

    def mapper(
        selected_graph_answer: str | None,
        graph_store: GraphStore,
        context: dict[str, Any] | None = None,
    ) -> HotpotQARelationSpanMapping:
        del graph_store
        payload = _context_payload(context)
        base_mapping = payload["base_mapping"]
        selected_node = _selected_node_from_context(selected_graph_answer, payload)
        mapped_answer = str(base_mapping.get("mapped_answer") or "")
        return _mapping(
            mapper_name="identity_relation_span",
            selected_graph_answer=selected_graph_answer,
            mapped_answer=mapped_answer,
            decision_reason="identity_preserves_base_mapping",
            source_nodes=[selected_node],
            candidate_spans=[],
            selected_span=mapped_answer,
            selected_span_reason="base_mapping",
            fallback_occurred=True,
            fallback_target="base_mapping",
            fallback_reason="identity_relation_span_mapper_does_not_extract",
            metadata={"strategy": "identity_base_mapping", "base_mapping": base_mapping},
        )

    return mapper


def make_clause_relation_mapper() -> RelationSpanMapper:
    """Return mapper that selects a deterministic short relation-like clause."""

    def mapper(
        selected_graph_answer: str | None,
        graph_store: GraphStore,
        context: dict[str, Any] | None = None,
    ) -> HotpotQARelationSpanMapping:
        payload = _context_payload(context)
        selected_node = _selected_node_payload(selected_graph_answer, graph_store, payload)
        source_nodes = _local_sentence_nodes(payload, selected_node=selected_node, graph_store=graph_store)
        candidates = _clause_candidates(source_nodes)
        if not candidates:
            return _fallback_mapping(
                mapper_name="clause_relation",
                selected_graph_answer=selected_graph_answer,
                source_nodes=source_nodes or [selected_node],
                base_mapping=payload["base_mapping"],
                candidate_spans=[],
                fallback_reason="no_clause_candidates",
            )
        chosen, ranking = _select_span_with_optional_ranker(candidates, payload)
        return _mapping(
            mapper_name="clause_relation",
            selected_graph_answer=selected_graph_answer,
            mapped_answer=chosen["text"],
            decision_reason="selected_shortest_relation_like_clause",
            source_nodes=_nodes_by_ids(source_nodes, [chosen["source_node_id"]]),
            candidate_spans=candidates,
            selected_span=chosen["text"],
            selected_span_reason=_selected_span_reason(ranking, default="shortest_non_empty_clause"),
            fallback_occurred=False,
            fallback_target=None,
            fallback_reason=None,
            metadata={
                "strategy": "clause_split",
                "candidate_count": len(candidates),
                "relation_span_ranking": ranking.to_dict() if ranking is not None else None,
            },
        )

    return mapper


def make_pattern_span_mapper() -> RelationSpanMapper:
    """Return mapper that extracts fixed-pattern local relation/span phrases."""

    def mapper(
        selected_graph_answer: str | None,
        graph_store: GraphStore,
        context: dict[str, Any] | None = None,
    ) -> HotpotQARelationSpanMapping:
        payload = _context_payload(context)
        selected_node = _selected_node_payload(selected_graph_answer, graph_store, payload)
        source_nodes = _local_sentence_nodes(payload, selected_node=selected_node, graph_store=graph_store)
        candidates, discovery = _discover_pattern_candidates(source_nodes, selected_node, payload)
        candidates, proposal = _apply_optional_proposal(
            candidates=candidates,
            source_nodes=source_nodes,
            selected_node=selected_node,
            payload=payload,
            discovery=discovery,
        )
        if not candidates:
            return _fallback_mapping(
                mapper_name="pattern_span",
                selected_graph_answer=selected_graph_answer,
                source_nodes=source_nodes or [selected_node],
                base_mapping=payload["base_mapping"],
                candidate_spans=[],
                fallback_reason="no_fixed_pattern_span_found",
                discovery=discovery,
                proposal=proposal,
            )
        chosen, ranking = _select_span_with_optional_ranker(candidates, payload)
        return _mapping(
            mapper_name="pattern_span",
            selected_graph_answer=selected_graph_answer,
            mapped_answer=chosen["text"],
            decision_reason="selected_fixed_pattern_span",
            source_nodes=_nodes_by_ids(source_nodes, [chosen["source_node_id"]]),
            candidate_spans=candidates,
            selected_span=chosen["text"],
            selected_span_reason=_selected_span_reason(ranking, default=chosen["strategy"]),
            fallback_occurred=False,
            fallback_target=None,
            fallback_reason=None,
            metadata={
                "strategy": "fixed_pattern_span",
                "candidate_count": len(candidates),
                "fixed_pattern_names": [name for name, _ in PATTERN_RULES],
                "relation_span_discovery": discovery.to_dict() if discovery is not None else None,
                "relation_span_proposal": proposal.to_dict() if proposal is not None else None,
                "relation_span_ranking": ranking.to_dict() if ranking is not None else None,
            },
        )

    return mapper


def make_clause_then_pattern_backoff_mapper() -> RelationSpanMapper:
    """Try fixed patterns first, then clause-level extraction."""

    def mapper(
        selected_graph_answer: str | None,
        graph_store: GraphStore,
        context: dict[str, Any] | None = None,
    ) -> HotpotQARelationSpanMapping:
        pattern = make_pattern_span_mapper()(selected_graph_answer, graph_store, context)
        if not pattern.fallback_occurred:
            return _copy_mapping(pattern, mapper_name="clause_then_pattern_backoff", strategy="pattern_first")
        clause = make_clause_relation_mapper()(selected_graph_answer, graph_store, context)
        if not clause.fallback_occurred:
            return _copy_mapping(
                clause,
                mapper_name="clause_then_pattern_backoff",
                strategy="clause_backoff",
                extra={"pattern_mapping": pattern.to_dict()},
            )
        return HotpotQARelationSpanMapping(
            relation_span_mapper_name="clause_then_pattern_backoff",
            selected_graph_answer=clause.selected_graph_answer,
            mapped_answer=clause.mapped_answer,
            decision_reason="pattern_and_clause_unavailable",
            source_node_ids=clause.source_node_ids,
            source_node_type=clause.source_node_type,
            source_text=clause.source_text,
            candidate_spans=pattern.candidate_spans + clause.candidate_spans,
            selected_span=clause.selected_span,
            selected_span_reason=clause.selected_span_reason,
            fallback_occurred=True,
            fallback_target="base_mapping",
            fallback_reason="pattern_and_clause_unavailable",
            metadata={
                "strategy": "base_mapping_backoff",
                "pattern_mapping": pattern.to_dict(),
                "clause_mapping": clause.to_dict(),
            },
        )

    return mapper


def make_relation_span_mapper_factory(mapper_name: str) -> RelationSpanMapperFactory:
    """Return a fresh relation/span mapper factory."""

    if mapper_name == "identity_relation_span":
        return make_identity_relation_span_mapper
    if mapper_name == "clause_relation":
        return make_clause_relation_mapper
    if mapper_name == "pattern_span":
        return make_pattern_span_mapper
    if mapper_name == "clause_then_pattern_backoff":
        return make_clause_then_pattern_backoff_mapper
    raise ValueError(
        f"Unknown HotpotQA relation/span mapper '{mapper_name}'. "
        f"Available mappers: {', '.join(RELATION_SPAN_MAPPER_NAMES)}"
    )


def _context_payload(context: dict[str, Any] | None) -> dict[str, Any]:
    payload = context if isinstance(context, dict) else {}
    answer_type_label = str(payload.get("answer_type_label") or "")
    if answer_type_label != "descriptive_span_or_relation":
        raise ValueError("HotpotQA relation/span mapper requires answer_type_label='descriptive_span_or_relation'.")
    return {
        "query_text": str(payload.get("query_text") or ""),
        "answer_type_label": answer_type_label,
        "answer_selection": payload.get("answer_selection") if isinstance(payload.get("answer_selection"), dict) else {},
        "base_mapping": payload.get("base_mapping") if isinstance(payload.get("base_mapping"), dict) else {},
        "relation_span_discovery": (
            payload.get("relation_span_discovery") if callable(payload.get("relation_span_discovery")) else None
        ),
        "relation_span_discovery_name": str(payload.get("relation_span_discovery_name") or ""),
        "relation_span_ranker": payload.get("relation_span_ranker") if callable(payload.get("relation_span_ranker")) else None,
        "relation_span_ranker_name": str(payload.get("relation_span_ranker_name") or ""),
        "relation_span_proposal": (
            payload.get("relation_span_proposal") if callable(payload.get("relation_span_proposal")) else None
        ),
        "relation_span_proposal_name": str(payload.get("relation_span_proposal_name") or ""),
    }


def _selected_node_from_context(selected_graph_answer: str | None, payload: dict[str, Any]) -> dict[str, Any]:
    answer_selection = payload["answer_selection"]
    for node in answer_selection.get("candidate_nodes", []) or []:
        if isinstance(node, dict) and node.get("node_id") == selected_graph_answer:
            return dict(node)
    return {"node_id": selected_graph_answer, "node_type": None, "text": ""}


def _selected_node_payload(
    selected_graph_answer: str | None,
    graph_store: GraphStore,
    payload: dict[str, Any],
) -> dict[str, Any]:
    context_node = _selected_node_from_context(selected_graph_answer, payload)
    if selected_graph_answer is None:
        return context_node
    attrs = graph_store.get_node_attributes(selected_graph_answer)
    if not attrs:
        return context_node
    node = {
        "node_id": selected_graph_answer,
        "node_type": str(attrs.get("node_type") or context_node.get("node_type") or ""),
        "text": str(attrs.get("text") or attrs.get("name") or context_node.get("text") or ""),
        "source": context_node.get("source"),
        "step_idx": context_node.get("step_idx"),
        "source_level": "selected_sentence",
    }
    return _with_title_context(node, graph_store)


def _local_sentence_nodes(
    payload: dict[str, Any],
    *,
    selected_node: dict[str, Any],
    graph_store: GraphStore,
) -> list[dict[str, Any]]:
    answer_selection = payload["answer_selection"]
    nodes: list[dict[str, Any]] = []
    if selected_node.get("node_id") and selected_node.get("node_type") == "sentence":
        selected = dict(selected_node)
        selected["source_level"] = "selected_sentence"
        nodes.append(_with_title_context(selected, graph_store))
    for node in answer_selection.get("candidate_nodes", []) or []:
        if not isinstance(node, dict):
            continue
        if node.get("source") != "path":
            continue
        if node.get("node_type") == "sentence":
            nodes.append(_with_title_context(dict(node), graph_store))
    return _dedupe_nodes(nodes)


def _clause_candidates(source_nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for node_index, node in enumerate(source_nodes):
        text = str(node.get("text") or "")
        for span_index, raw_span in enumerate(CLAUSE_SPLIT_RE.split(text)):
            span = _clean_span(raw_span)
            if not _is_usable_span(span):
                continue
            candidates.append(
                _span_payload(
                    text=span,
                    strategy="clause_split",
                    source_node=node,
                    node_index=node_index,
                    span_index=span_index,
                )
            )
    return candidates


def _pattern_candidates(source_nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for node_index, node in enumerate(source_nodes):
        text = str(node.get("text") or "")
        for rule_index, (rule_name, pattern) in enumerate(PATTERN_RULES):
            for match_index, match in enumerate(pattern.finditer(text)):
                span = _clean_span(match.group(1))
                if not _is_usable_span(span):
                    continue
                candidates.append(
                    _span_payload(
                        text=span,
                        strategy=rule_name,
                        source_node=node,
                        node_index=node_index,
                        span_index=(rule_index * 1000) + match_index,
                    )
                )
    return candidates


def _discover_pattern_candidates(
    source_nodes: list[dict[str, Any]],
    selected_node: dict[str, Any],
    payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], HotpotQARelationSpanDiscoveryResult | None]:
    discovery = payload.get("relation_span_discovery")
    if discovery is None:
        return _pattern_candidates(source_nodes), None
    discovery_result = discovery(
        {
            "query_text": payload.get("query_text", ""),
            "selected_node": selected_node,
            "source_nodes": source_nodes,
            "relation_span_discovery_name": payload.get("relation_span_discovery_name"),
            "whitelist_fields": [
                "query_text",
                "selected_sentence_text",
                "path_touched_local_sentence_text",
                "title_text",
                "parent_title_text",
                "local_structure_statistics",
            ],
        }
    )
    return list(discovery_result.candidate_spans), discovery_result


def _apply_optional_proposal(
    *,
    candidates: list[dict[str, Any]],
    source_nodes: list[dict[str, Any]],
    selected_node: dict[str, Any],
    payload: dict[str, Any],
    discovery: HotpotQARelationSpanDiscoveryResult | None,
) -> tuple[list[dict[str, Any]], HotpotQARelationSpanProposalResult | None]:
    proposal = payload.get("relation_span_proposal")
    if proposal is None:
        return candidates, None
    proposal_result = proposal(
        {
            "query_text": payload.get("query_text", ""),
            "selected_node": selected_node,
            "source_nodes": source_nodes,
            "base_candidate_spans": candidates,
            "relation_span_discovery_name": payload.get("relation_span_discovery_name"),
            "relation_span_discovery_result": discovery.to_dict() if discovery is not None else None,
            "whitelist_fields": [
                "query_text",
                "selected_sentence_text",
                "path_touched_local_sentence_text",
                "title_text",
                "parent_title_text",
                "local_structure_statistics",
                "discovery_candidate_spans",
            ],
        }
    )
    return list(proposal_result.candidate_spans), proposal_result


def _span_payload(
    *,
    text: str,
    strategy: str,
    source_node: dict[str, Any],
    node_index: int,
    span_index: int,
) -> dict[str, Any]:
    tokens = TOKEN_RE.findall(text)
    return {
        "text": text,
        "strategy": strategy,
        "source_node_id": source_node.get("node_id"),
        "source_node_type": source_node.get("node_type"),
        "source_text": source_node.get("text"),
        "source_level": source_node.get("source_level"),
        "title_text": source_node.get("title_text") or source_node.get("parent_title_text") or "",
        "token_count": len(tokens),
        "char_count": len(text),
        "node_index": node_index,
        "span_index": span_index,
    }


def _rank_spans(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        candidates,
        key=lambda item: (
            int(item.get("token_count", 0)),
            int(item.get("char_count", 0)),
            int(item.get("node_index", 0)),
            int(item.get("span_index", 0)),
            str(item.get("text", "")),
        ),
    )


def _select_span_with_optional_ranker(
    candidates: list[dict[str, Any]],
    payload: dict[str, Any],
) -> tuple[dict[str, Any], HotpotQARelationSpanRankingResult | None]:
    ranker = payload.get("relation_span_ranker")
    if ranker is None:
        return _rank_spans(candidates)[0], None
    ranking = ranker(
        candidates,
        {
            "relation_span_ranker_name": payload.get("relation_span_ranker_name"),
            "batch28_default_ranked_spans": _rank_spans(candidates),
            "whitelist_fields": [
                "candidate_span.text",
                "candidate_span.strategy",
                "candidate_span.source_node_id",
                "candidate_span.token_count",
                "candidate_span.char_count",
                "candidate_span.node_index",
                "candidate_span.span_index",
            ],
        },
    )
    if ranking.selected_span_payload is None:
        return _rank_spans(candidates)[0], ranking
    return dict(ranking.selected_span_payload), ranking


def _selected_span_reason(
    ranking: HotpotQARelationSpanRankingResult | None,
    *,
    default: str,
) -> str:
    return ranking.selected_span_reason if ranking is not None else default


def _fallback_mapping(
    *,
    mapper_name: str,
    selected_graph_answer: str | None,
    source_nodes: list[dict[str, Any]],
    base_mapping: dict[str, Any],
    candidate_spans: list[dict[str, Any]],
    fallback_reason: str,
    discovery: HotpotQARelationSpanDiscoveryResult | None = None,
    proposal: HotpotQARelationSpanProposalResult | None = None,
) -> HotpotQARelationSpanMapping:
    mapped_answer = str(base_mapping.get("mapped_answer") or "")
    return _mapping(
        mapper_name=mapper_name,
        selected_graph_answer=selected_graph_answer,
        mapped_answer=mapped_answer,
        decision_reason=fallback_reason,
        source_nodes=source_nodes,
        candidate_spans=candidate_spans,
        selected_span=mapped_answer,
        selected_span_reason="base_mapping",
        fallback_occurred=True,
        fallback_target="base_mapping",
        fallback_reason=fallback_reason,
        metadata={
            "strategy": "base_mapping_fallback",
            "base_mapping": base_mapping,
            "relation_span_discovery": discovery.to_dict() if discovery is not None else None,
            "relation_span_proposal": proposal.to_dict() if proposal is not None else None,
        },
    )


def _mapping(
    *,
    mapper_name: str,
    selected_graph_answer: str | None,
    mapped_answer: str,
    decision_reason: str,
    source_nodes: list[dict[str, Any]],
    candidate_spans: list[dict[str, Any]],
    selected_span: str,
    selected_span_reason: str,
    fallback_occurred: bool,
    fallback_target: str | None,
    fallback_reason: str | None,
    metadata: dict[str, Any],
) -> HotpotQARelationSpanMapping:
    public_nodes = [_public_node(node) for node in source_nodes]
    source_node_ids = [str(node.get("node_id")) for node in source_nodes if node.get("node_id")]
    first_node = source_nodes[0] if source_nodes else {}
    return HotpotQARelationSpanMapping(
        relation_span_mapper_name=mapper_name,
        selected_graph_answer=selected_graph_answer,
        mapped_answer=mapped_answer,
        decision_reason=decision_reason,
        source_node_ids=source_node_ids,
        source_node_type=first_node.get("node_type"),
        source_text=str(first_node.get("text") or ""),
        candidate_spans=candidate_spans,
        selected_span=selected_span,
        selected_span_reason=selected_span_reason,
        fallback_occurred=fallback_occurred,
        fallback_target=fallback_target,
        fallback_reason=fallback_reason,
        metadata={
            **metadata,
            "source_nodes": public_nodes,
            "candidate_spans": candidate_spans,
            "selected_span": selected_span,
            "selected_span_reason": selected_span_reason,
            "fixed_pattern_names": [name for name, _ in PATTERN_RULES],
            "whitelist_fields": [
                "query_text",
                "selected_node.title_text",
                "selected_node.sentence_text",
                "path_touched_local_nodes",
                "local_structure_statistics",
            ],
        },
    )


def _copy_mapping(
    base: HotpotQARelationSpanMapping,
    *,
    mapper_name: str,
    strategy: str,
    extra: dict[str, Any] | None = None,
) -> HotpotQARelationSpanMapping:
    return HotpotQARelationSpanMapping(
        relation_span_mapper_name=mapper_name,
        selected_graph_answer=base.selected_graph_answer,
        mapped_answer=base.mapped_answer,
        decision_reason=base.decision_reason,
        source_node_ids=list(base.source_node_ids),
        source_node_type=base.source_node_type,
        source_text=base.source_text,
        candidate_spans=list(base.candidate_spans),
        selected_span=base.selected_span,
        selected_span_reason=base.selected_span_reason,
        fallback_occurred=base.fallback_occurred,
        fallback_target=base.fallback_target,
        fallback_reason=base.fallback_reason,
        metadata={"strategy": strategy, "base_mapping": base.to_dict(), **(extra or {})},
    )


def _nodes_by_ids(nodes: list[dict[str, Any]], node_ids: list[str]) -> list[dict[str, Any]]:
    selected = [node for node in nodes if node.get("node_id") in node_ids]
    return selected or nodes[:1]


def _clean_span(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" \t\n\r\"'`.,;:()[]{}")


def _is_usable_span(text: str) -> bool:
    tokens = TOKEN_RE.findall(text)
    return 1 <= len(tokens) <= 12 and len(text) >= 2


def _dedupe_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for node in nodes:
        node_id = str(node.get("node_id") or "")
        if not node_id or node_id in seen:
            continue
        seen.add(node_id)
        deduped.append(node)
    return deduped


def _public_node(node: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_id": node.get("node_id"),
        "node_type": node.get("node_type"),
        "text": node.get("text"),
        "source": node.get("source"),
        "step_idx": node.get("step_idx"),
        "parent_title_text": node.get("parent_title_text"),
        "title_text": node.get("title_text"),
        "source_level": node.get("source_level"),
    }


def _with_title_context(node: dict[str, Any], graph_store: GraphStore) -> dict[str, Any]:
    payload = dict(node)
    node_type = str(payload.get("node_type") or "")
    node_id = str(payload.get("node_id") or "")
    if node_type == "title":
        payload["title_text"] = str(payload.get("text") or "")
        payload["parent_title_node_id"] = node_id
        payload["parent_title_text"] = str(payload.get("text") or "")
        payload.setdefault("source_level", "selected_sentence" if payload.get("source") == "selected" else "path_touched_sentence")
        return payload
    if node_type != "sentence" or not node_id:
        return payload
    parent_id = _parent_title_node_id(node_id, graph_store)
    payload["parent_title_node_id"] = parent_id
    payload["parent_title_text"] = _node_text(parent_id, graph_store) if parent_id is not None else ""
    payload["title_text"] = payload["parent_title_text"]
    payload.setdefault("source_level", "path_touched_sentence")
    return payload


def _parent_title_node_id(node_id: str, graph_store: GraphStore) -> str | None:
    incoming = sorted(graph_store.get_incoming_edges(node_id), key=lambda edge: edge.edge_id)
    for edge in incoming:
        if edge.relation == "title_to_sentence" and _node_type(edge.src, graph_store) == "title":
            return edge.src
    outgoing = sorted(graph_store.get_outgoing_edges(node_id), key=lambda edge: edge.edge_id)
    for edge in outgoing:
        if edge.relation == "sentence_to_title" and _node_type(edge.dst, graph_store) == "title":
            return edge.dst
    return None


def _node_type(node_id: str, graph_store: GraphStore) -> str:
    attrs = graph_store.get_node_attributes(node_id)
    return str(attrs.get("node_type") or "")


def _node_text(node_id: str | None, graph_store: GraphStore) -> str:
    if node_id is None:
        return ""
    attrs = graph_store.get_node_attributes(node_id)
    return str(attrs.get("text") or attrs.get("name") or "")
