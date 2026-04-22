"""Non-privileged entity/title-like answer mappers for HotpotQA.

Mappers operate only on the selected answer node and its local graph
neighborhood. They do not read gold answers, supporting facts, lexical overlap
against the question, or scan the full graph.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from graph.graph_store import GraphStore


@dataclass(slots=True)
class HotpotQAEntityTitleMapping:
    """Mapped entity/title-like answer from one selected graph answer node."""

    mapper_name: str
    selected_graph_answer: str | None
    mapped_answer: str
    source_node_id: str | None
    source_node_type: str | None
    source_text: str
    fallback_occurred: bool
    fallback_target: str | None
    fallback_reason: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""

        return asdict(self)


EntityTitleMapper = Callable[[str | None, GraphStore, dict[str, Any] | None], HotpotQAEntityTitleMapping]
EntityTitleMapperFactory = Callable[[], EntityTitleMapper]

ENTITY_TITLE_MAPPER_NAMES = (
    "identity",
    "parent_title",
    "capitalized_span",
    "parent_title_or_span",
)

_CAPITALIZED_SKIP = {
    "A",
    "An",
    "He",
    "His",
    "It",
    "No",
    "She",
    "The",
    "They",
    "This",
    "Those",
    "These",
}
_CONNECTOR_TOKENS = {"&", "and", "de", "del", "di", "du", "for", "in", "la", "of", "the", "to", "van", "von"}


def make_identity_mapper() -> EntityTitleMapper:
    """Return the baseline mapper: selected node text unchanged."""

    def mapper(
        selected_graph_answer: str | None,
        graph_store: GraphStore,
        answer_selection: dict[str, Any] | None = None,
    ) -> HotpotQAEntityTitleMapping:
        del answer_selection
        node = _selected_node_payload(selected_graph_answer, graph_store)
        if node["fallback_reason"]:
            return _mapping(
                mapper_name="identity",
                selected_graph_answer=selected_graph_answer,
                node=node,
                mapped_answer="",
                fallback_occurred=True,
                fallback_target="empty",
                fallback_reason=node["fallback_reason"],
                metadata={"strategy": "identity_node_text"},
            )
        return _mapping(
            mapper_name="identity",
            selected_graph_answer=selected_graph_answer,
            node=node,
            mapped_answer=node["text"],
            fallback_occurred=False,
            fallback_target=None,
            fallback_reason=None,
            metadata={"strategy": "identity_node_text"},
        )

    return mapper


def make_parent_title_mapper() -> EntityTitleMapper:
    """Map a selected sentence to its adjacent parent title when available."""

    def mapper(
        selected_graph_answer: str | None,
        graph_store: GraphStore,
        answer_selection: dict[str, Any] | None = None,
    ) -> HotpotQAEntityTitleMapping:
        del answer_selection
        selected_node = _selected_node_payload(selected_graph_answer, graph_store)
        if selected_node["fallback_reason"]:
            return _fallback_to_identity(
                mapper_name="parent_title",
                selected_graph_answer=selected_graph_answer,
                graph_store=graph_store,
                reason=selected_node["fallback_reason"],
            )
        if selected_node["node_type"] != "sentence":
            return _fallback_to_identity(
                mapper_name="parent_title",
                selected_graph_answer=selected_graph_answer,
                graph_store=graph_store,
                reason="selected_node_not_sentence",
            )

        parent_title_id = _parent_title_node_id(selected_graph_answer, graph_store)
        if parent_title_id is None:
            return _fallback_to_identity(
                mapper_name="parent_title",
                selected_graph_answer=selected_graph_answer,
                graph_store=graph_store,
                reason="no_parent_title_found",
            )

        parent_node = _selected_node_payload(parent_title_id, graph_store)
        if parent_node["fallback_reason"] or parent_node["node_type"] != "title":
            return _fallback_to_identity(
                mapper_name="parent_title",
                selected_graph_answer=selected_graph_answer,
                graph_store=graph_store,
                reason="parent_title_node_invalid",
            )

        return _mapping(
            mapper_name="parent_title",
            selected_graph_answer=selected_graph_answer,
            node=parent_node,
            mapped_answer=parent_node["text"],
            fallback_occurred=False,
            fallback_target=None,
            fallback_reason=None,
            metadata={
                "strategy": "adjacent_parent_title",
                "selected_node": _public_node_summary(selected_node),
                "local_neighborhood_only": True,
            },
        )

    return mapper


def make_capitalized_span_mapper() -> EntityTitleMapper:
    """Map a selected sentence to a local title-like capitalized span."""

    def mapper(
        selected_graph_answer: str | None,
        graph_store: GraphStore,
        answer_selection: dict[str, Any] | None = None,
    ) -> HotpotQAEntityTitleMapping:
        del answer_selection
        node = _selected_node_payload(selected_graph_answer, graph_store)
        if node["fallback_reason"]:
            return _fallback_to_identity(
                mapper_name="capitalized_span",
                selected_graph_answer=selected_graph_answer,
                graph_store=graph_store,
                reason=node["fallback_reason"],
            )
        if node["node_type"] == "title":
            return _mapping(
                mapper_name="capitalized_span",
                selected_graph_answer=selected_graph_answer,
                node=node,
                mapped_answer=node["text"],
                fallback_occurred=False,
                fallback_target=None,
                fallback_reason=None,
                metadata={"strategy": "title_node_identity", "local_text_only": True},
            )
        if node["node_type"] != "sentence":
            return _fallback_to_identity(
                mapper_name="capitalized_span",
                selected_graph_answer=selected_graph_answer,
                graph_store=graph_store,
                reason="selected_node_not_sentence_or_title",
            )

        candidates = _title_like_span_candidates(node["text"])
        if not candidates:
            return _fallback_to_identity(
                mapper_name="capitalized_span",
                selected_graph_answer=selected_graph_answer,
                graph_store=graph_store,
                reason="no_capitalized_span_found",
            )
        chosen = candidates[0]
        return _mapping(
            mapper_name="capitalized_span",
            selected_graph_answer=selected_graph_answer,
            node=node,
            mapped_answer=chosen["text"],
            fallback_occurred=False,
            fallback_target=None,
            fallback_reason=None,
            metadata={
                "strategy": chosen["strategy"],
                "candidate_count": len(candidates),
                "candidates": candidates,
                "local_text_only": True,
            },
        )

    return mapper


def make_parent_title_or_span_mapper() -> EntityTitleMapper:
    """Try a local title-like span, then back off to parent title."""

    def mapper(
        selected_graph_answer: str | None,
        graph_store: GraphStore,
        answer_selection: dict[str, Any] | None = None,
    ) -> HotpotQAEntityTitleMapping:
        span_mapping = make_capitalized_span_mapper()(selected_graph_answer, graph_store, answer_selection)
        if not span_mapping.fallback_occurred:
            return HotpotQAEntityTitleMapping(
                mapper_name="parent_title_or_span",
                selected_graph_answer=span_mapping.selected_graph_answer,
                mapped_answer=span_mapping.mapped_answer,
                source_node_id=span_mapping.source_node_id,
                source_node_type=span_mapping.source_node_type,
                source_text=span_mapping.source_text,
                fallback_occurred=False,
                fallback_target=None,
                fallback_reason=None,
                metadata={"strategy": "capitalized_span_first", "base_mapping": span_mapping.to_dict()},
            )

        title_mapping = make_parent_title_mapper()(selected_graph_answer, graph_store, answer_selection)
        if not title_mapping.fallback_occurred:
            return HotpotQAEntityTitleMapping(
                mapper_name="parent_title_or_span",
                selected_graph_answer=title_mapping.selected_graph_answer,
                mapped_answer=title_mapping.mapped_answer,
                source_node_id=title_mapping.source_node_id,
                source_node_type=title_mapping.source_node_type,
                source_text=title_mapping.source_text,
                fallback_occurred=False,
                fallback_target=None,
                fallback_reason=None,
                metadata={
                    "strategy": "parent_title_backoff",
                    "span_mapping": span_mapping.to_dict(),
                    "base_mapping": title_mapping.to_dict(),
                },
            )

        identity_mapping = make_identity_mapper()(selected_graph_answer, graph_store, answer_selection)
        return HotpotQAEntityTitleMapping(
            mapper_name="parent_title_or_span",
            selected_graph_answer=identity_mapping.selected_graph_answer,
            mapped_answer=identity_mapping.mapped_answer,
            source_node_id=identity_mapping.source_node_id,
            source_node_type=identity_mapping.source_node_type,
            source_text=identity_mapping.source_text,
            fallback_occurred=True,
            fallback_target="identity",
            fallback_reason="capitalized_span_and_parent_title_unavailable",
            metadata={
                "strategy": "identity_fallback",
                "span_mapping": span_mapping.to_dict(),
                "parent_title_mapping": title_mapping.to_dict(),
                "base_mapping": identity_mapping.to_dict(),
            },
        )

    return mapper


def make_entity_title_mapper_factory(mapper_name: str) -> EntityTitleMapperFactory:
    """Return a fresh mapper factory for the requested mapper name."""

    if mapper_name == "identity":
        return make_identity_mapper
    if mapper_name == "parent_title":
        return make_parent_title_mapper
    if mapper_name == "capitalized_span":
        return make_capitalized_span_mapper
    if mapper_name == "parent_title_or_span":
        return make_parent_title_or_span_mapper
    raise ValueError(
        f"Unknown HotpotQA entity/title mapper '{mapper_name}'. "
        f"Available mappers: {', '.join(ENTITY_TITLE_MAPPER_NAMES)}"
    )


def _fallback_to_identity(
    *,
    mapper_name: str,
    selected_graph_answer: str | None,
    graph_store: GraphStore,
    reason: str,
) -> HotpotQAEntityTitleMapping:
    identity_mapping = make_identity_mapper()(selected_graph_answer, graph_store, None)
    return HotpotQAEntityTitleMapping(
        mapper_name=mapper_name,
        selected_graph_answer=identity_mapping.selected_graph_answer,
        mapped_answer=identity_mapping.mapped_answer,
        source_node_id=identity_mapping.source_node_id,
        source_node_type=identity_mapping.source_node_type,
        source_text=identity_mapping.source_text,
        fallback_occurred=True,
        fallback_target="identity",
        fallback_reason=reason,
        metadata={"strategy": "identity_fallback", "base_mapping": identity_mapping.to_dict()},
    )


def _mapping(
    *,
    mapper_name: str,
    selected_graph_answer: str | None,
    node: dict[str, Any],
    mapped_answer: str,
    fallback_occurred: bool,
    fallback_target: str | None,
    fallback_reason: str | None,
    metadata: dict[str, Any],
) -> HotpotQAEntityTitleMapping:
    return HotpotQAEntityTitleMapping(
        mapper_name=mapper_name,
        selected_graph_answer=selected_graph_answer,
        mapped_answer=mapped_answer,
        source_node_id=node.get("node_id"),
        source_node_type=node.get("node_type"),
        source_text=str(node.get("text") or ""),
        fallback_occurred=fallback_occurred,
        fallback_target=fallback_target,
        fallback_reason=fallback_reason,
        metadata=metadata,
    )


def _parent_title_node_id(selected_graph_answer: str | None, graph_store: GraphStore) -> str | None:
    if selected_graph_answer is None:
        return None

    incoming = sorted(graph_store.get_incoming_edges(selected_graph_answer), key=lambda edge: edge.edge_id)
    for edge in incoming:
        if edge.relation == "title_to_sentence" and _node_type(edge.src, graph_store) == "title":
            return edge.src

    outgoing = sorted(graph_store.get_outgoing_edges(selected_graph_answer), key=lambda edge: edge.edge_id)
    for edge in outgoing:
        if edge.relation == "sentence_to_title" and _node_type(edge.dst, graph_store) == "title":
            return edge.dst
    return None


def _selected_node_payload(selected_graph_answer: str | None, graph_store: GraphStore) -> dict[str, Any]:
    if not selected_graph_answer or selected_graph_answer == "UNKNOWN":
        return {
            "node_id": selected_graph_answer,
            "node_type": None,
            "text": "",
            "metadata": {},
            "fallback_reason": "empty_or_unknown_answer",
        }
    attrs = graph_store.get_node_attributes(selected_graph_answer)
    if not attrs:
        return {
            "node_id": selected_graph_answer,
            "node_type": None,
            "text": "",
            "metadata": {},
            "fallback_reason": "unknown_node",
        }
    node_type = str(attrs.get("node_type", ""))
    if node_type == "question":
        return {
            "node_id": selected_graph_answer,
            "node_type": node_type,
            "text": "",
            "metadata": dict(attrs.get("metadata") or {}),
            "fallback_reason": "question_node",
        }
    return {
        "node_id": selected_graph_answer,
        "node_type": node_type or None,
        "text": str(attrs.get("text") or attrs.get("name") or ""),
        "metadata": dict(attrs.get("metadata") or {}),
        "fallback_reason": None,
    }


def _node_type(node_id: str, graph_store: GraphStore) -> str:
    attrs = graph_store.get_node_attributes(node_id)
    return str(attrs.get("node_type") or "")


def _public_node_summary(node: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_id": node.get("node_id"),
        "node_type": node.get("node_type"),
        "text": node.get("text"),
        "metadata": dict(node.get("metadata") or {}),
    }


def _title_like_span_candidates(text: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for span in re.findall(r'"([^"]{1,100})"|“([^”]{1,100})”|\'([^\']{1,100})\'', text):
        cleaned = _clean_span(next(item for item in span if item))
        if cleaned:
            candidates.append({"text": cleaned, "strategy": "quoted_title_like_span"})

    # Capitalized phrases may include short title connectors such as "of" or "the".
    pattern = r"\b[A-Z][A-Za-z0-9'.]*(?:(?:\s+(?:[A-Z][A-Za-z0-9'.]*|&|and|de|del|di|du|for|in|la|of|the|to|van|von))){0,7}\b"
    for match in re.finditer(pattern, text):
        cleaned = _clean_span(match.group(0))
        if not cleaned:
            continue
        tokens = cleaned.split()
        if not tokens or len(tokens) > 8:
            continue
        if len(tokens) == 1 and tokens[0] in _CAPITALIZED_SKIP:
            continue
        if all(token.lower() in _CONNECTOR_TOKENS for token in tokens):
            continue
        candidates.append({"text": cleaned, "strategy": "capitalized_title_like_span"})

    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate["text"].lower()
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return sorted(unique, key=lambda item: (_strategy_priority(item["strategy"]), len(item["text"].split()), len(item["text"]), item["text"]))


def _strategy_priority(strategy: str) -> int:
    order = {
        "quoted_title_like_span": 0,
        "capitalized_title_like_span": 1,
    }
    return order.get(strategy, 99)


def _clean_span(text: str) -> str:
    cleaned = text.strip(" \t\n\r\"'“”[]{} .?!,;:")
    if not re.search(r"[A-Za-z0-9]", cleaned):
        return ""
    return cleaned
