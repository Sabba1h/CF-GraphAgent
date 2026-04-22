"""Non-privileged HotpotQA answer extractors.

Extractors operate only on the currently selected graph answer node text. They
do not read gold answers, supporting facts, or scan other graph nodes.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from graph.graph_store import GraphStore


@dataclass(slots=True)
class HotpotQAAnswerExtraction:
    """Extracted text answer from one selected graph answer node."""

    extractor_name: str
    source_node_id: str | None
    source_node_type: str | None
    source_text: str
    extracted_answer: str
    fallback_occurred: bool
    fallback_target: str | None
    fallback_reason: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""

        return asdict(self)


AnswerExtractor = Callable[[str | None, GraphStore, dict[str, Any] | None], HotpotQAAnswerExtraction]
AnswerExtractorFactory = Callable[[], AnswerExtractor]

ANSWER_EXTRACTOR_NAMES = (
    "full_sentence",
    "clause_trim",
    "answer_like_span",
    "sentence_title_backoff",
)
_CAPITALIZED_PHRASE_SKIP = {"A", "An", "He", "It", "She", "The", "They", "This", "Those", "These"}


def make_full_sentence_extractor() -> AnswerExtractor:
    """Return baseline extractor: title text or full sentence text."""

    def extractor(
        selected_graph_answer: str | None,
        graph_store: GraphStore,
        answer_selection: dict[str, Any] | None = None,
    ) -> HotpotQAAnswerExtraction:
        del answer_selection
        node = _selected_node_payload(selected_graph_answer, graph_store)
        if node["fallback_reason"]:
            return _fallback_extraction(
                extractor_name="full_sentence",
                node=node,
                target="empty",
                reason=node["fallback_reason"],
                extracted_answer="",
            )
        return HotpotQAAnswerExtraction(
            extractor_name="full_sentence",
            source_node_id=selected_graph_answer,
            source_node_type=node["node_type"],
            source_text=node["text"],
            extracted_answer=node["text"],
            fallback_occurred=False,
            fallback_target=None,
            fallback_reason=None,
            metadata={"strategy": "identity_node_text"},
        )

    return extractor


def make_clause_trim_extractor() -> AnswerExtractor:
    """Return extractor that chooses a deterministic short clause from one node."""

    def extractor(
        selected_graph_answer: str | None,
        graph_store: GraphStore,
        answer_selection: dict[str, Any] | None = None,
    ) -> HotpotQAAnswerExtraction:
        del answer_selection
        node = _selected_node_payload(selected_graph_answer, graph_store)
        if node["fallback_reason"]:
            return _fallback_extraction(
                extractor_name="clause_trim",
                node=node,
                target="empty",
                reason=node["fallback_reason"],
                extracted_answer="",
            )
        if node["node_type"] == "title":
            return _fallback_extraction(
                extractor_name="clause_trim",
                node=node,
                target="title",
                reason="title_node_has_no_sentence_clause",
                extracted_answer=node["text"],
            )

        clauses = _split_clauses(node["text"])
        if not clauses:
            return _fallback_extraction(
                extractor_name="clause_trim",
                node=node,
                target="full_sentence",
                reason="trim_after_split_empty",
                extracted_answer=node["text"],
            )
        chosen = _rank_short_texts(clauses)[0]
        if not chosen or chosen == node["text"]:
            return _fallback_extraction(
                extractor_name="clause_trim",
                node=node,
                target="full_sentence",
                reason="no_shorter_clause_found",
                extracted_answer=node["text"],
            )
        return HotpotQAAnswerExtraction(
            extractor_name="clause_trim",
            source_node_id=selected_graph_answer,
            source_node_type=node["node_type"],
            source_text=node["text"],
            extracted_answer=chosen,
            fallback_occurred=False,
            fallback_target=None,
            fallback_reason=None,
            metadata={"strategy": "shortest_clause", "candidate_count": len(clauses), "clauses": clauses},
        )

    return extractor


def make_answer_like_span_extractor() -> AnswerExtractor:
    """Return local string-rule extractor for answer-like spans."""

    def extractor(
        selected_graph_answer: str | None,
        graph_store: GraphStore,
        answer_selection: dict[str, Any] | None = None,
    ) -> HotpotQAAnswerExtraction:
        del answer_selection
        node = _selected_node_payload(selected_graph_answer, graph_store)
        if node["fallback_reason"]:
            return _fallback_extraction(
                extractor_name="answer_like_span",
                node=node,
                target="empty",
                reason=node["fallback_reason"],
                extracted_answer="",
            )
        if node["node_type"] == "title":
            return _fallback_extraction(
                extractor_name="answer_like_span",
                node=node,
                target="title",
                reason="title_node_already_answer_like",
                extracted_answer=node["text"],
            )

        candidates = _answer_like_candidates(node["text"])
        if not candidates:
            fallback = make_clause_trim_extractor()(selected_graph_answer, graph_store, None)
            return _fallback_extraction(
                extractor_name="answer_like_span",
                node=node,
                target="full_sentence" if fallback.fallback_occurred else "clause_trim",
                reason="no_answer_like_span_found",
                extracted_answer=fallback.extracted_answer,
                extra_metadata={"clause_trim_fallback": fallback.to_dict()},
            )
        chosen = candidates[0]
        return HotpotQAAnswerExtraction(
            extractor_name="answer_like_span",
            source_node_id=selected_graph_answer,
            source_node_type=node["node_type"],
            source_text=node["text"],
            extracted_answer=chosen["text"],
            fallback_occurred=False,
            fallback_target=None,
            fallback_reason=None,
            metadata={
                "strategy": chosen["strategy"],
                "candidate_count": len(candidates),
                "candidates": candidates,
            },
        )

    return extractor


def make_sentence_title_backoff_extractor() -> AnswerExtractor:
    """Return clause trimmer with metadata title backoff for empty trims."""

    def extractor(
        selected_graph_answer: str | None,
        graph_store: GraphStore,
        answer_selection: dict[str, Any] | None = None,
    ) -> HotpotQAAnswerExtraction:
        base = make_clause_trim_extractor()(selected_graph_answer, graph_store, answer_selection)
        if not base.fallback_occurred or base.extracted_answer:
            return HotpotQAAnswerExtraction(
                extractor_name="sentence_title_backoff",
                source_node_id=base.source_node_id,
                source_node_type=base.source_node_type,
                source_text=base.source_text,
                extracted_answer=base.extracted_answer,
                fallback_occurred=base.fallback_occurred,
                fallback_target=base.fallback_target,
                fallback_reason=base.fallback_reason,
                metadata={"base_extraction": base.to_dict()},
            )

        node = _selected_node_payload(selected_graph_answer, graph_store)
        title = str(node["metadata"].get("title") or "")
        if title:
            return _fallback_extraction(
                extractor_name="sentence_title_backoff",
                node=node,
                target="title",
                reason="clause_trim_empty_backoff_to_node_title_metadata",
                extracted_answer=title,
                extra_metadata={"base_extraction": base.to_dict()},
            )
        return _fallback_extraction(
            extractor_name="sentence_title_backoff",
            node=node,
            target="empty",
            reason="clause_trim_empty_and_no_title_metadata",
            extracted_answer="",
            extra_metadata={"base_extraction": base.to_dict()},
        )

    return extractor


def make_answer_extractor_factory(extractor_name: str) -> AnswerExtractorFactory:
    """Return a fresh extractor factory for the requested extractor name."""

    if extractor_name == "full_sentence":
        return make_full_sentence_extractor
    if extractor_name == "clause_trim":
        return make_clause_trim_extractor
    if extractor_name == "answer_like_span":
        return make_answer_like_span_extractor
    if extractor_name == "sentence_title_backoff":
        return make_sentence_title_backoff_extractor
    raise ValueError(
        f"Unknown HotpotQA answer extractor '{extractor_name}'. "
        f"Available extractors: {', '.join(ANSWER_EXTRACTOR_NAMES)}"
    )


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


def _fallback_extraction(
    *,
    extractor_name: str,
    node: dict[str, Any],
    target: str,
    reason: str,
    extracted_answer: str,
    extra_metadata: dict[str, Any] | None = None,
) -> HotpotQAAnswerExtraction:
    metadata = {
        "strategy": "fallback",
        "fallback_occurred": True,
        "fallback_target": target,
        "fallback_reason": reason,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return HotpotQAAnswerExtraction(
        extractor_name=extractor_name,
        source_node_id=node.get("node_id"),
        source_node_type=node.get("node_type"),
        source_text=str(node.get("text") or ""),
        extracted_answer=extracted_answer,
        fallback_occurred=True,
        fallback_target=target,
        fallback_reason=reason,
        metadata=metadata,
    )


def _split_clauses(text: str) -> list[str]:
    normalized = text.strip()
    if not normalized:
        return []
    parts = re.split(r"[,;:\-\u2013\u2014()\"'“”]+", normalized)
    return [_clean_span(part) for part in parts if _clean_span(part)]


def _answer_like_candidates(text: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for span in re.findall(r'"([^"]{1,80})"|“([^”]{1,80})”|\'([^\']{1,80})\'', text):
        cleaned = _clean_span(next(item for item in span if item))
        if cleaned:
            candidates.append({"text": cleaned, "strategy": "quoted_span"})

    for match in re.finditer(r"\(([^()]{1,80})\)", text):
        cleaned = _clean_span(match.group(1))
        if cleaned:
            candidates.append({"text": cleaned, "strategy": "parenthetical_span"})

    for match in re.finditer(r"\b(?:1[5-9]\d{2}|20\d{2}|\d+(?:\.\d+)?)\b", text):
        candidates.append({"text": match.group(0), "strategy": "numeric_or_year_span"})

    for match in re.finditer(r"\b[A-Z][A-Za-z0-9'.]*(?:\s+[A-Z][A-Za-z0-9'.]*){0,4}\b", text):
        cleaned = _clean_span(match.group(0))
        first_token = cleaned.split()[0] if cleaned else ""
        if cleaned and first_token not in _CAPITALIZED_PHRASE_SKIP and len(cleaned.split()) <= 5:
            candidates.append({"text": cleaned, "strategy": "capitalized_phrase"})

    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate["text"].lower()
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return sorted(unique, key=lambda item: (_strategy_priority(item["strategy"]), len(item["text"]), item["text"]))


def _rank_short_texts(items: list[str]) -> list[str]:
    return sorted(items, key=lambda item: (len(item.split()), len(item), item))


def _strategy_priority(strategy: str) -> int:
    order = {
        "quoted_span": 0,
        "parenthetical_span": 1,
        "numeric_or_year_span": 2,
        "capitalized_phrase": 3,
    }
    return order.get(strategy, 99)


def _clean_span(text: str) -> str:
    cleaned = text.strip(" \t\n\r\"'“”[]{} .?!")
    if not re.search(r"[A-Za-z0-9]", cleaned):
        return ""
    return cleaned
