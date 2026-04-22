"""Candidate span discovery variants for HotpotQA relation/span answers.

Discovery variants operate only on the selected sentence and path-touched local
sentences supplied by the relation/span mapper context. They do not read gold
answers, supporting facts, benchmark metadata, or scan the full graph.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class HotpotQARelationSpanDiscoveryResult:
    """Candidate spans discovered from local relation/span text."""

    discovery_variant_name: str
    candidate_spans: list[dict[str, Any]]
    discovery_reason: str
    fallback_occurred: bool
    fallback_reason: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""

        return asdict(self)


RelationSpanDiscovery = Callable[[dict[str, Any]], HotpotQARelationSpanDiscoveryResult]
RelationSpanDiscoveryFactory = Callable[[], RelationSpanDiscovery]

RELATION_SPAN_DISCOVERY_NAMES = (
    "baseline_pattern_discovery",
    "multi_window_pattern_discovery",
    "path_touched_sentence_discovery",
    "pattern_plus_local_context_discovery",
)

# Fixed discovery patterns inherited from batch 28. This batch widens candidate
# windows and sources; it does not tune the pattern set from run results.
DISCOVERY_PATTERN_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
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


def make_baseline_pattern_discovery() -> RelationSpanDiscovery:
    """Return batch-28 pattern-only discovery over selected/local sentences."""

    def discovery(context: dict[str, Any]) -> HotpotQARelationSpanDiscoveryResult:
        source_nodes = _source_nodes(context, include_all_path_sentences=True)
        candidates = _pattern_candidates(source_nodes, include_windows=False)
        return _result(
            discovery_variant_name="baseline_pattern_discovery",
            candidate_spans=candidates,
            discovery_reason="fixed_batch28_pattern_candidates",
            fallback_reason="no_fixed_pattern_span_found" if not candidates else None,
            metadata={"strategy": "fixed_pattern_only", "fixed_pattern_names": _pattern_names()},
        )

    return discovery


def make_multi_window_pattern_discovery() -> RelationSpanDiscovery:
    """Return pattern discovery plus small local windows around each hit."""

    def discovery(context: dict[str, Any]) -> HotpotQARelationSpanDiscoveryResult:
        source_nodes = _source_nodes(context, include_all_path_sentences=False)
        candidates = _pattern_candidates(source_nodes, include_windows=True)
        return _result(
            discovery_variant_name="multi_window_pattern_discovery",
            candidate_spans=candidates,
            discovery_reason="pattern_hits_with_local_token_windows",
            fallback_reason="no_pattern_or_window_candidates" if not candidates else None,
            metadata={"strategy": "selected_sentence_pattern_windows", "fixed_pattern_names": _pattern_names()},
        )

    return discovery


def make_path_touched_sentence_discovery() -> RelationSpanDiscovery:
    """Return clause/window candidates from every path-touched local sentence."""

    def discovery(context: dict[str, Any]) -> HotpotQARelationSpanDiscoveryResult:
        source_nodes = _source_nodes(context, include_all_path_sentences=True)
        candidates = _pattern_candidates(source_nodes, include_windows=True) + _clause_candidates(source_nodes)
        candidates = _dedupe_spans(candidates)
        return _result(
            discovery_variant_name="path_touched_sentence_discovery",
            candidate_spans=candidates,
            discovery_reason="path_touched_sentence_patterns_and_clauses",
            fallback_reason="no_path_touched_sentence_candidates" if not candidates else None,
            metadata={"strategy": "path_touched_local_sentence_discovery", "fixed_pattern_names": _pattern_names()},
        )

    return discovery


def make_pattern_plus_local_context_discovery() -> RelationSpanDiscovery:
    """Return pattern hits plus selected/path local context windows."""

    def discovery(context: dict[str, Any]) -> HotpotQARelationSpanDiscoveryResult:
        source_nodes = _source_nodes(context, include_all_path_sentences=True)
        candidates = _pattern_candidates(source_nodes, include_windows=True)
        candidates.extend(_local_context_candidates(source_nodes))
        candidates = _dedupe_spans(candidates)
        return _result(
            discovery_variant_name="pattern_plus_local_context_discovery",
            candidate_spans=candidates,
            discovery_reason="fixed_patterns_plus_local_context_windows",
            fallback_reason="no_pattern_or_context_candidates" if not candidates else None,
            metadata={"strategy": "pattern_plus_local_context", "fixed_pattern_names": _pattern_names()},
        )

    return discovery


def make_relation_span_discovery_factory(discovery_name: str) -> RelationSpanDiscoveryFactory:
    """Return a fresh relation/span discovery factory."""

    if discovery_name == "baseline_pattern_discovery":
        return make_baseline_pattern_discovery
    if discovery_name == "multi_window_pattern_discovery":
        return make_multi_window_pattern_discovery
    if discovery_name == "path_touched_sentence_discovery":
        return make_path_touched_sentence_discovery
    if discovery_name == "pattern_plus_local_context_discovery":
        return make_pattern_plus_local_context_discovery
    raise ValueError(
        f"Unknown HotpotQA relation/span discovery '{discovery_name}'. "
        f"Available variants: {', '.join(RELATION_SPAN_DISCOVERY_NAMES)}"
    )


def _result(
    *,
    discovery_variant_name: str,
    candidate_spans: list[dict[str, Any]],
    discovery_reason: str,
    fallback_reason: str | None,
    metadata: dict[str, Any],
) -> HotpotQARelationSpanDiscoveryResult:
    return HotpotQARelationSpanDiscoveryResult(
        discovery_variant_name=discovery_variant_name,
        candidate_spans=list(candidate_spans),
        discovery_reason=discovery_reason,
        fallback_occurred=not candidate_spans,
        fallback_reason=fallback_reason,
        metadata={
            **metadata,
            **_source_level_summary(candidate_spans),
            "candidate_count": len(candidate_spans),
            "whitelist_fields": [
                "query_text",
                "selected_sentence_text",
                "path_touched_local_sentence_text",
                "title_text",
                "parent_title_text",
                "local_structure_statistics",
            ],
        },
    )


def _source_nodes(context: dict[str, Any], *, include_all_path_sentences: bool) -> list[dict[str, Any]]:
    selected_node = context.get("selected_node") if isinstance(context.get("selected_node"), dict) else {}
    source_nodes = context.get("source_nodes") if isinstance(context.get("source_nodes"), list) else []
    nodes: list[dict[str, Any]] = []
    if selected_node.get("node_type") == "sentence":
        selected = dict(selected_node)
        selected["source_level"] = "selected_sentence"
        nodes.append(selected)
    if include_all_path_sentences:
        for node in source_nodes:
            if not isinstance(node, dict) or node.get("node_type") != "sentence":
                continue
            candidate = dict(node)
            candidate["source_level"] = (
                "selected_sentence"
                if selected_node.get("node_id") and node.get("node_id") == selected_node.get("node_id")
                else "path_touched_sentence"
            )
            nodes.append(candidate)
    return _dedupe_nodes(nodes)


def _pattern_candidates(source_nodes: list[dict[str, Any]], *, include_windows: bool) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for node_index, node in enumerate(source_nodes):
        text = str(node.get("text") or "")
        tokens = _token_spans(text)
        for rule_index, (rule_name, pattern) in enumerate(DISCOVERY_PATTERN_RULES):
            for match_index, match in enumerate(pattern.finditer(text)):
                span = _clean_span(match.group(1))
                if not _is_usable_span(span):
                    continue
                base_payload = _span_payload(
                    text=span,
                    strategy=rule_name,
                    source_node=node,
                    node_index=node_index,
                    span_index=(rule_index * 1000) + match_index,
                    discovery_reason="fixed_pattern_match",
                )
                candidates.append(base_payload)
                if include_windows:
                    candidates.extend(
                        _window_candidates(
                            text=text,
                            tokens=tokens,
                            match_start=match.start(1),
                            match_end=match.end(1),
                            source_node=node,
                            node_index=node_index,
                            span_index=(rule_index * 1000) + match_index,
                            strategy=f"{rule_name}_window",
                        )
                    )
    return _dedupe_spans(candidates)


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
                    span_index=9000 + span_index,
                    discovery_reason="local_clause_split",
                )
            )
    return _dedupe_spans(candidates)


def _local_context_candidates(source_nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for node_index, node in enumerate(source_nodes):
        text = str(node.get("text") or "")
        tokens = _token_spans(text)
        for span_index, width in enumerate((3, 5, 8)):
            if len(tokens) < width:
                continue
            for start in range(0, len(tokens) - width + 1):
                if start > 3:
                    break
                span = _clean_span(text[tokens[start][1] : tokens[start + width - 1][2]])
                if not _is_usable_span(span):
                    continue
                candidates.append(
                    _span_payload(
                        text=span,
                        strategy=f"context_window_{width}",
                        source_node=node,
                        node_index=node_index,
                        span_index=12000 + (span_index * 100) + start,
                        discovery_reason="local_context_window",
                    )
                )
    return _dedupe_spans(candidates)


def _window_candidates(
    *,
    text: str,
    tokens: list[tuple[str, int, int]],
    match_start: int,
    match_end: int,
    source_node: dict[str, Any],
    node_index: int,
    span_index: int,
    strategy: str,
) -> list[dict[str, Any]]:
    if not tokens:
        return []
    token_indices = [
        index
        for index, (_, start, end) in enumerate(tokens)
        if not (end <= match_start or start >= match_end)
    ]
    if not token_indices:
        return []
    left = min(token_indices)
    right = max(token_indices)
    candidates: list[dict[str, Any]] = []
    for window_index, radius in enumerate((1, 2)):
        start_index = max(0, left - radius)
        end_index = min(len(tokens) - 1, right + radius)
        span = _clean_span(text[tokens[start_index][1] : tokens[end_index][2]])
        if not _is_usable_span(span):
            continue
        candidates.append(
            _span_payload(
                text=span,
                strategy=strategy,
                source_node=source_node,
                node_index=node_index,
                span_index=span_index + 10000 + window_index,
                discovery_reason=f"pattern_window_radius_{radius}",
            )
        )
    return candidates


def _span_payload(
    *,
    text: str,
    strategy: str,
    source_node: dict[str, Any],
    node_index: int,
    span_index: int,
    discovery_reason: str,
) -> dict[str, Any]:
    tokens = TOKEN_RE.findall(text)
    return {
        "text": text,
        "strategy": strategy,
        "source_node_id": source_node.get("node_id"),
        "source_node_type": source_node.get("node_type"),
        "source_text": source_node.get("text"),
        "source_level": source_node.get("source_level") or "path_touched_sentence",
        "discovery_reason": discovery_reason,
        "token_count": len(tokens),
        "char_count": len(text),
        "node_index": node_index,
        "span_index": span_index,
    }


def _token_spans(text: str) -> list[tuple[str, int, int]]:
    return [(match.group(0), match.start(), match.end()) for match in TOKEN_RE.finditer(text)]


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


def _dedupe_spans(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for span in candidates:
        key = (
            str(span.get("text") or "").lower(),
            str(span.get("source_node_id") or ""),
            str(span.get("strategy") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(span)
    return deduped


def _source_level_summary(candidate_spans: list[dict[str, Any]]) -> dict[str, int]:
    selected_count = sum(1 for span in candidate_spans if span.get("source_level") == "selected_sentence")
    path_count = sum(1 for span in candidate_spans if span.get("source_level") == "path_touched_sentence")
    return {
        "from_selected_sentence_count": selected_count,
        "from_path_touched_sentence_count": path_count,
    }


def _pattern_names() -> list[str]:
    return [name for name, _ in DISCOVERY_PATTERN_RULES]
