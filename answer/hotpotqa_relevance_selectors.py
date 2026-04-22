"""Non-privileged HotpotQA relevance-oriented answer selectors.

Selectors here choose only from nodes touched by the actual rollout path via
``collect_path_candidate_nodes``. They do not scan the full graph and do not
read gold answers, supporting facts, lexical overlap, or external retrieval
signals.
"""

from __future__ import annotations

from collections import Counter

from answer.hotpotqa_answer_selector import (
    AnswerSelector,
    AnswerSelectorFactory,
    HotpotQAAnswerSelection,
    collect_path_candidate_nodes,
)
from core.experiment_result import ExperimentResult
from graph.graph_store import GraphStore

RELEVANCE_SELECTOR_NAMES = (
    "dominant_title_region",
    "recent_relevant_region",
)


def make_dominant_title_region_selector() -> AnswerSelector:
    """Select the latest sentence from the most visited touched title region."""

    def selector(result: ExperimentResult, graph_store: GraphStore) -> HotpotQAAnswerSelection:
        raw_answer = _raw_graph_answer(result)
        candidate_nodes = collect_path_candidate_nodes(result=result, graph_store=graph_store, raw_graph_answer=raw_answer)
        dominant_region = _dominant_title_region(candidate_nodes, graph_store)
        selected = _latest_sentence_in_region(candidate_nodes, graph_store, dominant_region)
        if selected is None:
            selected = _latest_non_question_in_region(candidate_nodes, graph_store, dominant_region)
        if selected is None:
            selected = _latest_by_type(candidate_nodes, "sentence") or _latest_by_type(candidate_nodes, "title")
        selected_id = selected["node_id"] if selected is not None else raw_answer
        return HotpotQAAnswerSelection(
            selector_name="dominant_title_region",
            raw_graph_answer=raw_answer,
            selected_graph_answer=selected_id,
            selection_source=selected["node_type"] if selected is not None else "raw_final_node",
            candidate_nodes=candidate_nodes,
            metadata={
                "dominant_title_region": dominant_region,
                "fallback_to_raw": selected is None,
                "path_touched_only": True,
            },
        )

    return selector


def make_recent_relevant_region_selector() -> AnswerSelector:
    """Select the latest sentence in the most recent local sentence region."""

    def selector(result: ExperimentResult, graph_store: GraphStore) -> HotpotQAAnswerSelection:
        raw_answer = _raw_graph_answer(result)
        candidate_nodes = collect_path_candidate_nodes(result=result, graph_store=graph_store, raw_graph_answer=raw_answer)
        recent_region = _recent_sentence_region(candidate_nodes, graph_store)
        selected = _latest_sentence_in_region(candidate_nodes, graph_store, recent_region)
        if selected is None:
            selected = _latest_by_type(candidate_nodes, "sentence") or _latest_by_type(candidate_nodes, "title")
        selected_id = selected["node_id"] if selected is not None else raw_answer
        return HotpotQAAnswerSelection(
            selector_name="recent_relevant_region",
            raw_graph_answer=raw_answer,
            selected_graph_answer=selected_id,
            selection_source=selected["node_type"] if selected is not None else "raw_final_node",
            candidate_nodes=candidate_nodes,
            metadata={
                "recent_title_region": recent_region,
                "fallback_to_raw": selected is None,
                "path_touched_only": True,
            },
        )

    return selector


def make_relevance_selector_factory(selector_name: str) -> AnswerSelectorFactory:
    """Return a fresh relevance-selector factory."""

    if selector_name == "dominant_title_region":
        return make_dominant_title_region_selector
    if selector_name == "recent_relevant_region":
        return make_recent_relevant_region_selector
    raise ValueError(
        f"Unknown HotpotQA relevance selector '{selector_name}'. "
        f"Available selectors: {', '.join(RELEVANCE_SELECTOR_NAMES)}"
    )


def _dominant_title_region(candidate_nodes: list[dict], graph_store: GraphStore) -> str | None:
    counts: Counter[str] = Counter()
    latest_index: dict[str, int] = {}
    for index, node in enumerate(candidate_nodes):
        region = _title_region_key(node, graph_store)
        if region is None:
            continue
        counts[region] += 1
        latest_index[region] = index
    if not counts:
        return None
    return sorted(counts, key=lambda key: (-counts[key], -latest_index[key], key))[0]


def _recent_sentence_region(candidate_nodes: list[dict], graph_store: GraphStore) -> str | None:
    sentence_regions: list[str] = []
    for node in candidate_nodes:
        if node.get("node_type") != "sentence":
            continue
        region = _title_region_key(node, graph_store)
        if region is not None:
            sentence_regions.append(region)
    if not sentence_regions:
        return None

    suffix_region = sentence_regions[-1]
    suffix_count = 0
    for region in reversed(sentence_regions):
        if region != suffix_region:
            break
        suffix_count += 1
    if suffix_count >= 2:
        return suffix_region
    return sentence_regions[-1]


def _latest_sentence_in_region(
    candidate_nodes: list[dict],
    graph_store: GraphStore,
    region: str | None,
) -> dict | None:
    if region is None:
        return None
    for node in reversed(candidate_nodes):
        if node.get("node_type") == "sentence" and _title_region_key(node, graph_store) == region:
            return node
    return None


def _latest_non_question_in_region(
    candidate_nodes: list[dict],
    graph_store: GraphStore,
    region: str | None,
) -> dict | None:
    if region is None:
        return None
    for node in reversed(candidate_nodes):
        if node.get("node_type") != "question" and _title_region_key(node, graph_store) == region:
            return node
    return None


def _latest_by_type(candidate_nodes: list[dict], node_type: str) -> dict | None:
    for node in reversed(candidate_nodes):
        if node.get("node_type") == node_type:
            return node
    return None


def _title_region_key(node: dict, graph_store: GraphStore) -> str | None:
    node_type = node.get("node_type")
    if node_type == "title":
        return str(node.get("node_id"))

    metadata = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
    metadata_title = metadata.get("title")
    node_id = node.get("node_id")
    if node_type == "sentence" and node_id:
        parent_id = _parent_title_node_id(str(node_id), graph_store)
        if parent_id is not None:
            return parent_id
    if metadata_title:
        return f"title::{metadata_title}"
    return None


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


def _raw_graph_answer(result: ExperimentResult) -> str | None:
    return result.metadata.get("raw_graph_answer") or result.final_answer
