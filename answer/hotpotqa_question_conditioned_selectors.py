"""Question-conditioned HotpotQA answer selectors.

Selectors choose only from path-touched nodes and use only query text plus
whitelisted title/sentence text. They do not read gold answers, supporting
facts, benchmark metadata, or scan the full graph.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from answer.hotpotqa_answer_selector import (
    AnswerSelector,
    AnswerSelectorFactory,
    HotpotQAAnswerSelection,
    collect_path_candidate_nodes,
)
from core.experiment_result import ExperimentResult
from graph.graph_store import GraphStore
from relevance.hotpotqa_question_conditioned_scorer import RelevanceScore, score_candidate_text

QUESTION_CONDITIONED_SELECTOR_NAMES = (
    "overlap_guided_sentence",
    "overlap_plus_region",
)


def make_overlap_guided_sentence_selector(*, scorer_name: str = "title_sentence_hybrid") -> AnswerSelector:
    """Select the path-touched sentence/title with highest query overlap score."""

    def selector(result: ExperimentResult, graph_store: GraphStore) -> HotpotQAAnswerSelection:
        raw_answer = _raw_graph_answer(result)
        query_text = _query_text(result)
        candidate_nodes = collect_path_candidate_nodes(result=result, graph_store=graph_store, raw_graph_answer=raw_answer)
        scored = _score_candidate_nodes(
            candidate_nodes,
            graph_store=graph_store,
            query_text=query_text,
            scorer_name=scorer_name,
            prefer_region=None,
        )
        selected = _best_scored_node(scored) or _latest_by_type(candidate_nodes, "sentence") or _latest_by_type(candidate_nodes, "title")
        selected_id = selected["node"]["node_id"] if selected is not None else raw_answer
        return HotpotQAAnswerSelection(
            selector_name="overlap_guided_sentence",
            raw_graph_answer=raw_answer,
            selected_graph_answer=selected_id,
            selection_source=selected["node"].get("node_type") if selected is not None else "raw_final_node",
            candidate_nodes=candidate_nodes,
            metadata={
                "fallback_to_raw": selected is None,
                "path_touched_only": True,
                "scorer_name": scorer_name,
                "selected_score": selected["score"].to_dict() if selected is not None else None,
                "score_component_summary": _score_component_summary(scored),
            },
        )

    return selector


def make_overlap_plus_region_selector(*, scorer_name: str = "title_sentence_hybrid") -> AnswerSelector:
    """Combine query overlap with a dominant touched-title region bonus."""

    def selector(result: ExperimentResult, graph_store: GraphStore) -> HotpotQAAnswerSelection:
        raw_answer = _raw_graph_answer(result)
        query_text = _query_text(result)
        candidate_nodes = collect_path_candidate_nodes(result=result, graph_store=graph_store, raw_graph_answer=raw_answer)
        dominant_region = _dominant_title_region(candidate_nodes, graph_store)
        scored = _score_candidate_nodes(
            candidate_nodes,
            graph_store=graph_store,
            query_text=query_text,
            scorer_name=scorer_name,
            prefer_region=dominant_region,
        )
        selected = _best_scored_node(scored) or _latest_by_type(candidate_nodes, "sentence") or _latest_by_type(candidate_nodes, "title")
        selected_id = selected["node"]["node_id"] if selected is not None else raw_answer
        return HotpotQAAnswerSelection(
            selector_name="overlap_plus_region",
            raw_graph_answer=raw_answer,
            selected_graph_answer=selected_id,
            selection_source=selected["node"].get("node_type") if selected is not None else "raw_final_node",
            candidate_nodes=candidate_nodes,
            metadata={
                "dominant_title_region": dominant_region,
                "fallback_to_raw": selected is None,
                "path_touched_only": True,
                "scorer_name": scorer_name,
                "selected_score": selected["score"].to_dict() if selected is not None else None,
                "score_component_summary": _score_component_summary(scored),
            },
        )

    return selector


def make_question_conditioned_selector_factory(
    selector_name: str,
    *,
    scorer_name: str = "title_sentence_hybrid",
) -> AnswerSelectorFactory:
    """Return a fresh question-conditioned selector factory."""

    if selector_name == "overlap_guided_sentence":
        return lambda: make_overlap_guided_sentence_selector(scorer_name=scorer_name)
    if selector_name == "overlap_plus_region":
        return lambda: make_overlap_plus_region_selector(scorer_name=scorer_name)
    raise ValueError(
        f"Unknown HotpotQA question-conditioned selector '{selector_name}'. "
        f"Available selectors: {', '.join(QUESTION_CONDITIONED_SELECTOR_NAMES)}"
    )


def _score_candidate_nodes(
    candidate_nodes: list[dict[str, Any]],
    *,
    graph_store: GraphStore,
    query_text: str,
    scorer_name: str,
    prefer_region: str | None,
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for index, node in enumerate(candidate_nodes):
        if node.get("node_type") not in {"sentence", "title"}:
            continue
        region = _title_region_key(node, graph_store)
        title_text, sentence_text = _node_title_sentence_text(node, graph_store)
        region_bonus = 1.0 if prefer_region is not None and region == prefer_region else 0.0
        recent_bonus = index / max(1, len(candidate_nodes) - 1) if len(candidate_nodes) > 1 else 1.0
        score = score_candidate_text(
            query_text=query_text,
            title_text=title_text,
            sentence_text=sentence_text,
            path_stats={"region_continuity": region_bonus, "recent_region": recent_bonus},
            scorer_name=scorer_name,
        )
        scored.append({"node": node, "score": score, "region": region, "path_index": index})
    return scored


def _best_scored_node(scored: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not scored:
        return None
    return sorted(
        scored,
        key=lambda item: (
            -float(item["score"].total_score),
            0 if item["node"].get("node_type") == "sentence" else 1,
            -int(item["path_index"]),
            str(item["node"].get("node_id")),
        ),
    )[0]


def _node_title_sentence_text(node: dict[str, Any], graph_store: GraphStore) -> tuple[str, str]:
    node_type = node.get("node_type")
    if node_type == "title":
        return str(node.get("text") or ""), ""
    if node_type == "sentence":
        parent_id = _parent_title_node_id(str(node.get("node_id")), graph_store)
        title_text = _node_text(parent_id, graph_store) if parent_id is not None else ""
        return title_text, str(node.get("text") or "")
    return "", ""


def _score_component_summary(scored: list[dict[str, Any]]) -> dict[str, Any]:
    if not scored:
        return {"candidate_count": 0, "max_total_score": 0.0, "avg_total_score": 0.0}
    totals = [float(item["score"].total_score) for item in scored]
    component_names = sorted({name for item in scored for name in item["score"].component_scores})
    return {
        "candidate_count": len(scored),
        "max_total_score": max(totals),
        "avg_total_score": sum(totals) / len(totals),
        "component_names": component_names,
        "component_description": scored[0]["score"].component_description,
    }


def _dominant_title_region(candidate_nodes: list[dict[str, Any]], graph_store: GraphStore) -> str | None:
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


def _title_region_key(node: dict[str, Any], graph_store: GraphStore) -> str | None:
    if node.get("node_type") == "title":
        return str(node.get("node_id"))
    if node.get("node_type") == "sentence":
        return _parent_title_node_id(str(node.get("node_id")), graph_store)
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


def _latest_by_type(candidate_nodes: list[dict[str, Any]], node_type: str) -> dict[str, Any] | None:
    for node in reversed(candidate_nodes):
        if node.get("node_type") == node_type:
            return {"node": node, "score": RelevanceScore("", 0.0, {}, {}), "path_index": 0}
    return None


def _node_type(node_id: str, graph_store: GraphStore) -> str:
    attrs = graph_store.get_node_attributes(node_id)
    return str(attrs.get("node_type") or "")


def _node_text(node_id: str | None, graph_store: GraphStore) -> str:
    if node_id is None:
        return ""
    attrs = graph_store.get_node_attributes(node_id)
    return str(attrs.get("text") or attrs.get("name") or "")


def _query_text(result: ExperimentResult) -> str:
    return str(result.metadata.get("query") or result.config.metadata.get("query") or "")


def _raw_graph_answer(result: ExperimentResult) -> str | None:
    return result.metadata.get("raw_graph_answer") or result.final_answer
