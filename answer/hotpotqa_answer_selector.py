"""Non-privileged HotpotQA graph answer selectors."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from core.experiment_result import ExperimentResult
from graph.graph_store import GraphStore


@dataclass(slots=True)
class HotpotQAAnswerSelection:
    """Selected graph-level answer before text projection."""

    selector_name: str
    raw_graph_answer: str | None
    selected_graph_answer: str | None
    selection_source: str
    candidate_nodes: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""

        return asdict(self)


AnswerSelector = Callable[[ExperimentResult, GraphStore], HotpotQAAnswerSelection]
AnswerSelectorFactory = Callable[[], AnswerSelector]

ANSWER_SELECTOR_NAMES = (
    "raw_final_node",
    "latest_sentence",
    "prefer_sentence_over_title",
    "latest_non_question",
)


def make_raw_final_node_selector() -> AnswerSelector:
    """Return the current selector baseline: use env final answer unchanged."""

    def selector(result: ExperimentResult, graph_store: GraphStore) -> HotpotQAAnswerSelection:
        raw_answer = _raw_graph_answer(result)
        candidate_nodes = collect_path_candidate_nodes(result=result, graph_store=graph_store, raw_graph_answer=raw_answer)
        return HotpotQAAnswerSelection(
            selector_name="raw_final_node",
            raw_graph_answer=raw_answer,
            selected_graph_answer=raw_answer,
            selection_source="raw_final_node",
            candidate_nodes=candidate_nodes,
        )

    return selector


def make_latest_sentence_selector() -> AnswerSelector:
    """Prefer the latest sentence reached by the actual path."""

    def selector(result: ExperimentResult, graph_store: GraphStore) -> HotpotQAAnswerSelection:
        raw_answer = _raw_graph_answer(result)
        candidate_nodes = collect_path_candidate_nodes(result=result, graph_store=graph_store, raw_graph_answer=raw_answer)
        selected = _latest_by_type(candidate_nodes, "sentence") or _latest_by_type(candidate_nodes, "title")
        selected_id = selected["node_id"] if selected is not None else raw_answer
        return HotpotQAAnswerSelection(
            selector_name="latest_sentence",
            raw_graph_answer=raw_answer,
            selected_graph_answer=selected_id,
            selection_source=selected["node_type"] if selected is not None else "raw_final_node",
            candidate_nodes=candidate_nodes,
            metadata={"fallback_to_raw": selected is None},
        )

    return selector


def make_prefer_sentence_over_title_selector() -> AnswerSelector:
    """Prefer any reached sentence over reached title nodes, with stable ordering."""

    def selector(result: ExperimentResult, graph_store: GraphStore) -> HotpotQAAnswerSelection:
        raw_answer = _raw_graph_answer(result)
        candidate_nodes = collect_path_candidate_nodes(result=result, graph_store=graph_store, raw_graph_answer=raw_answer)
        selected = _first_by_type(candidate_nodes, "sentence") or _first_by_type(candidate_nodes, "title")
        selected_id = selected["node_id"] if selected is not None else raw_answer
        return HotpotQAAnswerSelection(
            selector_name="prefer_sentence_over_title",
            raw_graph_answer=raw_answer,
            selected_graph_answer=selected_id,
            selection_source=selected["node_type"] if selected is not None else "raw_final_node",
            candidate_nodes=candidate_nodes,
            metadata={"fallback_to_raw": selected is None},
        )

    return selector


def make_latest_non_question_selector() -> AnswerSelector:
    """Return the latest reached node that is not a question node."""

    def selector(result: ExperimentResult, graph_store: GraphStore) -> HotpotQAAnswerSelection:
        raw_answer = _raw_graph_answer(result)
        candidate_nodes = collect_path_candidate_nodes(result=result, graph_store=graph_store, raw_graph_answer=raw_answer)
        selected = None
        for candidate in reversed(candidate_nodes):
            if candidate.get("node_type") != "question":
                selected = candidate
                break
        selected_id = selected["node_id"] if selected is not None else raw_answer
        return HotpotQAAnswerSelection(
            selector_name="latest_non_question",
            raw_graph_answer=raw_answer,
            selected_graph_answer=selected_id,
            selection_source=selected["node_type"] if selected is not None else "raw_final_node",
            candidate_nodes=candidate_nodes,
            metadata={"fallback_to_raw": selected is None},
        )

    return selector


def make_answer_selector_factory(selector_name: str) -> AnswerSelectorFactory:
    """Return a fresh selector factory for the requested selector name."""

    if selector_name == "raw_final_node":
        return make_raw_final_node_selector
    if selector_name == "latest_sentence":
        return make_latest_sentence_selector
    if selector_name == "prefer_sentence_over_title":
        return make_prefer_sentence_over_title_selector
    if selector_name == "latest_non_question":
        return make_latest_non_question_selector
    raise ValueError(
        f"Unknown HotpotQA answer selector '{selector_name}'. "
        f"Available selectors: {', '.join(ANSWER_SELECTOR_NAMES)}"
    )


def collect_path_candidate_nodes(
    *,
    result: ExperimentResult,
    graph_store: GraphStore,
    raw_graph_answer: str | None,
) -> list[dict[str, Any]]:
    """Collect nodes touched by the actual path without scanning the full graph."""

    nodes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for trace in result.step_traces:
        for node_id in _trace_expanded_nodes(trace):
            if node_id not in seen:
                nodes.append(
                    _node_payload(
                        node_id=node_id,
                        graph_store=graph_store,
                        source="path",
                        step_idx=getattr(trace, "step_idx", None),
                    )
                )
                seen.add(node_id)
    if raw_graph_answer and raw_graph_answer not in seen:
        nodes.append(
            _node_payload(
                node_id=raw_graph_answer,
                graph_store=graph_store,
                source="raw_final_node",
                step_idx=None,
            )
        )
    return nodes


def _trace_expanded_nodes(trace: Any) -> list[str]:
    info = trace.metadata.get("info", {}) if trace.metadata else {}
    expanded_edge = info.get("expanded_edge") if isinstance(info, dict) else None
    if isinstance(expanded_edge, dict):
        return [str(expanded_edge["src"]), str(expanded_edge["dst"])]

    observation = trace.metadata.get("observation", {}) if trace.metadata else {}
    candidates = observation.get("candidate_actions", []) if isinstance(observation, dict) else []
    trace_candidate_id = _trace_candidate_id(trace.action)
    for candidate in candidates:
        if int(candidate.get("candidate_id", -1)) != trace_candidate_id:
            continue
        if candidate.get("action_type") != "EXPAND_EDGE":
            return []
        metadata = candidate.get("metadata") or {}
        src = metadata.get("src")
        dst = metadata.get("dst")
        if src is not None and dst is not None:
            return [str(src), str(dst)]
    return []


def _node_payload(*, node_id: str, graph_store: GraphStore, source: str, step_idx: int | None) -> dict[str, Any]:
    attrs = graph_store.get_node_attributes(node_id)
    node_metadata = attrs.get("metadata") if isinstance(attrs.get("metadata"), dict) else {}
    return {
        "node_id": node_id,
        "node_type": str(attrs.get("node_type", "")),
        "text": str(attrs.get("text") or attrs.get("name") or ""),
        "source": source,
        "step_idx": step_idx,
        "metadata": dict(node_metadata),
    }


def _raw_graph_answer(result: ExperimentResult) -> str | None:
    return result.metadata.get("raw_graph_answer") or result.final_answer


def _latest_by_type(candidate_nodes: list[dict[str, Any]], node_type: str) -> dict[str, Any] | None:
    for candidate in reversed(candidate_nodes):
        if candidate.get("node_type") == node_type:
            return candidate
    return None


def _first_by_type(candidate_nodes: list[dict[str, Any]], node_type: str) -> dict[str, Any] | None:
    for candidate in candidate_nodes:
        if candidate.get("node_type") == node_type:
            return candidate
    return None


def _trace_candidate_id(action: Any) -> int:
    if isinstance(action, dict):
        return int(action.get("candidate_id", -1))
    return int(action)
