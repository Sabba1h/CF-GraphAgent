"""Project graph-internal answers into text."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from graph.graph_store import GraphStore


@dataclass(slots=True)
class GraphAnswerProjection:
    """Text projection for one graph-level answer."""

    raw_graph_answer: str | None
    projected_answer_text: str
    node_type: str | None = None
    projection_source: str = "fallback"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary copy."""

        return asdict(self)


class GraphAnswerProjector:
    """Map graph node ids to answer text using GraphStore node attributes."""

    def project(self, *, raw_graph_answer: str | None, graph_store: GraphStore) -> GraphAnswerProjection:
        """Project a raw graph answer into text without using benchmark supervision."""

        if not raw_graph_answer or raw_graph_answer == "UNKNOWN":
            return GraphAnswerProjection(
                raw_graph_answer=raw_graph_answer,
                projected_answer_text="",
                projection_source="fallback",
                metadata={"projection_fallback_reason": "empty_or_unknown_answer"},
            )

        node_attrs = graph_store.get_node_attributes(raw_graph_answer)
        if not node_attrs:
            return GraphAnswerProjection(
                raw_graph_answer=raw_graph_answer,
                projected_answer_text="",
                projection_source="fallback",
                metadata={"projection_fallback_reason": "unknown_node"},
            )

        node_type = str(node_attrs.get("node_type", ""))
        if node_type == "question":
            return GraphAnswerProjection(
                raw_graph_answer=raw_graph_answer,
                projected_answer_text="",
                node_type=node_type,
                projection_source="fallback",
                metadata={"projection_fallback_reason": "question_node"},
            )

        if node_type == "title":
            return GraphAnswerProjection(
                raw_graph_answer=raw_graph_answer,
                projected_answer_text=str(node_attrs.get("text") or node_attrs.get("name") or ""),
                node_type=node_type,
                projection_source="title",
                metadata={"node_id": raw_graph_answer},
            )

        if node_type == "sentence":
            return GraphAnswerProjection(
                raw_graph_answer=raw_graph_answer,
                projected_answer_text=str(node_attrs.get("text") or node_attrs.get("name") or ""),
                node_type=node_type,
                projection_source="sentence",
                metadata={"node_id": raw_graph_answer},
            )

        return GraphAnswerProjection(
            raw_graph_answer=raw_graph_answer,
            projected_answer_text=str(node_attrs.get("text") or node_attrs.get("name") or ""),
            node_type=node_type or None,
            projection_source="node_text",
            metadata={"node_id": raw_graph_answer},
        )
