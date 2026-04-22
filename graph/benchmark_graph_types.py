"""Serializable benchmark local graph schemas."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class BenchmarkGraphNode:
    """A node in a benchmark-local graph spec."""

    node_id: str
    node_type: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BenchmarkGraphNode":
        """Create a node from a dictionary payload."""

        return cls(
            node_id=str(payload["node_id"]),
            node_type=str(payload["node_type"]),
            text=str(payload.get("text", "")),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(slots=True)
class BenchmarkGraphEdge:
    """An edge in a benchmark-local graph spec."""

    edge_id: str
    src: str
    dst: str
    relation: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BenchmarkGraphEdge":
        """Create an edge from a dictionary payload."""

        return cls(
            edge_id=str(payload["edge_id"]),
            src=str(payload["src"]),
            dst=str(payload["dst"]),
            relation=str(payload["relation"]),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(slots=True)
class BenchmarkLocalGraph:
    """A conservative, serializable local graph built from one benchmark example."""

    graph_id: str
    dataset_name: str
    question_id: str
    question: str
    nodes: list[BenchmarkGraphNode] = field(default_factory=list)
    edges: list[BenchmarkGraphEdge] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""

        return {
            "graph_id": self.graph_id,
            "dataset_name": self.dataset_name,
            "question_id": self.question_id,
            "question": self.question,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BenchmarkLocalGraph":
        """Create a local graph from a dictionary payload."""

        return cls(
            graph_id=str(payload["graph_id"]),
            dataset_name=str(payload["dataset_name"]),
            question_id=str(payload["question_id"]),
            question=str(payload["question"]),
            nodes=[BenchmarkGraphNode.from_dict(item) for item in payload.get("nodes", [])],
            edges=[BenchmarkGraphEdge.from_dict(item) for item in payload.get("edges", [])],
            metadata=dict(payload.get("metadata", {})),
        )
