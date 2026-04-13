"""Graph storage primitives for the stage-1 environment."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import networkx as nx


@dataclass(slots=True)
class EdgeRecord:
    """Stable edge schema stored in the base graph."""

    edge_id: str
    src: str
    dst: str
    relation: str
    confidence: float = 1.0
    source: str = "unknown"
    timestamp: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary payload compatible with NetworkX edge attrs."""

        return asdict(self)


class GraphStore:
    """Thin wrapper around a NetworkX MultiDiGraph with edge_id indexing."""

    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()
        self._edge_index: dict[str, EdgeRecord] = {}

    def add_node(self, node_id: str, **attrs: Any) -> None:
        """Add or update a node."""

        self.graph.add_node(node_id, **attrs)

    def add_edge(self, edge: EdgeRecord) -> None:
        """Add an edge with a globally unique edge_id."""

        if edge.edge_id in self._edge_index:
            raise ValueError(f"Duplicate edge_id '{edge.edge_id}' is not allowed.")
        self.add_node(edge.src, name=edge.src)
        self.add_node(edge.dst, name=edge.dst)
        self.graph.add_edge(edge.src, edge.dst, key=edge.edge_id, **edge.to_dict())
        self._edge_index[edge.edge_id] = edge

    def get_edge_by_id(self, edge_id: str) -> EdgeRecord | None:
        """Return an edge by its stable edge_id."""

        return self._edge_index.get(edge_id)

    def has_edge_by_id(self, edge_id: str) -> bool:
        """Check whether an edge exists."""

        return edge_id in self._edge_index

    def get_outgoing_edges(self, node_id: str) -> list[EdgeRecord]:
        """Return outgoing edges for a node."""

        return [self._edge_index[key] for _, _, key in self.graph.out_edges(node_id, keys=True)]

    def get_incoming_edges(self, node_id: str) -> list[EdgeRecord]:
        """Return incoming edges for a node."""

        return [self._edge_index[key] for _, _, key in self.graph.in_edges(node_id, keys=True)]

    def get_neighbors(self, node_id: str, direction: str = "both") -> list[str]:
        """Return neighbor node ids in the requested direction."""

        neighbors: set[str] = set()
        if direction not in {"in", "out", "both"}:
            raise ValueError("direction must be one of {'in', 'out', 'both'}.")
        if direction in {"out", "both"}:
            neighbors.update(dst for _, dst in self.graph.out_edges(node_id))
        if direction in {"in", "both"}:
            neighbors.update(src for src, _ in self.graph.in_edges(node_id))
        return sorted(neighbors)

    def iter_node_ids(self) -> list[str]:
        """Return all node ids in insertion order."""

        return list(self.graph.nodes())

    def iter_edges(self) -> list[EdgeRecord]:
        """Return all edge records in insertion order."""

        return list(self._edge_index.values())

    def get_node_attributes(self, node_id: str) -> dict[str, Any]:
        """Return node attributes when the node exists."""

        if node_id not in self.graph:
            return {}
        return dict(self.graph.nodes[node_id])

    def export_subgraph_summary(
        self,
        edge_ids: Iterable[str],
        *,
        max_edges: int = 10,
        max_nodes: int = 10,
    ) -> dict[str, Any]:
        """Export a small structured summary of selected edges and nodes."""

        selected_edges: list[EdgeRecord] = []
        nodes: set[str] = set()
        for edge_id in edge_ids:
            edge = self.get_edge_by_id(edge_id)
            if edge is None:
                continue
            selected_edges.append(edge)
            nodes.add(edge.src)
            nodes.add(edge.dst)

        ordered_edges = sorted(selected_edges, key=lambda item: item.edge_id)
        edge_payloads = [
            {
                "edge_id": edge.edge_id,
                "src": edge.src,
                "relation": edge.relation,
                "dst": edge.dst,
                "confidence": edge.confidence,
                "source": edge.source,
                "timestamp": edge.timestamp,
            }
            for edge in ordered_edges[:max_edges]
        ]
        edge_descriptions = [f"{edge.src} -[{edge.relation}]-> {edge.dst}" for edge in ordered_edges[:max_edges]]
        return {
            "node_count": len(nodes),
            "edge_count": len(ordered_edges),
            "nodes": sorted(nodes)[:max_nodes],
            "edges": edge_payloads,
            "edge_descriptions": edge_descriptions,
            "text_summary": "; ".join(edge_descriptions) if edge_descriptions else "Empty working subgraph",
        }
