"""Thin graph backend wrapper for the reference environment."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from graph.graph_store import EdgeRecord, GraphStore


class GraphBackend:
    """Minimal backend facade over GraphStore."""

    def __init__(self, graph_store: GraphStore) -> None:
        self.graph_store = graph_store

    def get_edge_by_id(self, edge_id: str) -> EdgeRecord | None:
        """Return an edge by stable edge_id."""

        return self.graph_store.get_edge_by_id(edge_id)

    def has_edge_by_id(self, edge_id: str) -> bool:
        """Check whether an edge exists."""

        return self.graph_store.has_edge_by_id(edge_id)

    def export_subgraph_summary(self, edge_ids: Iterable[str]) -> dict[str, Any]:
        """Export a structured summary for selected edge ids."""

        return self.graph_store.export_subgraph_summary(edge_ids)
