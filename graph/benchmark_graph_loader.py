"""Load benchmark local graph specs into GraphStore."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from graph.benchmark_graph_types import BenchmarkLocalGraph
from graph.graph_store import EdgeRecord, GraphStore


def local_graph_to_graph_store(local_graph: BenchmarkLocalGraph) -> GraphStore:
    """Convert a BenchmarkLocalGraph into the existing GraphStore backend."""

    graph_store = GraphStore()
    for node in local_graph.nodes:
        graph_store.add_node(
            node.node_id,
            name=node.text,
            node_type=node.node_type,
            text=node.text,
            metadata=dict(node.metadata),
            graph_id=local_graph.graph_id,
            dataset_name=local_graph.dataset_name,
            question_id=local_graph.question_id,
        )

    for edge in local_graph.edges:
        edge_record = EdgeRecord(
            edge_id=edge.edge_id,
            src=edge.src,
            dst=edge.dst,
            relation=edge.relation,
            confidence=1.0,
            source="benchmark_local_graph",
            timestamp=None,
        )
        graph_store.add_edge(edge_record)
        graph_store.graph.edges[edge.src, edge.dst, edge.edge_id]["metadata"] = dict(edge.metadata)
        graph_store.graph.edges[edge.src, edge.dst, edge.edge_id]["graph_id"] = local_graph.graph_id
        graph_store.graph.edges[edge.src, edge.dst, edge.edge_id]["dataset_name"] = local_graph.dataset_name
        graph_store.graph.edges[edge.src, edge.dst, edge.edge_id]["question_id"] = local_graph.question_id

    return graph_store


def compute_graph_stats(local_graph: BenchmarkLocalGraph) -> dict[str, Any]:
    """Compute simple graph stats for inspection and smoke tests."""

    node_type_counts = Counter(node.node_type for node in local_graph.nodes)
    relation_counts = Counter(edge.relation for edge in local_graph.edges)
    return {
        "graph_id": local_graph.graph_id,
        "dataset_name": local_graph.dataset_name,
        "question_id": local_graph.question_id,
        "node_count": len(local_graph.nodes),
        "edge_count": len(local_graph.edges),
        "node_type_counts": dict(sorted(node_type_counts.items())),
        "relation_counts": dict(sorted(relation_counts.items())),
        "supporting_facts_in_metadata": "supporting_facts" in local_graph.metadata,
        "privileged_edges_injected": bool(local_graph.metadata.get("privileged_edges_injected", False)),
    }


def save_local_graph_json(local_graph: BenchmarkLocalGraph, output_path: str | Path) -> Path:
    """Save one local graph spec to JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(local_graph.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_local_graph_json(input_path: str | Path) -> BenchmarkLocalGraph:
    """Load one local graph spec from JSON."""

    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Local graph JSON must contain one object.")
    return BenchmarkLocalGraph.from_dict(payload)
