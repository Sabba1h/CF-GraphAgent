"""Inspect one 2WikiMultiHopQA local graph built from benchmark context."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.benchmarks.twowiki import load_twowiki
from graph.benchmark_graph_builder import build_from_benchmark_example
from graph.benchmark_graph_loader import compute_graph_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a 2WikiMultiHopQA benchmark local graph.")
    parser.add_argument("--path", required=True, help="Path to a 2WikiMultiHopQA JSON or JSONL file.")
    parser.add_argument("--index", type=int, default=0, help="Record index to inspect.")
    parser.add_argument("--split", default=None, help="Optional split label stored in metadata.")
    parser.add_argument("--max-nodes", type=int, default=8, help="Maximum nodes to print.")
    parser.add_argument("--max-edges", type=int, default=12, help="Maximum edges to print.")
    args = parser.parse_args()

    example = load_twowiki(args.path, indices=[args.index], split=args.split)[0]
    local_graph = build_from_benchmark_example(example)
    _print_graph(local_graph, max_nodes=args.max_nodes, max_edges=args.max_edges)


def _print_graph(local_graph, *, max_nodes: int, max_edges: int) -> None:
    stats = compute_graph_stats(local_graph)
    print("=== 2WikiMultiHopQA Local Graph ===")
    print(f"graph_id: {local_graph.graph_id}")
    print(f"question: {local_graph.question}")
    print(f"stats: {stats}")
    print(f"supporting_facts(metadata only): {local_graph.metadata.get('supporting_facts', [])}")
    print("nodes:")
    for node in local_graph.nodes[:max_nodes]:
        print(f"- {node.node_id} | {node.node_type} | {node.text}")
    print("edges:")
    for edge in local_graph.edges[:max_edges]:
        print(f"- {edge.edge_id} | {edge.src} -[{edge.relation}]-> {edge.dst}")


if __name__ == "__main__":
    main()
