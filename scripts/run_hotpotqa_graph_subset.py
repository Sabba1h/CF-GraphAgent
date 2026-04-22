"""Run a graph-backed HotpotQA subset ingestion smoke test."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.benchmarks import parse_indices
from data.benchmarks.hotpotqa import load_hotpotqa
from graph.benchmark_graph_builder import build_from_benchmark_example
from graph.benchmark_graph_loader import compute_graph_stats, local_graph_to_graph_store


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate HotpotQA local graph build and GraphStore loading.")
    parser.add_argument("--path", required=True, help="Path to a HotpotQA JSON or JSONL file.")
    parser.add_argument("--limit", type=int, default=3, help="Maximum number of examples to load.")
    parser.add_argument("--indices", default=None, help="Comma-separated record indices to load.")
    parser.add_argument("--split", default=None, help="Optional split label stored in metadata.")
    args = parser.parse_args()

    examples = load_hotpotqa(args.path, limit=args.limit, indices=parse_indices(args.indices), split=args.split)
    print("=== HotpotQA Graph Subset Dry Run ===")
    print(f"examples: {len(examples)}")
    for index, example in enumerate(examples):
        local_graph = build_from_benchmark_example(example)
        graph_store = local_graph_to_graph_store(local_graph)
        stats = compute_graph_stats(local_graph)
        print(
            f"[{index}] id={example.question_id} graph={local_graph.graph_id} "
            f"nodes={stats['node_count']} edges={stats['edge_count']} "
            f"graph_store_nodes={len(graph_store.iter_node_ids())} graph_store_edges={len(graph_store.iter_edges())}"
        )


if __name__ == "__main__":
    main()
