"""Build benchmark local graphs from local 2WikiMultiHopQA files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.benchmarks import parse_indices
from data.benchmarks.twowiki import load_twowiki
from graph.benchmark_graph_builder import build_from_benchmark_example
from graph.benchmark_graph_loader import compute_graph_stats, save_local_graph_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local graph specs from 2WikiMultiHopQA examples.")
    parser.add_argument("--path", required=True, help="Path to a 2WikiMultiHopQA JSON or JSONL file.")
    parser.add_argument("--limit", type=int, default=1, help="Maximum number of examples to build.")
    parser.add_argument("--indices", default=None, help="Comma-separated record indices to build.")
    parser.add_argument("--split", default=None, help="Optional split label stored in metadata.")
    parser.add_argument("--output-dir", default=None, help="Optional directory for graph JSON specs.")
    args = parser.parse_args()

    examples = load_twowiki(args.path, limit=args.limit, indices=parse_indices(args.indices), split=args.split)
    for index, example in enumerate(examples):
        local_graph = build_from_benchmark_example(example)
        stats = compute_graph_stats(local_graph)
        print(f"=== 2WikiMultiHopQA Graph {index} ===")
        print(f"graph_id: {stats['graph_id']}")
        print(f"nodes: {stats['node_count']} | edges: {stats['edge_count']}")
        print(f"node_type_counts: {stats['node_type_counts']}")
        print(f"relation_counts: {stats['relation_counts']}")
        print(f"supporting_facts_in_metadata: {stats['supporting_facts_in_metadata']}")
        print(f"privileged_edges_injected: {stats['privileged_edges_injected']}")
        if args.output_dir:
            output_path = Path(args.output_dir) / f"{_safe_filename(local_graph.graph_id)}.json"
            print(f"saved: {save_local_graph_json(local_graph, output_path)}")


def _safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


if __name__ == "__main__":
    main()
