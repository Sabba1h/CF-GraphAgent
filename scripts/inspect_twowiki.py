"""Inspect local 2WikiMultiHopQA records and their TaskSample conversion."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.benchmarks import parse_indices
from data.benchmarks.twowiki import load_twowiki


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect local 2WikiMultiHopQA examples.")
    parser.add_argument("--path", required=True, help="Path to a 2WikiMultiHopQA JSON or JSONL file.")
    parser.add_argument("--limit", type=int, default=3, help="Maximum number of examples to inspect.")
    parser.add_argument("--indices", default=None, help="Comma-separated record indices to inspect.")
    parser.add_argument("--split", default=None, help="Optional split label stored in metadata.")
    args = parser.parse_args()

    examples = load_twowiki(
        args.path,
        limit=args.limit,
        indices=parse_indices(args.indices),
        split=args.split,
    )
    for index, example in enumerate(examples):
        task = example.to_task_sample()
        print(f"=== 2WikiMultiHopQA Example {index} ===")
        print(f"question_id: {example.question_id}")
        print(f"question: {example.question}")
        print(f"answer(raw): {example.answer}")
        print(f"answer(normalized): {example.normalized_answer}")
        print(f"context_titles: {example.context_titles}")
        print(f"supporting_facts: {example.supporting_facts}")
        print(f"evidences: {example.metadata.get('evidences')}")
        print(f"task.query: {task.query}")
        print(f"task.ground_truth: {task.ground_truth}")
        print(f"task.dataset_name: {task.dataset_name}")
        print(f"task.metadata.keys: {sorted(task.metadata.keys())}")


if __name__ == "__main__":
    main()
