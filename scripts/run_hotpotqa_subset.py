"""Run a minimal HotpotQA ingestion and TaskSample conversion dry-run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.benchmarks import parse_indices
from data.benchmarks.hotpotqa import load_hotpotqa


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate HotpotQA subset ingestion.")
    parser.add_argument("--path", required=True, help="Path to a HotpotQA JSON or JSONL file.")
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of examples to load.")
    parser.add_argument("--indices", default=None, help="Comma-separated record indices to load.")
    parser.add_argument("--split", default=None, help="Optional split label stored in metadata.")
    args = parser.parse_args()

    examples = load_hotpotqa(
        args.path,
        limit=args.limit,
        indices=parse_indices(args.indices),
        split=args.split,
    )
    tasks = [example.to_task_sample() for example in examples]
    print("=== HotpotQA Subset Dry Run ===")
    print(f"examples: {len(examples)}")
    print(f"tasks: {len(tasks)}")
    for index, task in enumerate(tasks):
        print(
            f"[{index}] id={task.metadata.get('question_id')} "
            f"dataset={task.dataset_name} titles={len(task.metadata.get('context_titles', []))} "
            f"supporting_facts={len(task.metadata.get('supporting_facts', []))}"
        )


if __name__ == "__main__":
    main()
