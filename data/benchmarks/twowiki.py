"""2WikiMultiHopQA local file loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from core.task import TaskSample
from data.benchmarks.answer_normalization import normalize_answer
from data.benchmarks.common import (
    BenchmarkExample,
    context_titles,
    load_json_records,
    normalize_context,
    normalize_supporting_facts,
    select_subset,
)

DATASET_NAME = "twowiki"


def record_to_benchmark_example(record: dict[str, Any], *, split: str | None = None) -> BenchmarkExample:
    """Convert one 2WikiMultiHopQA record into a common benchmark example."""

    question_id = str(record.get("_id") or record.get("id") or record.get("qid") or record.get("question_id") or "")
    question = str(record.get("question", ""))
    raw_answer = "" if record.get("answer") is None else str(record.get("answer"))
    context = normalize_context(record.get("context", record.get("contexts", record.get("paragraphs"))))
    supporting_facts = normalize_supporting_facts(record.get("supporting_facts"))
    metadata = {
        "dataset_name": DATASET_NAME,
        "question_id": question_id,
        "raw_answer": raw_answer,
        "normalized_answer": normalize_answer(raw_answer),
        "question_type": record.get("type"),
        "answer_type": record.get("answer_type"),
        "split": split,
        "context_titles": context_titles(context),
        "evidences": record.get("evidences", record.get("evidence", [])),
        "decomposition": record.get("decomposition"),
        "raw_record_keys": sorted(record.keys()),
    }
    return BenchmarkExample(
        dataset_name=DATASET_NAME,
        question_id=question_id,
        question=question,
        answer=raw_answer,
        normalized_answer=metadata["normalized_answer"],
        context=context,
        supporting_facts=supporting_facts,
        metadata=metadata,
    )


def load_twowiki(
    path: str | Path,
    *,
    limit: int | None = None,
    indices: Iterable[int] | None = None,
    split: str | None = None,
) -> list[BenchmarkExample]:
    """Load a local 2WikiMultiHopQA JSON/JSONL file as BenchmarkExample objects."""

    records = load_json_records(path)
    examples = [record_to_benchmark_example(record, split=split) for record in records]
    return select_subset(examples, limit=limit, indices=indices)


def load_twowiki_task_samples(
    path: str | Path,
    *,
    limit: int | None = None,
    indices: Iterable[int] | None = None,
    split: str | None = None,
) -> list[TaskSample]:
    """Load 2WikiMultiHopQA and convert records into TaskSample objects."""

    return [example.to_task_sample() for example in load_twowiki(path, limit=limit, indices=indices, split=split)]
