"""HotpotQA local file loader."""

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

DATASET_NAME = "hotpotqa"


def record_to_benchmark_example(record: dict[str, Any], *, split: str | None = None) -> BenchmarkExample:
    """Convert one HotpotQA record into a common benchmark example."""

    question_id = str(record.get("_id") or record.get("id") or record.get("question_id") or "")
    question = str(record.get("question", ""))
    raw_answer = "" if record.get("answer") is None else str(record.get("answer"))
    context, context_warnings = _normalize_hotpot_context(record.get("context"))
    supporting_facts, supporting_fact_warnings = _normalize_hotpot_supporting_facts(record.get("supporting_facts"))
    metadata = {
        "dataset_name": DATASET_NAME,
        "question_id": question_id,
        "raw_answer": raw_answer,
        "raw_context": record.get("context"),
        "raw_supporting_facts": record.get("supporting_facts"),
        "normalized_answer": normalize_answer(raw_answer),
        "question_type": record.get("type"),
        "level": record.get("level"),
        "split": split,
        "context_titles": context_titles(context),
        "warnings": context_warnings + supporting_fact_warnings,
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


def load_hotpotqa(
    path: str | Path,
    *,
    limit: int | None = None,
    indices: Iterable[int] | None = None,
    split: str | None = None,
) -> list[BenchmarkExample]:
    """Load a local HotpotQA JSON/JSONL file as BenchmarkExample objects."""

    records = load_json_records(path)
    examples = [record_to_benchmark_example(record, split=split) for record in records]
    return select_subset(examples, limit=limit, indices=indices)


def load_hotpotqa_task_samples(
    path: str | Path,
    *,
    limit: int | None = None,
    indices: Iterable[int] | None = None,
    split: str | None = None,
) -> list[TaskSample]:
    """Load HotpotQA and convert records into TaskSample objects."""

    return [example.to_task_sample() for example in load_hotpotqa(path, limit=limit, indices=indices, split=split)]


def _normalize_hotpot_context(context: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Normalize HotpotQA context, including fullwiki columnar records."""

    if not _is_columnar_context(context):
        return normalize_context(context), []

    titles = _as_list(context.get("title"))
    sentence_rows = _as_list(context.get("sentences"))
    paired_count = min(len(titles), len(sentence_rows))
    warnings = _length_warning(
        field="context",
        lengths={"title": len(titles), "sentences": len(sentence_rows)},
        paired_count=paired_count,
    )
    normalized = [
        {
            "title": str(titles[index]),
            "sentences": _as_sentence_list(sentence_rows[index]),
            "raw": {"title": titles[index], "sentences": sentence_rows[index]},
        }
        for index in range(paired_count)
    ]
    return normalized, warnings


def _normalize_hotpot_supporting_facts(
    supporting_facts: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Normalize HotpotQA supporting facts, including fullwiki columnar records."""

    if not _is_columnar_supporting_facts(supporting_facts):
        return normalize_supporting_facts(supporting_facts), []

    titles = _as_list(supporting_facts.get("title"))
    sent_ids = _as_list(supporting_facts.get("sent_id"))
    paired_count = min(len(titles), len(sent_ids))
    warnings = _length_warning(
        field="supporting_facts",
        lengths={"title": len(titles), "sent_id": len(sent_ids)},
        paired_count=paired_count,
    )
    normalized = [
        {
            "title": str(titles[index]),
            "sent_id": sent_ids[index],
            "sentence_index": sent_ids[index],
            "raw": {"title": titles[index], "sent_id": sent_ids[index]},
        }
        for index in range(paired_count)
    ]
    return normalized, warnings


def _is_columnar_context(value: Any) -> bool:
    return isinstance(value, dict) and isinstance(value.get("title"), list) and isinstance(value.get("sentences"), list)


def _is_columnar_supporting_facts(value: Any) -> bool:
    return isinstance(value, dict) and isinstance(value.get("title"), list) and isinstance(value.get("sent_id"), list)


def _length_warning(*, field: str, lengths: dict[str, int], paired_count: int) -> list[dict[str, Any]]:
    if len(set(lengths.values())) <= 1:
        return []
    return [
        {
            "field": field,
            "reason": "column_length_mismatch",
            "lengths": lengths,
            "paired_count": paired_count,
        }
    ]


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if value is None:
        return []
    return [value]


def _as_sentence_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if value is None:
        return []
    return [value]
