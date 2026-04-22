"""Common benchmark ingestion schemas and helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, TypeVar

from core.task import TaskSample

T = TypeVar("T")


@dataclass(slots=True)
class BenchmarkExample:
    """A JSON-serializable benchmark QA example detached from live env state."""

    dataset_name: str
    question_id: str
    question: str
    answer: str
    normalized_answer: str
    context: list[dict[str, Any]] = field(default_factory=list)
    supporting_facts: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def context_titles(self) -> list[str]:
        """Return context titles in their original order."""

        return [str(item.get("title", "")) for item in self.context if item.get("title") is not None]

    def to_task_sample(self) -> TaskSample:
        """Convert the benchmark example into the core TaskSample schema."""

        metadata = {
            **self.metadata,
            "dataset_name": self.dataset_name,
            "question_id": self.question_id,
            "raw_answer": self.answer,
            "normalized_answer": self.normalized_answer,
            "context": self.context,
            "context_titles": self.context_titles,
            "supporting_facts": self.supporting_facts,
        }
        return TaskSample(
            query=self.question,
            ground_truth=self.answer,
            dataset_name=self.dataset_name,
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain JSON-serializable dictionary."""

        return asdict(self)


def load_json_records(path: str | Path) -> list[dict[str, Any]]:
    """Load benchmark records from a local JSON or JSONL file."""

    input_path = Path(path)
    if input_path.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        with input_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    payload = json.loads(stripped)
                    if not isinstance(payload, dict):
                        raise ValueError(f"JSONL records must be objects: {input_path}")
                    records.append(payload)
        return records

    with input_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        return _ensure_record_list(payload, input_path)
    if isinstance(payload, dict):
        for key in ("data", "examples", "records"):
            value = payload.get(key)
            if isinstance(value, list):
                return _ensure_record_list(value, input_path)
        return [payload]
    raise ValueError(f"Unsupported benchmark file payload in {input_path}: {type(payload).__name__}")


def select_subset(items: list[T], *, limit: int | None = None, indices: Iterable[int] | None = None) -> list[T]:
    """Select examples by explicit indices, then optional limit."""

    selected = items
    if indices is not None:
        selected = []
        for index in indices:
            if index < 0 or index >= len(items):
                raise IndexError(f"Subset index {index} out of range for {len(items)} records.")
            selected.append(items[index])
    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be non-negative.")
        selected = selected[:limit]
    return list(selected)


def parse_indices(value: str | None) -> list[int] | None:
    """Parse a comma-separated index string for CLI scripts."""

    if value is None or value.strip() == "":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def normalize_context(context: Any) -> list[dict[str, Any]]:
    """Preserve benchmark context as title plus sentence/paragraph structure."""

    if context is None:
        return []
    if isinstance(context, dict):
        iterable = context.items()
        return [
            {
                "title": str(title),
                "sentences": _as_sentence_list(sentences),
                "raw": {str(title): sentences},
            }
            for title, sentences in iterable
        ]

    normalized: list[dict[str, Any]] = []
    if not isinstance(context, list):
        return [{"title": "", "sentences": _as_sentence_list(context), "raw": context}]

    for item in context:
        if isinstance(item, dict):
            title = item.get("title") or item.get("name") or item.get("page") or ""
            sentences = item.get("sentences", item.get("paragraph", item.get("text", [])))
            normalized.append({"title": str(title), "sentences": _as_sentence_list(sentences), "raw": item})
            continue
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            title, sentences = item[0], item[1]
            normalized.append({"title": str(title), "sentences": _as_sentence_list(sentences), "raw": item})
            continue
        normalized.append({"title": "", "sentences": _as_sentence_list(item), "raw": item})
    return normalized


def normalize_supporting_facts(supporting_facts: Any) -> list[dict[str, Any]]:
    """Preserve supporting facts as title plus sentence index when available."""

    if supporting_facts is None:
        return []
    facts = supporting_facts if isinstance(supporting_facts, list) else [supporting_facts]
    normalized: list[dict[str, Any]] = []
    for fact in facts:
        if isinstance(fact, dict):
            title = fact.get("title", fact.get("page", ""))
            sentence_index = fact.get("sent_id", fact.get("sentence_index", fact.get("index")))
            normalized.append({"title": str(title), "sentence_index": sentence_index, "raw": fact})
            continue
        if isinstance(fact, (list, tuple)) and len(fact) >= 2:
            normalized.append({"title": str(fact[0]), "sentence_index": fact[1], "raw": fact})
            continue
        normalized.append({"title": "", "sentence_index": None, "raw": fact})
    return normalized


def context_titles(context: list[dict[str, Any]]) -> list[str]:
    """Return context titles without flattening paragraph text."""

    return [str(item.get("title", "")) for item in context if item.get("title") is not None]


def _ensure_record_list(payload: list[Any], path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError(f"Benchmark records must be objects in {path}.")
        records.append(item)
    return records


def _as_sentence_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if value is None:
        return []
    return [value]
