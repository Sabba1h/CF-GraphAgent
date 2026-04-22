"""Task sample schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TaskSample:
    """A single graph reasoning task."""

    query: str
    ground_truth: str | None = None
    dataset_name: str | None = None
    seed_entities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
