"""Error analysis for HotpotQA graph-backed evaluation records."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from data.benchmarks.answer_normalization import normalize_answer
from evaluation.hotpotqa_metrics import normalize_answer_source_type
from evaluation.hotpotqa_subset_evaluator import HotpotQAGraphEvalRecord

PROJECTED_ANSWER_TYPES = {"empty", "multi_token", "other", "single_token", "yes_no"}


@dataclass(slots=True)
class GraphStructureBucketConfig:
    """Transparent graph-size thresholds for error buckets."""

    node_small_max: int = 50
    node_medium_max: int = 100
    edge_small_max: int = 120
    edge_medium_max: int = 250


@dataclass(slots=True)
class PathBehaviorConfig:
    """Path behavior thresholds used by the first error analysis harness."""

    min_expands_before_answer: int = 1


@dataclass(slots=True)
class HotpotQAGraphErrorRecord:
    """One HotpotQA graph-backed record enriched with error buckets."""

    question_id: str
    graph_id: str
    policy_name: str
    reward_mode: str
    gold_answer: str
    raw_graph_answer: str | None
    selected_graph_answer: str | None
    projected_answer: str
    normalized_projected_answer: str
    exact_match: float
    f1: float
    projected_eval_score: float
    answer_source_type: str
    projected_answer_type: str
    base_total_reward: float
    total_reward: float
    step_count: int
    graph_node_count: int
    graph_edge_count: int
    path_summary: dict[str, Any]
    answer_selector_name: str = "raw_final_node"
    answer_extractor_name: str = "full_sentence"
    answer_mapper_name: str = "identity"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""

        return asdict(self)


@dataclass(slots=True)
class HotpotQAErrorAnalysisResult:
    """Error analysis output for a HotpotQA graph-backed subset."""

    records: list[HotpotQAGraphErrorRecord]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""

        return {
            "records": [record.to_dict() for record in self.records],
            "summary": self.summary,
        }


def analyze_hotpotqa_error_records(
    records: list[HotpotQAGraphEvalRecord | dict[str, Any]],
    *,
    graph_bucket_config: GraphStructureBucketConfig | None = None,
    path_behavior_config: PathBehaviorConfig | None = None,
) -> HotpotQAErrorAnalysisResult:
    """Build error records and bucket summaries from subset eval records."""

    graph_bucket_config = graph_bucket_config or GraphStructureBucketConfig()
    path_behavior_config = path_behavior_config or PathBehaviorConfig()
    error_records = [
        error_record_from_eval_record(
            _ensure_eval_record(record),
            graph_bucket_config=graph_bucket_config,
            path_behavior_config=path_behavior_config,
        )
        for record in records
    ]
    return HotpotQAErrorAnalysisResult(
        records=error_records,
        summary=summarize_error_records(
            error_records,
            graph_bucket_config=graph_bucket_config,
            path_behavior_config=path_behavior_config,
        ),
    )


def error_record_from_eval_record(
    record: HotpotQAGraphEvalRecord,
    *,
    graph_bucket_config: GraphStructureBucketConfig | None = None,
    path_behavior_config: PathBehaviorConfig | None = None,
) -> HotpotQAGraphErrorRecord:
    """Convert one eval record into an enriched error-analysis record."""

    graph_bucket_config = graph_bucket_config or GraphStructureBucketConfig()
    path_behavior_config = path_behavior_config or PathBehaviorConfig()
    path_summary = dict(record.metadata.get("path_summary") or {})
    path_summary.setdefault("step_count", record.step_count)
    path_summary.setdefault("expand_count", 0)
    path_summary["early_answer"] = is_early_answer(path_summary, path_behavior_config)
    path_summary["fallback_answer"] = record.answer_source_type == "fallback"

    metadata = {
        "graph_node_bucket": graph_count_bucket(
            record.graph_node_count,
            small_max=graph_bucket_config.node_small_max,
            medium_max=graph_bucket_config.node_medium_max,
        ),
        "graph_edge_bucket": graph_count_bucket(
            record.graph_edge_count,
            small_max=graph_bucket_config.edge_small_max,
            medium_max=graph_bucket_config.edge_medium_max,
        ),
        "is_failure": float(record.exact_match) < 1.0,
        "source_metadata": record.metadata,
    }
    return HotpotQAGraphErrorRecord(
        question_id=record.question_id,
        graph_id=record.graph_id,
        policy_name=record.policy_name,
        reward_mode=record.reward_mode,
        gold_answer=record.gold_answer,
        raw_graph_answer=record.raw_graph_answer,
        selected_graph_answer=record.selected_graph_answer,
        projected_answer=record.projected_answer,
        normalized_projected_answer=record.normalized_projected_answer,
        exact_match=float(record.exact_match),
        f1=float(record.f1),
        projected_eval_score=float(record.projected_eval_score),
        answer_source_type=normalize_answer_source_type(record.answer_source_type),
        projected_answer_type=classify_projected_answer(record.projected_answer),
        base_total_reward=float(record.base_total_reward),
        total_reward=float(record.total_reward),
        step_count=int(record.step_count),
        graph_node_count=int(record.graph_node_count),
        graph_edge_count=int(record.graph_edge_count),
        path_summary=path_summary,
        answer_selector_name=record.answer_selector_name,
        answer_extractor_name=record.answer_extractor_name,
        answer_mapper_name=record.answer_mapper_name,
        metadata=metadata,
    )


def summarize_error_records(
    records: list[HotpotQAGraphErrorRecord],
    *,
    graph_bucket_config: GraphStructureBucketConfig | None = None,
    path_behavior_config: PathBehaviorConfig | None = None,
) -> dict[str, Any]:
    """Return all supported bucket summaries."""

    graph_bucket_config = graph_bucket_config or GraphStructureBucketConfig()
    path_behavior_config = path_behavior_config or PathBehaviorConfig()
    return {
        "sample_count": len(records),
        "failure_count": sum(1 for record in records if record.exact_match < 1.0),
        "failure_rate": _safe_rate(sum(1 for record in records if record.exact_match < 1.0), len(records)),
        "answer_source_buckets": bucket_summary(records, lambda record: record.answer_source_type),
        "projected_answer_type_buckets": bucket_summary(records, lambda record: record.projected_answer_type),
        "graph_structure_buckets": {
            "node_count": bucket_summary(
                records,
                lambda record: graph_count_bucket(
                    record.graph_node_count,
                    small_max=graph_bucket_config.node_small_max,
                    medium_max=graph_bucket_config.node_medium_max,
                ),
            ),
            "edge_count": bucket_summary(
                records,
                lambda record: graph_count_bucket(
                    record.graph_edge_count,
                    small_max=graph_bucket_config.edge_small_max,
                    medium_max=graph_bucket_config.edge_medium_max,
                ),
            ),
            "thresholds": asdict(graph_bucket_config),
        },
        "path_buckets": {
            "step_count": bucket_summary(records, lambda record: f"steps_{record.step_count}"),
            "expand_count": bucket_summary(records, lambda record: expand_count_bucket(record.path_summary)),
            "early_answer": bucket_summary(
                records,
                lambda record: "early_answer"
                if is_early_answer(record.path_summary, path_behavior_config)
                else "not_early_answer",
            ),
            "fallback_answer": bucket_summary(
                records,
                lambda record: "fallback_answer" if record.path_summary.get("fallback_answer") else "not_fallback_answer",
            ),
            "thresholds": asdict(path_behavior_config),
        },
        "reward_modes": sorted({record.reward_mode for record in records}),
        "policy_names": sorted({record.policy_name for record in records}),
        "answer_selector_names": sorted({record.answer_selector_name for record in records}),
        "answer_extractor_names": sorted({record.answer_extractor_name for record in records}),
        "answer_mapper_names": sorted({record.answer_mapper_name for record in records}),
    }


def bucket_summary(records: list[HotpotQAGraphErrorRecord], bucket_fn) -> dict[str, dict[str, float | int]]:
    """Aggregate count and metrics for one bucket dimension."""

    grouped: dict[str, list[HotpotQAGraphErrorRecord]] = defaultdict(list)
    for record in records:
        grouped[str(bucket_fn(record))].append(record)
    return {bucket: _metric_summary(items) for bucket, items in sorted(grouped.items())}


def classify_projected_answer(answer: str | None) -> str:
    """Classify projected answer text into a fixed enum."""

    normalized = normalize_answer(answer)
    if not normalized:
        return "empty"
    if normalized in {"yes", "no"}:
        return "yes_no"
    token_count = len(normalized.split())
    if token_count == 1:
        return "single_token"
    if token_count > 1:
        return "multi_token"
    return "other"


def graph_count_bucket(count: int, *, small_max: int, medium_max: int) -> str:
    """Bucket graph size by centralized, configurable thresholds."""

    if count <= small_max:
        return "small"
    if count <= medium_max:
        return "medium"
    return "large"


def expand_count_bucket(path_summary: dict[str, Any]) -> str:
    """Bucket the number of expanded edges in a short path."""

    expand_count = int(path_summary.get("expand_count") or 0)
    if expand_count == 0:
        return "expand_0"
    if expand_count == 1:
        return "expand_1"
    return "expand_2_plus"


def is_early_answer(path_summary: dict[str, Any], config: PathBehaviorConfig) -> bool:
    """Return true when ANSWER occurs before enough EXPAND_EDGE actions."""

    answer_step_idx = path_summary.get("answer_step_idx")
    if answer_step_idx is None:
        return False
    expand_count_before_answer = int(path_summary.get("expand_count_before_answer") or 0)
    return expand_count_before_answer < config.min_expands_before_answer


def load_eval_records_jsonl(path: str | Path) -> list[HotpotQAGraphEvalRecord]:
    """Load subset eval records from JSONL."""

    records: list[HotpotQAGraphEvalRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(_ensure_eval_record(json.loads(line)))
    return records


def save_error_analysis_outputs(
    result: HotpotQAErrorAnalysisResult,
    output_dir: str | Path,
) -> tuple[Path, Path, Path]:
    """Save all error records, failed records, and summary JSON."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    records_path = output_path / "hotpotqa_error_records.jsonl"
    failures_path = output_path / "hotpotqa_failure_records.jsonl"
    summary_path = output_path / "hotpotqa_error_summary.json"
    _write_records_jsonl(records_path, [record.to_dict() for record in result.records])
    _write_records_jsonl(
        failures_path,
        [record.to_dict() for record in result.records if record.exact_match < 1.0],
    )
    summary_path.write_text(
        json.dumps(result.summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return records_path, failures_path, summary_path


def _ensure_eval_record(record: HotpotQAGraphEvalRecord | dict[str, Any]) -> HotpotQAGraphEvalRecord:
    if isinstance(record, HotpotQAGraphEvalRecord):
        return record
    return HotpotQAGraphEvalRecord(
        question_id=str(record.get("question_id") or ""),
        graph_id=str(record.get("graph_id") or ""),
        reward_mode=record.get("reward_mode", "baseline"),
        gold_answer=str(record.get("gold_answer") or ""),
        raw_graph_answer=record.get("raw_graph_answer"),
        selected_graph_answer=record.get("selected_graph_answer"),
        projected_answer=str(record.get("projected_answer") or ""),
        normalized_projected_answer=str(record.get("normalized_projected_answer") or ""),
        projected_eval_score=float(record.get("projected_eval_score") or 0.0),
        exact_match=float(record.get("exact_match") or 0.0),
        f1=float(record.get("f1") or 0.0),
        answer_source_type=normalize_answer_source_type(record.get("answer_source_type")),
        base_total_reward=float(record.get("base_total_reward") or 0.0),
        total_reward=float(record.get("total_reward") or 0.0),
        step_count=int(record.get("step_count") or 0),
        graph_node_count=int(record.get("graph_node_count") or 0),
        graph_edge_count=int(record.get("graph_edge_count") or 0),
        policy_name=str(record.get("policy_name") or record.get("metadata", {}).get("policy_name") or "baseline"),
        answer_selector_name=str(
            record.get("answer_selector_name")
            or record.get("metadata", {}).get("answer_selector_name")
            or "raw_final_node"
        ),
        answer_extractor_name=str(
            record.get("answer_extractor_name")
            or record.get("metadata", {}).get("answer_extractor_name")
            or "full_sentence"
        ),
        answer_mapper_name=str(
            record.get("answer_mapper_name")
            or record.get("metadata", {}).get("answer_mapper_name")
            or "identity"
        ),
        metadata=dict(record.get("metadata") or {}),
    )


def _metric_summary(records: list[HotpotQAGraphErrorRecord]) -> dict[str, float | int]:
    return {
        "count": len(records),
        "exact_match": _average(record.exact_match for record in records),
        "avg_f1": _average(record.f1 for record in records),
        "avg_projected_eval_score": _average(record.projected_eval_score for record in records),
        "failure_count": sum(1 for record in records if record.exact_match < 1.0),
    }


def _average(values: Any) -> float:
    items = [float(value) for value in values]
    if not items:
        return 0.0
    return sum(items) / len(items)


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _write_records_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
