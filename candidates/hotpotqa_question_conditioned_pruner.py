"""Candidate-stage question-conditioned pruning/ranking for HotpotQA.

The candidate-stage logic is opt-in and whitelist-based. It reads only the
query text plus candidate-local text/type/relation fields already exposed by
the baseline candidate generator. It does not read gold answers, supporting
facts, benchmark metadata, or raw node metadata.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

from candidates.generator import CandidateGenerator
from core.actions import ActionType, CandidateAction
from graph.graph_store import GraphStore
from memory.working_memory import WorkingMemory
from relevance.hotpotqa_question_conditioned_scorer import (
    TOKENIZATION_DESCRIPTION,
    RelevanceScore,
    score_candidate_text,
)

CandidateStageMode = Literal[
    "baseline_generator",
    "overlap_pruned_generator",
    "overlap_ranked_generator",
    "hybrid_prune_then_rank_generator",
]

CANDIDATE_STAGE_VARIANTS = (
    "baseline_generator",
    "overlap_pruned_generator",
    "overlap_ranked_generator",
    "hybrid_prune_then_rank_generator",
)


@dataclass(slots=True)
class CandidateStageDecision:
    """Score and keep/drop decision for one expand candidate."""

    candidate_id: int
    original_index: int
    action_type: str
    relation: str
    src_node_type: str
    dst_node_type: str
    score: float
    score_components: dict[str, float]
    component_description: dict[str, Any]
    kept: bool
    pruning_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""

        return asdict(self)


@dataclass(slots=True)
class CandidateStageSummary:
    """Step-level before/after summary for candidate-stage filtering."""

    mode: CandidateStageMode
    original_expand_count: int
    kept_expand_count: int
    pruning_ratio: float
    top_candidate_score: float
    top_k_scores: list[float]
    kept_relation_counts: dict[str, int]
    kept_node_type_counts: dict[str, int]
    scorer_name: str
    component_description: dict[str, Any]
    decisions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""

        return asdict(self)


class HotpotQAQuestionConditionedCandidateGenerator(CandidateGenerator):
    """Opt-in candidate generator with question-conditioned candidate-stage logic."""

    def __init__(
        self,
        *,
        top_k: int = 5,
        max_seed_nodes: int = 3,
        mode: CandidateStageMode = "overlap_ranked_generator",
        scorer_name: str = "title_sentence_hybrid",
        prune_threshold: float = 0.0,
        min_keep: int = 1,
        pool_k: int | None = None,
    ) -> None:
        super().__init__(top_k=top_k, max_seed_nodes=max_seed_nodes)
        if mode not in CANDIDATE_STAGE_VARIANTS:
            raise ValueError(
                f"Unknown candidate-stage mode '{mode}'. "
                f"Available modes: {', '.join(CANDIDATE_STAGE_VARIANTS)}"
            )
        self.mode: CandidateStageMode = mode
        self.scorer_name = scorer_name
        self.prune_threshold = prune_threshold
        self.min_keep = max(0, min_keep)
        self.pool_k = pool_k

    def generate(
        self,
        *,
        query: str,
        graph_store: GraphStore,
        working_memory: WorkingMemory,
    ) -> list[CandidateAction]:
        """Generate candidates and apply opt-in candidate-stage filtering."""

        if self.mode == "baseline_generator":
            return super().generate(query=query, graph_store=graph_store, working_memory=working_memory)

        original_top_k = self.top_k
        self.top_k = self.pool_k or max(self.top_k, 25)
        try:
            pooled_actions = super().generate(query=query, graph_store=graph_store, working_memory=working_memory)
        finally:
            self.top_k = original_top_k

        expand_actions = [action for action in pooled_actions if action.action_type == ActionType.EXPAND_EDGE]
        terminal_actions = [action for action in pooled_actions if action.action_type != ActionType.EXPAND_EDGE]
        kept_actions, summary = apply_candidate_stage_filter(
            query_text=query,
            expand_actions=expand_actions,
            mode=self.mode,
            top_k=original_top_k,
            scorer_name=self.scorer_name,
            prune_threshold=self.prune_threshold,
            min_keep=self.min_keep,
        )
        renumbered: list[CandidateAction] = []
        for action in kept_actions + terminal_actions:
            renumbered.append(replace(action, candidate_id=len(renumbered)))

        return renumbered


def make_candidate_generator_factory(
    *,
    mode: CandidateStageMode,
    top_k: int = 5,
    max_seed_nodes: int = 3,
    scorer_name: str = "title_sentence_hybrid",
    prune_threshold: float = 0.0,
    min_keep: int = 1,
    pool_k: int | None = None,
) -> Any:
    """Return a fresh candidate generator factory for evaluator injection."""

    if mode == "baseline_generator":
        return lambda: CandidateGenerator(top_k=top_k, max_seed_nodes=max_seed_nodes)
    return lambda: HotpotQAQuestionConditionedCandidateGenerator(
        top_k=top_k,
        max_seed_nodes=max_seed_nodes,
        mode=mode,
        scorer_name=scorer_name,
        prune_threshold=prune_threshold,
        min_keep=min_keep,
        pool_k=pool_k,
    )


def apply_candidate_stage_filter(
    *,
    query_text: str,
    expand_actions: list[CandidateAction],
    mode: CandidateStageMode,
    top_k: int,
    scorer_name: str = "title_sentence_hybrid",
    prune_threshold: float = 0.0,
    min_keep: int = 1,
) -> tuple[list[CandidateAction], CandidateStageSummary]:
    """Apply deterministic candidate-stage pruning/ranking to expand actions."""

    scored = [
        _score_action(
            action,
            query_text=query_text,
            original_index=index,
            scorer_name=scorer_name,
        )
        for index, action in enumerate(expand_actions)
    ]
    ranked = sorted(scored, key=_ranking_key)
    if mode == "overlap_pruned_generator":
        kept_scored = _pruned_candidates(scored, ranked, prune_threshold=prune_threshold, min_keep=min_keep)
        kept_scored = sorted(kept_scored, key=lambda item: item["original_index"])[:top_k]
    elif mode == "overlap_ranked_generator":
        kept_scored = ranked[:top_k]
    elif mode == "hybrid_prune_then_rank_generator":
        kept_scored = _pruned_candidates(scored, ranked, prune_threshold=prune_threshold, min_keep=min_keep)
        kept_scored = sorted(kept_scored, key=_ranking_key)[:top_k]
    elif mode == "baseline_generator":
        kept_scored = scored[:top_k]
    else:
        raise ValueError(f"Unsupported candidate-stage mode '{mode}'.")

    kept_indices = {int(item["original_index"]) for item in kept_scored}
    decisions = [
        _decision(
            item,
            kept=int(item["original_index"]) in kept_indices,
            pruning_reason=None if int(item["original_index"]) in kept_indices else _pruning_reason(item, prune_threshold),
        )
        for item in scored
    ]
    summary = _summary(mode=mode, scored=scored, kept_scored=kept_scored, scorer_name=scorer_name, decisions=decisions)
    kept_actions = [_with_candidate_stage_metadata(item["action"], item, summary) for item in kept_scored]
    return kept_actions, summary


def summarize_candidate_stage_from_records(records: list[Any]) -> dict[str, Any]:
    """Aggregate candidate-stage summaries from HotpotQA eval records."""

    summaries: list[dict[str, Any]] = []
    for record in records:
        for step_summary in _candidate_stage_summaries_from_record(record):
            summaries.append(step_summary)
    if not summaries:
        return {
            "available": False,
            "step_count": 0,
            "avg_original_expand_count": 0.0,
            "avg_kept_expand_count": 0.0,
            "avg_pruning_ratio": 0.0,
            "max_top_candidate_score": 0.0,
            "component_description": {**TOKENIZATION_DESCRIPTION},
        }
    relation_counts: Counter[str] = Counter()
    node_type_counts: Counter[str] = Counter()
    for summary in summaries:
        relation_counts.update({str(key): int(value) for key, value in summary.get("kept_relation_counts", {}).items()})
        node_type_counts.update({str(key): int(value) for key, value in summary.get("kept_node_type_counts", {}).items()})
    return {
        "available": True,
        "step_count": len(summaries),
        "avg_original_expand_count": _average(float(summary.get("original_expand_count", 0)) for summary in summaries),
        "avg_kept_expand_count": _average(float(summary.get("kept_expand_count", 0)) for summary in summaries),
        "avg_pruning_ratio": _average(float(summary.get("pruning_ratio", 0.0)) for summary in summaries),
        "max_top_candidate_score": max(float(summary.get("top_candidate_score", 0.0)) for summary in summaries),
        "avg_top_candidate_score": _average(float(summary.get("top_candidate_score", 0.0)) for summary in summaries),
        "kept_relation_counts": dict(sorted(relation_counts.items())),
        "kept_node_type_counts": dict(sorted(node_type_counts.items())),
        "component_description": summaries[0].get("component_description", {**TOKENIZATION_DESCRIPTION}),
    }


def _score_action(
    action: CandidateAction,
    *,
    query_text: str,
    original_index: int,
    scorer_name: str,
) -> dict[str, Any]:
    title_text, sentence_text = _candidate_title_sentence_text(action)
    score = score_candidate_text(
        query_text=query_text,
        title_text=title_text,
        sentence_text=sentence_text,
        scorer_name=scorer_name,
    )
    return {
        "action": action,
        "original_index": original_index,
        "score": score,
        "relation": str(action.metadata.get("relation", "")),
        "src_node_type": str(action.metadata.get("src_node_type", "")),
        "dst_node_type": str(action.metadata.get("dst_node_type", "")),
    }


def _candidate_title_sentence_text(action: CandidateAction) -> tuple[str, str]:
    metadata = action.metadata
    src_type = str(metadata.get("src_node_type") or "")
    dst_type = str(metadata.get("dst_node_type") or "")
    src_text = str(metadata.get("src_text") or "")
    dst_text = str(metadata.get("dst_text") or "")
    title_text = src_text if src_type == "title" else ""
    sentence_text = src_text if src_type == "sentence" else ""
    if dst_type == "title":
        title_text = dst_text
    if dst_type == "sentence":
        sentence_text = dst_text
    return title_text, sentence_text


def _ranking_key(item: dict[str, Any]) -> tuple[float, int, str, str, str, int]:
    action: CandidateAction = item["action"]
    score: RelevanceScore = item["score"]
    metadata = action.metadata
    return (
        -float(score.total_score),
        int(item["original_index"]),
        str(item["relation"]),
        str(metadata.get("src", "")),
        str(metadata.get("dst", "")),
        int(action.candidate_id),
    )


def _pruned_candidates(
    scored: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    *,
    prune_threshold: float,
    min_keep: int,
) -> list[dict[str, Any]]:
    kept = [item for item in scored if float(item["score"].total_score) > prune_threshold]
    if len(kept) >= min_keep:
        return kept
    fallback = ranked[: min(min_keep, len(ranked))]
    fallback_indices = {int(item["original_index"]) for item in fallback}
    merged = [item for item in scored if int(item["original_index"]) in fallback_indices or item in kept]
    return merged


def _decision(item: dict[str, Any], *, kept: bool, pruning_reason: str | None) -> CandidateStageDecision:
    action: CandidateAction = item["action"]
    score: RelevanceScore = item["score"]
    return CandidateStageDecision(
        candidate_id=action.candidate_id,
        original_index=int(item["original_index"]),
        action_type=action.action_type.value,
        relation=str(item["relation"]),
        src_node_type=str(item["src_node_type"]),
        dst_node_type=str(item["dst_node_type"]),
        score=float(score.total_score),
        score_components=dict(score.component_scores),
        component_description=dict(score.component_description),
        kept=kept,
        pruning_reason=pruning_reason,
    )


def _pruning_reason(item: dict[str, Any], prune_threshold: float) -> str:
    score: RelevanceScore = item["score"]
    if float(score.total_score) <= prune_threshold:
        return f"score_lte_threshold:{prune_threshold}"
    return "ranked_out_by_top_k"


def _summary(
    *,
    mode: CandidateStageMode,
    scored: list[dict[str, Any]],
    kept_scored: list[dict[str, Any]],
    scorer_name: str,
    decisions: list[CandidateStageDecision],
) -> CandidateStageSummary:
    original_count = len(scored)
    kept_count = len(kept_scored)
    top_scores = [float(item["score"].total_score) for item in kept_scored]
    relation_counts = Counter(str(item["relation"]) for item in kept_scored)
    node_type_counts: Counter[str] = Counter()
    for item in kept_scored:
        node_type_counts.update([str(item["src_node_type"]), str(item["dst_node_type"])])
    component_description = (
        kept_scored[0]["score"].component_description
        if kept_scored
        else {
            **TOKENIZATION_DESCRIPTION,
            "formula": "candidate-stage scorer received no EXPAND_EDGE candidates",
        }
    )
    return CandidateStageSummary(
        mode=mode,
        original_expand_count=original_count,
        kept_expand_count=kept_count,
        pruning_ratio=0.0 if original_count == 0 else (original_count - kept_count) / original_count,
        top_candidate_score=max(top_scores) if top_scores else 0.0,
        top_k_scores=top_scores,
        kept_relation_counts=dict(sorted(relation_counts.items())),
        kept_node_type_counts={key: value for key, value in sorted(node_type_counts.items()) if key},
        scorer_name=scorer_name,
        component_description=component_description,
        decisions=[decision.to_dict() for decision in decisions],
    )


def _with_candidate_stage_metadata(
    action: CandidateAction,
    item: dict[str, Any],
    summary: CandidateStageSummary,
) -> CandidateAction:
    score: RelevanceScore = item["score"]
    metadata = dict(action.metadata)
    metadata.update(
        {
            "candidate_stage_mode": summary.mode,
            "candidate_stage_score": float(score.total_score),
            "candidate_stage_score_components": dict(score.component_scores),
            "candidate_stage_component_description": dict(score.component_description),
            "candidate_stage_summary": summary.to_dict(),
        }
    )
    return replace(action, score=float(score.total_score), metadata=metadata)


def _candidate_stage_summaries_from_record(record: Any) -> list[dict[str, Any]]:
    metadata = getattr(record, "metadata", {}) if not isinstance(record, dict) else record.get("metadata", {})
    path_summary = metadata.get("path_summary", {}) if isinstance(metadata, dict) else {}
    selected_actions = path_summary.get("selected_actions", []) if isinstance(path_summary, dict) else []
    summaries: list[dict[str, Any]] = []
    # Eval records do not store every observation directly in path_summary, so
    # use candidate summaries copied into selected candidates when available.
    for item in selected_actions:
        if isinstance(item, dict) and isinstance(item.get("candidate_stage_summary"), dict):
            summaries.append(item["candidate_stage_summary"])

    answer_selection = metadata.get("answer_selection") if isinstance(metadata, dict) else None
    if isinstance(answer_selection, dict):
        for node in answer_selection.get("candidate_nodes", []) or []:
            if isinstance(node, dict):
                summary = node.get("candidate_stage_summary")
                if isinstance(summary, dict):
                    summaries.append(summary)
    # Most records expose candidate-stage summaries through step trace metadata;
    # scripts pass ExperimentResult-derived records, whose source metadata keeps
    # reward summaries but not all observations. Fall back to explicit metadata.
    explicit = metadata.get("candidate_stage_summaries") if isinstance(metadata, dict) else None
    if isinstance(explicit, list):
        summaries.extend(summary for summary in explicit if isinstance(summary, dict))
    return _dedupe_summaries(summaries)


def _dedupe_summaries(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for summary in summaries:
        key = repr(
            (
                summary.get("mode"),
                summary.get("original_expand_count"),
                summary.get("kept_expand_count"),
                tuple(summary.get("top_k_scores", [])),
                tuple(sorted((summary.get("kept_relation_counts") or {}).items())),
            )
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(summary)
    return deduped


def _average(values: Any) -> float:
    items = list(values)
    return 0.0 if not items else sum(items) / len(items)

