"""HotpotQA-specific answer projection and evaluation adapter."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from answer.evaluator import AnswerEvaluator
from answer.graph_answer_projector import GraphAnswerProjection, GraphAnswerProjector
from answer.hotpotqa_answer_extractor import (
    AnswerExtractor,
    HotpotQAAnswerExtraction,
    make_full_sentence_extractor,
)
from core.episode_result import EvalResult
from data.benchmarks.answer_normalization import normalize_answer
from graph.graph_store import GraphStore


@dataclass(slots=True)
class HotpotQAAnswerAlignment:
    """Projected HotpotQA answer and minimal evaluation output."""

    raw_graph_answer: str | None
    projected_answer: str
    normalized_projected_answer: str
    gold_answer: str | None
    normalized_gold_answer: str
    projected_eval_score: float
    is_correct: bool | None
    projection: GraphAnswerProjection
    extraction: HotpotQAAnswerExtraction
    eval_result: EvalResult
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary copy."""

        return asdict(self)


class HotpotQAAnswerAdapter:
    """Align graph-backed HotpotQA answers to text-level evaluation."""

    def __init__(
        self,
        *,
        projector: GraphAnswerProjector | None = None,
        evaluator: AnswerEvaluator | None = None,
    ) -> None:
        self.projector = projector or GraphAnswerProjector()
        self.evaluator = evaluator or AnswerEvaluator()

    def align(
        self,
        *,
        raw_graph_answer: str | None,
        graph_store: GraphStore,
        gold_answer: str | None,
        answer_extractor: AnswerExtractor | None = None,
        extraction: HotpotQAAnswerExtraction | None = None,
    ) -> HotpotQAAnswerAlignment:
        """Project a graph answer to text and evaluate normalized text."""

        projection = self.projector.project(raw_graph_answer=raw_graph_answer, graph_store=graph_store)
        resolved_extraction = extraction
        if resolved_extraction is None:
            extractor = answer_extractor or make_full_sentence_extractor()
            resolved_extraction = extractor(raw_graph_answer, graph_store, None)
        projected_answer = resolved_extraction.extracted_answer
        normalized_projected = normalize_answer(projected_answer)
        normalized_gold = normalize_answer(gold_answer)
        eval_result = self.evaluator.evaluate(
            predicted_answer=normalized_projected,
            ground_truth=normalized_gold if gold_answer is not None else None,
        )
        return HotpotQAAnswerAlignment(
            raw_graph_answer=raw_graph_answer,
            projected_answer=projected_answer,
            normalized_projected_answer=normalized_projected,
            gold_answer=gold_answer,
            normalized_gold_answer=normalized_gold,
            projected_eval_score=eval_result.score,
            is_correct=eval_result.is_correct,
            projection=projection,
            extraction=resolved_extraction,
            eval_result=eval_result,
            metadata={
                "projection_source": projection.projection_source,
                "node_type": projection.node_type,
                "extractor_name": resolved_extraction.extractor_name,
                "answer_extraction": resolved_extraction.to_dict(),
                **projection.metadata,
            },
        )
