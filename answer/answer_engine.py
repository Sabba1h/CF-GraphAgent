"""Answer generation for the stage-1 environment."""

from __future__ import annotations

from dataclasses import replace

from answer.evaluator import AnswerEvaluator
from graph.graph_store import EdgeRecord, GraphStore
from memory.working_memory import WorkingMemory
from schemas import AnswerResult


class AnswerEngine:
    """Stage-1 answer engine with a stable future-proof interface."""

    def answer(
        self,
        *,
        query: str,
        graph_store: GraphStore,
        working_memory: WorkingMemory,
    ) -> AnswerResult:
        """Generate a placeholder answer from the current working subgraph."""

        working_edges = self._working_edges(graph_store=graph_store, working_memory=working_memory)
        answer, evidence_edge_ids = self._infer_answer(query=query, working_edges=working_edges)
        reasoning = "No supporting edge found in the working subgraph."

        if answer:
            reasoning = f"Selected answer '{answer}' from working subgraph evidence."
        else:
            answer = "UNKNOWN"

        return AnswerResult(
            answer=answer,
            evidence_edge_ids=evidence_edge_ids,
            reasoning=reasoning,
        )

    def generate_answer(
        self,
        *,
        query: str,
        graph_store: GraphStore,
        working_memory: WorkingMemory,
        ground_truth: str | None = None,
    ) -> AnswerResult:
        """Compatibility wrapper that returns answer plus simple eval fields."""

        answer_result = self.answer(query=query, graph_store=graph_store, working_memory=working_memory)
        eval_result = AnswerEvaluator().evaluate(predicted_answer=answer_result.answer, ground_truth=ground_truth)
        return replace(
            answer_result,
            reward=eval_result.score,
            is_correct=eval_result.is_correct,
            reasoning=f"{answer_result.reasoning} Evaluation applied: {eval_result.reason}.",
            metadata={**answer_result.metadata, "eval": eval_result.to_dict()},
        )

    def _working_edges(self, *, graph_store: GraphStore, working_memory: WorkingMemory) -> list[EdgeRecord]:
        return [
            edge
            for edge_id in sorted(working_memory.working_edge_ids)
            if (edge := graph_store.get_edge_by_id(edge_id)) is not None
        ]

    def _infer_answer(self, *, query: str, working_edges: list[EdgeRecord]) -> tuple[str | None, list[str]]:
        lowered_query = query.lower()
        preferred_relations = []
        if "born" in lowered_query:
            preferred_relations.append("born")
        if "director" in lowered_query or "directed" in lowered_query:
            preferred_relations.append("direct")

        for keyword in preferred_relations:
            for edge in working_edges:
                if keyword in edge.relation.lower():
                    return edge.dst, [edge.edge_id]

        if working_edges:
            fallback_edge = working_edges[-1]
            return fallback_edge.dst, [fallback_edge.edge_id]
        return None, []
