"""State transition engine for the reference graph environment."""

from __future__ import annotations

from typing import Any

from answer.answer_engine import AnswerEngine
from answer.evaluator import AnswerEvaluator
from candidates.generator import CandidateGenerator
from core.actions import ActionType, CandidateAction
from core.episode_result import TransitionResult
from core.state import GraphEpisodeState
from core.task import TaskSample
from graph.graph_backend import GraphBackend
from reward.reward_engine import RewardEngine


class TransitionEngine:
    """Move the existing stage-1 reset/step logic out of CFGraphEnv."""

    def __init__(
        self,
        *,
        graph_backend: GraphBackend,
        candidate_generator: CandidateGenerator,
        reward_engine: RewardEngine,
        answer_engine: AnswerEngine,
        answer_evaluator: AnswerEvaluator,
    ) -> None:
        self.graph_backend = graph_backend
        self.candidate_generator = candidate_generator
        self.reward_engine = reward_engine
        self.answer_engine = answer_engine
        self.answer_evaluator = answer_evaluator

    def reset(self, *, task: TaskSample, max_steps: int) -> tuple[GraphEpisodeState, list[str]]:
        """Initialize episode state and first candidate set."""

        state = GraphEpisodeState(task=task, max_steps=max_steps)
        seed_nodes = task.seed_entities or self.candidate_generator.find_seed_nodes(
            task.query,
            self.graph_backend.graph_store,
        )
        state.initialize_frontier(seed_nodes)
        initial_candidates = self.candidate_generator.generate(
            query=state.query,
            graph_store=self.graph_backend.graph_store,
            working_memory=state,
        )
        state.set_latest_candidates(initial_candidates)
        return state, seed_nodes

    def step(self, *, state: GraphEpisodeState, candidate_id: int) -> TransitionResult:
        """Apply one selected candidate_id and advance state."""

        current_candidates = [item.to_dict() for item in state.latest_candidate_actions]
        candidate = self._get_candidate(state=state, candidate_id=candidate_id)

        terminated = False
        truncated = False
        info: dict[str, Any] = {"candidate_id": candidate_id}
        final_answer: str | None = None
        termination_reason: str | None = None

        state.increment_step()

        if candidate is None:
            reward_result = self.reward_engine.reward_for_expand(is_valid=False, is_repeated=False)
            info["error"] = f"Invalid candidate_id {candidate_id}. Available ids: {self._candidate_ids(state)}"
            selected_action = {"candidate_id": candidate_id, "action_type": "INVALID"}
        elif candidate.action_type == ActionType.EXPAND_EDGE:
            reward_result, info = self._handle_expand(state=state, candidate=candidate, info=info)
            selected_action = candidate.to_dict()
        elif candidate.action_type == ActionType.ANSWER:
            answer_result = self.answer_engine.answer(
                query=state.query,
                graph_store=self.graph_backend.graph_store,
                working_memory=state,
            )
            eval_result = self.answer_evaluator.evaluate(
                predicted_answer=answer_result.answer,
                ground_truth=state.ground_truth,
            )
            reward_result = self.reward_engine.reward_for_answer(eval_result)
            terminated = True
            final_answer = answer_result.answer
            termination_reason = "answer"
            info.update(
                {
                    "answer": answer_result.answer,
                    "answer_reasoning": answer_result.reasoning,
                    "eval_score": eval_result.score,
                    "eval_reason": eval_result.reason,
                    "is_correct": eval_result.is_correct,
                    "evidence_edge_ids": answer_result.evidence_edge_ids,
                }
            )
            selected_action = candidate.to_dict()
        elif candidate.action_type == ActionType.STOP:
            reward_result = self.reward_engine.reward_for_stop()
            terminated = True
            termination_reason = "stop"
            selected_action = candidate.to_dict()
        else:
            reward_result = self.reward_engine.reward_for_expand(is_valid=False, is_repeated=False)
            info["error"] = f"Unsupported action type {candidate.action_type.value} in stage 1."
            selected_action = candidate.to_dict()

        state.add_action_record(
            {
                "step_index": state.steps_used,
                "candidate_id": candidate_id,
                "action_type": selected_action["action_type"],
                "description": selected_action.get("description", info.get("error", "")),
                "reward": reward_result.reward,
                "reward_reason": reward_result.reason,
            }
        )

        if not terminated and state.steps_left <= 0:
            truncated = True
            termination_reason = "max_steps"

        if not terminated and not truncated:
            next_candidates = self.candidate_generator.generate(
                query=state.query,
                graph_store=self.graph_backend.graph_store,
                working_memory=state,
            )
            state.set_latest_candidates(next_candidates)
        else:
            state.set_latest_candidates([])

        state.final_answer = final_answer
        state.termination_reason = termination_reason
        info["reward_breakdown"] = reward_result.breakdown.to_dict() if reward_result.breakdown else None

        return TransitionResult(
            candidate_id=candidate_id,
            candidate_actions=current_candidates,
            selected_action=selected_action,
            reward_result=reward_result,
            terminated=terminated,
            truncated=truncated,
            info=info,
            final_answer=final_answer,
            termination_reason=termination_reason,
        )

    def _get_candidate(self, *, state: GraphEpisodeState, candidate_id: int) -> CandidateAction | None:
        for candidate in state.latest_candidate_actions:
            if candidate.candidate_id == candidate_id:
                return candidate
        return None

    def _candidate_ids(self, state: GraphEpisodeState) -> list[int]:
        return [candidate.candidate_id for candidate in state.latest_candidate_actions]

    def _handle_expand(
        self,
        *,
        state: GraphEpisodeState,
        candidate: CandidateAction,
        info: dict[str, Any],
    ):
        edge_id = candidate.edge_id
        if edge_id is None:
            info["error"] = "Expand candidate is missing edge_id."
            return self.reward_engine.reward_for_expand(is_valid=False, is_repeated=False), info

        edge = self.graph_backend.get_edge_by_id(edge_id)
        if edge is None:
            info["error"] = f"Edge '{edge_id}' does not exist."
            return self.reward_engine.reward_for_expand(is_valid=False, is_repeated=False), info

        is_repeated = edge.edge_id in state.working_edge_ids
        if not is_repeated:
            state.accept_edge(edge)
            info["expanded_edge_id"] = edge.edge_id
            info["expanded_edge"] = {
                "edge_id": edge.edge_id,
                "src": edge.src,
                "relation": edge.relation,
                "dst": edge.dst,
            }
        else:
            info["warning"] = f"Edge '{edge.edge_id}' is already in the working subgraph."
        return self.reward_engine.reward_for_expand(is_valid=True, is_repeated=is_repeated), info
