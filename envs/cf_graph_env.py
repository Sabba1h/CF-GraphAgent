"""Gymnasium-style environment for stage-1 Agentic Graph RAG."""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from answer.answer_engine import AnswerEngine
from answer.evaluator import AnswerEvaluator
from candidates.generator import CandidateGenerator
from core.state import GraphEpisodeState
from core.task import TaskSample
from core.transition import TransitionEngine
from graph.graph_backend import GraphBackend
from graph.graph_store import GraphStore
from memory.trajectory_logger import TrajectoryLogger
from memory.working_memory import WorkingMemory
from observation.renderer import ObservationRenderer
from reward.reward_engine import RewardEngine
from schemas import ObservationDict


class CFGraphEnv(gym.Env):
    """Minimal stage-1 environment with structured state and candidate actions."""

    metadata = {"render_modes": ["text"]}

    def __init__(
        self,
        *,
        graph_store: GraphStore,
        query: str | None = None,
        ground_truth: str | None = None,
        max_steps: int = 4,
        candidate_top_k: int = 5,
    ) -> None:
        super().__init__()
        self.graph_store = graph_store
        self.graph_backend = GraphBackend(graph_store)
        self.default_query = query
        self.default_ground_truth = ground_truth
        self.max_steps = max_steps
        self.query: str | None = None
        self.ground_truth: str | None = None

        self.candidate_generator = CandidateGenerator(top_k=candidate_top_k)
        self.observation_renderer = ObservationRenderer()
        self.reward_engine = RewardEngine()
        self.answer_engine = AnswerEngine()
        self.answer_evaluator = AnswerEvaluator()
        self.transition_engine = TransitionEngine(
            graph_backend=self.graph_backend,
            candidate_generator=self.candidate_generator,
            reward_engine=self.reward_engine,
            answer_engine=self.answer_engine,
            answer_evaluator=self.answer_evaluator,
        )
        self.trajectory_logger = TrajectoryLogger()

        self.state: GraphEpisodeState | None = None
        self.working_memory: WorkingMemory | None = None
        self.last_observation: ObservationDict | None = None
        self.last_text_observation: str = ""

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObservationDict, dict[str, Any]]:
        """Reset the environment state and generate the first candidate set."""

        super().reset(seed=seed)
        options = options or {}
        query = options.get("query", self.default_query)
        ground_truth = options.get("ground_truth", self.default_ground_truth)
        if not query:
            raise ValueError("CFGraphEnv.reset() requires a query either in the constructor or reset options.")

        self.query = str(query)
        self.ground_truth = ground_truth
        task = TaskSample(
            query=self.query,
            ground_truth=self.ground_truth,
            seed_entities=list(options.get("seed_entities", [])),
        )
        self.state, seed_nodes = self.transition_engine.reset(task=task, max_steps=self.max_steps)
        self._sync_working_memory_view()

        self.trajectory_logger.reset()
        self.trajectory_logger.start_episode(query=self.query, ground_truth=self.ground_truth)

        observation = self._render_observation()
        info = {
            "seed_nodes": seed_nodes,
            "steps_left": self.state.steps_left,
            "text_observation": self.last_text_observation,
            "trajectory": self.trajectory_logger.as_dict(),
        }
        return observation, info

    def step(self, action: int | dict[str, int]) -> tuple[ObservationDict, float, bool, bool, dict[str, Any]]:
        """Apply one selected candidate_id and advance the environment."""

        if self.query is None or self.state is None:
            raise RuntimeError("CFGraphEnv.reset() must be called before step().")

        candidate_id, action_payload = self._normalize_action(action)
        del action_payload
        transition = self.transition_engine.step(state=self.state, candidate_id=candidate_id)
        self._sync_working_memory_view()

        observation = self._render_observation()
        working_summary = self.graph_backend.export_subgraph_summary(self.state.working_edge_ids)
        info = transition.info
        info.update(
            {
                "steps_left": self.state.steps_left,
                "working_subgraph_summary": working_summary,
                "text_observation": self.last_text_observation,
            }
        )

        self.trajectory_logger.log_step(
            step_index=self.state.steps_used,
            candidate_actions=transition.candidate_actions,
            selected_action=transition.selected_action,
            reward=transition.reward_result.reward,
            reward_reason=transition.reward_result.reason,
            working_subgraph_summary=working_summary,
            terminated=transition.terminated,
            truncated=transition.truncated,
            info=info,
        )

        if transition.terminated or transition.truncated:
            final_score = sum(step.reward for step in self.trajectory_logger.current_summary().steps)
            self.state.final_score = final_score
            self.trajectory_logger.finalize(
                termination_reason=transition.termination_reason or "terminated",
                final_answer=transition.final_answer,
                final_score=final_score,
            )
        info["trajectory"] = self.trajectory_logger.as_dict()
        return observation, transition.reward_result.reward, transition.terminated, transition.truncated, info

    def render(self) -> str:
        """Return the latest text observation."""

        return self.last_text_observation

    def _normalize_action(self, action: int | dict[str, int]) -> tuple[int, dict[str, int] | None]:
        if type(action) is int:
            return action, None
        if isinstance(action, dict):
            if set(action.keys()) != {"candidate_id"}:
                raise ValueError("step(action) dict input must be exactly {'candidate_id': int}.")
            candidate_id = action["candidate_id"]
            if type(candidate_id) is not int:
                raise ValueError("step(action) candidate_id must be an int and must not be bool.")
            return candidate_id, action
        raise ValueError("step(action) only accepts int or exactly {'candidate_id': int}; bool is not accepted.")

    def _render_observation(self) -> ObservationDict:
        if self.query is None or self.state is None:
            raise RuntimeError("Environment is not initialized.")
        structured = self.observation_renderer.render_structured_observation(
            query=self.query,
            graph_store=self.graph_store,
            working_memory=self.state,
        )
        self.last_observation = structured
        self.last_text_observation = self.observation_renderer.render_text_observation(
            query=self.query,
            graph_store=self.graph_store,
            working_memory=self.state,
        )
        return structured

    def _sync_working_memory_view(self) -> None:
        """Expose a compatibility WorkingMemory view derived from state."""

        if self.state is None:
            self.working_memory = None
            return
        view = WorkingMemory(max_steps=self.state.max_steps)
        view.working_edge_ids = self.state.working_edge_ids
        view.visited_edge_ids = self.state.visited_edge_ids
        view.observed_edge_ids = self.state.observed_edge_ids
        view.visited_nodes = self.state.visited_nodes
        view.frontier_nodes = self.state.frontier_nodes
        view.action_history = self.state.action_history
        view.steps_used = self.state.steps_used
        view.latest_candidate_actions = self.state.latest_candidate_actions
        self.working_memory = view
