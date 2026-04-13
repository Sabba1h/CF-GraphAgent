"""Single-episode replay engine built on the existing TransitionEngine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from answer.answer_engine import AnswerEngine
from answer.evaluator import AnswerEvaluator
from candidates.generator import CandidateGenerator
from core.actions import ActionType, CandidateAction
from core.episode_result import TransitionResult
from core.snapshots import StateSnapshot
from core.state import GraphEpisodeState
from core.task import TaskSample
from core.transition import TransitionEngine
from graph.graph_backend import GraphBackend
from replay.snapshot_manager import SnapshotManager
from reward.reward_engine import RewardEngine


@dataclass(slots=True)
class ReplayStep:
    """One replayed transition."""

    snapshot_before: StateSnapshot
    action: int
    transition: TransitionResult

    @property
    def reward(self) -> float:
        """Return the replayed transition reward."""

        return self.transition.reward_result.reward


@dataclass(slots=True)
class ReplayResult:
    """Result of a synchronous single-episode replay."""

    initial_snapshot: StateSnapshot
    steps: list[ReplayStep] = field(default_factory=list)
    final_state: GraphEpisodeState | None = None

    @property
    def total_reward(self) -> float:
        """Return accumulated replay reward."""

        return sum(step.reward for step in self.steps)

    @property
    def final_answer(self) -> str | None:
        """Return the final answer if replay terminated by ANSWER."""

        if not self.steps:
            return None
        return self.steps[-1].transition.final_answer


class ReplayEngine:
    """Restore snapshots and continue them through TransitionEngine."""

    def __init__(
        self,
        *,
        graph_backend: GraphBackend,
        transition_engine: TransitionEngine | None = None,
        candidate_generator: CandidateGenerator | None = None,
        reward_engine: RewardEngine | None = None,
        answer_engine: AnswerEngine | None = None,
        answer_evaluator: AnswerEvaluator | None = None,
        snapshot_manager: SnapshotManager | None = None,
    ) -> None:
        self.graph_backend = graph_backend
        self.candidate_generator = candidate_generator or CandidateGenerator()
        self.reward_engine = reward_engine or RewardEngine()
        self.answer_engine = answer_engine or AnswerEngine()
        self.answer_evaluator = answer_evaluator or AnswerEvaluator()
        self.transition_engine = transition_engine or TransitionEngine(
            graph_backend=graph_backend,
            candidate_generator=self.candidate_generator,
            reward_engine=self.reward_engine,
            answer_engine=self.answer_engine,
            answer_evaluator=self.answer_evaluator,
        )
        self.snapshot_manager = snapshot_manager or SnapshotManager()

    def restore_state(self, snapshot: StateSnapshot) -> GraphEpisodeState:
        """Restore a minimal GraphEpisodeState from a StateSnapshot."""

        max_steps = snapshot.max_steps or snapshot.step_index + snapshot.steps_left
        state = GraphEpisodeState(
            task=TaskSample(query=snapshot.query, ground_truth=snapshot.ground_truth),
            max_steps=max_steps,
        )
        state.working_edge_ids = set(snapshot.working_edge_ids)
        state.visited_edge_ids = set(snapshot.visited_edge_ids)
        state.observed_edge_ids = set(snapshot.observed_edge_ids)
        state.visited_nodes = set(snapshot.visited_nodes)
        state.frontier_nodes = set(snapshot.frontier_nodes)
        state.steps_used = snapshot.step_index
        state.latest_candidate_actions = [self._candidate_from_dict(item) for item in snapshot.candidate_actions]
        if not state.latest_candidate_actions and state.steps_left > 0:
            state.set_latest_candidates(
                self.candidate_generator.generate(
                    query=state.query,
                    graph_store=self.graph_backend.graph_store,
                    working_memory=state,
                )
            )
        return state

    def replay(self, *, snapshot: StateSnapshot, actions: list[int | dict[str, int]]) -> ReplayResult:
        """Replay one or more specified candidate-id actions from a snapshot."""

        state = self.restore_state(snapshot)
        result = ReplayResult(initial_snapshot=snapshot, final_state=state)

        for action in actions:
            candidate_id = self._normalize_action(action)
            snapshot_before = self.snapshot_manager.create_snapshot(state)
            transition = self.transition_engine.step(state=state, candidate_id=candidate_id)
            result.steps.append(ReplayStep(snapshot_before=snapshot_before, action=candidate_id, transition=transition))
            if transition.terminated or transition.truncated:
                break

        result.final_state = state
        return result

    def replay_one_step(self, *, snapshot: StateSnapshot, action: int | dict[str, int]) -> ReplayResult:
        """Replay exactly one specified action unless it is invalid."""

        return self.replay(snapshot=snapshot, actions=[action])

    def find_candidate_id(self, snapshot: StateSnapshot, action_type: ActionType) -> int:
        """Find the current candidate id for an action type in the snapshot."""

        for candidate in snapshot.candidate_actions:
            if candidate.get("action_type") == action_type.value:
                return int(candidate["candidate_id"])
        raise ValueError(f"Snapshot does not contain an {action_type.value} candidate.")

    def _candidate_from_dict(self, payload: dict[str, Any]) -> CandidateAction:
        action_type = payload.get("action_type")
        return CandidateAction(
            candidate_id=int(payload["candidate_id"]),
            action_type=action_type if isinstance(action_type, ActionType) else ActionType(str(action_type)),
            description=str(payload.get("description", "")),
            edge_id=payload.get("edge_id"),
            score=float(payload.get("score", 0.0)),
            metadata=dict(payload.get("metadata", {})),
        )

    def _normalize_action(self, action: int | dict[str, int]) -> int:
        if type(action) is int:
            return action
        if isinstance(action, dict) and set(action.keys()) == {"candidate_id"} and type(action["candidate_id"]) is int:
            return action["candidate_id"]
        raise ValueError("ReplayEngine actions must be int or exactly {'candidate_id': int}; bool is not accepted.")
