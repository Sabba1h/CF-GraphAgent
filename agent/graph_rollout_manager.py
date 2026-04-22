"""Minimal synchronous rollout manager for one graph environment."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agent.graph_action_parser import GraphActionParser
from agent.rollout_types import RewardMode, RolloutResult, RolloutStep
from core.actions import ActionType
from core.episode_result import CounterfactualComparisonResult, EvalResult, RewardBreakdown, RewardResult
from core.experiment_config import CounterfactualMode, ExperimentConfig
from core.experiment_result import ExperimentResult, ExperimentStepTrace
from core.snapshots import StateSnapshot
from core.state import GraphEpisodeState
from replay.counterfactual_runner import CounterfactualRunner
from replay.replay_engine import ReplayEngine
from replay.snapshot_manager import SnapshotManager
from reward.reward_engine import RewardEngine


PolicyFn = Callable[[dict[str, Any]], Any]


class GraphRolloutManager:
    """Run a single-env, single-episode synchronous rollout."""

    def __init__(
        self,
        *,
        action_parser: GraphActionParser | None = None,
        reward_engine: RewardEngine | None = None,
        snapshot_manager: SnapshotManager | None = None,
    ) -> None:
        self.action_parser = action_parser or GraphActionParser()
        self.reward_engine = reward_engine or RewardEngine()
        self.snapshot_manager = snapshot_manager or SnapshotManager()

    def run_episode(
        self,
        *,
        env,
        policy: PolicyFn,
        max_steps: int | None = None,
        reward_mode: RewardMode = "baseline",
        use_counterfactual_merge: bool | None = None,
        counterfactual_mode: CounterfactualMode = "replace",
    ) -> RolloutResult:
        """Run one episode using a caller-provided synchronous policy."""

        self._validate_reward_mode(reward_mode)
        self._validate_counterfactual_mode(counterfactual_mode)
        should_merge_counterfactual = reward_mode == "oracle_counterfactual" if use_counterfactual_merge is None else use_counterfactual_merge
        observation, _ = env.reset()
        result = RolloutResult(reward_mode=reward_mode)
        terminated = False
        truncated = False
        steps_used = 0

        while not terminated and not truncated:
            if max_steps is not None and steps_used >= max_steps:
                break
            raw_action = policy(observation)
            parsed_action = self.action_parser.parse(raw_action)
            pre_step_snapshot = self.snapshot_manager.create_snapshot(env.state) if reward_mode == "oracle_counterfactual" else None
            next_observation, reward, terminated, truncated, info = env.step(parsed_action.candidate_id)
            reward_breakdown = reward_breakdown_from_info(reward=reward, info=info)
            counterfactual_comparison = None
            counterfactual_reward = 0.0
            rollout_reward = reward
            if reward_mode == "oracle_counterfactual" and pre_step_snapshot is not None:
                reward_breakdown, counterfactual_reward, counterfactual_comparison = compute_oracle_counterfactual_reward(
                    env=env,
                    snapshot=pre_step_snapshot,
                    candidate_id=parsed_action.candidate_id,
                    base_reward=reward,
                    info=info,
                    reward_engine=self.reward_engine,
                    counterfactual_mode=counterfactual_mode,
                    use_counterfactual_merge=should_merge_counterfactual,
                )
                rollout_reward = reward_breakdown.total_reward
            result.steps.append(
                RolloutStep(
                    observation=observation,
                    action=raw_action,
                    reward=rollout_reward,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                    reward_mode=reward_mode,
                    base_reward=reward,
                    reward_breakdown=reward_breakdown,
                    counterfactual_reward=counterfactual_reward,
                    counterfactual_comparison=counterfactual_comparison,
                )
            )
            observation = next_observation
            steps_used += 1

        return result

    def run_experiment(self, *, env, policy: PolicyFn, config: ExperimentConfig | None = None) -> ExperimentResult:
        """Run a single-env rollout and return a unified experiment result."""

        config = config or ExperimentConfig()
        rollout = self.run_episode(
            env=env,
            policy=policy,
            max_steps=config.max_steps,
            reward_mode=config.reward_mode,
            use_counterfactual_merge=config.use_counterfactual_merge,
            counterfactual_mode=config.resolved_counterfactual_mode,
        )
        return rollout_result_to_experiment_result(rollout=rollout, config=config)

    def _validate_reward_mode(self, reward_mode: RewardMode) -> None:
        if reward_mode not in {"baseline", "oracle_counterfactual"}:
            raise ValueError("reward_mode must be either 'baseline' or 'oracle_counterfactual'.")

    def _validate_counterfactual_mode(self, counterfactual_mode: CounterfactualMode) -> None:
        if counterfactual_mode not in {"remove", "replace", "null", "mixed"}:
            raise ValueError("counterfactual_mode must be one of {'remove', 'replace', 'null', 'mixed'}.")


def deterministic_suffix_policy(state: GraphEpisodeState) -> int | None:
    """Choose deterministic suffix actions until terminal for oracle replay."""

    expand_candidates = [
        candidate
        for candidate in state.latest_candidate_actions
        if candidate.action_type == ActionType.EXPAND_EDGE and candidate.edge_id not in state.working_edge_ids
    ]
    if expand_candidates:
        ranked = sorted(
            expand_candidates,
            key=lambda candidate: (
                -_query_relation_priority(query=state.query, candidate=candidate),
                -candidate.score,
                candidate.candidate_id,
            ),
        )
        return ranked[0].candidate_id

    for action_type in (ActionType.ANSWER, ActionType.STOP):
        for candidate in state.latest_candidate_actions:
            if candidate.action_type == action_type:
                return candidate.candidate_id
    return None


def compute_oracle_counterfactual_reward(
    *,
    env,
    snapshot: StateSnapshot,
    candidate_id: int,
    base_reward: float,
    info: dict[str, Any],
    reward_engine: RewardEngine | None = None,
    counterfactual_mode: CounterfactualMode = "replace",
    use_counterfactual_merge: bool = True,
) -> tuple[RewardBreakdown, float, CounterfactualComparisonResult]:
    """Compute opt-in oracle counterfactual reward for one rollout step."""

    reward_engine = reward_engine or RewardEngine()
    resolved_mode = "remove" if counterfactual_mode == "mixed" else counterfactual_mode
    replay_engine = ReplayEngine(graph_backend=env.graph_backend)
    runner = CounterfactualRunner(
        replay_engine=replay_engine,
        base_snapshot=snapshot,
        original_actions=[candidate_id],
    )
    comparisons = {
        "remove": runner.remove_action(0, suffix_policy=deterministic_suffix_policy),
        "null": runner.null_action(0, suffix_policy=deterministic_suffix_policy),
    }
    replacement_action = _replacement_action(snapshot=snapshot, candidate_id=candidate_id)
    if replacement_action is not None:
        comparisons["replace"] = runner.replace_action(
            0,
            replacement_action,
            suffix_policy=deterministic_suffix_policy,
        )

    primary_mode = resolved_mode if resolved_mode in comparisons else "null"
    primary_comparison = comparisons[primary_mode]
    primary_comparison.metadata["all_comparisons"] = {
        mode: comparison.to_dict() for mode, comparison in comparisons.items()
    }
    primary_comparison.metadata["requested_counterfactual_mode"] = counterfactual_mode
    primary_comparison.metadata["resolved_counterfactual_mode"] = resolved_mode
    base_reward_result = RewardResult(
        reward=base_reward,
        reason="rollout_env_reward",
        breakdown=reward_breakdown_from_info(reward=base_reward, info=info),
    )
    counterfactual_reward = reward_engine.compute_counterfactual_bonus(primary_comparison)
    if use_counterfactual_merge:
        merged_reward = reward_engine.merge_counterfactual_reward(
            base_reward=base_reward_result,
            comparison=primary_comparison,
        )
        return merged_reward.breakdown, merged_reward.counterfactual_bonus, primary_comparison

    base_breakdown = base_reward_result.breakdown or RewardBreakdown(total_reward=base_reward)
    breakdown = RewardBreakdown(
        task_reward=base_breakdown.task_reward,
        process_reward=base_breakdown.process_reward,
        constraint_penalty=base_breakdown.constraint_penalty,
        counterfactual_reward=counterfactual_reward,
        total_reward=base_breakdown.total_reward,
    )
    return breakdown, counterfactual_reward, primary_comparison


def reward_breakdown_from_info(*, reward: float, info: dict[str, Any]) -> RewardBreakdown:
    """Build RewardBreakdown from env info without changing reward semantics."""

    payload = info.get("reward_breakdown") or {}
    return RewardBreakdown(
        task_reward=float(payload.get("task_reward", 0.0)),
        process_reward=float(payload.get("process_reward", 0.0)),
        constraint_penalty=float(payload.get("constraint_penalty", 0.0)),
        counterfactual_reward=float(payload.get("counterfactual_reward", 0.0)),
        total_reward=float(payload.get("total_reward", reward)),
    )


def _replacement_action(*, snapshot: StateSnapshot, candidate_id: int) -> int | None:
    original = None
    for candidate in snapshot.candidate_actions:
        if int(candidate["candidate_id"]) == candidate_id:
            original = candidate
            break
    preferred_action = ActionType.STOP.value
    if original is not None and original.get("action_type") == ActionType.STOP.value:
        preferred_action = ActionType.ANSWER.value
    for candidate in snapshot.candidate_actions:
        if candidate.get("action_type") == preferred_action and int(candidate["candidate_id"]) != candidate_id:
            return int(candidate["candidate_id"])
    return None


def _query_relation_priority(*, query: str, candidate) -> float:
    relation = str(candidate.metadata.get("relation", "")).lower()
    query_lower = query.lower()
    priority = 0.0
    if "born" in query_lower and "born" in relation:
        priority += 10.0
    if ("director" in query_lower or "directed" in query_lower) and "direct" in relation:
        priority += 5.0
    return priority


def rollout_result_to_experiment_result(*, rollout: RolloutResult, config: ExperimentConfig) -> ExperimentResult:
    """Convert a rollout result into the unified experiment result format."""

    traces: list[ExperimentStepTrace] = []
    reward_summaries: list[dict[str, Any]] = []
    counterfactual_summaries: list[dict[str, Any]] = []

    for step_idx, step in enumerate(rollout.steps):
        base_reward = step.base_reward if step.base_reward is not None else step.reward
        trace = ExperimentStepTrace(
            step_idx=step_idx,
            action=step.action,
            reward_mode=step.reward_mode,
            base_reward=base_reward,
            reward=step.reward,
            reward_breakdown=step.reward_breakdown,
            counterfactual_reward=step.counterfactual_reward,
            counterfactual_comparison=step.counterfactual_comparison if config.record_counterfactual_metadata else None,
            terminated=step.terminated,
            truncated=step.truncated,
            metadata={"info": step.info, "observation": step.observation} if config.record_step_traces else {},
        )
        if config.record_step_traces:
            traces.append(trace)
        reward_summaries.append(
            {
                "step_idx": step_idx,
                "base_reward": base_reward,
                "reward": step.reward,
                "counterfactual_reward": step.counterfactual_reward,
                "reward_breakdown": step.reward_breakdown.to_dict() if step.reward_breakdown else None,
            }
        )
        if step.counterfactual_comparison is not None:
            counterfactual_summaries.append(
                {
                    "step_idx": step_idx,
                    "counterfactual_reward": step.counterfactual_reward,
                    "comparison": step.counterfactual_comparison.to_dict()
                    if config.record_counterfactual_metadata
                    else None,
                }
            )

    base_total_reward = rollout.base_total_reward
    merged_counterfactual_reward_sum = (
        sum(step.counterfactual_reward for step in rollout.steps) if config.use_counterfactual_merge else 0.0
    )
    total_reward = base_total_reward + merged_counterfactual_reward_sum
    final_step = rollout.steps[-1] if rollout.steps else None
    final_info = final_step.info if final_step else {}
    final_answer = final_info.get("answer")
    if final_answer is None:
        trajectory = final_info.get("trajectory") or {}
        final_answer = trajectory.get("final_answer")
    final_eval = None
    if "eval_score" in final_info:
        final_eval = EvalResult(
            score=float(final_info["eval_score"]),
            is_correct=final_info.get("is_correct"),
            reason=str(final_info.get("eval_reason", "answer_eval")),
            details={"final_answer": final_answer},
        )

    return ExperimentResult(
        config=config,
        final_answer=final_answer,
        final_eval=final_eval,
        total_reward=total_reward,
        base_total_reward=base_total_reward,
        step_traces=traces,
        reward_summaries=reward_summaries,
        counterfactual_summaries=counterfactual_summaries,
        metadata={
            "rollout_reward_mode": rollout.reward_mode,
            "merged_counterfactual_reward_sum": merged_counterfactual_reward_sum,
            "query": traces[0].metadata.get("observation", {}).get("query", "") if traces else "",
        },
    )
