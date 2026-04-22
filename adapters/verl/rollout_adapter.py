"""Minimal verl rollout adapter shell."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from adapters.verl.action_bridge import VerlActionBridge
from adapters.verl.batch_backend import BatchedGraphBackend
from adapters.verl.prompt_builder import VerlPromptBuilder
from adapters.verl.reward_bridge import VerlRewardBridge, VerlRewardOutput
from agent.graph_rollout_manager import compute_oracle_counterfactual_reward, reward_breakdown_from_info
from agent.rollout_types import RewardMode
from core.episode_result import CounterfactualComparisonResult, EvalResult, RewardBreakdown
from core.experiment_config import ExperimentConfig
from core.experiment_result import ExperimentResult, ExperimentStepTrace
from replay.snapshot_manager import SnapshotManager


VerlPolicyFn = Callable[[list[str], list[dict[str, Any]]], Sequence[Any]]


@dataclass(slots=True)
class VerlBatchStep:
    """One synchronous batched adapter step."""

    prompts: list[str]
    actions: list[Any]
    rewards: list[float]
    terminated: list[bool]
    truncated: list[bool]
    infos: list[dict[str, Any]] = field(default_factory=list)
    reward_mode: RewardMode = "baseline"
    base_rewards: list[float] = field(default_factory=list)
    reward_breakdowns: list[RewardBreakdown] = field(default_factory=list)
    reward_outputs: list[VerlRewardOutput] = field(default_factory=list)
    counterfactual_comparisons: list[CounterfactualComparisonResult | None] = field(default_factory=list)


@dataclass(slots=True)
class VerlRolloutResult:
    """Result returned by the minimal mock rollout adapter."""

    steps: list[VerlBatchStep] = field(default_factory=list)
    final_active_mask: list[bool] = field(default_factory=list)
    reward_mode: RewardMode = "baseline"

    @property
    def total_rewards(self) -> list[float]:
        """Return accumulated rewards per batch element."""

        if not self.steps:
            return []
        totals = [0.0] * len(self.steps[0].rewards)
        for step in self.steps:
            for idx, reward in enumerate(step.rewards):
                totals[idx] += reward
        return totals


class VerlRolloutAdapter:
    """Organize batch, prompt, action, and reward bridges for smoke rollouts."""

    def __init__(
        self,
        *,
        batch_backend: BatchedGraphBackend,
        prompt_builder: VerlPromptBuilder | None = None,
        action_bridge: VerlActionBridge | None = None,
        reward_bridge: VerlRewardBridge | None = None,
        snapshot_manager: SnapshotManager | None = None,
    ) -> None:
        self.batch_backend = batch_backend
        self.prompt_builder = prompt_builder or batch_backend.prompt_builder
        self.action_bridge = action_bridge or VerlActionBridge()
        self.reward_bridge = reward_bridge or VerlRewardBridge()
        self.snapshot_manager = snapshot_manager or SnapshotManager()

    def run_rollout(
        self,
        *,
        policy: VerlPolicyFn,
        max_steps: int | None = None,
        reward_mode: RewardMode = "baseline",
        config: ExperimentConfig | None = None,
    ) -> VerlRolloutResult:
        """Run a synchronous mock rollout with a caller-provided policy."""

        if config is None:
            config = ExperimentConfig(
                reward_mode=reward_mode,
                counterfactual_mode="replace",
                use_counterfactual_merge=reward_mode == "oracle_counterfactual",
                max_steps=max_steps,
            )
        else:
            reward_mode = config.reward_mode
            max_steps = config.max_steps if config.max_steps is not None else max_steps
        self._validate_reward_mode(reward_mode)
        observations, batched_state, _ = self.batch_backend.batch_reset()
        result = VerlRolloutResult(reward_mode=reward_mode)
        steps_used = 0

        while any(batched_state.active_mask):
            if max_steps is not None and steps_used >= max_steps:
                break
            prompts = self.batch_backend.batch_render_prompts()
            raw_actions = list(policy(prompts, observations))
            if len(raw_actions) != batched_state.batch_size:
                raise ValueError("policy must return one action per batch element.")
            candidate_ids = self.action_bridge.batch_to_candidate_ids(raw_actions)
            active_mask_before = list(batched_state.active_mask)
            pre_step_snapshots = [
                self.snapshot_manager.create_snapshot(env.state) if reward_mode == "oracle_counterfactual" and active else None
                for env, active in zip(self.batch_backend.envs, active_mask_before, strict=True)
            ]
            observations, rewards, terminated, truncated, infos, batched_state = self.batch_backend.batch_step(
                candidate_ids
            )
            reward_breakdowns: list[RewardBreakdown] = []
            reward_outputs: list[VerlRewardOutput] = []
            counterfactual_comparisons: list[CounterfactualComparisonResult | None] = []
            bridged_rewards: list[float] = []
            for idx, (env, candidate_id, reward, info, snapshot, was_active) in enumerate(
                zip(
                    self.batch_backend.envs,
                    candidate_ids,
                    rewards,
                    infos,
                    pre_step_snapshots,
                    active_mask_before,
                    strict=True,
                )
            ):
                if reward_mode == "oracle_counterfactual" and snapshot is not None and was_active:
                    breakdown, _, comparison = compute_oracle_counterfactual_reward(
                        env=env,
                        snapshot=snapshot,
                        candidate_id=candidate_id,
                        base_reward=reward,
                        info=info,
                        counterfactual_mode=config.resolved_counterfactual_mode,
                        use_counterfactual_merge=config.use_counterfactual_merge,
                    )
                    reward_output = self.reward_bridge.bridge(
                        breakdown,
                        comparison=comparison,
                        config=config,
                    )
                else:
                    breakdown = reward_breakdown_from_info(reward=reward, info=info)
                    comparison = None
                    reward_output = self.reward_bridge.bridge(breakdown, config=config)
                reward_breakdowns.append(breakdown)
                reward_outputs.append(reward_output)
                counterfactual_comparisons.append(comparison)
                bridged_rewards.append(reward_output.reward)
            result.steps.append(
                VerlBatchStep(
                    prompts=prompts,
                    actions=raw_actions,
                    rewards=bridged_rewards,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    reward_mode=reward_mode,
                    base_rewards=rewards,
                    reward_breakdowns=reward_breakdowns,
                    reward_outputs=reward_outputs,
                    counterfactual_comparisons=counterfactual_comparisons,
                )
            )
            steps_used += 1

        result.final_active_mask = list(batched_state.active_mask)
        return result

    def run_experiment(self, *, policy: VerlPolicyFn, config: ExperimentConfig | None = None) -> ExperimentResult:
        """Run a single-env adapter rollout and return a unified experiment result."""

        if len(self.batch_backend.envs) != 1:
            raise ValueError("VerlRolloutAdapter.run_experiment currently supports exactly one env.")
        config = config or ExperimentConfig()
        rollout = self.run_rollout(policy=policy, config=config)
        return _verl_rollout_to_experiment_result(rollout=rollout, config=config)

    def _validate_reward_mode(self, reward_mode: RewardMode) -> None:
        if reward_mode not in {"baseline", "oracle_counterfactual"}:
            raise ValueError("reward_mode must be either 'baseline' or 'oracle_counterfactual'.")


def _verl_rollout_to_experiment_result(*, rollout: VerlRolloutResult, config: ExperimentConfig) -> ExperimentResult:
    traces: list[ExperimentStepTrace] = []
    reward_summaries: list[dict[str, Any]] = []
    counterfactual_summaries: list[dict[str, Any]] = []

    for step_idx, step in enumerate(rollout.steps):
        base_reward = step.base_rewards[0] if step.base_rewards else step.rewards[0]
        breakdown = step.reward_breakdowns[0] if step.reward_breakdowns else None
        reward_output = step.reward_outputs[0] if step.reward_outputs else None
        reward = reward_output.reward if reward_output else step.rewards[0]
        comparison = step.counterfactual_comparisons[0] if step.counterfactual_comparisons else None
        trace = ExperimentStepTrace(
            step_idx=step_idx,
            action=step.actions[0],
            reward_mode=step.reward_mode,
            base_reward=base_reward,
            reward=reward,
            reward_breakdown=breakdown,
            counterfactual_reward=breakdown.counterfactual_reward if breakdown else 0.0,
            counterfactual_comparison=comparison,
            terminated=step.terminated[0],
            truncated=step.truncated[0],
            metadata={"metrics": reward_output.metrics if reward_output else {}, "info": step.infos[0]},
        )
        if config.record_step_traces:
            traces.append(trace)
        reward_summaries.append(
            {
                "step_idx": step_idx,
                "base_reward": base_reward,
                "reward": reward,
                "counterfactual_reward": trace.counterfactual_reward,
                "reward_breakdown": breakdown.to_dict() if breakdown else None,
            }
        )
        if trace.counterfactual_reward:
            counterfactual_summaries.append(
                {
                    "step_idx": step_idx,
                    "counterfactual_reward": trace.counterfactual_reward,
                    "metrics": reward_output.metrics if reward_output and config.record_counterfactual_metadata else None,
                }
            )

    base_total_reward = sum(step.base_rewards[0] if step.base_rewards else step.rewards[0] for step in rollout.steps)
    merged_counterfactual_reward_sum = (
        sum((step.reward_breakdowns[0].counterfactual_reward if step.reward_breakdowns else 0.0) for step in rollout.steps)
        if config.use_counterfactual_merge
        else 0.0
    )
    total_reward = base_total_reward + merged_counterfactual_reward_sum
    final_step = rollout.steps[-1] if rollout.steps else None
    final_info = final_step.infos[0] if final_step and final_step.infos else {}
    final_answer = final_info.get("answer")
    if final_answer is None:
        final_answer = (final_info.get("trajectory") or {}).get("final_answer")
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
        },
    )
