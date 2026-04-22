"""Reward logic for the stage-1 graph environment."""

from __future__ import annotations

from reward.counterfactual import CounterfactualRewardEngine
from schemas import AnswerResult, CounterfactualComparisonResult, EvalResult, RewardBreakdown, RewardResult


class RewardEngine:
    """Encapsulate stage-1 reward rules and future extension hooks."""

    def __init__(self, *, counterfactual_reward_engine: CounterfactualRewardEngine | None = None) -> None:
        self.counterfactual_reward_engine = counterfactual_reward_engine or CounterfactualRewardEngine()

    def reward_for_expand(self, *, is_valid: bool, is_repeated: bool) -> RewardResult:
        """Return the reward for an expand action."""

        if not is_valid:
            return self._result(total=-1.0, reason="invalid_action", constraint_penalty=-1.0)
        if is_repeated:
            return self._result(total=-0.2, reason="repeated_expand", constraint_penalty=-0.2)
        return self._result(total=0.1, reason="valid_expand", process_reward=0.1)

    def reward_for_stop(self) -> RewardResult:
        """Return the reward for stop."""

        return self._result(total=-0.5, reason="stop", constraint_penalty=-0.5)

    def reward_for_answer(self, eval_result: EvalResult | AnswerResult) -> RewardResult:
        """Return the reward for answer based on the evaluator result."""

        score = eval_result.score if isinstance(eval_result, EvalResult) else eval_result.reward
        return self._result(total=score, reason="answer", task_reward=score)

    def compute_counterfactual_bonus(self, comparison: CounterfactualComparisonResult | None = None) -> float:
        """Return a counterfactual delta only when explicitly requested."""

        if comparison is None:
            return 0.0
        return self.counterfactual_reward_engine.compute(comparison).counterfactual_reward

    def merge_counterfactual_reward(
        self,
        *,
        base_reward: RewardResult,
        comparison: CounterfactualComparisonResult,
    ) -> RewardResult:
        """Return a new RewardResult with an opt-in counterfactual delta."""

        counterfactual_result = self.counterfactual_reward_engine.compute(comparison)
        base_breakdown = base_reward.breakdown or RewardBreakdown(total_reward=base_reward.reward)
        breakdown = RewardBreakdown(
            task_reward=base_breakdown.task_reward,
            process_reward=base_breakdown.process_reward,
            constraint_penalty=base_breakdown.constraint_penalty,
            counterfactual_reward=counterfactual_result.counterfactual_reward,
            total_reward=base_breakdown.total_reward + counterfactual_result.counterfactual_reward,
        )
        return RewardResult(
            reward=breakdown.total_reward,
            reason=f"{base_reward.reason}+counterfactual",
            counterfactual_bonus=counterfactual_result.counterfactual_reward,
            breakdown=breakdown,
        )

    def compute_counterfactual_placeholder(self) -> float:
        """Compatibility placeholder for callers that only need zero by default."""

        return 0.0

    def _result(
        self,
        *,
        total: float,
        reason: str,
        task_reward: float = 0.0,
        process_reward: float = 0.0,
        constraint_penalty: float = 0.0,
    ) -> RewardResult:
        breakdown = RewardBreakdown(
            task_reward=task_reward,
            process_reward=process_reward,
            constraint_penalty=constraint_penalty,
            counterfactual_reward=0.0,
            total_reward=total,
        )
        return RewardResult(reward=total, reason=reason, breakdown=breakdown)
