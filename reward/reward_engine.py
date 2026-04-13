"""Reward logic for the stage-1 graph environment."""

from __future__ import annotations

from schemas import AnswerResult, EvalResult, RewardBreakdown, RewardResult


class RewardEngine:
    """Encapsulate stage-1 reward rules and future extension hooks."""

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

    def compute_counterfactual_bonus(self, *args, **kwargs) -> float:
        """Placeholder for stage-2 counterfactual reward extensions."""

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
