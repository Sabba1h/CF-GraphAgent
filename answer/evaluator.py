"""Answer evaluation utilities."""

from __future__ import annotations

from core.episode_result import EvalResult


class AnswerEvaluator:
    """Evaluate predicted answers with a stable, replaceable interface."""

    def evaluate(self, *, predicted_answer: str, ground_truth: str | None) -> EvalResult:
        """Score an answer using the current stage-1 exact-match rule."""

        if ground_truth is None:
            score = 0.2 if predicted_answer != "UNKNOWN" else 0.0
            return EvalResult(
                score=score,
                is_correct=None,
                reason="no_ground_truth_non_unknown" if score > 0 else "no_ground_truth_unknown",
                details={"predicted_answer": predicted_answer, "ground_truth": ground_truth},
            )

        is_correct = predicted_answer.lower() == ground_truth.lower()
        return EvalResult(
            score=1.0 if is_correct else 0.0,
            is_correct=is_correct,
            reason="exact_match" if is_correct else "exact_mismatch",
            details={"predicted_answer": predicted_answer, "ground_truth": ground_truth},
        )
