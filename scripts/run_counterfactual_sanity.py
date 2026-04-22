"""Run a small sanity check for oracle counterfactual reward signals."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.graph_rollout_manager import GraphRolloutManager
from core.experiment_config import CounterfactualMode, ExperimentConfig
from data.toy_data import build_toy_graph, get_demo_task
from envs.cf_graph_env import CFGraphEnv
from scripts.run_rollout_experiment import demo_policy


def build_env(max_steps: int = 5) -> CFGraphEnv:
    """Build the toy graph environment for sanity checks."""

    task = get_demo_task()
    return CFGraphEnv(
        graph_store=build_toy_graph(),
        query=task["query"],
        ground_truth=task["ground_truth"],
        max_steps=max_steps,
    )


def run_sanity(
    *,
    counterfactual_modes: list[CounterfactualMode] | None = None,
) -> dict[str, Any]:
    """Return step-wise oracle reward sanity statistics for each mode."""

    modes = counterfactual_modes or ["remove", "replace", "null"]
    baseline = GraphRolloutManager().run_experiment(
        env=build_env(),
        policy=demo_policy,
        config=ExperimentConfig(max_steps=5),
    )
    results: dict[str, Any] = {
        "baseline": {
            "base_total_reward": baseline.base_total_reward,
            "total_reward": baseline.total_reward,
            "final_answer": baseline.final_answer,
        },
        "modes": {},
    }

    for mode in modes:
        config = ExperimentConfig(
            reward_mode="oracle_counterfactual",
            counterfactual_mode=mode,
            use_counterfactual_merge=True,
            max_steps=5,
        )
        experiment = GraphRolloutManager().run_experiment(env=build_env(), policy=demo_policy, config=config)
        step_records = []
        deltas = []
        for trace in experiment.step_traces:
            comparison = trace.counterfactual_comparison
            if comparison is None:
                continue
            deltas.append(comparison.score_delta)
            step_records.append(
                {
                    "step_idx": trace.step_idx,
                    "original_score": comparison.original_score,
                    "counterfactual_score": comparison.counterfactual_score,
                    "score_delta": comparison.score_delta,
                    "counterfactual_reward": trace.counterfactual_reward,
                }
            )
        results["modes"][mode] = {
            "steps": step_records,
            "stats": summarize_deltas(deltas),
            "base_total_reward": experiment.base_total_reward,
            "total_reward": experiment.total_reward,
        }
    return results


def summarize_deltas(deltas: list[float]) -> dict[str, float | int]:
    """Return minimal distribution statistics for score deltas."""

    if not deltas:
        return {"count": 0, "nonzero_count": 0, "min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "count": len(deltas),
        "nonzero_count": sum(1 for delta in deltas if delta != 0.0),
        "min": min(deltas),
        "max": max(deltas),
        "mean": mean(deltas),
    }


def print_sanity_report(results: dict[str, Any]) -> None:
    """Print a compact human-readable sanity report."""

    baseline = results["baseline"]
    print("=== Counterfactual Sanity ===")
    print(f"Baseline Final Answer: {baseline['final_answer']}")
    print(f"Baseline Total Reward: {baseline['total_reward']}")
    for mode, payload in results["modes"].items():
        print(f"\nMode: {mode}")
        for step in payload["steps"]:
            print(
                f"Step {step['step_idx'] + 1}: "
                f"original={step['original_score']}, "
                f"counterfactual={step['counterfactual_score']}, "
                f"delta={step['score_delta']}, "
                f"cf_reward={step['counterfactual_reward']}"
            )
        stats = payload["stats"]
        print(
            "Stats: "
            f"count={stats['count']}, "
            f"nonzero={stats['nonzero_count']}, "
            f"min={stats['min']}, "
            f"max={stats['max']}, "
            f"mean={stats['mean']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run toy counterfactual reward sanity checks.")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["remove", "replace", "null"],
        default=["remove", "replace", "null"],
        help="Counterfactual modes to evaluate.",
    )
    args = parser.parse_args()
    print_sanity_report(run_sanity(counterfactual_modes=args.modes))


if __name__ == "__main__":
    main()
