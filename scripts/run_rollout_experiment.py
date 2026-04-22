"""Run baseline and oracle counterfactual rollout experiments on the toy graph."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.graph_rollout_manager import GraphRolloutManager
from core.experiment_config import ExperimentConfig, RewardMode
from core.experiment_result import ExperimentResult
from data.toy_data import build_toy_graph, get_demo_task
from envs.cf_graph_env import CFGraphEnv


def build_env(max_steps: int = 5) -> CFGraphEnv:
    """Build the toy graph environment used by experiment smoke runs."""

    task = get_demo_task()
    return CFGraphEnv(
        graph_store=build_toy_graph(),
        query=task["query"],
        ground_truth=task["ground_truth"],
        max_steps=max_steps,
    )


def demo_policy(observation: dict) -> int:
    """Follow the toy evidence path, then answer."""

    for edge_id in ("e1", "e2"):
        for candidate in observation["candidate_actions"]:
            if candidate["action_type"] == "EXPAND_EDGE" and candidate.get("edge_id") == edge_id:
                if "already in working subgraph" not in candidate["description"]:
                    return candidate["candidate_id"]
    for candidate in observation["candidate_actions"]:
        if candidate["action_type"] == "ANSWER":
            return candidate["candidate_id"]
    return observation["candidate_actions"][-1]["candidate_id"]


def run_single_experiment(reward_mode: RewardMode) -> ExperimentResult:
    """Run one configured single-env rollout experiment."""

    config = ExperimentConfig(
        reward_mode=reward_mode,
        counterfactual_mode="replace",
        use_counterfactual_merge=reward_mode == "oracle_counterfactual",
        max_steps=5,
        record_step_traces=True,
        record_counterfactual_metadata=True,
    )
    return GraphRolloutManager().run_experiment(env=build_env(max_steps=5), policy=demo_policy, config=config)


def print_result(result: ExperimentResult) -> None:
    """Print a compact per-step reward summary."""

    print(f"=== Rollout Experiment: {result.config.reward_mode} ===")
    print(f"Final Answer: {result.final_answer}")
    print(f"Base Total Reward: {result.base_total_reward}")
    print(f"Total Reward: {result.total_reward}")
    for trace in result.step_traces:
        breakdown = trace.reward_breakdown
        print(f"\nStep {trace.step_idx + 1}")
        print(f"Action: {trace.action}")
        print(f"Base Reward: {trace.base_reward}")
        print(f"Reward: {trace.reward}")
        if breakdown is not None:
            print(
                "Breakdown: "
                f"task={breakdown.task_reward}, "
                f"process={breakdown.process_reward}, "
                f"constraint={breakdown.constraint_penalty}, "
                f"counterfactual={breakdown.counterfactual_reward}, "
                f"total={breakdown.total_reward}"
            )
        if trace.counterfactual_comparison is not None:
            print(f"Counterfactual Reward: {trace.counterfactual_reward}")
            print(f"Score Delta: {trace.counterfactual_comparison.score_delta}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run toy rollout experiments.")
    parser.add_argument(
        "--mode",
        choices=["baseline", "oracle_counterfactual", "both"],
        default="both",
        help="Which reward mode to run.",
    )
    args = parser.parse_args()

    modes = ["baseline", "oracle_counterfactual"] if args.mode == "both" else [args.mode]
    for index, mode in enumerate(modes):
        if index:
            print()
        print_result(run_single_experiment(mode))


if __name__ == "__main__":
    main()
