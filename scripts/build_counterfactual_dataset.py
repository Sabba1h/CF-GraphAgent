"""Build a tiny JSONL counterfactual dataset from the toy rollout."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.graph_rollout_manager import GraphRolloutManager
from core.counterfactual_dataset import CounterfactualDatasetExample, examples_from_experiment_result, write_jsonl
from core.experiment_config import CounterfactualMode, ExperimentConfig, RewardMode
from data.toy_data import build_toy_graph, get_demo_task
from envs.cf_graph_env import CFGraphEnv
from scripts.run_rollout_experiment import demo_policy


def build_env(max_steps: int = 5) -> CFGraphEnv:
    """Build the toy graph environment for dataset export."""

    task = get_demo_task()
    return CFGraphEnv(
        graph_store=build_toy_graph(),
        query=task["query"],
        ground_truth=task["ground_truth"],
        max_steps=max_steps,
    )


def build_examples(
    *,
    reward_mode: RewardMode = "oracle_counterfactual",
    counterfactual_mode: CounterfactualMode = "replace",
) -> list[CounterfactualDatasetExample]:
    """Run one toy rollout and convert it into dataset examples."""

    config = ExperimentConfig(
        reward_mode=reward_mode,
        counterfactual_mode=counterfactual_mode,
        use_counterfactual_merge=reward_mode == "oracle_counterfactual",
        max_steps=5,
        record_step_traces=True,
        record_counterfactual_metadata=True,
    )
    result = GraphRolloutManager().run_experiment(env=build_env(), policy=demo_policy, config=config)
    return examples_from_experiment_result(
        result=result,
        task_id="toy_forrest_gump_director_birth_city",
        trajectory_id=f"toy_{reward_mode}_{counterfactual_mode}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build toy counterfactual dataset JSONL.")
    parser.add_argument(
        "--reward-mode",
        choices=["baseline", "oracle_counterfactual"],
        default="oracle_counterfactual",
    )
    parser.add_argument(
        "--counterfactual-mode",
        choices=["remove", "replace", "null", "mixed"],
        default="replace",
    )
    parser.add_argument(
        "--output",
        default="data/counterfactual_dataset.jsonl",
        help="Output JSONL path.",
    )
    args = parser.parse_args()

    examples = build_examples(reward_mode=args.reward_mode, counterfactual_mode=args.counterfactual_mode)
    output_path = write_jsonl(examples, args.output)
    print(f"Wrote {len(examples)} examples to {output_path}")


if __name__ == "__main__":
    main()
