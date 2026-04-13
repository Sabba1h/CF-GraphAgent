"""Run a minimal stage-1 episode on the toy graph."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.toy_data import build_toy_graph, get_demo_task
from envs.cf_graph_env import CFGraphEnv


def select_action(observation: dict, expand_budget_used: int) -> int:
    """Use a deterministic stage-1 policy for the demo."""

    candidates = observation["candidate_actions"]
    if expand_budget_used < 2:
        for candidate in candidates:
            if candidate["action_type"] == "EXPAND_EDGE" and "already in working subgraph" not in candidate["description"]:
                return candidate["candidate_id"]
    for candidate in candidates:
        if candidate["action_type"] == "ANSWER":
            return candidate["candidate_id"]
    return candidates[-1]["candidate_id"]


def main() -> None:
    task = get_demo_task()
    env = CFGraphEnv(graph_store=build_toy_graph(), query=task["query"], ground_truth=task["ground_truth"], max_steps=5)
    observation, info = env.reset()
    print("=== Stage-1 Demo ===")
    print(info["text_observation"])

    terminated = False
    truncated = False
    expand_budget_used = 0
    step_index = 0

    while not terminated and not truncated:
        action = select_action(observation, expand_budget_used=expand_budget_used)
        selected = next(item for item in observation["candidate_actions"] if item["candidate_id"] == action)
        if selected["action_type"] == "EXPAND_EDGE":
            expand_budget_used += 1
        observation, reward, terminated, truncated, info = env.step(action)
        step_index += 1
        print(f"\nStep {step_index}")
        print(f"Selected: [{selected['candidate_id']}] {selected['action_type']} | {selected['description']}")
        print(f"Reward: {reward}")
        print(f"Terminated: {terminated} | Truncated: {truncated}")
        print(f"Working Subgraph: {info['working_subgraph_summary']['text_summary']}")

    print("\n=== Episode Summary ===")
    print(f"Termination: {info['trajectory']['termination_reason']}")
    print(f"Final Answer: {info['trajectory']['final_answer']}")
    print(f"Final Score: {info['trajectory']['final_score']}")


if __name__ == "__main__":
    main()
