"""Print a detailed step-by-step trace for one stage-1 episode."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.toy_data import build_toy_graph, get_demo_task
from envs.cf_graph_env import CFGraphEnv


def choose_debug_action(observation: dict, expanded_steps: int) -> int:
    """Pick two expansions when possible, then answer."""

    candidates = observation["candidate_actions"]
    if expanded_steps < 2:
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
    print("=== Reset ===")
    print(info["text_observation"])

    terminated = False
    truncated = False
    expanded_steps = 0
    step_index = 0

    while not terminated and not truncated:
        print(f"\n=== Step {step_index + 1} Observation ===")
        print(f"Query: {observation['query']}")
        print(f"Working Summary: {observation['working_subgraph_summary']['text_summary']}")
        print(f"Frontier: {observation['frontier_nodes']}")
        print("Candidates:")
        for candidate in observation["candidate_actions"]:
            print(f"  [{candidate['candidate_id']}] {candidate['action_type']} | {candidate['description']}")

        action = choose_debug_action(observation, expanded_steps=expanded_steps)
        selected = next(item for item in observation["candidate_actions"] if item["candidate_id"] == action)
        if selected["action_type"] == "EXPAND_EDGE":
            expanded_steps += 1

        observation, reward, terminated, truncated, info = env.step({"candidate_id": action})
        step_index += 1

        print("Selected Action:")
        print(f"  [{selected['candidate_id']}] {selected['action_type']} | {selected['description']}")
        print(f"Reward: {reward}")
        print(f"Terminated: {terminated}")
        print(f"Truncated: {truncated}")
        print(f"Working Subgraph Summary: {info['working_subgraph_summary']['text_summary']}")

    print("\n=== Final Trajectory ===")
    trajectory = info["trajectory"]
    print(f"Termination: {trajectory['termination_reason']}")
    print(f"Final Answer: {trajectory['final_answer']}")
    print(f"Final Score: {trajectory['final_score']}")


if __name__ == "__main__":
    main()
