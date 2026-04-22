"""Tests for HotpotQA graph-backed experiment smoke flow."""

from __future__ import annotations

from core.experiment_result import ExperimentResult
from data.benchmarks.hotpotqa import record_to_benchmark_example
from graph.hotpotqa_graph_runtime import (
    build_hotpotqa_graph_runtime,
    make_hotpotqa_graph_policy,
    run_hotpotqa_graph_experiment_for_example,
)


def _record() -> dict:
    return {
        "id": "hp-runtime-1",
        "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
        "answer": "yes",
        "type": "comparison",
        "level": "medium",
        "context": {
            "title": ["Scott Derrickson", "Ed Wood"],
            "sentences": [
                ["Scott Derrickson is an American film director."],
                ["Ed Wood was an American filmmaker."],
            ],
        },
        "supporting_facts": {
            "title": ["Scott Derrickson", "Ed Wood"],
            "sent_id": [0, 0],
        },
    }


def test_hotpotqa_graph_runtime_builds_env_and_aligned_task_sample() -> None:
    example = record_to_benchmark_example(_record(), split="validation")
    runtime = build_hotpotqa_graph_runtime(example, max_steps=3, candidate_top_k=4)

    assert runtime.local_graph.graph_id == "hotpotqa::hp-runtime-1"
    assert runtime.task_sample.metadata["graph_id"] == runtime.local_graph.graph_id
    assert runtime.task_sample.metadata["graph_stats"]["node_count"] == len(runtime.local_graph.nodes)
    assert len(runtime.graph_store.iter_node_ids()) == len(runtime.local_graph.nodes)
    assert len(runtime.graph_store.iter_edges()) == len(runtime.local_graph.edges)

    observation, info = runtime.env.reset()
    assert observation["query"] == example.question
    assert info["steps_left"] == 3
    assert any(candidate["action_type"] == "EXPAND_EDGE" for candidate in observation["candidate_actions"])


def test_hotpotqa_graph_policy_uses_only_candidate_action_types() -> None:
    policy = make_hotpotqa_graph_policy(min_expand_steps=1)
    observation = {
        "candidate_actions": [
            {"candidate_id": 7, "action_type": "EXPAND_EDGE", "metadata": {"privileged": "ignored"}},
            {"candidate_id": 8, "action_type": "ANSWER"},
            {"candidate_id": 9, "action_type": "STOP"},
        ]
    }

    assert policy(observation) == 7
    assert policy(observation) == 8


def test_hotpotqa_graph_baseline_experiment_smoke() -> None:
    example = record_to_benchmark_example(_record(), split="validation")
    runtime, result = run_hotpotqa_graph_experiment_for_example(
        example,
        reward_mode="baseline",
        max_steps=3,
        candidate_top_k=4,
        min_expand_steps=1,
    )

    assert isinstance(result, ExperimentResult)
    assert result.config.reward_mode == "baseline"
    assert result.metadata["graph_id"] == runtime.local_graph.graph_id
    assert result.metadata["question_id"] == "hp-runtime-1"
    assert result.metadata["graph_backed"] is True
    assert len(result.step_traces) >= 1
    assert result.base_total_reward == result.total_reward


def test_hotpotqa_graph_oracle_experiment_smoke() -> None:
    example = record_to_benchmark_example(_record(), split="validation")
    runtime, result = run_hotpotqa_graph_experiment_for_example(
        example,
        reward_mode="oracle_counterfactual",
        max_steps=3,
        candidate_top_k=4,
        min_expand_steps=1,
    )

    assert isinstance(result, ExperimentResult)
    assert result.config.reward_mode == "oracle_counterfactual"
    assert result.metadata["graph_id"] == runtime.local_graph.graph_id
    assert len(result.step_traces) >= 1
    assert any(trace.counterfactual_comparison is not None for trace in result.step_traces)


def test_supporting_facts_remain_metadata_only_not_edges() -> None:
    example = record_to_benchmark_example(_record(), split="validation")
    runtime = build_hotpotqa_graph_runtime(example, max_steps=3)

    relations = {edge.relation for edge in runtime.local_graph.edges}
    assert "supporting_fact" not in relations
    assert "supporting_facts" not in relations
    assert runtime.local_graph.metadata["supporting_facts"] == example.supporting_facts
