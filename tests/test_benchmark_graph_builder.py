"""Tests for benchmark local graph construction."""

from __future__ import annotations

from data.benchmarks.hotpotqa import record_to_benchmark_example as hotpot_record_to_example
from data.benchmarks.twowiki import record_to_benchmark_example as twowiki_record_to_example
from graph.benchmark_graph_builder import build_from_benchmark_example
from graph.benchmark_graph_loader import compute_graph_stats


def _hotpot_columnar_record() -> dict:
    return {
        "id": "hp-graph-1",
        "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
        "answer": "yes",
        "type": "comparison",
        "level": "medium",
        "context": {
            "title": ["Scott Derrickson", "Ed Wood"],
            "sentences": [
                ["Scott Derrickson is an American director.", "He directed Doctor Strange."],
                ["Ed Wood was an American filmmaker."],
            ],
        },
        "supporting_facts": {
            "title": ["Scott Derrickson", "Ed Wood"],
            "sent_id": [0, 0],
        },
    }


def _twowiki_record() -> dict:
    return {
        "_id": "tw-graph-1",
        "question": "What is the capital of the country where Example Person was born?",
        "answer": "London",
        "type": "compositional",
        "answer_type": "span",
        "evidences": [["Example Person", "born in", "England"], ["England", "capital", "London"]],
        "supporting_facts": [["Example Person", 0], ["England", 1]],
        "context": [
            ["Example Person", ["Example Person was born in England."]],
            ["England", ["London is the capital of England.", "It is a large city."]],
        ],
    }


def test_hotpotqa_fixture_builds_conservative_local_graph() -> None:
    example = hotpot_record_to_example(_hotpot_columnar_record(), split="validation")
    local_graph = build_from_benchmark_example(example)
    stats = compute_graph_stats(local_graph)

    assert local_graph.graph_id == "hotpotqa::hp-graph-1"
    assert stats["node_type_counts"] == {"question": 1, "sentence": 3, "title": 2}
    assert stats["relation_counts"]["question_to_title"] == 2
    assert stats["relation_counts"]["title_to_sentence"] == 3
    assert stats["relation_counts"]["sentence_to_title"] == 3
    assert stats["relation_counts"]["next_sentence"] == 1
    assert local_graph.metadata["supporting_facts"] == example.supporting_facts
    assert local_graph.metadata["privileged_edges_injected"] is False
    assert all("support" not in edge.relation for edge in local_graph.edges)


def test_twowiki_fixture_builds_conservative_local_graph() -> None:
    example = twowiki_record_to_example(_twowiki_record(), split="dev")
    local_graph = build_from_benchmark_example(example)
    stats = compute_graph_stats(local_graph)

    assert local_graph.graph_id == "twowiki::tw-graph-1"
    assert stats["node_type_counts"] == {"question": 1, "sentence": 3, "title": 2}
    assert stats["relation_counts"]["question_to_title"] == 2
    assert stats["relation_counts"]["next_sentence"] == 1
    assert local_graph.metadata["benchmark_metadata"]["evidences"] == [
        ["Example Person", "born in", "England"],
        ["England", "capital", "London"],
    ]
    assert all("support" not in edge.relation for edge in local_graph.edges)


def test_question_connects_to_all_titles_without_lexical_filtering() -> None:
    example = hotpot_record_to_example(_hotpot_columnar_record())
    local_graph = build_from_benchmark_example(example)
    question_edges = [edge for edge in local_graph.edges if edge.relation == "question_to_title"]

    title_nodes = [node for node in local_graph.nodes if node.node_type == "title"]
    assert len(question_edges) == len(title_nodes)
    assert {edge.dst for edge in question_edges} == {node.node_id for node in title_nodes}
