"""Tests for benchmark local graph loading into GraphStore."""

from __future__ import annotations

import json

from data.benchmarks.hotpotqa import record_to_benchmark_example
from graph.benchmark_graph_builder import build_from_benchmark_example
from graph.benchmark_graph_loader import (
    compute_graph_stats,
    load_local_graph_json,
    local_graph_to_graph_store,
    save_local_graph_json,
)
from graph.benchmark_graph_types import BenchmarkLocalGraph
from graph.graph_store import GraphStore


def _record() -> dict:
    return {
        "id": "hp-loader-1",
        "question": "Who directed the film?",
        "answer": "Jane Doe",
        "context": {
            "title": ["Film Article"],
            "sentences": [["The film was directed by Jane Doe.", "It was released in 2001."]],
        },
        "supporting_facts": {"title": ["Film Article"], "sent_id": [0]},
    }


def test_local_graph_is_json_serializable_and_round_trips(tmp_path) -> None:
    local_graph = build_from_benchmark_example(record_to_benchmark_example(_record()))

    payload = local_graph.to_dict()
    json.dumps(payload)
    assert BenchmarkLocalGraph.from_dict(payload).to_dict() == payload

    output_path = save_local_graph_json(local_graph, tmp_path / "graph.json")
    loaded = load_local_graph_json(output_path)
    assert loaded.to_dict() == payload


def test_local_graph_converts_to_graph_store_with_metadata() -> None:
    local_graph = build_from_benchmark_example(record_to_benchmark_example(_record()))
    graph_store = local_graph_to_graph_store(local_graph)

    assert isinstance(graph_store, GraphStore)
    assert len(graph_store.iter_node_ids()) == len(local_graph.nodes)
    assert len(graph_store.iter_edges()) == len(local_graph.edges)

    question_node = graph_store.get_node_attributes("hotpotqa::hp-loader-1::question")
    assert question_node["node_type"] == "question"
    assert question_node["metadata"]["question_id"] == "hp-loader-1"

    question_to_title = next(edge for edge in local_graph.edges if edge.relation == "question_to_title")
    edge_record = graph_store.get_edge_by_id(question_to_title.edge_id)
    assert edge_record is not None
    assert edge_record.relation == "question_to_title"
    edge_attrs = graph_store.graph.edges[question_to_title.src, question_to_title.dst, question_to_title.edge_id]
    assert edge_attrs["metadata"]["title_index"] == 0
    assert edge_attrs["dataset_name"] == "hotpotqa"
    assert edge_attrs["question_id"] == "hp-loader-1"


def test_compute_graph_stats_reports_counts_and_metadata_flags() -> None:
    local_graph = build_from_benchmark_example(record_to_benchmark_example(_record()))
    stats = compute_graph_stats(local_graph)

    assert stats["node_count"] == 4
    assert stats["edge_count"] == 6
    assert stats["node_type_counts"] == {"question": 1, "sentence": 2, "title": 1}
    assert stats["relation_counts"] == {
        "next_sentence": 1,
        "question_to_title": 1,
        "sentence_to_title": 2,
        "title_to_sentence": 2,
    }
    assert stats["supporting_facts_in_metadata"] is True
    assert stats["privileged_edges_injected"] is False
