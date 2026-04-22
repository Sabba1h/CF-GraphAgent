"""Build conservative benchmark-local graphs from benchmark examples."""

from __future__ import annotations

from typing import Any

from data.benchmarks.common import BenchmarkExample
from graph.benchmark_graph_types import BenchmarkGraphEdge, BenchmarkGraphNode, BenchmarkLocalGraph

BUILDER_VERSION = "benchmark_local_graph_v1"


def build_from_benchmark_example(example: BenchmarkExample) -> BenchmarkLocalGraph:
    """Build a transparent local graph from benchmark context.

    The first version intentionally connects the question to every title node
    and avoids lexical or supporting-fact supervision edges.
    """

    graph_id = _graph_id(example)
    nodes: list[BenchmarkGraphNode] = []
    edges: list[BenchmarkGraphEdge] = []
    edge_index = 0

    question_node_id = f"{graph_id}::question"
    nodes.append(
        BenchmarkGraphNode(
            node_id=question_node_id,
            node_type="question",
            text=example.question,
            metadata={
                "dataset_name": example.dataset_name,
                "question_id": example.question_id,
            },
        )
    )

    for title_index, context_item in enumerate(example.context):
        title = str(context_item.get("title", ""))
        title_node_id = f"{graph_id}::title::{title_index}"
        nodes.append(
            BenchmarkGraphNode(
                node_id=title_node_id,
                node_type="title",
                text=title,
                metadata={
                    "dataset_name": example.dataset_name,
                    "question_id": example.question_id,
                    "title_index": title_index,
                    "title": title,
                    "raw": context_item.get("raw"),
                },
            )
        )

        edges.append(
            _edge(
                graph_id=graph_id,
                edge_index=edge_index,
                src=question_node_id,
                dst=title_node_id,
                relation="question_to_title",
                metadata={
                    "dataset_name": example.dataset_name,
                    "question_id": example.question_id,
                    "title_index": title_index,
                    "title": title,
                },
            )
        )
        edge_index += 1

        previous_sentence_node_id: str | None = None
        for sentence_index, sentence in enumerate(_sentences(context_item)):
            sentence_node_id = f"{graph_id}::sentence::{title_index}::{sentence_index}"
            nodes.append(
                BenchmarkGraphNode(
                    node_id=sentence_node_id,
                    node_type="sentence",
                    text=str(sentence),
                    metadata={
                        "dataset_name": example.dataset_name,
                        "question_id": example.question_id,
                        "title_index": title_index,
                        "sentence_index": sentence_index,
                        "title": title,
                        "raw": sentence,
                    },
                )
            )

            edges.append(
                _edge(
                    graph_id=graph_id,
                    edge_index=edge_index,
                    src=title_node_id,
                    dst=sentence_node_id,
                    relation="title_to_sentence",
                    metadata={
                        "dataset_name": example.dataset_name,
                        "question_id": example.question_id,
                        "title_index": title_index,
                        "sentence_index": sentence_index,
                        "title": title,
                    },
                )
            )
            edge_index += 1
            edges.append(
                _edge(
                    graph_id=graph_id,
                    edge_index=edge_index,
                    src=sentence_node_id,
                    dst=title_node_id,
                    relation="sentence_to_title",
                    metadata={
                        "dataset_name": example.dataset_name,
                        "question_id": example.question_id,
                        "title_index": title_index,
                        "sentence_index": sentence_index,
                        "title": title,
                    },
                )
            )
            edge_index += 1

            if previous_sentence_node_id is not None:
                edges.append(
                    _edge(
                        graph_id=graph_id,
                        edge_index=edge_index,
                        src=previous_sentence_node_id,
                        dst=sentence_node_id,
                        relation="next_sentence",
                        metadata={
                            "dataset_name": example.dataset_name,
                            "question_id": example.question_id,
                            "title_index": title_index,
                            "from_sentence_index": sentence_index - 1,
                            "to_sentence_index": sentence_index,
                            "title": title,
                        },
                    )
                )
                edge_index += 1
            previous_sentence_node_id = sentence_node_id

    return BenchmarkLocalGraph(
        graph_id=graph_id,
        dataset_name=example.dataset_name,
        question_id=example.question_id,
        question=example.question,
        nodes=nodes,
        edges=edges,
        metadata={
            "builder_version": BUILDER_VERSION,
            "dataset_name": example.dataset_name,
            "question_id": example.question_id,
            "raw_answer": example.answer,
            "normalized_answer": example.normalized_answer,
            "context_titles": example.context_titles,
            "supporting_facts": example.supporting_facts,
            "benchmark_metadata": dict(example.metadata),
            "privileged_edges_injected": False,
        },
    )


def _graph_id(example: BenchmarkExample) -> str:
    question_id = example.question_id or "unknown"
    return f"{example.dataset_name}::{question_id}"


def _sentences(context_item: dict[str, Any]) -> list[Any]:
    sentences = context_item.get("sentences", [])
    if isinstance(sentences, list):
        return sentences
    if isinstance(sentences, tuple):
        return list(sentences)
    if sentences is None:
        return []
    return [sentences]


def _edge(
    *,
    graph_id: str,
    edge_index: int,
    src: str,
    dst: str,
    relation: str,
    metadata: dict[str, Any],
) -> BenchmarkGraphEdge:
    return BenchmarkGraphEdge(
        edge_id=f"{graph_id}::edge::{edge_index}",
        src=src,
        dst=dst,
        relation=relation,
        metadata=metadata,
    )
