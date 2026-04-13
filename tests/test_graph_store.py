"""Tests for GraphStore."""

from graph.graph_store import EdgeRecord, GraphStore


def test_graph_store_add_and_lookup_edges() -> None:
    graph_store = GraphStore()
    graph_store.add_node("A", name="A")
    graph_store.add_node("B", name="B")
    edge = EdgeRecord(edge_id="edge-1", src="A", dst="B", relation="related_to", confidence=0.9, source="test")
    graph_store.add_edge(edge)

    assert graph_store.has_edge_by_id("edge-1") is True
    assert graph_store.get_edge_by_id("edge-1") == edge
    assert graph_store.get_outgoing_edges("A")[0].edge_id == "edge-1"
    assert graph_store.get_incoming_edges("B")[0].edge_id == "edge-1"
    assert graph_store.get_neighbors("A", direction="out") == ["B"]


def test_graph_store_export_subgraph_summary() -> None:
    graph_store = GraphStore()
    graph_store.add_edge(EdgeRecord(edge_id="edge-1", src="A", dst="B", relation="related_to", source="test"))
    summary = graph_store.export_subgraph_summary(["edge-1"])

    assert summary["edge_count"] == 1
    assert summary["node_count"] == 2
    assert summary["text_summary"] == "A -[related_to]-> B"
