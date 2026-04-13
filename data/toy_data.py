"""Toy graph and query fixtures for stage-1 demos and tests."""

from __future__ import annotations

from graph.graph_store import EdgeRecord, GraphStore


def build_toy_graph() -> GraphStore:
    """Create a small multi-relational graph for stage-1 validation."""

    graph_store = GraphStore()
    for node_id in [
        "Forrest Gump",
        "Robert Zemeckis",
        "Chicago",
        "Tom Hanks",
        "Drama",
        "Back to the Future",
        "Alan Silvestri",
    ]:
        graph_store.add_node(node_id, name=node_id)

    for edge in [
        EdgeRecord(
            edge_id="e1",
            src="Forrest Gump",
            dst="Robert Zemeckis",
            relation="directed_by",
            confidence=0.99,
            source="toy_data",
            timestamp="1994",
        ),
        EdgeRecord(
            edge_id="e2",
            src="Robert Zemeckis",
            dst="Chicago",
            relation="born_in",
            confidence=0.95,
            source="toy_data",
            timestamp="1951",
        ),
        EdgeRecord(
            edge_id="e3",
            src="Tom Hanks",
            dst="Forrest Gump",
            relation="acted_in",
            confidence=0.98,
            source="toy_data",
            timestamp="1994",
        ),
        EdgeRecord(
            edge_id="e4",
            src="Forrest Gump",
            dst="Drama",
            relation="has_genre",
            confidence=0.87,
            source="toy_data",
            timestamp="1994",
        ),
        EdgeRecord(
            edge_id="e5",
            src="Robert Zemeckis",
            dst="Back to the Future",
            relation="directed",
            confidence=0.96,
            source="toy_data",
            timestamp="1985",
        ),
        EdgeRecord(
            edge_id="e6",
            src="Back to the Future",
            dst="Alan Silvestri",
            relation="music_by",
            confidence=0.83,
            source="toy_data",
            timestamp="1985",
        ),
    ]:
        graph_store.add_edge(edge)
    return graph_store


def get_demo_task() -> dict[str, str]:
    """Return a query that benefits from two expansions."""

    return {
        "query": "Which city was the director of Forrest Gump born in?",
        "ground_truth": "Chicago",
    }
