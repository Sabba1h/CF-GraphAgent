"""Candidate action generation for the stage-1 graph environment."""

from __future__ import annotations

import re

from core.actions import ActionType, CandidateAction
from graph.graph_store import EdgeRecord, GraphStore
from memory.working_memory import WorkingMemory

_STOPWORDS = {
    "a",
    "an",
    "and",
    "by",
    "in",
    "is",
    "of",
    "on",
    "the",
    "to",
    "was",
    "what",
    "which",
    "who",
    "where",
}


def _tokenize(text: str) -> set[str]:
    normalized = text.lower().replace("_", " ")
    return {
        token
        for token in re.findall(r"[A-Za-z0-9]+", normalized)
        if token and token not in _STOPWORDS
    }


class CandidateGenerator:
    """Generate per-step candidate actions from the current frontier."""

    def __init__(self, *, top_k: int = 5, max_seed_nodes: int = 3) -> None:
        self.top_k = top_k
        self.max_seed_nodes = max_seed_nodes

    def find_seed_nodes(self, query: str, graph_store: GraphStore) -> list[str]:
        """Select a small seed set by simple string matching on nodes and relations."""

        query_tokens = _tokenize(query)
        node_scores: dict[str, float] = {}
        for node_id in graph_store.iter_node_ids():
            node_text = " ".join([node_id, graph_store.get_node_attributes(node_id).get("name", "")])
            overlap = len(query_tokens & _tokenize(node_text))
            if overlap > 0:
                node_scores[node_id] = float(overlap)

        for edge in graph_store.iter_edges():
            relation_overlap = len(query_tokens & _tokenize(edge.relation))
            if relation_overlap > 0:
                node_scores[edge.src] = node_scores.get(edge.src, 0.0) + relation_overlap * 0.5
                node_scores[edge.dst] = node_scores.get(edge.dst, 0.0) + relation_overlap * 0.25

        ranked_nodes = sorted(node_scores.items(), key=lambda item: (-item[1], item[0]))
        if ranked_nodes:
            return [node_id for node_id, _ in ranked_nodes[: self.max_seed_nodes]]
        return graph_store.iter_node_ids()[: self.max_seed_nodes]

    def generate(
        self,
        *,
        query: str,
        graph_store: GraphStore,
        working_memory: WorkingMemory,
    ) -> list[CandidateAction]:
        """Generate candidate expansions plus ANSWER and STOP."""

        query_tokens = _tokenize(query)
        candidate_edges: dict[str, tuple[EdgeRecord, float]] = {}
        frontier_nodes = sorted(working_memory.frontier_nodes) or self.find_seed_nodes(query, graph_store)

        for node_id in frontier_nodes:
            for edge in graph_store.get_outgoing_edges(node_id) + graph_store.get_incoming_edges(node_id):
                score = self._score_edge(edge=edge, query_tokens=query_tokens, working_memory=working_memory)
                current = candidate_edges.get(edge.edge_id)
                if current is None or score > current[1]:
                    candidate_edges[edge.edge_id] = (edge, score)

        ranked_edges = sorted(candidate_edges.values(), key=lambda item: (-item[1], item[0].edge_id))
        actions: list[CandidateAction] = []
        for edge, score in ranked_edges[: self.top_k]:
            src_attrs = graph_store.get_node_attributes(edge.src)
            dst_attrs = graph_store.get_node_attributes(edge.dst)
            actions.append(
                CandidateAction(
                    candidate_id=len(actions),
                    action_type=ActionType.EXPAND_EDGE,
                    edge_id=edge.edge_id,
                    score=score,
                    description=self._build_edge_description(edge=edge, repeated=edge.edge_id in working_memory.working_edge_ids),
                    metadata={
                        "src": edge.src,
                        "dst": edge.dst,
                        "relation": edge.relation,
                        "src_text": self._node_text(src_attrs, edge.src),
                        "dst_text": self._node_text(dst_attrs, edge.dst),
                        "src_node_type": str(src_attrs.get("node_type") or ""),
                        "dst_node_type": str(dst_attrs.get("node_type") or ""),
                    },
                )
            )

        actions.append(
            CandidateAction(
                candidate_id=len(actions),
                action_type=ActionType.ANSWER,
                description="Generate an answer from the current working subgraph and end the episode.",
                score=0.0,
            )
        )
        actions.append(
            CandidateAction(
                candidate_id=len(actions),
                action_type=ActionType.STOP,
                description="Stop exploration without answering and end the episode.",
                score=0.0,
            )
        )
        return actions

    def _score_edge(
        self,
        *,
        edge: EdgeRecord,
        query_tokens: set[str],
        working_memory: WorkingMemory,
    ) -> float:
        edge_tokens = _tokenize(f"{edge.src} {edge.relation} {edge.dst}")
        overlap = len(query_tokens & edge_tokens)
        repeat_penalty = 0.2 if edge.edge_id in working_memory.working_edge_ids else 0.0
        frontier_bonus = 0.1 if edge.src in working_memory.frontier_nodes or edge.dst in working_memory.frontier_nodes else 0.0
        return edge.confidence + overlap + frontier_bonus - repeat_penalty

    def _build_edge_description(self, *, edge: EdgeRecord, repeated: bool) -> str:
        base = f"Expand edge {edge.edge_id}: {edge.src} -[{edge.relation}]-> {edge.dst}"
        if repeated:
            return f"{base} (already in working subgraph)"
        return base

    def _node_text(self, attrs: dict[str, object], fallback: str) -> str:
        """Expose only whitelisted node text for transparent relevance scoring."""

        return str(attrs.get("text") or attrs.get("name") or fallback)
