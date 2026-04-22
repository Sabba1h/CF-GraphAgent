"""Non-privileged yes/no answer mappers for HotpotQA.

These mappers are only used after answer-type routing has produced the
``yes_no`` label. They do not read gold answers, supporting facts, benchmark
metadata, or scan the full graph.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from graph.graph_store import GraphStore


@dataclass(slots=True)
class HotpotQAYesNoMapping:
    """A yes/no mapping decision with explicit evidence metadata."""

    yesno_mapper_name: str
    selected_graph_answer: str | None
    mapped_answer: str
    yesno_decision: str | None
    decision_reason: str
    source_node_id: str | None
    source_node_type: str | None
    source_text: str
    evidence_node_ids: list[str]
    evidence_strength: float
    positive_evidence_count: int
    negative_evidence_count: int
    conflict_detected: bool
    fallback_occurred: bool
    fallback_target: str | None
    fallback_reason: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""

        return asdict(self)


YesNoMapper = Callable[[str | None, GraphStore, dict[str, Any] | None], HotpotQAYesNoMapping]
YesNoMapperFactory = Callable[[], YesNoMapper]

YESNO_MAPPER_NAMES = (
    "identity_yesno",
    "sentence_polarity",
    "title_sentence_consistency",
    "abstain_backoff_yesno",
)

NEGATIVE_TERMS = {"different", "false", "neither", "never", "no", "nor", "not", "unlike", "without"}
NEGATIVE_PHRASES = (
    "not the same",
    "not same",
    "are different",
    "were different",
    "is different",
    "was different",
    "different from",
)
POSITIVE_TERMS = {"also", "both", "common", "same", "share", "shared", "similar", "true", "yes"}
POSITIVE_PHRASES = ("the same", "same as", "both are", "both were", "also known", "in common")
QUERY_YESNO_TERMS = {"are", "did", "do", "does", "is", "was", "were"}
TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class EvidenceCounts:
    """Local polarity evidence counts."""

    positive_count: int = 0
    negative_count: int = 0
    positive_terms: list[str] = field(default_factory=list)
    negative_terms: list[str] = field(default_factory=list)

    @property
    def evidence_strength(self) -> float:
        """Return a transparent strength score."""

        return float(self.positive_count + self.negative_count)

    @property
    def conflict_detected(self) -> bool:
        """Return whether positive and negative cues coexist."""

        return self.positive_count > 0 and self.negative_count > 0


def make_identity_yesno_mapper() -> YesNoMapper:
    """Return baseline yes/no mapper: preserve existing base answer mapping."""

    def mapper(
        selected_graph_answer: str | None,
        graph_store: GraphStore,
        context: dict[str, Any] | None = None,
    ) -> HotpotQAYesNoMapping:
        del graph_store
        payload = _context_payload(context)
        base_mapping = payload["base_mapping"]
        selected_node = _selected_node_from_context(selected_graph_answer, payload)
        return _mapping(
            mapper_name="identity_yesno",
            selected_graph_answer=selected_graph_answer,
            mapped_answer=str(base_mapping.get("mapped_answer") or ""),
            yesno_decision=None,
            decision_reason="identity_preserves_base_mapping",
            selected_node=selected_node,
            evidence_nodes=[],
            evidence=EvidenceCounts(),
            fallback_occurred=True,
            fallback_target="base_mapping",
            fallback_reason="identity_yesno_mapper_does_not_decide",
            metadata={
                "strategy": "identity_base_mapping",
                "base_mapping": base_mapping,
                "answer_type_label": payload["answer_type_label"],
            },
        )

    return mapper


def make_sentence_polarity_mapper() -> YesNoMapper:
    """Return mapper using local selected/path sentence polarity cues."""

    def mapper(
        selected_graph_answer: str | None,
        graph_store: GraphStore,
        context: dict[str, Any] | None = None,
    ) -> HotpotQAYesNoMapping:
        payload = _context_payload(context)
        selected_node = _selected_node_payload(selected_graph_answer, graph_store, payload)
        evidence_nodes = _path_title_sentence_nodes(payload, selected_node=selected_node)
        evidence = _evidence_counts(query_text=payload["query_text"], evidence_nodes=evidence_nodes)
        return _decision_mapping(
            mapper_name="sentence_polarity",
            selected_graph_answer=selected_graph_answer,
            selected_node=selected_node,
            evidence_nodes=evidence_nodes,
            evidence=evidence,
            base_mapping=payload["base_mapping"],
            answer_type_label=payload["answer_type_label"],
            insufficient_target="yes",
            insufficient_reason="no_explicit_polarity_cue_default_yes",
        )

    return mapper


def make_title_sentence_consistency_yesno_mapper() -> YesNoMapper:
    """Return mapper combining selected sentence, parent title, and path evidence."""

    def mapper(
        selected_graph_answer: str | None,
        graph_store: GraphStore,
        context: dict[str, Any] | None = None,
    ) -> HotpotQAYesNoMapping:
        payload = _context_payload(context)
        selected_node = _selected_node_payload(selected_graph_answer, graph_store, payload)
        evidence_nodes = _dedupe_nodes(
            _path_title_sentence_nodes(payload, selected_node=selected_node)
            + _selected_local_title_sentence_nodes(selected_graph_answer, graph_store, selected_node)
        )
        evidence = _evidence_counts(query_text=payload["query_text"], evidence_nodes=evidence_nodes)
        if evidence.positive_count == 0 and evidence.negative_count == 0 and _has_local_sentence(evidence_nodes):
            evidence = EvidenceCounts(
                positive_count=1,
                negative_count=0,
                positive_terms=["local_sentence_present"],
                negative_terms=[],
            )
        return _decision_mapping(
            mapper_name="title_sentence_consistency",
            selected_graph_answer=selected_graph_answer,
            selected_node=selected_node,
            evidence_nodes=evidence_nodes,
            evidence=evidence,
            base_mapping=payload["base_mapping"],
            answer_type_label=payload["answer_type_label"],
            insufficient_target="base_mapping",
            insufficient_reason="insufficient_local_consistency_evidence",
        )

    return mapper


def make_abstain_backoff_yesno_mapper() -> YesNoMapper:
    """Return polarity mapper that backs off to base mapping when evidence is weak."""

    def mapper(
        selected_graph_answer: str | None,
        graph_store: GraphStore,
        context: dict[str, Any] | None = None,
    ) -> HotpotQAYesNoMapping:
        payload = _context_payload(context)
        selected_node = _selected_node_payload(selected_graph_answer, graph_store, payload)
        evidence_nodes = _path_title_sentence_nodes(payload, selected_node=selected_node)
        evidence = _evidence_counts(query_text=payload["query_text"], evidence_nodes=evidence_nodes)
        return _decision_mapping(
            mapper_name="abstain_backoff_yesno",
            selected_graph_answer=selected_graph_answer,
            selected_node=selected_node,
            evidence_nodes=evidence_nodes,
            evidence=evidence,
            base_mapping=payload["base_mapping"],
            answer_type_label=payload["answer_type_label"],
            insufficient_target="base_mapping",
            insufficient_reason="abstained_due_to_insufficient_evidence",
        )

    return mapper


def make_yesno_mapper_factory(mapper_name: str) -> YesNoMapperFactory:
    """Return a fresh yes/no mapper factory."""

    if mapper_name == "identity_yesno":
        return make_identity_yesno_mapper
    if mapper_name == "sentence_polarity":
        return make_sentence_polarity_mapper
    if mapper_name == "title_sentence_consistency":
        return make_title_sentence_consistency_yesno_mapper
    if mapper_name == "abstain_backoff_yesno":
        return make_abstain_backoff_yesno_mapper
    raise ValueError(
        f"Unknown HotpotQA yes/no mapper '{mapper_name}'. "
        f"Available mappers: {', '.join(YESNO_MAPPER_NAMES)}"
    )


def _decision_mapping(
    *,
    mapper_name: str,
    selected_graph_answer: str | None,
    selected_node: dict[str, Any],
    evidence_nodes: list[dict[str, Any]],
    evidence: EvidenceCounts,
    base_mapping: dict[str, Any],
    answer_type_label: str,
    insufficient_target: str,
    insufficient_reason: str,
) -> HotpotQAYesNoMapping:
    decision = _yesno_decision(evidence)
    if decision is None:
        if insufficient_target == "yes":
            return _mapping(
                mapper_name=mapper_name,
                selected_graph_answer=selected_graph_answer,
                mapped_answer="yes",
                yesno_decision="yes",
                decision_reason=insufficient_reason,
                selected_node=selected_node,
                evidence_nodes=evidence_nodes,
                evidence=evidence,
                fallback_occurred=True,
                fallback_target="yes",
                fallback_reason=insufficient_reason,
                metadata={
                    "strategy": "weak_yes_backoff",
                    "answer_type_label": answer_type_label,
                    "base_mapping": base_mapping,
                },
            )
        return _mapping(
            mapper_name=mapper_name,
            selected_graph_answer=selected_graph_answer,
            mapped_answer=str(base_mapping.get("mapped_answer") or ""),
            yesno_decision=None,
            decision_reason=insufficient_reason,
            selected_node=selected_node,
            evidence_nodes=evidence_nodes,
            evidence=evidence,
            fallback_occurred=True,
            fallback_target="base_mapping",
            fallback_reason=insufficient_reason,
            metadata={
                "strategy": "base_mapping_backoff",
                "answer_type_label": answer_type_label,
                "base_mapping": base_mapping,
            },
        )

    return _mapping(
        mapper_name=mapper_name,
        selected_graph_answer=selected_graph_answer,
        mapped_answer=decision,
        yesno_decision=decision,
        decision_reason=_decision_reason(decision, evidence),
        selected_node=selected_node,
        evidence_nodes=evidence_nodes,
        evidence=evidence,
        fallback_occurred=False,
        fallback_target=None,
        fallback_reason=None,
        metadata={
            "strategy": "local_polarity_decision",
            "answer_type_label": answer_type_label,
            "base_mapping": base_mapping,
        },
    )


def _yesno_decision(evidence: EvidenceCounts) -> str | None:
    if evidence.negative_count > evidence.positive_count:
        return "no"
    if evidence.positive_count > evidence.negative_count:
        return "yes"
    if evidence.conflict_detected:
        return "no"
    return None


def _decision_reason(decision: str, evidence: EvidenceCounts) -> str:
    if evidence.conflict_detected:
        return f"{decision}_from_conflicting_local_polarity"
    return f"{decision}_from_local_polarity"


def _mapping(
    *,
    mapper_name: str,
    selected_graph_answer: str | None,
    mapped_answer: str,
    yesno_decision: str | None,
    decision_reason: str,
    selected_node: dict[str, Any],
    evidence_nodes: list[dict[str, Any]],
    evidence: EvidenceCounts,
    fallback_occurred: bool,
    fallback_target: str | None,
    fallback_reason: str | None,
    metadata: dict[str, Any],
) -> HotpotQAYesNoMapping:
    return HotpotQAYesNoMapping(
        yesno_mapper_name=mapper_name,
        selected_graph_answer=selected_graph_answer,
        mapped_answer=mapped_answer,
        yesno_decision=yesno_decision,
        decision_reason=decision_reason,
        source_node_id=selected_node.get("node_id"),
        source_node_type=selected_node.get("node_type"),
        source_text=str(selected_node.get("text") or ""),
        evidence_node_ids=[str(node.get("node_id")) for node in evidence_nodes if node.get("node_id")],
        evidence_strength=evidence.evidence_strength,
        positive_evidence_count=evidence.positive_count,
        negative_evidence_count=evidence.negative_count,
        conflict_detected=evidence.conflict_detected,
        fallback_occurred=fallback_occurred,
        fallback_target=fallback_target,
        fallback_reason=fallback_reason,
        metadata={
            **metadata,
            "positive_terms": evidence.positive_terms,
            "negative_terms": evidence.negative_terms,
            "evidence_nodes": [_public_node(node) for node in evidence_nodes],
            "whitelist_fields": [
                "query_text",
                "selected_node.title_text",
                "selected_node.sentence_text",
                "path_touched_local_nodes",
                "selected_node_local_neighborhood",
            ],
        },
    )


def _context_payload(context: dict[str, Any] | None) -> dict[str, Any]:
    payload = context if isinstance(context, dict) else {}
    answer_type_label = str(payload.get("answer_type_label") or "")
    if answer_type_label != "yes_no":
        raise ValueError("HotpotQA yes/no mapper requires answer_type_label='yes_no'.")
    return {
        "query_text": str(payload.get("query_text") or ""),
        "answer_type_label": answer_type_label,
        "answer_selection": payload.get("answer_selection") if isinstance(payload.get("answer_selection"), dict) else {},
        "base_mapping": payload.get("base_mapping") if isinstance(payload.get("base_mapping"), dict) else {},
    }


def _selected_node_from_context(selected_graph_answer: str | None, payload: dict[str, Any]) -> dict[str, Any]:
    answer_selection = payload["answer_selection"]
    for node in answer_selection.get("candidate_nodes", []) or []:
        if isinstance(node, dict) and node.get("node_id") == selected_graph_answer:
            return dict(node)
    return {"node_id": selected_graph_answer, "node_type": None, "text": ""}


def _selected_node_payload(
    selected_graph_answer: str | None,
    graph_store: GraphStore,
    payload: dict[str, Any],
) -> dict[str, Any]:
    context_node = _selected_node_from_context(selected_graph_answer, payload)
    if selected_graph_answer is None:
        return context_node
    attrs = graph_store.get_node_attributes(selected_graph_answer)
    if not attrs:
        return context_node
    return {
        "node_id": selected_graph_answer,
        "node_type": str(attrs.get("node_type") or context_node.get("node_type") or ""),
        "text": str(attrs.get("text") or attrs.get("name") or context_node.get("text") or ""),
        "source": context_node.get("source"),
        "step_idx": context_node.get("step_idx"),
    }


def _path_title_sentence_nodes(
    payload: dict[str, Any],
    *,
    selected_node: dict[str, Any],
) -> list[dict[str, Any]]:
    answer_selection = payload["answer_selection"]
    nodes: list[dict[str, Any]] = []
    for node in answer_selection.get("candidate_nodes", []) or []:
        if not isinstance(node, dict):
            continue
        if node.get("source") != "path":
            continue
        if node.get("node_type") in {"sentence", "title"}:
            nodes.append(dict(node))
    if selected_node.get("node_id") and selected_node.get("node_type") in {"sentence", "title"}:
        nodes.append(dict(selected_node))
    return _dedupe_nodes(nodes)


def _selected_local_title_sentence_nodes(
    selected_graph_answer: str | None,
    graph_store: GraphStore,
    selected_node: dict[str, Any],
) -> list[dict[str, Any]]:
    if selected_graph_answer is None:
        return []
    nodes = [dict(selected_node)] if selected_node.get("node_type") in {"sentence", "title"} else []
    for edge in graph_store.get_incoming_edges(selected_graph_answer) + graph_store.get_outgoing_edges(selected_graph_answer):
        if edge.relation not in {"title_to_sentence", "sentence_to_title", "next_sentence"}:
            continue
        neighbor_id = edge.src if edge.dst == selected_graph_answer else edge.dst
        attrs = graph_store.get_node_attributes(neighbor_id)
        node_type = str(attrs.get("node_type") or "")
        if node_type not in {"sentence", "title"}:
            continue
        nodes.append(
            {
                "node_id": neighbor_id,
                "node_type": node_type,
                "text": str(attrs.get("text") or attrs.get("name") or ""),
                "source": "selected_local_neighborhood",
                "relation": edge.relation,
            }
        )
    return _dedupe_nodes(nodes)


def _evidence_counts(*, query_text: str, evidence_nodes: list[dict[str, Any]]) -> EvidenceCounts:
    texts = [str(node.get("text") or "") for node in evidence_nodes] + [query_text]
    positive_terms: list[str] = []
    negative_terms: list[str] = []
    for text in texts:
        normalized_text = " ".join(_tokens(text))
        token_set = set(normalized_text.split())
        positive_terms.extend(sorted(POSITIVE_TERMS & token_set))
        negative_terms.extend(sorted(NEGATIVE_TERMS & token_set))
        positive_terms.extend(phrase for phrase in POSITIVE_PHRASES if phrase in normalized_text)
        negative_terms.extend(phrase for phrase in NEGATIVE_PHRASES if phrase in normalized_text)
    query_tokens = set(_tokens(query_text))
    if QUERY_YESNO_TERMS & query_tokens and evidence_nodes:
        positive_terms.append("yesno_question_with_local_evidence")
    return EvidenceCounts(
        positive_count=len(positive_terms),
        negative_count=len(negative_terms),
        positive_terms=positive_terms,
        negative_terms=negative_terms,
    )


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _has_local_sentence(nodes: list[dict[str, Any]]) -> bool:
    return any(node.get("node_type") == "sentence" for node in nodes)


def _dedupe_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for node in nodes:
        node_id = str(node.get("node_id") or "")
        if not node_id or node_id in seen:
            continue
        seen.add(node_id)
        deduped.append(node)
    return deduped


def _public_node(node: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_id": node.get("node_id"),
        "node_type": node.get("node_type"),
        "text": node.get("text"),
        "source": node.get("source"),
        "step_idx": node.get("step_idx"),
        "relation": node.get("relation"),
    }
