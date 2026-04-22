"""Tests for HotpotQA relation/span proposal mechanism upgrades."""

from __future__ import annotations

import json
from pathlib import Path

from answer.hotpotqa_answer_selector import make_answer_selector_factory
from answer.hotpotqa_entity_title_mapper import make_entity_title_mapper_factory
from answer.hotpotqa_relation_span_discovery import make_relation_span_discovery_factory
from answer.hotpotqa_relation_span_mapper import make_pattern_span_mapper, make_relation_span_mapper_factory
from answer.hotpotqa_relation_span_proposal import (
    make_baseline_discovery_proposal,
    make_query_conditioned_plus_constrained_proposal,
    make_query_conditioned_proposal,
    make_relation_span_proposal_factory,
)
from answer.hotpotqa_relation_span_ranker import make_relation_span_ranker_factory
from answer.hotpotqa_yesno_mapper import make_yesno_mapper_factory
from data.benchmarks.hotpotqa import record_to_benchmark_example
from graph.graph_store import EdgeRecord, GraphStore
from graph.hotpotqa_graph_runtime import run_hotpotqa_graph_experiment_for_example
from graph.hotpotqa_policy_variants import make_policy_factory
from relevance.hotpotqa_question_conditioned_scorer import TOKENIZATION_DESCRIPTION
from scripts.compare_hotpotqa_relation_span_proposals import compare_hotpotqa_relation_span_proposals


def _graph_store() -> GraphStore:
    graph_store = GraphStore()
    graph_store.add_node("title-1", node_type="title", text="First Title")
    graph_store.add_node("sentence-1", node_type="sentence", text="He was the acting director of the office.")
    graph_store.add_edge(EdgeRecord(edge_id="e1", src="title-1", dst="sentence-1", relation="title_to_sentence"))
    graph_store.add_edge(EdgeRecord(edge_id="e2", src="sentence-1", dst="title-1", relation="sentence_to_title"))
    return graph_store


def _context(proposal_name: str | None = None) -> dict:
    context = {
        "query_text": "What was his role?",
        "answer_type_label": "descriptive_span_or_relation",
        "answer_selection": {
            "candidate_nodes": [
                {
                    "node_id": "sentence-1",
                    "node_type": "sentence",
                    "text": "He was the acting director of the office.",
                    "source": "path",
                    "step_idx": 1,
                }
            ]
        },
        "base_mapping": {"mapped_answer": "He was the acting director of the office."},
        "relation_span_discovery": make_relation_span_discovery_factory("baseline_pattern_discovery")(),
        "relation_span_discovery_name": "baseline_pattern_discovery",
        "relation_span_ranker": make_relation_span_ranker_factory("pattern_priority")(),
        "relation_span_ranker_name": "pattern_priority",
    }
    if proposal_name is not None:
        context["relation_span_proposal"] = make_relation_span_proposal_factory(proposal_name)()
        context["relation_span_proposal_name"] = proposal_name
    return context


def _proposal_context() -> dict:
    return {
        "query_text": "What was his role?",
        "selected_node": {
            "node_id": "sentence-1",
            "node_type": "sentence",
            "text": "He was the acting director of the office.",
            "source_level": "selected_sentence",
            "parent_title_text": "First Title",
            "title_text": "First Title",
            "step_idx": 1,
        },
        "source_nodes": [
            {
                "node_id": "sentence-1",
                "node_type": "sentence",
                "text": "He was the acting director of the office.",
                "source_level": "selected_sentence",
                "parent_title_text": "First Title",
                "title_text": "First Title",
                "step_idx": 1,
            }
        ],
        "base_candidate_spans": [
            {
                "text": "acting director of the office",
                "strategy": "was_the",
                "source_node_id": "sentence-1",
                "source_node_type": "sentence",
                "source_text": "He was the acting director of the office.",
                "source_level": "selected_sentence",
                "token_count": 5,
                "char_count": 29,
                "node_index": 0,
                "span_index": 0,
            }
        ],
        "gold_answer": "must not be used",
        "supporting_facts": ["must not be used"],
    }


def _fixture_record(answer: str = "complex descriptive relation answer phrase example") -> dict:
    return {
        "id": f"hp-proposal-{answer[:8].replace(' ', '-')}",
        "question": "What was the person's role?",
        "answer": answer,
        "context": {
            "title": ["First Title", "Other Page"],
            "sentences": [
                ["He was the acting director of the office."],
                ["Other Page is unrelated."],
            ],
        },
        "supporting_facts": {"title": ["First Title"], "sent_id": [0]},
    }


def _write_fixture(path: Path) -> Path:
    data_path = path / "hotpotqa_relation_span_proposal_fixture.json"
    data_path.write_text(
        json.dumps([_fixture_record(), _fixture_record("First Title")], ensure_ascii=False),
        encoding="utf-8",
    )
    return data_path


def test_proposals_are_deterministic_and_do_not_serialize_gold() -> None:
    context = _proposal_context()
    for factory in (
        make_baseline_discovery_proposal,
        make_query_conditioned_proposal,
        make_query_conditioned_plus_constrained_proposal,
    ):
        proposal = factory()
        first = proposal(context)
        second = proposal({**context, "gold_answer": "different ignored gold"})
        assert first.to_dict() == second.to_dict()
        serialized = json.dumps(first.to_dict(), ensure_ascii=False)
        assert "must not be used" not in serialized

    query_conditioned = make_query_conditioned_proposal()(_proposal_context())
    assert query_conditioned.metadata["scorer_normalization"] == TOKENIZATION_DESCRIPTION


def test_proposal_can_create_shorter_candidate_without_changing_default_path() -> None:
    graph_store = _graph_store()
    baseline = make_pattern_span_mapper()("sentence-1", graph_store, _context())
    proposed = make_pattern_span_mapper()(
        "sentence-1",
        graph_store,
        _context("query_conditioned_plus_constrained_proposal"),
    )

    assert baseline.selected_span == "acting director of the office"
    assert "acting director" in {span["text"] for span in proposed.candidate_spans}
    assert proposed.selected_span == "acting director"
    assert proposed.metadata["relation_span_proposal"]["proposal_variant_name"] == "query_conditioned_plus_constrained_proposal"


def test_runtime_routes_proposal_only_for_relation_span() -> None:
    relation_example = record_to_benchmark_example(_fixture_record(), split="validation")
    entity_example = record_to_benchmark_example(_fixture_record("First Title"), split="validation")

    _, relation_result = run_hotpotqa_graph_experiment_for_example(
        relation_example,
        answer_selector=make_answer_selector_factory("latest_sentence")(),
        answer_selector_name="latest_sentence",
        answer_mapper=make_entity_title_mapper_factory("parent_title")(),
        answer_mapper_name="parent_title",
        yesno_mapper=make_yesno_mapper_factory("sentence_polarity")(),
        yesno_mapper_name="sentence_polarity",
        relation_span_mapper=make_relation_span_mapper_factory("pattern_span")(),
        relation_span_mapper_name="pattern_span",
        relation_span_discovery=make_relation_span_discovery_factory("baseline_pattern_discovery")(),
        relation_span_discovery_name="baseline_pattern_discovery",
        relation_span_ranker=make_relation_span_ranker_factory("pattern_priority")(),
        relation_span_ranker_name="pattern_priority",
        relation_span_proposal=make_relation_span_proposal_factory("query_conditioned_plus_constrained_proposal")(),
        relation_span_proposal_name="query_conditioned_plus_constrained_proposal",
        policy=make_policy_factory("sentence_chain", min_expand_steps=2)(),
        policy_name="sentence_chain",
        min_expand_steps=2,
    )
    _, entity_result = run_hotpotqa_graph_experiment_for_example(
        entity_example,
        answer_selector=make_answer_selector_factory("latest_sentence")(),
        answer_selector_name="latest_sentence",
        answer_mapper=make_entity_title_mapper_factory("parent_title")(),
        answer_mapper_name="parent_title",
        yesno_mapper=make_yesno_mapper_factory("sentence_polarity")(),
        yesno_mapper_name="sentence_polarity",
        relation_span_mapper=make_relation_span_mapper_factory("pattern_span")(),
        relation_span_mapper_name="pattern_span",
        relation_span_discovery=make_relation_span_discovery_factory("baseline_pattern_discovery")(),
        relation_span_discovery_name="baseline_pattern_discovery",
        relation_span_ranker=make_relation_span_ranker_factory("pattern_priority")(),
        relation_span_ranker_name="pattern_priority",
        relation_span_proposal=make_relation_span_proposal_factory("query_conditioned_plus_constrained_proposal")(),
        relation_span_proposal_name="query_conditioned_plus_constrained_proposal",
        policy=make_policy_factory("sentence_chain", min_expand_steps=2)(),
        policy_name="sentence_chain",
        min_expand_steps=2,
    )

    assert relation_result.metadata["relation_span_mapper_applied"] is True
    assert relation_result.metadata["relation_span_proposal_applied"] is True
    assert relation_result.metadata["relation_span_mapping"]["metadata"]["relation_span_proposal"]
    assert entity_result.metadata["relation_span_mapper_applied"] is False
    assert entity_result.metadata["relation_span_proposal_applied"] is False


def test_compare_relation_span_proposals_script_smoke(tmp_path: Path) -> None:
    data_path = _write_fixture(tmp_path)
    output_dir = tmp_path / "relation-span-proposal-output"

    comparison = compare_hotpotqa_relation_span_proposals(
        path=data_path,
        split="validation",
        limit=2,
        reward_mode="baseline",
        proposals=["baseline_discovery_proposal", "query_conditioned_plus_constrained_proposal"],
        min_expand_steps=2,
        output_dir=output_dir,
    )

    assert comparison["fixed_relation_span_mapper_name"] == "pattern_span"
    assert comparison["fixed_relation_span_discovery_name"] == "baseline_pattern_discovery"
    assert comparison["fixed_relation_span_ranker_name"] == "pattern_priority"
    assert comparison["sample_indices"] == [0, 1]
    summary = comparison["proposals"]["query_conditioned_plus_constrained_proposal"]["relation_span_proposal_summary"]
    assert "proposal_direct_success_count" in summary
    assert "fallback_recovery_count" in summary
    assert "useful_candidate_coverage" in summary
    assert "no_useful_candidate_span_found_rate" in summary
    assert (output_dir / "query_conditioned_plus_constrained_proposal" / "hotpotqa_graph_eval_records.jsonl").exists()
    assert (output_dir / "hotpotqa_relation_span_proposal_comparison_summary.json").exists()
