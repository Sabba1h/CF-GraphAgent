"""Tests for question-conditioned HotpotQA relevance variants."""

from __future__ import annotations

import json
from pathlib import Path

from answer.hotpotqa_question_conditioned_selectors import (
    make_overlap_guided_sentence_selector,
    make_overlap_plus_region_selector,
)
from core.experiment_config import ExperimentConfig
from core.experiment_result import ExperimentResult, ExperimentStepTrace
from graph.graph_store import EdgeRecord, GraphStore
from graph.hotpotqa_question_conditioned_policies import make_overlap_guided_region_policy
from relevance.hotpotqa_question_conditioned_scorer import (
    TOKENIZATION_DESCRIPTION,
    compute_title_sentence_hybrid_score,
    compute_token_overlap_score,
    normalize_relevance_tokens,
)
from scripts.compare_hotpotqa_question_conditioned_relevance import (
    compare_hotpotqa_question_conditioned_relevance,
)


def _candidate(candidate_id: int, relation: str, src: str, dst: str, *, dst_text: str) -> dict:
    return {
        "candidate_id": candidate_id,
        "action_type": "EXPAND_EDGE",
        "description": f"Expand {src} -[{relation}]-> {dst}",
        "metadata": {
            "relation": relation,
            "src": src,
            "dst": dst,
            "src_text": src,
            "dst_text": dst_text,
            "src_node_type": "title",
            "dst_node_type": "sentence",
        },
    }


def _observation(candidates: list[dict]) -> dict:
    return {
        "query": "Which page discusses First Title?",
        "candidate_actions": candidates
        + [
            {"candidate_id": 90, "action_type": "ANSWER", "description": "Answer", "metadata": {}},
            {"candidate_id": 91, "action_type": "STOP", "description": "Stop", "metadata": {}},
        ],
    }


def _graph_store() -> GraphStore:
    graph_store = GraphStore()
    graph_store.add_node("title-1", node_type="title", text="First Title")
    graph_store.add_node("title-2", node_type="title", text="Unrelated Page")
    graph_store.add_node("sentence-1", node_type="sentence", text="This sentence discusses First Title.")
    graph_store.add_node("sentence-2", node_type="sentence", text="A different unrelated sentence.")
    graph_store.add_node("global-sentence", node_type="sentence", text="First Title exists globally but is not path touched.")
    graph_store.add_edge(EdgeRecord(edge_id="e-title-s1", src="title-1", dst="sentence-1", relation="title_to_sentence"))
    graph_store.add_edge(EdgeRecord(edge_id="e-s1-title", src="sentence-1", dst="title-1", relation="sentence_to_title"))
    graph_store.add_edge(EdgeRecord(edge_id="e-title2-s2", src="title-2", dst="sentence-2", relation="title_to_sentence"))
    return graph_store


def _trace(step_idx: int, action: int, src: str, dst: str) -> ExperimentStepTrace:
    return ExperimentStepTrace(
        step_idx=step_idx,
        action=action,
        reward_mode="baseline",
        base_reward=0.1,
        reward=0.1,
        reward_breakdown=None,
        metadata={
            "info": {
                "expanded_edge": {
                    "edge_id": f"e{step_idx}",
                    "src": src,
                    "relation": "title_to_sentence",
                    "dst": dst,
                }
            },
            "observation": {
                "query": "Which page discusses First Title?",
                "candidate_actions": [
                    {
                        "candidate_id": action,
                        "action_type": "EXPAND_EDGE",
                        "metadata": {"src": src, "dst": dst, "relation": "title_to_sentence"},
                    }
                ],
            },
        },
    )


def _experiment_result() -> ExperimentResult:
    return ExperimentResult(
        config=ExperimentConfig(metadata={"query": "Which page discusses First Title?"}),
        final_answer="title-2",
        step_traces=[
            _trace(0, 0, "title-2", "sentence-2"),
            _trace(1, 1, "title-1", "sentence-1"),
        ],
        metadata={"query": "Which page discusses First Title?", "raw_graph_answer": "title-2"},
    )


def _fixture_record(answer: str = "First Title") -> dict:
    return {
        "id": "hp-question-conditioned-fixture-1",
        "question": "Which page discusses First Title?",
        "answer": answer,
        "context": {
            "title": ["First Title", "Unrelated Page"],
            "sentences": [
                ["This sentence discusses First Title."],
                ["A different unrelated sentence."],
            ],
        },
        "supporting_facts": {"title": ["First Title"], "sent_id": [0]},
    }


def _write_fixture(path: Path) -> Path:
    data_path = path / "hotpotqa_question_conditioned_fixture.json"
    data_path.write_text(json.dumps([_fixture_record()], ensure_ascii=False), encoding="utf-8")
    return data_path


def test_scorer_normalization_and_components_are_fixed_and_deterministic() -> None:
    assert normalize_relevance_tokens("The Chief, of Protocol!") == ["chief", "protocol"]

    first = compute_token_overlap_score(
        query_text="Which page discusses First Title?",
        title_text="First Title",
        sentence_text="This sentence discusses First Title.",
        path_stats={"region_continuity": 1.0, "recent_region": 1.0, "ignored": 999},
    )
    second = compute_token_overlap_score(
        query_text="Which page discusses First Title?",
        title_text="First Title",
        sentence_text="This sentence discusses First Title.",
        path_stats={"region_continuity": 1.0, "recent_region": 1.0, "ignored": 999},
    )
    hybrid = compute_title_sentence_hybrid_score(
        query_text="Which page discusses First Title?",
        title_text="First Title",
        sentence_text="This sentence discusses First Title.",
    )

    assert first.to_dict() == second.to_dict()
    assert first.total_score > 0.0
    assert hybrid.component_scores["title_overlap"] > 0.0
    assert "lowercasing" in TOKENIZATION_DESCRIPTION
    assert "STOPWORDS" in first.component_description["stopword_handling"]


def test_overlap_guided_policy_uses_whitelisted_candidate_fields() -> None:
    observation = _observation(
        [
            _candidate(5, "title_to_sentence", "title-2", "sentence-2", dst_text="A different unrelated sentence."),
            _candidate(1, "title_to_sentence", "title-1", "sentence-1", dst_text="This sentence discusses First Title."),
        ]
    )

    policy = make_overlap_guided_region_policy(min_expand_steps=1)

    assert policy(observation) == 1
    assert policy(observation) == 1


def test_overlap_guided_selectors_choose_only_path_touched_nodes() -> None:
    graph_store = _graph_store()
    result = _experiment_result()

    sentence_selection = make_overlap_guided_sentence_selector()(result, graph_store)
    region_selection = make_overlap_plus_region_selector()(result, graph_store)

    assert sentence_selection.selected_graph_answer == "sentence-1"
    assert region_selection.selected_graph_answer in {"sentence-1", "title-1"}
    assert sentence_selection.selected_graph_answer != "global-sentence"
    assert sentence_selection.metadata["selected_score"]["component_scores"]
    summary = sentence_selection.metadata["score_component_summary"]
    assert summary["component_description"]["lowercasing"] is True


def test_question_conditioned_comparison_smoke_uses_fixed_mapper_extractor_and_samples(tmp_path: Path) -> None:
    data_path = _write_fixture(tmp_path)
    output_dir = tmp_path / "question-conditioned-output"

    comparison = compare_hotpotqa_question_conditioned_relevance(
        path=data_path,
        split="validation",
        limit=1,
        reward_mode="baseline",
        variants=["baseline_structural", "overlap_guided_selector"],
        output_dir=output_dir,
    )

    assert comparison["fixed_mapper_name"] == "parent_title"
    assert comparison["fixed_extractor_name"] == "full_sentence"
    assert comparison["sample_indices"] == [0]
    assert comparison["variant_order"] == ["baseline_structural", "overlap_guided_selector"]
    assert comparison["scorer_name"] == "title_sentence_hybrid"
    for variant_name in comparison["variant_order"]:
        payload = comparison["variants"][variant_name]
        assert "target_failure_buckets" in payload["comparison_metrics"]
        assert "scorer_summary" in payload
        assert (output_dir / variant_name / "parent_title" / "hotpotqa_graph_eval_records.jsonl").exists()
        assert (output_dir / variant_name / "identity" / "hotpotqa_graph_eval_records.jsonl").exists()
    selector_summary = comparison["variants"]["overlap_guided_selector"]["scorer_summary"]
    assert selector_summary["available"] is True
    assert "component_description" in selector_summary
    assert (output_dir / "hotpotqa_question_conditioned_relevance_summary.json").exists()
