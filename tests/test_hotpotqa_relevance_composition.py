"""Tests for HotpotQA relevance composition comparison."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.compare_hotpotqa_relevance_composition import (
    COMPOSITION_VARIANT,
    DEFAULT_SCORER_NAME,
    compare_hotpotqa_relevance_composition,
    variant_config,
)


def _fixture_record(answer: str = "First Title") -> dict:
    return {
        "id": "hp-relevance-composition-1",
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
    data_path = path / "hotpotqa_relevance_composition_fixture.json"
    data_path.write_text(json.dumps([_fixture_record()], ensure_ascii=False), encoding="utf-8")
    return data_path


def test_relevance_composition_configs_are_explicit() -> None:
    baseline = variant_config("baseline_structural")
    candidate_only = variant_config("candidate_only")
    selector_only = variant_config("selector_only")
    composed = variant_config("candidate_plus_selector")

    assert baseline == {
        "candidate_generator": "baseline_generator",
        "policy": "sentence_chain",
        "selector": "latest_sentence",
    }
    assert candidate_only["candidate_generator"] == "overlap_ranked_generator"
    assert candidate_only["selector"] == "latest_sentence"
    assert selector_only["candidate_generator"] == "baseline_generator"
    assert selector_only["selector"] == "overlap_guided_sentence"
    assert composed["candidate_generator"] == "overlap_ranked_generator"
    assert composed["selector"] == "overlap_guided_sentence"


def test_relevance_composition_smoke_uses_fixed_mapper_extractor_samples_and_scorer(tmp_path: Path) -> None:
    data_path = _write_fixture(tmp_path)
    output_dir = tmp_path / "composition-output"

    comparison = compare_hotpotqa_relevance_composition(
        path=data_path,
        split="validation",
        limit=1,
        reward_mode="baseline",
        variants=["baseline_structural", "candidate_only", "selector_only", "candidate_plus_selector"],
        output_dir=output_dir,
    )

    assert comparison["fixed_mapper_name"] == "parent_title"
    assert comparison["fixed_extractor_name"] == "full_sentence"
    assert comparison["sample_indices"] == [0]
    assert comparison["scorer_name"] == DEFAULT_SCORER_NAME
    assert comparison["scorer_recipe_consistency"]["shared_scorer_name"] == DEFAULT_SCORER_NAME
    assert comparison["variant_order"] == [
        "baseline_structural",
        "candidate_only",
        "selector_only",
        "candidate_plus_selector",
    ]

    for variant_name in comparison["variant_order"]:
        payload = comparison["variants"][variant_name]
        assert payload["mapper_name"] == "parent_title"
        assert payload["extractor_name"] == "full_sentence"
        assert payload["comparison_metrics"]["target_failure_buckets"].keys() == {
            "path_touched_wrong_region",
            "selected_sentence_not_relevant",
        }
        assert (output_dir / variant_name / "parent_title" / "hotpotqa_graph_eval_records.jsonl").exists()
        assert (output_dir / variant_name / "identity" / "hotpotqa_graph_eval_records.jsonl").exists()

    candidate_payload = comparison["variants"]["candidate_only"]
    selector_payload = comparison["variants"]["selector_only"]
    composed_payload = comparison["variants"][COMPOSITION_VARIANT]
    assert candidate_payload["candidate_generator_name"] == "overlap_ranked_generator"
    assert selector_payload["selector_name"] == "overlap_guided_sentence"
    assert composed_payload["candidate_generator_name"] == "overlap_ranked_generator"
    assert composed_payload["selector_name"] == "overlap_guided_sentence"

    gain_summary = comparison["composition_gain_summary"]
    assert gain_summary["available"] is True
    assert gain_summary["composition_variant"] == COMPOSITION_VARIANT
    assert gain_summary["single_layer_variants"] == ["candidate_only", "selector_only"]
    assert "metric_judgments" in gain_summary
    assert "selected_sentence_contains_gold_rate" in gain_summary["metric_judgments"]
    assert gain_summary["judgment"] in {
        "composition_outperforms_best_single_layer_on_target_metrics",
        "mixed_composition_signal",
        "no_clear_composition_gain_over_best_single_layer",
    }
    assert (output_dir / "hotpotqa_relevance_composition_summary.json").exists()
