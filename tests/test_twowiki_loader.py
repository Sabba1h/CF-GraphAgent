"""Tests for 2WikiMultiHopQA benchmark ingestion."""

from __future__ import annotations

import json

from data.benchmarks.twowiki import load_twowiki, load_twowiki_task_samples


def _twowiki_records() -> list[dict]:
    return [
        {
            "_id": "tw-1",
            "question": "What is the capital of the country where Example Person was born?",
            "answer": "London",
            "type": "compositional",
            "answer_type": "span",
            "evidences": [["Example Person", "born in", "England"], ["England", "capital", "London"]],
            "supporting_facts": [["Example Person", 0], ["England", 1]],
            "context": [
                ["Example Person", ["Example Person was born in England."]],
                ["England", ["London is the capital of England."]],
            ],
        },
        {
            "_id": "tw-2",
            "question": "Which film came first?",
            "answer": "Film A",
            "type": "comparison",
            "answer_type": "entity",
            "evidences": [["Film A", "release year", "1990"], ["Film B", "release year", "2000"]],
            "supporting_facts": [["Film A", 0], ["Film B", 0]],
            "context": [["Film A", ["Film A was released in 1990."]], ["Film B", ["Film B was released in 2000."]]],
        },
    ]


def test_twowiki_loader_preserves_evidences_and_task_metadata(tmp_path) -> None:
    path = tmp_path / "twowiki.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in _twowiki_records()), encoding="utf-8")

    examples = load_twowiki(path, limit=1, split="train")

    assert len(examples) == 1
    example = examples[0]
    assert example.dataset_name == "twowiki"
    assert example.question_id == "tw-1"
    assert example.answer == "London"
    assert example.normalized_answer == "london"
    assert example.context[1]["title"] == "England"
    assert example.supporting_facts[0]["sentence_index"] == 0
    assert example.metadata["evidences"] == [
        ["Example Person", "born in", "England"],
        ["England", "capital", "London"],
    ]

    task = example.to_task_sample()
    assert task.dataset_name == "twowiki"
    assert task.ground_truth == "London"
    assert task.metadata["raw_answer"] == "London"
    assert task.metadata["normalized_answer"] == "london"
    assert task.metadata["context_titles"] == ["Example Person", "England"]
    assert task.metadata["evidences"] == example.metadata["evidences"]
    assert task.metadata["split"] == "train"


def test_twowiki_task_sample_subset_indices(tmp_path) -> None:
    path = tmp_path / "twowiki.json"
    path.write_text(json.dumps(_twowiki_records()), encoding="utf-8")

    tasks = load_twowiki_task_samples(path, indices=[1])

    assert len(tasks) == 1
    assert tasks[0].metadata["question_id"] == "tw-2"
    assert tasks[0].dataset_name == "twowiki"
    assert tasks[0].metadata["question_type"] == "comparison"
