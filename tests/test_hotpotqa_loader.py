"""Tests for HotpotQA benchmark ingestion."""

from __future__ import annotations

import json

from data.benchmarks.hotpotqa import load_hotpotqa, load_hotpotqa_task_samples


def _hotpot_records() -> list[dict]:
    return [
        {
            "_id": "hp-1",
            "question": "Which city was the director of Forrest Gump born in?",
            "answer": " Chicago! ",
            "type": "bridge",
            "level": "easy",
            "supporting_facts": [["Forrest Gump", 0], ["Robert Zemeckis", 1]],
            "context": [
                ["Forrest Gump", ["Forrest Gump was directed by Robert Zemeckis."]],
                ["Robert Zemeckis", ["Robert Zemeckis was born in Chicago."]],
            ],
        },
        {
            "_id": "hp-2",
            "question": "Who composed the music for Back to the Future?",
            "answer": "Alan Silvestri",
            "type": "bridge",
            "level": "medium",
            "supporting_facts": [["Back to the Future", 0]],
            "context": [["Back to the Future", ["The film's music was composed by Alan Silvestri."]]],
        },
    ]


def test_hotpotqa_loader_preserves_raw_answer_context_and_supporting_facts(tmp_path) -> None:
    path = tmp_path / "hotpot.json"
    path.write_text(json.dumps(_hotpot_records()), encoding="utf-8")

    examples = load_hotpotqa(path, limit=1, split="dev")

    assert len(examples) == 1
    example = examples[0]
    assert example.dataset_name == "hotpotqa"
    assert example.question_id == "hp-1"
    assert example.answer == " Chicago! "
    assert example.normalized_answer == "chicago"
    assert example.context[0]["title"] == "Forrest Gump"
    assert example.context[0]["sentences"] == ["Forrest Gump was directed by Robert Zemeckis."]
    assert example.supporting_facts[1]["title"] == "Robert Zemeckis"
    assert example.supporting_facts[1]["sentence_index"] == 1

    task = example.to_task_sample()
    assert task.dataset_name == "hotpotqa"
    assert task.query == example.question
    assert task.ground_truth == " Chicago! "
    assert task.metadata["raw_answer"] == " Chicago! "
    assert task.metadata["normalized_answer"] == "chicago"
    assert task.metadata["context_titles"] == ["Forrest Gump", "Robert Zemeckis"]
    assert task.metadata["supporting_facts"] == example.supporting_facts
    assert task.metadata["context"] == example.context
    assert task.metadata["split"] == "dev"


def test_hotpotqa_task_sample_subset_indices(tmp_path) -> None:
    path = tmp_path / "hotpot.json"
    path.write_text(json.dumps({"data": _hotpot_records()}), encoding="utf-8")

    tasks = load_hotpotqa_task_samples(path, indices=[1])

    assert len(tasks) == 1
    assert tasks[0].metadata["question_id"] == "hp-2"
    assert tasks[0].dataset_name == "hotpotqa"
    assert tasks[0].ground_truth == "Alan Silvestri"


def test_hotpotqa_loader_handles_fullwiki_columnar_context_and_supporting_facts(tmp_path) -> None:
    path = tmp_path / "hotpot_columnar.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "hp-col-1",
                    "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
                    "answer": "yes",
                    "type": "comparison",
                    "level": "medium",
                    "context": {
                        "title": ["Scott Derrickson", "Ed Wood"],
                        "sentences": [
                            ["Scott Derrickson is an American film director."],
                            ["Ed Wood was an American filmmaker."],
                        ],
                    },
                    "supporting_facts": {
                        "title": ["Scott Derrickson", "Ed Wood"],
                        "sent_id": [0, 0],
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    example = load_hotpotqa(path, limit=1, split="validation")[0]

    assert example.context_titles == ["Scott Derrickson", "Ed Wood"]
    assert example.context == [
        {
            "title": "Scott Derrickson",
            "sentences": ["Scott Derrickson is an American film director."],
            "raw": {
                "title": "Scott Derrickson",
                "sentences": ["Scott Derrickson is an American film director."],
            },
        },
        {
            "title": "Ed Wood",
            "sentences": ["Ed Wood was an American filmmaker."],
            "raw": {
                "title": "Ed Wood",
                "sentences": ["Ed Wood was an American filmmaker."],
            },
        },
    ]
    assert example.supporting_facts == [
        {
            "title": "Scott Derrickson",
            "sent_id": 0,
            "sentence_index": 0,
            "raw": {"title": "Scott Derrickson", "sent_id": 0},
        },
        {
            "title": "Ed Wood",
            "sent_id": 0,
            "sentence_index": 0,
            "raw": {"title": "Ed Wood", "sent_id": 0},
        },
    ]
    assert example.metadata["raw_context"]["title"] == ["Scott Derrickson", "Ed Wood"]
    assert example.metadata["raw_supporting_facts"]["sent_id"] == [0, 0]
    assert example.metadata["warnings"] == []

    task = example.to_task_sample()
    assert task.metadata["context_titles"] == ["Scott Derrickson", "Ed Wood"]
    assert len(task.metadata["supporting_facts"]) == 2


def test_hotpotqa_loader_warns_and_uses_shortest_length_for_column_mismatch(tmp_path) -> None:
    path = tmp_path / "hotpot_mismatch.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "hp-col-2",
                    "question": "Question?",
                    "answer": "Answer",
                    "context": {
                        "title": ["Doc A", "Doc B"],
                        "sentences": [["Only one sentence row."]],
                    },
                    "supporting_facts": {
                        "title": ["Doc A"],
                        "sent_id": [0, 1],
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    example = load_hotpotqa(path, limit=1)[0]

    assert example.context_titles == ["Doc A"]
    assert example.supporting_facts == [
        {
            "title": "Doc A",
            "sent_id": 0,
            "sentence_index": 0,
            "raw": {"title": "Doc A", "sent_id": 0},
        }
    ]
    assert example.metadata["warnings"] == [
        {
            "field": "context",
            "reason": "column_length_mismatch",
            "lengths": {"title": 2, "sentences": 1},
            "paired_count": 1,
        },
        {
            "field": "supporting_facts",
            "reason": "column_length_mismatch",
            "lengths": {"title": 1, "sent_id": 2},
            "paired_count": 1,
        },
    ]
