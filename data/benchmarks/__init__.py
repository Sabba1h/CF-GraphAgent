"""Benchmark dataset loaders."""

from data.benchmarks.answer_normalization import normalize_answer
from data.benchmarks.common import BenchmarkExample, parse_indices
from data.benchmarks.hotpotqa import load_hotpotqa, load_hotpotqa_task_samples
from data.benchmarks.twowiki import load_twowiki, load_twowiki_task_samples

__all__ = [
    "BenchmarkExample",
    "load_hotpotqa",
    "load_hotpotqa_task_samples",
    "load_twowiki",
    "load_twowiki_task_samples",
    "normalize_answer",
    "parse_indices",
]
