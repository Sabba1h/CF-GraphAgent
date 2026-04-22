"""Compatibility schema exports.

New code should prefer the modules under core/. This file remains as a
compatibility shim for existing stage-1 imports.
"""

from __future__ import annotations

from typing import Any

from core.actions import ActionType, CandidateAction
from core.episode_result import (
    AnswerResult,
    CounterfactualComparisonResult,
    EpisodeSummary,
    EvalResult,
    RewardBreakdown,
    RewardResult,
    StepRecord,
    TransitionResult,
)
from core.counterfactual_dataset import CounterfactualDatasetExample
from core.experiment_config import CounterfactualMode, ExperimentConfig, RewardMode
from core.experiment_result import ExperimentResult, ExperimentStepTrace
from core.snapshots import StateSnapshot
from core.state import GraphEpisodeState
from core.task import TaskSample

ObservationDict = dict[str, Any]

__all__ = [
    "ActionType",
    "AnswerResult",
    "CandidateAction",
    "CounterfactualComparisonResult",
    "CounterfactualDatasetExample",
    "CounterfactualMode",
    "EpisodeSummary",
    "EvalResult",
    "ExperimentConfig",
    "ExperimentResult",
    "ExperimentStepTrace",
    "GraphEpisodeState",
    "ObservationDict",
    "RewardBreakdown",
    "RewardMode",
    "RewardResult",
    "StateSnapshot",
    "StepRecord",
    "TaskSample",
    "TransitionResult",
]
