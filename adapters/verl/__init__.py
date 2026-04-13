"""Minimal verl adapter skeletons.

This package defines bridge interfaces only. It does not integrate with a
real verl trainer or distributed rollout runtime.
"""

from adapters.verl.action_bridge import VerlActionBridge
from adapters.verl.batch_backend import BatchedGraphBackend
from adapters.verl.batch_state import BatchedEpisodeState
from adapters.verl.prompt_builder import VerlPromptBuilder
from adapters.verl.reward_bridge import VerlRewardBridge, VerlRewardOutput
from adapters.verl.rollout_adapter import VerlBatchStep, VerlRolloutAdapter, VerlRolloutResult
from adapters.verl.trainer_hooks import VerlTrainerHooks

__all__ = [
    "BatchedEpisodeState",
    "BatchedGraphBackend",
    "VerlActionBridge",
    "VerlBatchStep",
    "VerlPromptBuilder",
    "VerlRewardBridge",
    "VerlRewardOutput",
    "VerlRolloutAdapter",
    "VerlRolloutResult",
    "VerlTrainerHooks",
]
