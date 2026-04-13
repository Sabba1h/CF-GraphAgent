"""Tests for core schema compatibility imports."""


def test_core_and_schema_imports_are_available() -> None:
    from core import CandidateAction as CoreCandidateAction
    from core import EvalResult, GraphEpisodeState, RewardBreakdown, StateSnapshot, TaskSample
    from core.transition import TransitionEngine
    from graph.graph_backend import GraphBackend
    from observation.prompt_renderer import PromptRenderer
    from observation.structured_view import StructuredObservationView
    from observation.text_renderer import TextObservationRenderer
    from agent.graph_action_parser import GraphActionParser
    from agent.graph_rollout_manager import GraphRolloutManager
    from schemas import CandidateAction as CompatCandidateAction

    assert CoreCandidateAction is CompatCandidateAction
    assert TransitionEngine is not None
    assert GraphBackend is not None
    assert PromptRenderer is not None
    assert StructuredObservationView is not None
    assert TextObservationRenderer is not None
    assert GraphActionParser is not None
    assert GraphRolloutManager is not None
    assert EvalResult(score=1.0, is_correct=True, reason="ok").score == 1.0
    assert RewardBreakdown(total_reward=0.1).total_reward == 0.1
    assert TaskSample(query="q").query == "q"
    assert GraphEpisodeState(task=TaskSample(query="q"), max_steps=1).steps_left == 1
    assert StateSnapshot(query="q", step_index=0).step_index == 0
