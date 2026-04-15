import pytest
from pydantic import ValidationError
from stem_agent.models import ResearchSignals, AgentConfig, ProbeResult, VALID_TOOLS


def test_research_signals_valid():
    signals = ResearchSignals(
        patterns=["break into sub-questions"],
        tool_patterns=["web_search"],
        failure_modes=["hallucinated citations"],
        topology="single",
    )
    assert signals.topology == "single"


def test_research_signals_invalid_topology():
    with pytest.raises(ValidationError):
        ResearchSignals(
            patterns=[],
            tool_patterns=[],
            failure_modes=[],
            topology="invalid_topology",
        )


def test_agent_config_valid():
    config = AgentConfig(
        system_prompt="You are a research agent.",
        tools=["web_search"],
        topology="single",
        probe_threshold=0.4,
    )
    assert config.probe_threshold == 0.4


def test_agent_config_invalid_tool_triggers_apoptosis():
    """Invalid tool name → ValidationError → apoptosis in DifferentiationController."""
    with pytest.raises(ValidationError):
        AgentConfig(
            system_prompt="You are a research agent.",
            tools=["nonexistent_tool"],
            topology="single",
            probe_threshold=0.4,
        )


def test_agent_config_threshold_out_of_range():
    with pytest.raises(ValidationError):
        AgentConfig(
            system_prompt="You are a research agent.",
            tools=["web_search"],
            topology="single",
            probe_threshold=1.5,
        )


def test_probe_result_valid():
    result = ProbeResult(
        score=0.6,
        failed_questions=["What year did X happen?"],
        failure_reasons=["Expected '1969', got '1970'"],
    )
    assert result.score == 0.6


def test_valid_tools_set_is_non_empty():
    assert len(VALID_TOOLS) > 0
    assert "web_search" in VALID_TOOLS
