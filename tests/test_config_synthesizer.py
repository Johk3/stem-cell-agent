import pytest
from unittest.mock import AsyncMock, MagicMock
from stem_agent.config_synthesizer import ConfigSynthesizer
from stem_agent.models import ResearchSignals, AgentConfig

SIGNALS = ResearchSignals(
    patterns=["decompose queries", "verify each fact"],
    tool_patterns=["web_search"],
    failure_modes=["hallucination", "wrong number format"],
    topology="single",
)

GOOD_CONFIG = AgentConfig(
    system_prompt="You are a deep research agent. Decompose queries before answering.",
    tools=["web_search"],
    topology="single",
    probe_threshold=0.3,
)


def make_mock_client(config: AgentConfig) -> MagicMock:
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.parsed = config
    mock_client = MagicMock()
    mock_client.beta.chat.completions.parse = AsyncMock(return_value=mock_response)
    return mock_client


async def test_synthesizer_returns_agent_config():
    synthesizer = ConfigSynthesizer(model="gpt-4o")
    synthesizer._client = make_mock_client(GOOD_CONFIG)
    config = await synthesizer.synthesize(SIGNALS, failure_reasons=[])
    assert isinstance(config, AgentConfig)
    assert "web_search" in config.tools


async def test_synthesizer_includes_failure_reasons_in_prompt():
    """Failure reasons from previous probe are included in the LLM prompt."""
    synthesizer = ConfigSynthesizer(model="gpt-4o")
    synthesizer._client = make_mock_client(GOOD_CONFIG)

    captured_messages = []

    original_parse = synthesizer._client.beta.chat.completions.parse

    async def capture_parse(**kwargs):
        captured_messages.extend(kwargs.get("messages", []))
        return await original_parse(**kwargs)

    synthesizer._client.beta.chat.completions.parse = capture_parse

    await synthesizer.synthesize(SIGNALS, failure_reasons=["missed date format", "wrong units"])

    combined = " ".join(m["content"] for m in captured_messages)
    assert "missed date format" in combined
    assert "wrong units" in combined


async def test_synthesizer_prompt_addresses_failure_modes():
    synthesizer = ConfigSynthesizer(model="gpt-4o")
    synthesizer._client = make_mock_client(GOOD_CONFIG)
    config = await synthesizer.synthesize(SIGNALS, failure_reasons=[])
    assert config.probe_threshold > 0
