import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from stem_agent.signal_reader import SignalReader
from stem_agent.models import ResearchSignals


def make_mock_signals() -> ResearchSignals:
    return ResearchSignals(
        patterns=["decompose into sub-questions", "verify facts"],
        tool_patterns=["web_search"],
        failure_modes=["hallucinated citations", "missed numerical answer"],
        topology="single",
    )


async def test_signal_reader_returns_research_signals():
    reader = SignalReader(model="gpt-4o")
    mock_signals = make_mock_signals()
    with patch.object(reader, "_run_searches", new_callable=AsyncMock, return_value="search results"):
        with patch.object(reader, "_synthesize", new_callable=AsyncMock, return_value=mock_signals):
            result = await reader.read()
    assert isinstance(result, ResearchSignals)
    assert result.topology in ("single", "orchestrator+subagents")


async def test_signal_reader_synthesize_calls_openai():
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.parsed = make_mock_signals()

    mock_client = MagicMock()
    mock_client.beta.chat.completions.parse = AsyncMock(return_value=mock_response)

    reader = SignalReader(model="gpt-4o")
    reader._client = mock_client

    result = await reader._synthesize("web search results text")

    assert isinstance(result, ResearchSignals)
    mock_client.beta.chat.completions.parse.assert_called_once()
    call_kwargs = mock_client.beta.chat.completions.parse.call_args.kwargs
    assert call_kwargs["response_format"] is ResearchSignals


async def test_signal_reader_passes_search_results_to_synthesize():
    reader = SignalReader(model="gpt-4o")
    captured = {}

    async def fake_synthesize(search_results: str) -> ResearchSignals:
        captured["search_results"] = search_results
        return make_mock_signals()

    with patch.object(reader, "_run_searches", new_callable=AsyncMock, return_value="my results"):
        with patch.object(reader, "_synthesize", side_effect=fake_synthesize):
            await reader.read()

    assert captured["search_results"] == "my results"
