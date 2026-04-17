import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from stem_agent.probe_runner import ProbeRunner, build_tools, normalize
from stem_agent.models import AgentConfig, ProbeResult
from evaluation.gaia_loader import GAIAQuestion

CONFIG = AgentConfig(
    system_prompt="You are a research agent.",
    tools=["web_search"],
    topology="single",
    probe_threshold=0.4,
)

QUESTIONS = [
    GAIAQuestion(task_id=f"q{i}", question=f"Q{i}?", answer=str(i), level=1, file_name="")
    for i in range(5)
]


def test_normalize_strips_and_lowercases():
    assert normalize("  Paris  ") == "paris"
    assert normalize("42") == "42"
    assert normalize("NEW YORK") == "new york"


def test_build_tools_web_search():
    tools = build_tools(["web_search"])
    assert len(tools) == 1


async def test_probe_runner_perfect_score():
    """Agent always returns correct answer → score 1.0."""
    call_count = 0

    async def mock_run(agent, question):
        nonlocal call_count
        result = MagicMock()
        result.final_output = str(call_count)
        call_count += 1
        return result

    with patch("stem_agent.probe_runner.Runner.run", side_effect=mock_run):
        runner = ProbeRunner()
        result = await runner.run(CONFIG, QUESTIONS)

    assert result.score == pytest.approx(1.0)
    assert result.failed_questions == []
    assert result.failure_reasons == []


async def test_probe_runner_partial_score():
    """First 3 correct, last 2 wrong → score 0.6."""
    call_count = 0

    async def mock_run(agent, question):
        nonlocal call_count
        result = MagicMock()
        result.final_output = str(call_count) if call_count < 3 else "wrong_answer"
        call_count += 1
        return result

    with patch("stem_agent.probe_runner.Runner.run", side_effect=mock_run):
        runner = ProbeRunner()
        result = await runner.run(CONFIG, QUESTIONS)

    assert result.score == pytest.approx(0.6)
    assert len(result.failed_questions) == 2
    assert len(result.failure_reasons) == 2


async def test_probe_runner_zero_score():
    mock_result = MagicMock()
    mock_result.final_output = "completely_wrong"

    with patch("stem_agent.probe_runner.Runner.run", new_callable=AsyncMock, return_value=mock_result):
        runner = ProbeRunner()
        result = await runner.run(CONFIG, QUESTIONS)

    assert result.score == pytest.approx(0.0)
    assert result.score < CONFIG.probe_threshold
    assert len(result.failed_questions) == 5
