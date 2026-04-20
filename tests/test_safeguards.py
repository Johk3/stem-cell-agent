import pytest
from unittest.mock import AsyncMock, MagicMock
from stem_agent.differentiation import DifferentiationController
from stem_agent.models import ResearchSignals, AgentConfig, ProbeResult
from evaluation.gaia_loader import GAIAQuestion

SIGNALS = ResearchSignals(
    patterns=["decompose"],
    tool_patterns=["web_search"],
    failure_modes=["hallucination"],
    topology="single",
)

GOOD_CONFIG = AgentConfig(
    system_prompt="You are a research agent.",
    tools=["web_search"],
    topology="single",
    probe_threshold=0.4,
)

PROBE_QUESTIONS = [
    GAIAQuestion(task_id=f"q{i}", question=f"Q{i}?", answer=str(i), level=1, file_name="")
    for i in range(5)
]


def make_controller(signal_reader, synthesizer, probe_runner, max_attempts=3):
    return DifferentiationController(
        signal_reader=signal_reader,
        config_synthesizer=synthesizer,
        probe_runner=probe_runner,
        probe_questions=PROBE_QUESTIONS,
        max_attempts=max_attempts,
    )


async def test_commit_on_good_probe_score():
    signal_reader = MagicMock()
    signal_reader.read = AsyncMock(return_value=SIGNALS)

    synthesizer = MagicMock()
    synthesizer.synthesize = AsyncMock(return_value=GOOD_CONFIG)

    probe_runner = MagicMock()
    probe_runner.run = AsyncMock(return_value=ProbeResult(
        score=0.8, failed_questions=[], failure_reasons=[]
    ))

    controller = make_controller(signal_reader, synthesizer, probe_runner)
    config = await controller.differentiate()

    assert config.tools == GOOD_CONFIG.tools
    outcomes = [entry["outcome"] for entry in controller.log]
    assert "candidate" in outcomes
    assert "commit" in outcomes


async def test_apoptosis_on_synthesis_exception():
    """Exception from synthesizer → apoptosis, config stack unchanged, retry."""
    signal_reader = MagicMock()
    signal_reader.read = AsyncMock(return_value=SIGNALS)

    call_count = 0

    async def synthesize_side_effect(signals, failure_reasons):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ValueError("Invalid tools: {'bad_tool'}")
        return GOOD_CONFIG

    synthesizer = MagicMock()
    synthesizer.synthesize = AsyncMock(side_effect=synthesize_side_effect)

    probe_runner = MagicMock()
    probe_runner.run = AsyncMock(return_value=ProbeResult(
        score=0.8, failed_questions=[], failure_reasons=[]
    ))

    controller = make_controller(signal_reader, synthesizer, probe_runner)
    config = await controller.differentiate()

    assert config.tools == GOOD_CONFIG.tools
    outcomes = [entry["outcome"] for entry in controller.log]
    assert "apoptosis" in outcomes
    assert "commit" in outcomes


async def test_retrograde_migration_on_low_probe_score():
    """Low probe score → rollback, retry with failure_reasons passed to synthesizer."""
    signal_reader = MagicMock()
    signal_reader.read = AsyncMock(return_value=SIGNALS)

    synthesizer = MagicMock()
    synthesizer.synthesize = AsyncMock(return_value=GOOD_CONFIG)

    probe_call_count = 0

    async def probe_side_effect(config, questions):
        nonlocal probe_call_count
        probe_call_count += 1
        if probe_call_count == 1:
            return ProbeResult(
                score=0.1,
                failed_questions=["Q0?"],
                failure_reasons=["Expected '0', got 'wrong'"],
            )
        return ProbeResult(score=0.8, failed_questions=[], failure_reasons=[])

    probe_runner = MagicMock()
    probe_runner.run = AsyncMock(side_effect=probe_side_effect)

    controller = make_controller(signal_reader, synthesizer, probe_runner)
    config = await controller.differentiate()

    assert config.tools == GOOD_CONFIG.tools
    outcomes = [entry["outcome"] for entry in controller.log]
    assert "retrograde" in outcomes
    assert "candidate" in outcomes
    assert "commit" in outcomes

    second_call = synthesizer.synthesize.call_args_list[1]
    assert second_call.kwargs["failure_reasons"] == ["Expected '0', got 'wrong'"]


async def test_abort_after_max_attempts():
    signal_reader = MagicMock()
    signal_reader.read = AsyncMock(return_value=SIGNALS)

    synthesizer = MagicMock()
    synthesizer.synthesize = AsyncMock(return_value=GOOD_CONFIG)

    probe_runner = MagicMock()
    probe_runner.run = AsyncMock(return_value=ProbeResult(
        score=0.0,
        failed_questions=["Q0?"],
        failure_reasons=["always wrong"],
    ))

    controller = make_controller(signal_reader, synthesizer, probe_runner, max_attempts=3)

    with pytest.raises(RuntimeError, match="Differentiation failed"):
        await controller.differentiate()

    assert len(controller.log) == 3
    assert all(entry["outcome"] == "retrograde" for entry in controller.log)


async def test_log_saved_to_disk(tmp_path):
    signal_reader = MagicMock()
    signal_reader.read = AsyncMock(return_value=SIGNALS)

    synthesizer = MagicMock()
    synthesizer.synthesize = AsyncMock(return_value=GOOD_CONFIG)

    probe_runner = MagicMock()
    probe_runner.run = AsyncMock(return_value=ProbeResult(
        score=0.8, failed_questions=[], failure_reasons=[]
    ))

    controller = DifferentiationController(
        signal_reader=signal_reader,
        config_synthesizer=synthesizer,
        probe_runner=probe_runner,
        probe_questions=PROBE_QUESTIONS,
        log_dir=str(tmp_path),
    )
    await controller.differentiate()

    log_files = list(tmp_path.glob("differentiation_*.json"))
    assert len(log_files) == 1
