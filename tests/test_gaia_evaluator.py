import pytest
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from evaluation.gaia_evaluator import GAIAEvaluator
from evaluation.gaia_loader import GAIAQuestion
from evaluation.report import generate_report

QUESTIONS = [
    GAIAQuestion(task_id="q1", question="What is 2+2?", answer="4", level=1, file_name=""),
    GAIAQuestion(task_id="q2", question="Capital of France?", answer="Paris", level=1, file_name=""),
    GAIAQuestion(task_id="q3", question="What year did WWII end?", answer="1945", level=1, file_name=""),
]


async def test_evaluator_perfect_score():
    answers = ["4", "Paris", "1945"]
    call_count = 0

    async def mock_run(agent, question):
        nonlocal call_count
        result = MagicMock()
        result.final_output = answers[call_count]
        call_count += 1
        return result

    with patch("evaluation.gaia_evaluator.Runner.run", side_effect=mock_run):
        evaluator = GAIAEvaluator()
        results = await evaluator.evaluate(MagicMock(), QUESTIONS)

    assert results["score"] == pytest.approx(1.0)
    assert results["correct"] == 3
    assert results["total"] == 3


async def test_evaluator_partial_score():
    answers = ["4", "wrong", "1945"]
    call_count = 0

    async def mock_run(agent, question):
        nonlocal call_count
        result = MagicMock()
        result.final_output = answers[call_count]
        call_count += 1
        return result

    with patch("evaluation.gaia_evaluator.Runner.run", side_effect=mock_run):
        evaluator = GAIAEvaluator()
        results = await evaluator.evaluate(MagicMock(), QUESTIONS)

    assert results["score"] == pytest.approx(2 / 3)
    assert results["correct"] == 2
    assert len(results["per_question"]) == 3


def test_report_generates_summary(tmp_path):
    before = {
        "score": 0.3,
        "correct": 3,
        "total": 10,
        "per_question": [
            {"task_id": "q1", "question": "Q1?", "expected": "A", "predicted": "A", "correct": True},
        ],
    }
    after = {
        "score": 0.5,
        "correct": 5,
        "total": 10,
        "per_question": [
            {"task_id": "q1", "question": "Q1?", "expected": "A", "predicted": "A", "correct": True},
        ],
    }
    output = str(tmp_path / "report.json")
    summary = generate_report(before, after, differentiation_log=[], output_path=output)

    assert "+20.0%" in summary or "0.2" in summary
    assert Path(output).exists()
    data = json.loads(Path(output).read_text())
    assert data["delta"]["score"] == pytest.approx(0.2)
