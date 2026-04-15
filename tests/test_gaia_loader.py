import pytest
from unittest.mock import patch
from evaluation.gaia_loader import GAIALoader, GAIAQuestion


def make_mock_dataset():
    level1 = [
        {
            "task_id": f"task_{i}",
            "Question": f"Question {i}?",
            "Level": 1,
            "Final answer": f"Answer{i}",
            "file_name": "",
        }
        for i in range(20)
    ]
    level2 = [
        {
            "task_id": "task_l2",
            "Question": "Level 2 question?",
            "Level": 2,
            "Final answer": "L2Answer",
            "file_name": "",
        }
    ]
    return level1 + level2


def test_loader_filters_to_level1_only():
    with patch("evaluation.gaia_loader.load_dataset", return_value=make_mock_dataset()):
        loader = GAIALoader()
        questions = loader.load()
    assert len(questions) == 20
    assert all(q.level == 1 for q in questions)


def test_loader_split_no_overlap():
    with patch("evaluation.gaia_loader.load_dataset", return_value=make_mock_dataset()):
        loader = GAIALoader(n_probe=5)
        probe, evaluation = loader.split()
    assert len(probe) == 5
    assert len(evaluation) == 15
    probe_ids = {q.task_id for q in probe}
    eval_ids = {q.task_id for q in evaluation}
    assert probe_ids.isdisjoint(eval_ids)


def test_gaia_question_has_required_fields():
    with patch("evaluation.gaia_loader.load_dataset", return_value=make_mock_dataset()):
        loader = GAIALoader()
        questions = loader.load()
    q = questions[0]
    assert hasattr(q, "task_id")
    assert hasattr(q, "question")
    assert hasattr(q, "answer")
    assert hasattr(q, "level")
    assert hasattr(q, "file_name")


def test_loader_split_is_deterministic():
    with patch("evaluation.gaia_loader.load_dataset", return_value=make_mock_dataset()):
        loader = GAIALoader(n_probe=5, seed=42)
        probe1, _ = loader.split()
    with patch("evaluation.gaia_loader.load_dataset", return_value=make_mock_dataset()):
        loader2 = GAIALoader(n_probe=5, seed=42)
        probe2, _ = loader2.split()
    assert [q.task_id for q in probe1] == [q.task_id for q in probe2]
