from agents import Agent, Runner
from evaluation.gaia_loader import GAIAQuestion
from evaluation.scoring import is_correct


class GAIAEvaluator:
    async def evaluate(self, agent: Agent, questions: list[GAIAQuestion]) -> dict:
        correct = 0
        per_question = []

        for q in questions:
            result = await Runner.run(agent, q.question)
            correct_flag = is_correct(result.final_output, q.answer)
            if correct_flag:
                correct += 1
            per_question.append(
                {
                    "task_id": q.task_id,
                    "question": q.question,
                    "expected": q.answer,
                    "predicted": result.final_output,
                    "correct": correct_flag,
                }
            )

        total = len(questions)
        return {
            "score": correct / total if total else 0.0,
            "correct": correct,
            "total": total,
            "per_question": per_question,
        }
