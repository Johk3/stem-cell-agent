from agents import Agent, Runner
from evaluation.gaia_loader import GAIAQuestion


def normalize(answer: str) -> str:
    return answer.strip().lower()


class GAIAEvaluator:
    async def evaluate(self, agent: Agent, questions: list[GAIAQuestion]) -> dict:
        correct = 0
        per_question = []

        for q in questions:
            result = await Runner.run(agent, q.question)
            predicted = normalize(result.final_output)
            expected = normalize(q.answer)
            is_correct = predicted == expected
            if is_correct:
                correct += 1
            per_question.append(
                {
                    "task_id": q.task_id,
                    "question": q.question,
                    "expected": q.answer,
                    "predicted": result.final_output,
                    "correct": is_correct,
                }
            )

        total = len(questions)
        return {
            "score": correct / total if total else 0.0,
            "correct": correct,
            "total": total,
            "per_question": per_question,
        }
