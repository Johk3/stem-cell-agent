import asyncio
from agents import Agent, Runner
from evaluation.gaia_loader import GAIAQuestion
from evaluation.scoring import is_correct


class GAIAEvaluator:
    async def evaluate(self, agent: Agent, questions: list[GAIAQuestion]) -> dict:
        async def run_one(q: GAIAQuestion) -> dict:
            result = await Runner.run(agent, q.question)
            correct_flag = is_correct(result.final_output, q.answer)
            return {
                "task_id": q.task_id,
                "question": q.question,
                "expected": q.answer,
                "predicted": result.final_output,
                "correct": correct_flag,
            }

        per_question = await asyncio.gather(*[run_one(q) for q in questions])

        correct = sum(1 for r in per_question if r["correct"])
        total = len(questions)
        return {
            "score": correct / total if total else 0.0,
            "correct": correct,
            "total": total,
            "per_question": list(per_question),
        }
