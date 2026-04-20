import asyncio
import os
from agents import Agent, ModelSettings, Runner, WebSearchTool
from evaluation.gaia_loader import GAIAQuestion
from evaluation.scoring import extract_answer, is_correct
from stem_agent.models import AgentConfig, ProbeResult

TOOL_MAP = {
    "web_search": WebSearchTool,
}


def build_tools(tool_names: list[str]) -> list:
    """Instantiate SDK tool objects from tool name strings."""
    tools = []
    for name in tool_names:
        cls = TOOL_MAP.get(name)
        if cls is not None:
            tools.append(cls())
    return tools


def _make_failure_reason(question: str, expected: str, predicted: str) -> str:
    q_lower = question.lower()
    if any(w in q_lower for w in ("who", "name", "person", "author", "scientist")):
        qtype = "name lookup"
    elif any(w in q_lower for w in ("when", "year", "date")):
        qtype = "date/year lookup"
    elif any(w in q_lower for w in ("how many", "count", "number of")):
        qtype = "numeric count"
    elif any(w in q_lower for w in ("what is", "define", "meaning")):
        qtype = "definition/fact"
    else:
        qtype = "factual lookup"
    return (
        f"[{qtype}] Expected '{expected}', got '{predicted[:80]}' — "
        f"agent gave {'no answer' if not predicted.strip() else 'wrong answer'}"
    )


class ProbeRunner:
    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("OPENAI_CYTOPLASM_MODEL", "gpt-5.4-mini")

    async def run(
        self, config: AgentConfig, questions: list[GAIAQuestion]
    ) -> ProbeResult:
        agent = Agent(
            name="probe_agent",
            instructions=config.system_prompt,
            tools=build_tools(config.tools),
            model=self.model,
            model_settings=ModelSettings(temperature=0.0),
        )

        async def run_one(q: GAIAQuestion) -> tuple[bool, str, str, str]:
            result = await Runner.run(agent, q.question)
            answer = extract_answer(result.final_output)
            correct = is_correct(answer, q.answer)
            return correct, q.question, _make_failure_reason(q.question, q.answer, answer), answer

        outcomes = await asyncio.gather(*[run_one(q) for q in questions])

        correct = sum(1 for ok, _, _, _ in outcomes if ok)
        failed_questions = [q for ok, q, _, _ in outcomes if not ok]
        failure_reasons = [r for ok, _, r, _ in outcomes if not ok]
        few_shot = [
            {"question": q, "answer": a}
            for ok, q, _, a in outcomes if ok
        ]

        total = len(questions)
        return ProbeResult(
            score=correct / total if total else 0.0,
            failed_questions=failed_questions,
            failure_reasons=failure_reasons,
            few_shot_examples=few_shot,
        )
