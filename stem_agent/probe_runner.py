import os
from agents import Agent, Runner, WebSearchTool
from evaluation.gaia_loader import GAIAQuestion
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


def normalize(answer: str) -> str:
    return answer.strip().lower()


class ProbeRunner:
    def __init__(self):
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o")

    async def run(
        self, config: AgentConfig, questions: list[GAIAQuestion]
    ) -> ProbeResult:
        agent = Agent(
            name="probe_agent",
            instructions=config.system_prompt,
            tools=build_tools(config.tools),
            model=self.model,
        )

        correct = 0
        failed_questions: list[str] = []
        failure_reasons: list[str] = []

        for q in questions:
            result = await Runner.run(agent, q.question)
            predicted = normalize(result.final_output)
            expected = normalize(q.answer)
            if predicted == expected:
                correct += 1
            else:
                failed_questions.append(q.question)
                failure_reasons.append(
                    f"Expected '{q.answer}', got '{result.final_output}'"
                )

        total = len(questions)
        score = correct / total if total else 0.0
        return ProbeResult(
            score=score,
            failed_questions=failed_questions,
            failure_reasons=failure_reasons,
        )
