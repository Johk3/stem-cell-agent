import os
from openai import AsyncOpenAI
from agents import Agent, Runner, WebSearchTool

from stem_agent.models import ResearchSignals

SEARCH_QUERIES = [
    "GAIA benchmark top performing agent architecture tools 2024 2025",
    "deep research AI agent system prompt best practices chain of thought",
    "ReAct agent research task web search decomposition strategy",
    "common failure modes AI research agents GAIA benchmark wrong answers",
]

MAX_CHARS_PER_RESULT = 3000
MAX_TOTAL_CHARS = 10000

SYNTHESIS_PROMPT = """\
You have analyzed web search results about how deep research AI agents are built \
and evaluated on the GAIA benchmark.

Search results:
{search_results}

Extract structured signals. Use only these tool names: web_search, file_read, \
file_write, python_repl.

Return a JSON object with:
- patterns: list of general reasoning/research strategies that work well
- tool_patterns: list of specific tools and combinations that recur
- failure_modes: list of common ways research agents fail
- topology: "single" if a single agent works best, "orchestrator+subagents" if \
  multi-agent works better
"""


class SignalReader:
    def __init__(self, model: str = "gpt-5.4"):
        self.model = model
        self._client: AsyncOpenAI | None = None

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI()
        return self._client

    async def read(self) -> ResearchSignals:
        search_results = await self._run_searches()
        return await self._synthesize(search_results)

    async def _run_searches(self) -> str:
        search_agent = Agent(
            name="signal_searcher",
            instructions="Search the web and return the most relevant results verbatim.",
            tools=[WebSearchTool()],
            model=self.model,
        )
        results = []
        for query in SEARCH_QUERIES:
            result = await Runner.run(search_agent, query)
            output = result.final_output[:MAX_CHARS_PER_RESULT]
            results.append(f"Query: {query}\nResult: {output}")
        combined = "\n\n---\n\n".join(results)
        return combined[:MAX_TOTAL_CHARS]

    async def _synthesize(self, search_results: str) -> ResearchSignals:
        response = await self._get_client().beta.chat.completions.parse(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": SYNTHESIS_PROMPT.format(search_results=search_results),
                }
            ],
            response_format=ResearchSignals,
        )
        return response.choices[0].message.parsed
