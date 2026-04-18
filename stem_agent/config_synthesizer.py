from openai import AsyncOpenAI
from stem_agent.models import ResearchSignals, AgentConfig, VALID_TOOLS

SYNTHESIS_PROMPT = """\
You are designing a system prompt for a deep research AI agent evaluated on GAIA \
Level 1 benchmark questions. These questions require factual lookups, multi-step \
reasoning, and precise short answers (a name, number, or short phrase).

Environmental signals from top-performing agents:
- Reasoning patterns: {patterns}
- Known failure modes to address: {failure_modes}

Previous attempt failure reasons (empty = first attempt): {failure_reasons}

Available tools: {valid_tools}
Topology: single (only option)

Design an AgentConfig:
- system_prompt: 150-300 words. Focus on: always attempt every question, return \
  only the precise short answer (no explanation), search the web for factual \
  information, break multi-step questions into sub-searches, never refuse a \
  question due to content type.
- tools: use ["web_search"] — the only available tool.
- topology: "single"
- probe_threshold: 0.3

Return valid JSON matching the AgentConfig schema exactly.
"""


class ConfigSynthesizer:
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self._client: AsyncOpenAI | None = None

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI()
        return self._client

    async def synthesize(
        self,
        signals: ResearchSignals,
        failure_reasons: list[str],
    ) -> AgentConfig:
        prompt = SYNTHESIS_PROMPT.format(
            patterns=signals.patterns,
            tool_patterns=signals.tool_patterns,
            failure_modes=signals.failure_modes,
            topology=signals.topology,
            failure_reasons=failure_reasons if failure_reasons else ["none — first attempt"],
            valid_tools=sorted(VALID_TOOLS),
        )
        response = await self._get_client().beta.chat.completions.parse(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format=AgentConfig,
        )
        return response.choices[0].message.parsed
