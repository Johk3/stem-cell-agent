from openai import AsyncOpenAI
from stem_agent.models import ResearchSignals, AgentConfig, VALID_TOOLS

SYNTHESIS_PROMPT = """\
You are designing a configuration for a deep research AI agent that will be \
evaluated on the GAIA benchmark.

Environmental signals:
- Reasoning patterns: {patterns}
- Effective tool combinations: {tool_patterns}
- Known failure modes to address: {failure_modes}
- Recommended topology: {topology}

Previous attempt failure reasons (empty = first attempt): {failure_reasons}

Design an AgentConfig:
- system_prompt: Detailed instructions (200-400 words). Must explicitly address \
  the failure modes above. Include step-by-step research strategy.
- tools: Subset of {valid_tools} — choose only what's needed.
- topology: "single" or "orchestrator+subagents"
- probe_threshold: Minimum probe score to accept (use 0.3 for Level 1 — the probe \
  set is only 10 questions so variance is high; 0.3 means 3/10 correct)

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
