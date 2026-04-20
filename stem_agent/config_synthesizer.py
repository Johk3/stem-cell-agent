from openai import AsyncOpenAI
from stem_agent.models import ResearchSignals, AgentConfig, VALID_TOOLS

BASE_RESEARCH_PROMPT = """\
You are a research agent answering questions that require web search, \
multi-step reasoning, and precise short answers.

WORKFLOW:
1. Read the question carefully. Identify what type of answer is needed \
(a name, number, date, list, phrase).
2. If it is a factual question, search the web. Break multi-hop questions \
into separate searches — do NOT try to answer everything in one query.
3. If the first search is inconclusive, search again with different terms. \
Verify critical facts with a second search.
4. For logic puzzles, math, or reasoning tasks: work through the problem \
step by step WITHOUT searching. Only search for factual premises you are \
unsure about.
5. Never refuse a question. Always provide your best answer.
6. Follow any format instructions exactly (comma-separated, alphabetical, \
plain text numbers, etc.).

YOUR FINAL LINE MUST BE:
ANSWER: <your answer here>

The ANSWER line contains ONLY the answer. No explanation, no "the answer is", \
no extra words. Examples:
- ANSWER: Paris
- ANSWER: 42
- ANSWER: Braintree, Honolulu
"""

SYNTHESIS_PROMPT = """\
Below is a base system prompt for a research agent that answers GAIA \
benchmark questions. Improve it using the environmental signals gathered \
from web research about what makes research agents succeed and fail.

Base prompt:
{base_prompt}

Environmental signals:
- Successful patterns: {patterns}
- Tool usage patterns: {tool_patterns}
- Common failure modes to avoid: {failure_modes}

Your task: produce an IMPROVED version of the base prompt. You should:
- Weave in 2-5 concrete tips from the signals above (e.g. if a common \
failure is "not verifying dates", add a verification step)
- Strengthen the workflow with specific strategies that top agents use
- Do NOT remove or weaken the existing ANSWER: tag requirement
- Do NOT make the prompt excessively long (max 150 words added)
- Keep instructions actionable and specific, not vague

Return valid JSON matching the AgentConfig schema:
- system_prompt: the improved prompt
- tools: ["web_search"]
- topology: "single"
- probe_threshold: 0.3
"""

ADJUSTMENT_PROMPT = """\
Below is a system prompt for a research agent. It was tested on probe \
questions and some were answered incorrectly. Fix the prompt to address \
the failure patterns while keeping what already works.

Current prompt:
{current_prompt}

Probe failures:
{failure_reasons}

Environmental signals (for reference):
- Successful patterns: {patterns}
- Common failure modes: {failure_modes}

Your task: produce a REFINED version of the prompt. You may:
- Add 1-3 targeted sentences addressing the specific failure patterns above
- Strengthen instructions for question types that failed
- Do NOT remove or weaken any existing instructions
- Do NOT make the prompt significantly longer (max 100 words added)
- Keep the ANSWER: tag requirement exactly as is

Return valid JSON matching the AgentConfig schema:
- system_prompt: the refined prompt
- tools: ["web_search"]
- topology: "single"
- probe_threshold: 0.3
"""


class ConfigSynthesizer:
    def __init__(self, model: str = "gpt-5.4"):
        self.model = model
        self._client: AsyncOpenAI | None = None
        self._last_prompt: str | None = None

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI()
        return self._client

    _MAX_FAILURE_REASONS = 10
    _MAX_REASON_CHARS = 120

    async def synthesize(
        self,
        signals: ResearchSignals,
        failure_reasons: list[str],
    ) -> AgentConfig:
        trimmed = [
            r[: self._MAX_REASON_CHARS]
            for r in failure_reasons[-self._MAX_FAILURE_REASONS :]
        ]
        if trimmed and self._last_prompt:
            # retry path: fix the prompt based on what went wrong
            prompt = ADJUSTMENT_PROMPT.format(
                current_prompt=self._last_prompt,
                failure_reasons=trimmed,
                patterns=signals.patterns,
                failure_modes=signals.failure_modes,
            )
        else:
            # first attempt: improve the base prompt using signals
            prompt = SYNTHESIS_PROMPT.format(
                base_prompt=BASE_RESEARCH_PROMPT,
                patterns=signals.patterns,
                tool_patterns=signals.tool_patterns,
                failure_modes=signals.failure_modes,
            )

        response = await self._get_client().beta.chat.completions.parse(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format=AgentConfig,
        )
        config = response.choices[0].message.parsed
        self._last_prompt = config.system_prompt
        return config
