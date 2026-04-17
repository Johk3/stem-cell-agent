import os
from agents import Agent, WebSearchTool

from stem_agent.signal_reader import SignalReader
from stem_agent.config_synthesizer import ConfigSynthesizer
from stem_agent.probe_runner import ProbeRunner, build_tools
from stem_agent.differentiation import DifferentiationController
from stem_agent.models import AgentConfig
from evaluation.gaia_loader import GAIAQuestion

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

STEM_INSTRUCTIONS = (
    "Answer the user's question as accurately as possible. "
    "Search the web if you need current or factual information. "
    "Keep answers concise and direct."
)


class StemAgent:
    """Undifferentiated agent that specializes itself into a deep research agent."""

    def __init__(self, max_attempts: int = 3, log_dir: str = "logs"):
        self.model = MODEL
        self.max_attempts = max_attempts
        self.log_dir = log_dir
        self._differentiated_config: AgentConfig | None = None
        self._differentiation_log: list[dict] = []

    def build_baseline_agent(self) -> Agent:
        """The undifferentiated stem agent — minimal tools, generic prompt."""
        return Agent(
            name="stem_baseline",
            instructions=STEM_INSTRUCTIONS,
            tools=[WebSearchTool()],
            model=self.model,
        )

    def build_differentiated_agent(self) -> Agent:
        """The differentiated research agent built from synthesized config."""
        if self._differentiated_config is None:
            raise RuntimeError(
                "Agent has not differentiated yet. Call differentiate() first."
            )
        config = self._differentiated_config
        return Agent(
            name="research_agent",
            instructions=config.system_prompt,
            tools=build_tools(config.tools),
            model=self.model,
        )

    async def differentiate(self, probe_questions: list[GAIAQuestion]) -> AgentConfig:
        """Run the differentiation process. Returns committed AgentConfig."""
        controller = DifferentiationController(
            signal_reader=SignalReader(model=self.model),
            config_synthesizer=ConfigSynthesizer(model=self.model),
            probe_runner=ProbeRunner(),
            probe_questions=probe_questions,
            max_attempts=self.max_attempts,
            log_dir=self.log_dir,
        )
        self._differentiated_config = await controller.differentiate()
        self._differentiation_log = controller.log
        return self._differentiated_config

    @property
    def differentiation_log(self) -> list[dict]:
        return self._differentiation_log
