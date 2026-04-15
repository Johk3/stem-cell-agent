from typing import Literal
from pydantic import BaseModel, field_validator

VALID_TOOLS = {"web_search", "file_read", "file_write", "python_repl"}


class ResearchSignals(BaseModel):
    patterns: list[str]
    tool_patterns: list[str]
    failure_modes: list[str]
    topology: Literal["single", "orchestrator+subagents"]


class AgentConfig(BaseModel):
    system_prompt: str
    tools: list[str]
    topology: Literal["single", "orchestrator+subagents"]
    probe_threshold: float

    @field_validator("tools")
    @classmethod
    def validate_tools(cls, v: list[str]) -> list[str]:
        invalid = set(v) - VALID_TOOLS
        if invalid:
            raise ValueError(
                f"Invalid tools: {invalid}. Must be subset of {VALID_TOOLS}"
            )
        return v

    @field_validator("probe_threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"probe_threshold must be in [0, 1], got {v}")
        return v


class ProbeResult(BaseModel):
    score: float
    failed_questions: list[str]
    failure_reasons: list[str]
