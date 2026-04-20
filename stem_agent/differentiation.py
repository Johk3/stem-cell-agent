import json
from datetime import datetime, timezone
from pathlib import Path

from stem_agent.models import AgentConfig, ProbeResult
from stem_agent.signal_reader import SignalReader
from stem_agent.config_synthesizer import ConfigSynthesizer
from stem_agent.probe_runner import ProbeRunner
from evaluation.gaia_loader import GAIAQuestion


class DifferentiationController:
    def __init__(
        self,
        signal_reader: SignalReader,
        config_synthesizer: ConfigSynthesizer,
        probe_runner: ProbeRunner,
        probe_questions: list[GAIAQuestion],
        max_attempts: int = 3,
        log_dir: str = "logs",
        baseline_probe_score: float | None = None,
    ):
        self.signal_reader = signal_reader
        self.config_synthesizer = config_synthesizer
        self.probe_runner = probe_runner
        self.probe_questions = probe_questions
        self.max_attempts = max_attempts
        self.log_dir = Path(log_dir)
        self.baseline_probe_score = baseline_probe_score
        self.log: list[dict] = []
        self._config_stack: list[AgentConfig] = []

    _EARLY_STOP_SCORE = 0.9

    async def differentiate(self) -> AgentConfig:
        signals = await self.signal_reader.read()
        failure_reasons: list[str] = []
        candidates: list[tuple[AgentConfig, float, ProbeResult]] = []

        for attempt in range(self.max_attempts):
            # SYNTHESIZE
            try:
                config = await self.config_synthesizer.synthesize(
                    signals, failure_reasons=failure_reasons
                )
            except Exception as e:
                self._record(attempt, "apoptosis", str(e))
                continue

            # PROBE
            probe_result = await self.probe_runner.run(config, self.probe_questions)

            threshold = max(config.probe_threshold, (self.baseline_probe_score or 0.0))
            if probe_result.score >= threshold:
                candidates.append((config, probe_result.score, probe_result))
                self._record(attempt, "candidate", {
                    "score": probe_result.score,
                    "tools": config.tools,
                    "topology": config.topology,
                    "system_prompt": config.system_prompt,
                })
                if probe_result.score >= self._EARLY_STOP_SCORE:
                    break
            else:
                failure_reasons = probe_result.failure_reasons
                self._record(attempt, "retrograde", probe_result.score)

        if not candidates:
            self._save_log()
            raise RuntimeError(
                f"Differentiation failed after {self.max_attempts} attempts. "
                f"See logs for details."
            )

        # Best-of-N: pick highest scoring candidate
        best_config, best_score, best_probe = max(candidates, key=lambda c: c[1])

        self._config_stack.append(best_config)
        self._record("final", "commit", {
            "score": best_score,
            "tools": best_config.tools,
            "topology": best_config.topology,
            "system_prompt": best_config.system_prompt,
        })
        self._save_log()
        return best_config

    def _record(self, attempt: int, outcome: str, detail) -> None:
        self.log.append(
            {
                "attempt": attempt,
                "outcome": outcome,
                "detail": detail,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def _save_log(self) -> None:
        self.log_dir.mkdir(exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = self.log_dir / f"differentiation_{ts}.json"
        path.write_text(json.dumps(self.log, indent=2))
