import random
from dataclasses import dataclass
from datasets import load_dataset


@dataclass
class GAIAQuestion:
    task_id: str
    question: str
    answer: str
    level: int
    file_name: str


class GAIALoader:
    def __init__(self, n_probe: int = 10, seed: int = 42):
        self.n_probe = n_probe
        self.seed = seed
        self._questions: list[GAIAQuestion] | None = None

    def load(self) -> list[GAIAQuestion]:
        if self._questions is None:
            raw = load_dataset(
                "gaia-benchmark/GAIA",
                "2023_all",
                split="validation",
            )
            self._questions = [
                GAIAQuestion(
                    task_id=row["task_id"],
                    question=row["Question"],
                    answer=row["Final answer"],
                    level=int(row["Level"]),
                    file_name=row.get("file_name", ""),
                )
                for row in raw
                if int(row["Level"]) == 1
            ]
        return self._questions

    def split(self) -> tuple[list[GAIAQuestion], list[GAIAQuestion]]:
        questions = self.load()
        rng = random.Random(self.seed)
        shuffled = questions[:]
        rng.shuffle(shuffled)
        return shuffled[: self.n_probe], shuffled[self.n_probe :]
