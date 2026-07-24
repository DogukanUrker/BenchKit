"""BoolQ benchmark - yes/no reading comprehension."""

import json
import re
from pathlib import Path

from benchkit.benchmarks.base import Task
from benchkit.benchmarks.utils import strip_think_tags

DATASET = Path(__file__).parent.parent / "datasets" / "boolq.jsonl"

SYSTEM = (
    "Read the passage and answer the question. "
    "Reply with ONLY YES or NO."
)


def _extract_answer(response: str) -> str | None:
    text = strip_think_tags(response)
    answers = re.findall(r"\b(?:yes|no)\b", text.lower())
    return answers[-1] if answers else None


class BoolQ:
    name = "boolq"

    def load_tasks(self) -> list[Task]:
        tasks = []
        with open(DATASET) as f:
            for line in f:
                d = json.loads(line)
                tasks.append(
                    Task(
                        id=d["task_id"],
                        prompt=d["question"],
                        metadata={
                            "passage": d["passage"],
                            "answer": d["answer"],
                        },
                    )
                )
        return tasks

    def build_prompt(self, task: Task) -> str:
        passage = task.metadata["passage"]
        return f"{SYSTEM}\n\nPassage:\n{passage}\n\nQuestion: {task.prompt}"

    def evaluate(self, task: Task, response: str) -> bool:
        return _extract_answer(response) == task.metadata["answer"]
