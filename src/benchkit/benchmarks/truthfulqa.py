"""TruthfulQA MC1 benchmark - truthful multiple choice QA."""

import json
from pathlib import Path

from benchkit.benchmarks.base import Task
from benchkit.benchmarks.mcq import extract_choice
from benchkit.benchmarks.utils import strip_think_tags

DATASET = Path(__file__).parent.parent / "datasets" / "truthfulqa.jsonl"

SYSTEM = (
    "Answer the following multiple choice question. "
    "Reply with ONLY the letter (A, B, C, or D)."
)


def _format_choices(choices: list[str]) -> str:
    return "\n".join(f"{letter}) {choice}" for letter, choice in zip("ABCD", choices))


class TruthfulQA:
    name = "truthfulqa"

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
                            "choices": d["choices"],
                            "answer": d["answer"],
                        },
                    )
                )
        return tasks

    def build_prompt(self, task: Task) -> str:
        choices = _format_choices(task.metadata["choices"])
        return f"{SYSTEM}\n\n{task.prompt}\n\n{choices}"

    def evaluate(self, task: Task, response: str) -> bool:
        text = strip_think_tags(response)
        choice = extract_choice(text, task.metadata["choices"])
        return choice == task.metadata["answer"]
