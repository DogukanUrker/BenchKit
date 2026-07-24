"""PIQA benchmark - binary physical commonsense reasoning."""

import json
from pathlib import Path

from benchkit.benchmarks.base import Task
from benchkit.benchmarks.mcq import extract_choice
from benchkit.benchmarks.utils import strip_think_tags

DATASET = Path(__file__).parent.parent / "datasets" / "piqa.jsonl"

SYSTEM = (
    "Choose the most appropriate solution for the goal. "
    "Reply with ONLY the letter (A or B)."
)


def _format_choices(choices: list[str]) -> str:
    return "\n".join(f"{letter}) {choice}" for letter, choice in zip("AB", choices))


class PIQA:
    name = "piqa"

    def load_tasks(self) -> list[Task]:
        tasks = []
        with open(DATASET) as f:
            for line in f:
                d = json.loads(line)
                tasks.append(
                    Task(
                        id=d["task_id"],
                        prompt=d["goal"],
                        metadata={
                            "choices": d["choices"],
                            "answer": d["answer"],
                        },
                    )
                )
        return tasks

    def build_prompt(self, task: Task) -> str:
        choices = _format_choices(task.metadata["choices"])
        return f"{SYSTEM}\n\nGoal: {task.prompt}\n\n{choices}"

    def evaluate(self, task: Task, response: str) -> bool:
        text = strip_think_tags(response)
        choice = extract_choice(text, task.metadata["choices"])
        return choice == task.metadata["answer"]
