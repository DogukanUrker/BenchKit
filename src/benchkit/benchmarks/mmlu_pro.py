"""MMLU-Pro benchmark - reasoning-heavy academic QA with ten options."""

import json

from benchkit.benchmarks.base import Task
from benchkit.benchmarks.downloads import MMLU_PRO as DATASET
from benchkit.benchmarks.mcq import extract_choice
from benchkit.benchmarks.utils import strip_think_tags

# Ten options per question, so the guess floor is 10% rather than MMLU's 25%.
# A handful of questions carry fewer after the authors pruned bad distractors,
# which is why the letter range is always derived from the option list.
LETTERS = "ABCDEFGHIJ"

SYSTEM = (
    "Answer the following multiple choice question about {category}. "
    "Reply with ONLY the letter of the correct option (A through {last})."
)


def _format_choices(choices: list[str]) -> str:
    return "\n".join(
        f"{letter}) {choice}" for letter, choice in zip(LETTERS, choices, strict=False)
    )


class MMLUPro:
    name = "mmlu-pro"
    decision_instructions = "Which option correctly answers the question?"
    task_count = 12032
    data_source = DATASET.source("MMLU-Pro test split, 14 categories")

    def load_tasks(self) -> list[Task]:
        tasks = []
        with open(DATASET.ensure(), encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                tasks.append(
                    Task(
                        id=d["task_id"],
                        prompt=d["question"],
                        metadata={
                            "choices": d["choices"],
                            "answer": d["answer"],
                            "category": d["category"],
                            "source": d.get("src", ""),
                        },
                    )
                )
        return tasks

    def build_prompt(self, task: Task) -> str:
        choices = task.metadata["choices"]
        category = task.metadata["category"].replace("_", " ")
        system = SYSTEM.format(category=category, last=LETTERS[len(choices) - 1])
        return f"{system}\n\n{task.prompt}\n\n{_format_choices(choices)}"

    def evaluate(self, task: Task, response: str) -> bool:
        text = strip_think_tags(response)
        letters = LETTERS[: len(task.metadata["choices"])]
        choice = extract_choice(text, task.metadata["choices"], letters=letters)
        return choice == task.metadata["answer"]
