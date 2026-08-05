"""MMLU-Pro benchmark - reasoning-heavy academic QA with ten options."""

import json
from pathlib import Path

from benchkit.benchmarks.base import Task
from benchkit.benchmarks.mcq import extract_choice
from benchkit.benchmarks.utils import strip_think_tags

DATASET = Path(__file__).parent.parent / "datasets" / "mmlu_pro.jsonl"

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

    def load_tasks(self) -> list[Task]:
        if not DATASET.exists():
            raise FileNotFoundError(
                f"{DATASET.name} is missing; build it with "
                "`uv run --with datasets python scripts/build_mmlu_pro.py`"
            )

        tasks = []
        with open(DATASET, encoding="utf-8") as f:
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
