"""MMLU-Pro benchmark - reasoning-heavy academic QA with ten options."""

import json
import random
from collections import defaultdict
from pathlib import Path

from benchkit.benchmarks.base import Task
from benchkit.benchmarks.mcq import extract_choice
from benchkit.benchmarks.utils import strip_think_tags

DATASET = Path(__file__).parent.parent / "datasets" / "mmlu_pro.jsonl"

# Ten options per question, so the guess floor is 10% rather than MMLU's 25%.
# A handful of questions carry fewer after the authors pruned bad distractors,
# which is why the letter range is always derived from the option list.
LETTERS = "ABCDEFGHIJ"

# The default suite runs a stratified slice: `PER_CATEGORY` questions from each
# of the 14 categories, sampled with a fixed seed so every run and every model
# sees the same tasks. `mmlu-pro-full` runs all of them.
PER_CATEGORY = 100
SEED = 0

SYSTEM = (
    "Answer the following multiple choice question about {category}. "
    "Reply with ONLY the letter of the correct option (A through {last})."
)


def _format_choices(choices: list[str]) -> str:
    return "\n".join(
        f"{letter}) {choice}" for letter, choice in zip(LETTERS, choices, strict=False)
    )


def _load_tasks() -> list[Task]:
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


def _stratified(tasks: list[Task], per_category: int) -> list[Task]:
    """Sample `per_category` tasks from each category, in dataset order."""
    by_category: dict[str, list[int]] = defaultdict(list)
    for index, task in enumerate(tasks):
        by_category[task.metadata["category"]].append(index)

    picked: list[int] = []
    for category in sorted(by_category):
        indices = by_category[category]
        if len(indices) <= per_category:
            picked.extend(indices)
            continue
        rng = random.Random(f"{SEED}:{category}")
        picked.extend(rng.sample(indices, per_category))

    return [tasks[index] for index in sorted(picked)]


class MMLUPro:
    name = "mmlu-pro"

    def load_tasks(self) -> list[Task]:
        return _stratified(_load_tasks(), PER_CATEGORY)

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


class MMLUProFull(MMLUPro):
    """Every MMLU-Pro question rather than the stratified default slice."""

    name = "mmlu-pro-full"

    def load_tasks(self) -> list[Task]:
        return _load_tasks()
