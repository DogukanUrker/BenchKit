"""MBPP benchmark - 500 Python programming tasks."""

import json
from pathlib import Path

from benchkit.benchmarks.base import Task
from benchkit.benchmarks.utils import strip_think_tags
from benchkit.executor import execute

DATASET = Path(__file__).parent.parent / "datasets" / "mbpp.jsonl"

SYSTEM = (
    "Write a Python function to solve the following task. "
    "Output ONLY the function. "
    "No markdown fences, no explanation."
)


def _extract_code(response: str) -> str:
    """Strip markdown fences if the model wrapped its output."""
    text = strip_think_tags(response).rstrip()

    if "```python" in text:
        text = text.split("```python", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]

    return text.strip()


class MBPP:
    name = "mbpp"

    def load_tasks(self) -> list[Task]:
        tasks = []
        with open(DATASET) as f:
            for line in f:
                d = json.loads(line)
                tasks.append(
                    Task(
                        id=d["task_id"],
                        prompt=d["prompt"],
                        metadata={"test": d["test"]},
                    )
                )
        return tasks

    def build_prompt(self, task: Task) -> str:
        return f"{SYSTEM}\n\n{task.prompt}"

    def evaluate(self, task: Task, response: str) -> bool:
        code = _extract_code(response)
        full = code + "\n\n" + task.metadata["test"] + "\n"
        return execute(full)
