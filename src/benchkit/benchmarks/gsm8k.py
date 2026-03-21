"""GSM8K benchmark - grade school math word problems."""

import json
from pathlib import Path

from benchkit.benchmarks.base import Task
from benchkit.benchmarks.utils import strip_think_tags

DATASET = Path(__file__).parent.parent / "datasets" / "gsm8k.jsonl"

SYSTEM = (
    "Solve the following math problem step by step. At the very end, write your "
    'final answer as a single number after "####".'
)


def _extract_leading_number(text: str) -> str | None:
    token = []
    started = False

    for ch in text.strip():
        if ch in "+-0123456789,.":
            token.append(ch)
            started = True
            continue
        if started:
            break

    value = "".join(token).strip(".,")
    return value if any(ch.isdigit() for ch in value) else None


def _extract_last_number(text: str) -> str | None:
    numbers = []
    current = []

    for ch in text:
        if ch in "+-0123456789,.":
            current.append(ch)
            continue
        if current:
            token = "".join(current).strip(".,")
            if any(c.isdigit() for c in token):
                numbers.append(token)
            current = []

    if current:
        token = "".join(current).strip(".,")
        if any(c.isdigit() for c in token):
            numbers.append(token)

    return numbers[-1] if numbers else None


def _parse_number(text: str | None) -> float | None:
    if not text:
        return None

    cleaned = text.replace(",", "").strip()
    if not cleaned:
        return None

    try:
        return float(cleaned)
    except ValueError:
        return None


class GSM8K:
    name = "gsm8k"

    def load_tasks(self) -> list[Task]:
        tasks = []
        with open(DATASET) as f:
            for line in f:
                d = json.loads(line)
                tasks.append(
                    Task(
                        id=d["task_id"],
                        prompt=d["question"],
                        metadata={"answer": d["answer"]},
                    )
                )
        return tasks

    def build_prompt(self, task: Task) -> str:
        return f"{SYSTEM}\n\n{task.prompt}"

    def evaluate(self, task: Task, response: str) -> bool:
        text = strip_think_tags(response)

        if "####" in text:
            predicted = _extract_leading_number(text.rsplit("####", 1)[1])
        else:
            predicted = _extract_last_number(text)

        return _parse_number(predicted) == _parse_number(task.metadata["answer"])
