"""V* Bench - find small details in high-resolution images (191 questions).

Two categories: direct attributes (the colour or material of a small object,
four options) and relative position (left or right of another object, two
options). The images are large on purpose, so a model or server that shrinks
them hard loses the detail the question is about; that is the measurement,
not a harness fault. Slices keep both categories' shares.
"""

import json
import re
from functools import cache

from benchkit.benchmarks.base import Task
from benchkit.benchmarks.mcq import extract_choice
from benchkit.benchmarks.utils import strip_think_tags
from benchkit.benchmarks.vision import category_scores, load_files, spread

DATASET = "craigwu/vstar_bench"
MANIFEST = "test_questions.jsonl"
REVISION = "d9ae62c903da0c98336e85c5ee89cd863b04b4da"

SYSTEM = "Answer the question about the image. Reply with ONLY the option letter."

_OPTION = re.compile(r"^\(([A-D])\)\s*(.+)$")


def parse_manifest(data: bytes) -> list[dict]:
    rows = []
    for line in data.decode("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        lines = [part.strip() for part in row["text"].splitlines() if part.strip()]
        options = [match for part in lines if (match := _OPTION.match(part))]
        letters = "".join(match.group(1) for match in options)
        if not options or letters != "ABCD"[: len(options)]:
            raise ValueError(f"V* {row['question_id']}: unexpected options")
        rows.append(
            {
                "task_id": f"vstar/{row['question_id']}",
                "category": row["category"],
                "question": lines[0],
                "choices": [match.group(2) for match in options],
                "answer": row["label"].strip().upper(),
                "image": row["image"],
            }
        )
    return rows


@cache
def _rows() -> tuple[dict, ...]:
    return tuple(load_files("vstar", DATASET, MANIFEST, REVISION, parse_manifest))


class VStar:
    name = "vstar"
    vision = True
    task_count = 191
    decision_instructions = (
        "Which option correctly answers the question about the image?"
    )

    def load_tasks(self) -> list[Task]:
        tasks = [
            Task(
                id=row["task_id"],
                prompt=row["question"],
                metadata={
                    "choices": row["choices"],
                    "answer": row["answer"],
                    "category": row["category"],
                    "images": [row["image"]],
                },
            )
            for row in _rows()
        ]
        return spread(tasks, lambda task: task.metadata["category"])

    def build_prompt(self, task: Task) -> str:
        choices = "\n".join(
            f"{letter}) {choice}"
            for letter, choice in zip("ABCD", task.metadata["choices"], strict=False)
        )
        return f"{SYSTEM}\n\n{task.prompt}\n\n{choices}"

    def result_metadata(self, variant: str | None = None) -> dict:
        return {"dataset_revision": REVISION}

    def evaluate(self, task: Task, response: str) -> bool:
        choices = task.metadata["choices"]
        text = strip_think_tags(response)
        choice = extract_choice(text, choices, letters="ABCD"[: len(choices)])
        return choice == task.metadata["answer"]

    def summary_fields(self, records: list[object]) -> dict:
        categories = {row["task_id"]: row["category"] for row in _rows()}
        scores = category_scores(records, categories)
        return {"category_scores": scores} if scores else {}
