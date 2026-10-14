"""MMVP - 150 pairs of look-alike images that CLIP-style encoders confuse.

Each pair asks the same two-option question about two images that differ in
one visual detail (orientation, count, state, colour), and the right answer
differs between them. The official score is pair accuracy: both questions of
a pair must be right, so a model that ignores the image sits at 25%, not 50%.
Plain per-question accuracy is the headline score so BenchKit's averages and
decision calibration stay comparable; pair accuracy is reported next to it.
Pairs are consecutive, so even-sized slices keep them whole.
"""

import csv
import io
import re
from functools import cache

from benchkit.benchmarks.base import Task
from benchkit.benchmarks.mcq import extract_choice
from benchkit.benchmarks.utils import strip_think_tags
from benchkit.benchmarks.vision import load_files

DATASET = "MMVP/MMVP"
MANIFEST = "Questions.csv"
REVISION = "37eafecab8a3940c50c2ade5b36de69dbc99a8cf"

SYSTEM = "Answer the question about the image. Reply with ONLY the letter (A or B)."

_OPTIONS = re.compile(r"\(a\)\s*(.+?)\s*\(b\)\s*(.+)", re.IGNORECASE)


def parse_manifest(data: bytes) -> list[dict]:
    rows = []
    for row in csv.DictReader(io.StringIO(data.decode("utf-8-sig"))):
        index = int(row["Index"])
        options = _OPTIONS.fullmatch(row["Options"].strip())
        if options is None:
            raise ValueError(f"MMVP row {index}: unexpected options {row['Options']}")
        rows.append(
            {
                "task_id": f"mmvp/{index}",
                "pair": (index + 1) // 2,
                "question": row["Question"].strip(),
                "choices": [options.group(1).strip(), options.group(2).strip()],
                "answer": row["Correct Answer"].strip().strip("()").upper(),
                "image": f"MMVP Images/{index}.jpg",
            }
        )
    return rows


@cache
def _rows() -> tuple[dict, ...]:
    return tuple(load_files("mmvp", DATASET, MANIFEST, REVISION, parse_manifest))


def pair_accuracy(records: list[object]) -> dict:
    """Share of pairs with both questions right, over pairs fully scored."""
    pair_of = {row["task_id"]: row["pair"] for row in _rows()}
    pairs: dict[int, list[float]] = {}
    for record in records:
        pair = pair_of.get(record.task_id)
        if pair is not None:
            pairs.setdefault(pair, []).append(float(getattr(record, "score", 0.0)))
    complete = [scores for scores in pairs.values() if len(scores) == 2]
    if not complete:
        return {}
    passed = sum(all(score >= 1.0 for score in scores) for scores in complete)
    return {
        "pair_accuracy": round(100 * passed / len(complete), 1),
        "pairs": len(complete),
    }


class MMVP:
    name = "mmvp"
    vision = True
    task_count = 300
    decision_instructions = (
        "Which option correctly answers the question about the image?"
    )

    def load_tasks(self) -> list[Task]:
        return [
            Task(
                id=row["task_id"],
                prompt=row["question"],
                metadata={
                    "choices": row["choices"],
                    "answer": row["answer"],
                    "images": [row["image"]],
                },
            )
            for row in _rows()
        ]

    def build_prompt(self, task: Task) -> str:
        choices = "\n".join(
            f"{letter}) {choice}"
            for letter, choice in zip("AB", task.metadata["choices"], strict=True)
        )
        return f"{SYSTEM}\n\n{task.prompt}\n\n{choices}"

    def result_metadata(self, variant: str | None = None) -> dict:
        return {"dataset_revision": REVISION}

    def evaluate(self, task: Task, response: str) -> bool:
        text = strip_think_tags(response)
        choice = extract_choice(text, task.metadata["choices"], letters="AB")
        return choice == task.metadata["answer"]

    def summary_fields(self, records: list[object]) -> dict:
        return pair_accuracy(records)
