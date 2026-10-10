"""ChartQA - questions about bar, line and pie charts (2,500 test images).

Half the questions are human-written and need reasoning across the chart;
half are machine-generated lookups. Scoring is the standard relaxed accuracy:
a numeric answer passes within 5% of the reference, anything else must match
exactly up to case. Slices interleave the two halves.

One deliberate leniency over the reference scorer: no test label carries a
"%", yet chat models write "6.8%" for a label of 6.8. A percent answer is
therefore read both as written and divided by 100, and passes if either fits.
"""

import re
from functools import cache

from benchkit.benchmarks.base import Task
from benchkit.benchmarks.utils import strip_think_tags
from benchkit.benchmarks.vision import category_scores, load_rows, spread
from benchkit.evaluation import EvaluationResult

DATASET = "HuggingFaceM4/ChartQA"
PARQUET = "data/test-00000-of-00001-e2cd0b7a0f9eb20d.parquet"
REVISION = "b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5"

INSTRUCTION = (
    "Answer the question about the chart. You may reason first, but end with "
    'a final line of the form "Answer: X", where X is a single word or a '
    "number without units."
)

_ANSWER = re.compile(r"answer\s*[:：]\s*(.+)", re.IGNORECASE)
_WRAPPERS = " \t`'\"*_$"


def _convert(index: int, row: dict) -> dict:
    return {
        "task_id": f"chartqa/{index}",
        "category": "augmented" if row["human_or_machine"] else "human",
        "question": row["query"],
        "answers": list(row["label"]),
    }


def extract_answer(response: str) -> str:
    """The text after the last "Answer:", else the last non-empty line."""
    text = strip_think_tags(response).strip()
    found = _ANSWER.findall(text)
    if found:
        answer = found[-1]
    else:
        lines = [line for line in text.splitlines() if line.strip()]
        answer = lines[-1] if lines else ""
    return answer.strip(_WRAPPERS).removesuffix(".").strip(_WRAPPERS)


def _to_floats(text: str) -> list[float]:
    """Numeric readings of an answer: a percent is tried over 100 and raw."""
    text = text.replace(",", "").strip()
    try:
        value = float(text.removesuffix("%").strip())
    except ValueError:
        return []
    return [value / 100.0, value] if text.endswith("%") else [value]


def relaxed_match(prediction: str, target: str, tolerance: float = 0.05) -> bool:
    expected = _to_floats(target)
    predicted = _to_floats(prediction)
    if predicted and expected and expected[0]:
        return any(
            abs(value - expected[0]) / abs(expected[0]) <= tolerance
            for value in predicted
        )
    return prediction.strip().lower() == target.strip().lower()


@cache
def _rows() -> tuple[dict, ...]:
    return tuple(load_rows("chartqa", DATASET, PARQUET, REVISION, _convert))


class ChartQA:
    name = "chartqa"
    vision = True
    task_count = 2500

    def load_tasks(self) -> list[Task]:
        tasks = [
            Task(
                id=row["task_id"],
                prompt=row["question"],
                metadata={
                    "answers": row["answers"],
                    "category": row["category"],
                    "images": [row["image"]],
                },
            )
            for row in _rows()
        ]
        return spread(tasks, lambda task: task.metadata["category"])

    def build_prompt(self, task: Task) -> str:
        return f"{INSTRUCTION}\n\nQuestion: {task.prompt}"

    def result_metadata(self, variant: str | None = None) -> dict:
        return {"dataset_revision": REVISION}

    def evaluate_with_feedback(self, task: Task, response: str) -> EvaluationResult:
        predicted = extract_answer(response)
        passed = any(relaxed_match(predicted, a) for a in task.metadata["answers"])
        return EvaluationResult(
            score=float(passed),
            feedback=""
            if passed
            else 'The final answer is wrong. End with one line "Answer: X".',
            details={"predicted_answer": predicted},
        )

    def evaluate(self, task: Task, response: str) -> bool:
        return self.evaluate_with_feedback(task, response).passed

    def summary_fields(self, records: list[object]) -> dict:
        categories = {row["task_id"]: row["category"] for row in _rows()}
        scores = category_scores(records, categories)
        return {"category_scores": scores} if scores else {}
