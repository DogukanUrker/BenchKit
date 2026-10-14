"""OCRBench - text recognition and text-centric VQA over 1,000 images.

Ten categories from scene text to handwritten math. Scoring follows the
official OCRBench evaluator: a task passes when any reference answer appears
in the response, case-insensitively. Handwritten math (HME100k) compares with
all whitespace removed and keeps case, since LaTeX is case-sensitive. The
question is sent as written, with no extra instructions, as upstream does.
"""

from functools import cache

from benchkit.benchmarks.base import Task
from benchkit.benchmarks.utils import strip_think_tags
from benchkit.benchmarks.vision import category_scores, load_rows, spread
from benchkit.evaluation import EvaluationResult

DATASET = "echo840/OCRBench"
PARQUET = "data/test-00000-of-00001.parquet"
REVISION = "92a54bd1384387c178d5a07140a2d85e0a3d12e1"


def _convert(index: int, row: dict) -> dict:
    return {
        "task_id": f"ocrbench/{index}",
        "source": row["dataset"],
        "category": row["question_type"],
        "question": row["question"],
        "answers": list(row["answer"]),
    }


def matches(response: str, answers: list[str], source: str) -> bool:
    predicted = strip_think_tags(response).strip().replace("\n", " ")
    if source == "HME100k":
        predicted = predicted.replace(" ", "")
        return any(answer.strip().replace(" ", "") in predicted for answer in answers)
    predicted = predicted.lower()
    return any(
        answer.lower().strip().replace("\n", " ") in predicted for answer in answers
    )


@cache
def _rows() -> tuple[dict, ...]:
    return tuple(load_rows("ocrbench", DATASET, PARQUET, REVISION, _convert))


class OCRBench:
    name = "ocrbench"
    vision = True
    task_count = 1000

    def load_tasks(self) -> list[Task]:
        tasks = [
            Task(
                id=row["task_id"],
                prompt=row["question"],
                metadata={
                    "answers": row["answers"],
                    "source": row["source"],
                    "category": row["category"],
                    "images": [row["image"]],
                },
            )
            for row in _rows()
        ]
        return spread(tasks, lambda task: task.metadata["category"])

    def build_prompt(self, task: Task) -> str:
        return task.prompt

    def result_metadata(self, variant: str | None = None) -> dict:
        return {"dataset_revision": REVISION}

    def evaluate_with_feedback(self, task: Task, response: str) -> EvaluationResult:
        passed = matches(response, task.metadata["answers"], task.metadata["source"])
        return EvaluationResult(
            score=float(passed),
            feedback="" if passed else "The answer does not match the image text.",
        )

    def evaluate(self, task: Task, response: str) -> bool:
        return matches(response, task.metadata["answers"], task.metadata["source"])

    def summary_fields(self, records: list[object]) -> dict:
        categories = {row["task_id"]: row["category"] for row in _rows()}
        scores = category_scores(records, categories)
        return {"category_scores": scores} if scores else {}
