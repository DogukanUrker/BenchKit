"""Code verifier - does a model-written HumanEval+ solution pass its tests?"""

import json
import re
from pathlib import Path

from benchkit.benchmarks.base import Task
from benchkit.benchmarks.utils import strip_think_tags

DATASET = Path(__file__).parent.parent / "datasets" / "code_verifier.jsonl"

# One neutral question for both harnesses, so neither is nudged either way.
QUESTION = "Does this solution correctly solve the problem?"


def _extract_answer(response: str) -> str | None:
    text = strip_think_tags(response)
    answers = re.findall(r"\b(?:yes|no)\b", text.lower())
    return answers[-1] if answers else None


class CodeVerifier:
    name = "code-verifier"
    decision_instructions = QUESTION
    task_count = 500

    def load_tasks(self) -> list[Task]:
        tasks = []
        with open(DATASET, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                tasks.append(
                    Task(
                        id=d["id"],
                        prompt=(
                            f"Problem:\n```python\n{d['prompt'].strip()}\n```\n\n"
                            f"Solution:\n```python\n"
                            f"{d['solution'].strip()}\n```"
                        ),
                        metadata={
                            "answer": "yes" if d["passes"] else "no",
                            "task_id": d["task_id"],
                            "model": d["model"],
                        },
                    )
                )
        return tasks

    def build_prompt(self, task: Task) -> str:
        return f"{task.prompt}\n\n{QUESTION} Reply with only YES or NO."

    def evaluate(self, task: Task, response: str) -> bool:
        return _extract_answer(response) == task.metadata["answer"]

    def summary_fields(self, records: list[object]) -> dict:
        """Score passing and failing solutions separately.

        The suite is balanced, so a verifier that always says "yes" scores
        50% overall; the per-class split shows which way a model leans.
        """
        answers = {task.id: task.metadata["answer"] for task in self.load_tasks()}
        passing = [r for r in records if answers.get(r.task_id) == "yes"]
        failing = [r for r in records if answers.get(r.task_id) == "no"]

        def accuracy(rows: list[object]) -> float | None:
            if not rows:
                return None
            return round(sum(r.passed for r in rows) / len(rows) * 100, 1)

        return {
            "verifier_pass_total": len(passing),
            "verifier_fail_total": len(failing),
            "verifier_pass_accuracy": accuracy(passing),
            "verifier_fail_accuracy": accuracy(failing),
        }
