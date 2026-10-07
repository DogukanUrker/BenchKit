"""BANKING77 intent routing - classify a customer message into one of 77 intents.

Routing is what decision models are deployed for, so the suite is shaped like
a router: every task offers all 77 intents (no option filtering) and the answer
is the intent label itself rather than an option letter. Chat models get the
same list and are scored on an exact match of the label they reply with.
"""

import json
from collections import Counter
from functools import cache
from pathlib import Path

from benchkit.benchmarks.base import Task
from benchkit.benchmarks.utils import strip_think_tags
from benchkit.evaluation import EvaluationResult

DATASETS = Path(__file__).parent.parent / "datasets"
DATASET = DATASETS / "banking77.jsonl"
INTENTS = DATASETS / "banking77_intents.json"

SYSTEM = (
    "Route the customer's message to exactly one of the 77 banking intents "
    "below. Reply with ONLY the intent name, exactly as written."
)

# Quotes, emphasis and heading marks a model may put around an exact label.
_WRAPPERS = " \t`'\"*#.,;:"


@cache
def intents() -> tuple[str, ...]:
    """The 77 intent labels in upstream order."""
    return tuple(json.loads(INTENTS.read_text(encoding="utf-8")))


@cache
def _gold() -> dict[str, str]:
    """Expected intent by task id."""
    with open(DATASET, encoding="utf-8") as f:
        return {row["task_id"]: row["answer"] for row in map(json.loads, f)}


def extract_intent(response: str) -> str | None:
    """Return the intent a response names, or None.

    The label must be exact up to case and surrounding quotes, backticks,
    emphasis or heading marks, optionally after an ``intent:`` prefix. The whole
    answer is tried first, then its last line, since a model may explain before
    committing.
    """
    labels = {label.casefold(): label for label in intents()}
    text = strip_think_tags(response).strip()
    lines = [line for line in text.splitlines() if line.strip()]
    for candidate in (text, lines[-1] if lines else ""):
        cleaned = candidate.strip(_WRAPPERS)
        if cleaned.casefold().startswith("intent:"):
            cleaned = cleaned[len("intent:") :].strip(_WRAPPERS)
        label = labels.get(cleaned.casefold())
        if label:
            return label
    return None


def macro_f1(pairs: list[tuple[str, str | None]]) -> float:
    """Unweighted mean F1 over every intent that is gold or predicted."""
    true_pos: Counter[str] = Counter()
    predicted: Counter[str] = Counter()
    gold: Counter[str] = Counter()
    for expected, guess in pairs:
        gold[expected] += 1
        if guess is not None:
            predicted[guess] += 1
            true_pos[guess] += guess == expected
    labels = set(gold) | set(predicted)
    if not labels:
        return 0.0
    scores = [
        2 * true_pos[label] / (gold[label] + predicted[label]) for label in labels
    ]
    return sum(scores) / len(scores)


class Banking77:
    name = "banking77"
    decision_instructions = "Which intent does the customer's message express?"

    def load_tasks(self) -> list[Task]:
        tasks = []
        with open(DATASET, encoding="utf-8") as f:
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
        listing = "\n".join(f"- {label}" for label in intents())
        return f"{SYSTEM}\n\nIntents:\n{listing}\n\nMessage: {task.prompt}"

    def decision_criteria(self, task: Task) -> dict[str, str]:
        """All 77 intents, keyed by label so the chosen key is the answer."""
        return {label: label.replace("_", " ").lower() for label in intents()}

    def evaluate_with_feedback(self, task: Task, response: str) -> EvaluationResult:
        predicted = extract_intent(response)
        expected = task.metadata["answer"]
        if predicted is None:
            feedback = (
                "The answer did not name one of the listed intents. Reply with "
                "exactly one intent name from the list and nothing else."
            )
        elif predicted != expected:
            feedback = (
                f"{predicted} is not the intent of this message. Re-read it and "
                "reply with exactly one intent name from the list."
            )
        else:
            feedback = ""
        return EvaluationResult(
            score=float(predicted == expected),
            feedback=feedback,
            details={"predicted_intent": predicted, "expected_intent": expected},
        )

    def evaluate(self, task: Task, response: str) -> bool:
        return extract_intent(response) == task.metadata["answer"]

    def job_metrics(self, records: list[object]) -> dict:
        """Macro-F1 over the scored tasks, for a router's per-intent quality.

        Tasks that never reached the verifier (timeouts, loop kills) count as
        no prediction, exactly as they count against accuracy.
        """
        gold = _gold()
        pairs = [
            (
                gold[record.task_id],
                (getattr(record, "workspace", None) or {}).get("predicted_intent"),
            )
            for record in records
            if record.task_id in gold
        ]
        return {"macro_f1": round(macro_f1(pairs) * 100, 1)} if pairs else {}
