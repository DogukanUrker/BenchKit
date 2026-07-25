"""Offline stand-in inference client used by ``benchkit --demo``.

Demo mode exists so the interface can be explored (and reviewed) without an
inference server: it answers prompts locally with a per-model skill level, so
scores, speeds and pass/fail streaks look like a real run.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import random
import time

from benchkit.benchmarks.base import Task
from benchkit.engine import benchmark, tasks_for

LETTERS = "ABCD"

DEMO_MODELS = [
    {"name": "demo-nano:0.5b", "size": 0.4e9, "skill": 0.30, "speed": 240.0},
    {"name": "demo-mini:3b", "size": 2.1e9, "skill": 0.55, "speed": 130.0},
    {"name": "demo-pro:8b", "size": 5.2e9, "skill": 0.76, "speed": 68.0},
    {"name": "demo-max:14b", "size": 9.4e9, "skill": 0.91, "speed": 34.0},
]


def _dataset_path(key: str):
    module = importlib.import_module(type(benchmark(key)).__module__)
    return getattr(module, "DATASET", None)


def _canonical_solutions(key: str) -> dict[str, str]:
    """Reference solutions from a dataset file, when it ships any."""
    path = _dataset_path(key)
    if path is None:
        return {}

    solutions: dict[str, str] = {}
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                solution = row.get("canonical_solution") or row.get("code")
                if solution:
                    solutions[row["task_id"]] = solution
    # ValueError covers both JSONDecodeError and UnicodeDecodeError.
    except (OSError, ValueError, KeyError):
        return {}
    return solutions


def _wrong_letter(answer: str) -> str:
    options = [letter for letter in LETTERS if letter != answer]
    return random.choice(options) if options else "A"


def _good_answer(task: Task, solution: str | None) -> str:
    metadata = task.metadata
    if solution is not None:
        return solution

    answer = metadata.get("answer")
    if isinstance(answer, bool):
        return "Yes" if answer else "No"
    if isinstance(answer, str) and answer.lower() in {"yes", "no"}:
        return answer.capitalize()
    if isinstance(answer, str) and answer.upper() in set(LETTERS):
        return f"The answer is {answer.upper()}."
    if answer is not None:
        return f"Working through it step by step.\n\n#### {answer}"
    return "A"


def _bad_answer(task: Task, has_solution: bool) -> str:
    metadata = task.metadata
    if has_solution or "test" in metadata:
        return "    raise NotImplementedError"

    answer = metadata.get("answer")
    if isinstance(answer, bool):
        return "No" if answer else "Yes"
    if isinstance(answer, str) and answer.lower() in {"yes", "no"}:
        return "No" if answer.lower() == "yes" else "Yes"
    if isinstance(answer, str) and answer.upper() in set(LETTERS):
        return f"The answer is {_wrong_letter(answer.upper())}."
    if answer is not None:
        return "Let me compute that.\n\n#### 42"
    return "B"


class DemoClient:
    """Mimics :class:`benchkit.client.InferenceClient` without a server."""

    host = "demo://offline"
    label = "Demo mode"
    provider = "demo"
    timeout = 0.0

    def __init__(self, speed: float = 1.0) -> None:
        self.speed = max(speed, 0.01)
        self._answers: dict[str, tuple[Task, str | None]] = {}
        self._primed: set[str] = set()

    def prime(self, benchmark_keys: list[str]) -> None:
        """Index prompts for the benchmarks about to run."""
        for key in benchmark_keys:
            if key in self._primed:
                continue
            bench = benchmark(key)
            solutions = _canonical_solutions(key)
            for task in tasks_for(key):
                prompt = bench.build_prompt(task)
                solution = solutions.get(task.id) or task.metadata.get(
                    "canonical_solution"
                )
                self._answers[prompt] = (task, solution)
            self._primed.add(key)

    def list_models(self) -> list[dict]:
        return [
            {"name": model["name"], "size": model["size"], "status": "demo"}
            for model in DEMO_MODELS
        ]

    def unload_model(self, model: str) -> None:
        time.sleep(0.05 / self.speed)

    def _profile(self, model: str) -> dict:
        for entry in DEMO_MODELS:
            if entry["name"] == model:
                return entry
        return DEMO_MODELS[-1]

    def generate(self, model: str, prompt: str) -> dict:
        profile = self._profile(model)
        entry = self._answers.get(prompt)

        # Deterministic per model+prompt so repeat runs stay comparable.
        digest = hashlib.sha256(f"{model}{prompt}".encode()).hexdigest()
        roll = int(digest[:8], 16) / 0xFFFFFFFF
        random.seed(digest)

        if entry is None:
            response = "A"
        else:
            task, solution = entry
            correct = roll < profile["skill"]
            response = (
                _good_answer(task, solution)
                if correct
                else _bad_answer(task, solution is not None)
            )

        tokens = 40 + int(roll * 260)
        tok_s = profile["speed"] * (0.85 + roll * 0.3)
        elapsed = tokens / tok_s
        time.sleep(min(elapsed / (40 * self.speed), 0.35))

        return {
            "response": response,
            "tok_s": tok_s,
            "eval_count": tokens,
            "eval_duration_ns": int(elapsed * 1e9),
            "response_time_s": elapsed,
            "done_reason": "stop",
        }


def is_demo_client(client: object) -> bool:
    return isinstance(client, DemoClient)
