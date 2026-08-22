"""RULER effective-context benchmark with runtime-generated prompts."""

from __future__ import annotations

import math
import threading
from collections.abc import Callable

from benchkit.benchmarks.base import Task
from benchkit.benchmarks.ruler_generator import (
    CONTEXT_BUCKETS,
    DEFAULT_SAMPLES_PER_KIND,
    PRACTICAL_SAMPLES_PER_KIND,
    TASK_KINDS,
    TASKS_PER_BUCKET,
    context_label,
    fit_prompt,
    generate_tasks,
)
from benchkit.benchmarks.utils import strip_think_tags


class RULER:
    """Consumer-friendly RULER suite with repeated task-family coverage."""

    name = "ruler"
    samples_per_kind = PRACTICAL_SAMPLES_PER_KIND
    tasks_per_variant = samples_per_kind * len(TASK_KINDS)
    task_count = tasks_per_variant * len(CONTEXT_BUCKETS)
    context_buckets = CONTEXT_BUCKETS
    variant_labels = tuple(context_label(bucket) for bucket in CONTEXT_BUCKETS)
    variant_count = len(CONTEXT_BUCKETS)
    list_task_count = f"{tasks_per_variant} × 6"
    include_in_overall = False
    list_note = "practical · 13 tasks × 3 samples · 4k–128k"
    evaluation_activity = "checking retrieved values"

    def __init__(self) -> None:
        self._unit_hints: dict[tuple[str, str, int], int] = {}
        self._hint_lock = threading.Lock()

    def load_tasks(self) -> list[Task]:
        return generate_tasks(samples_per_kind=self.samples_per_kind)

    def variants(self, client: object, model: str) -> list[str]:
        """Return buckets that fit the model's discovered served context."""
        limit: int | None = None
        discover = getattr(client, "context_length", None)
        if callable(discover):
            try:
                value = discover(model)
                if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                    limit = value
            except Exception:
                limit = None
        return [
            context_label(bucket)
            for bucket in CONTEXT_BUCKETS
            if limit is None or bucket <= limit
        ]

    def tasks_for_variant(self, tasks: list[Task], variant: str) -> list[Task]:
        return [task for task in tasks if task.metadata.get("variant") == variant]

    def result_metadata(self, variant: str | None) -> dict:
        if not variant:
            return {"include_in_overall": self.include_in_overall}
        context_tokens = next(
            (bucket for bucket in CONTEXT_BUCKETS if context_label(bucket) == variant),
            0,
        )
        return {
            "context_tokens": context_tokens,
            "context_label": variant,
            "include_in_overall": self.include_in_overall,
        }

    def build_prompt(self, task: Task) -> str:
        prompt, _, _ = fit_prompt(task)
        return prompt

    def build_prompt_for(self, task: Task, client: object, model: str) -> str:
        tokenizer = getattr(client, "tokenize", None)
        token_count: Callable[[str], int | None] | None = None
        if callable(tokenizer):

            def count(prompt: str) -> int | None:
                tokens = tokenizer(model, prompt)
                return len(tokens) if isinstance(tokens, list) else None

            token_count = count

        cache_key = (
            model,
            str(task.metadata["task_type"]),
            int(task.metadata["context_tokens"]),
        )
        with self._hint_lock:
            initial_units = self._unit_hints.get(cache_key)
        prompt, units, _ = fit_prompt(
            task,
            token_count,
            initial_units=initial_units,
        )
        with self._hint_lock:
            self._unit_hints[cache_key] = units
        return prompt

    def evaluate(self, task: Task, response: str) -> float:
        """Return RULER's fraction of reference strings found in the answer."""
        text = strip_think_tags(response).casefold()
        answers = [str(answer).casefold() for answer in task.metadata["answers"]]
        if not answers:
            return 0.0
        return sum(answer in text for answer in answers) / len(answers)

    def task_statistics(self, records: list[object]) -> list[dict]:
        """Return per-task mean scores and Wilson 95% intervals."""
        grouped: dict[str, list[float]] = {}
        for record in records:
            parts = str(getattr(record, "task_id", "")).split("/")
            if len(parts) >= 4:
                grouped.setdefault(parts[2], []).append(
                    float(getattr(record, "score", 0.0))
                )
        rows = []
        z = 1.959963984540054
        for kind, scores in sorted(grouped.items()):
            n = len(scores)
            mean = sum(scores) / n
            denominator = 1 + z * z / n
            center = (mean + z * z / (2 * n)) / denominator
            margin = (
                z * math.sqrt(mean * (1 - mean) / n + z * z / (4 * n * n)) / denominator
            )
            rows.append(
                {
                    "task": kind,
                    "samples": n,
                    "score": round(mean * 100, 1),
                    "ci95_low": round(max(0.0, center - margin) * 100, 1),
                    "ci95_high": round(min(1.0, center + margin) * 100, 1),
                }
            )
        return rows


class RULERFull(RULER):
    """Complete RULER suite for research-scale, explicitly opted-in runs."""

    name = "ruler-full"
    samples_per_kind = DEFAULT_SAMPLES_PER_KIND
    tasks_per_variant = TASKS_PER_BUCKET
    task_count = tasks_per_variant * len(CONTEXT_BUCKETS)
    list_task_count = f"{tasks_per_variant:,} × 6"
    list_note = "full · 13 tasks × 500 samples · 4k–128k · extremely slow"
