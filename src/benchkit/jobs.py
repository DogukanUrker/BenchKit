"""Job planning: what a run will do before anything is generated.

Slice parsing, the cached benchmark registry, ``JobSpec`` and the functions that
expand, route and count jobs, plus the per-task questions the engine asks before
it generates (the prompt, how many options a task offers, how many repairs it
gets). Nothing here talks to a model.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, replace
from typing import Literal

from benchkit import decision
from benchkit.benchmarks import REGISTRY
from benchkit.benchmarks.base import Task

MAX_REPAIR_ATTEMPTS = 10


class SliceError(ValueError):
    """Raised when a task slice specification cannot be parsed."""


def parse_slice(spec: str | None, total: int) -> tuple[int, int]:
    """Resolve a slice spec into ``(start, end)`` indices for a task list.

    Accepted forms: ``"20"`` (first 20), ``"-20"`` (last 20) and ``"5-15"``
    (range, 0-indexed, end exclusive). Blank or ``None`` means every task.
    """
    if spec is None:
        return 0, total

    text = spec.strip()
    if not text:
        return 0, total

    try:
        if text.startswith("-"):
            count = int(text[1:])
            if count <= 0:
                raise ValueError(text)
            return max(0, total - count), total

        if "-" in text:
            head, tail = text.split("-", 1)
            start, end = int(head), int(tail)
            if start < 0 or end <= start:
                raise ValueError(text)
            # `total <= 0` means the task count is not known yet, so only the
            # syntax can be checked.
            if 0 < total <= start:
                raise ValueError(text)
            return min(start, total), min(end, total)

        count = int(text)
        if count <= 0:
            raise ValueError(text)
        return 0, min(count, total)
    except ValueError as exc:
        raise SliceError(f"Invalid slice: {spec}") from exc


def slice_label(spec: str | None) -> str:
    """Human readable description of a slice spec."""
    if spec is None or not spec.strip():
        return "all tasks"

    text = spec.strip()
    if text.startswith("-"):
        return f"last {text[1:]}"
    if "-" in text:
        head, tail = text.split("-", 1)
        return f"tasks {head}-{tail}"
    return f"first {text}"


_BENCHMARKS: dict[str, object] = {}
_TASKS: dict[str, list[Task]] = {}


def benchmark(key: str):
    """Return a cached benchmark instance for a registry key."""
    if key not in _BENCHMARKS:
        _BENCHMARKS[key] = REGISTRY[key]()
    return _BENCHMARKS[key]


def tasks_for(key: str) -> list[Task]:
    """Return the (cached) full task list for a registry key."""
    if key not in _TASKS:
        _TASKS[key] = benchmark(key).load_tasks()
    return _TASKS[key]


def task_count(key: str) -> int:
    declared = getattr(benchmark(key), "task_count", None)
    if isinstance(declared, int):
        return declared
    return len(tasks_for(key))


def slice_task_count(key: str) -> int:
    """Return the task count to which a user-provided slice is applied."""
    per_variant = getattr(benchmark(key), "tasks_per_variant", None)
    if isinstance(per_variant, int):
        return per_variant
    return task_count(key)


@dataclass(frozen=True)
class JobSpec:
    """One model run against one benchmark, optionally sliced."""

    model: str
    benchmark: str
    slice_spec: str | None = None
    variant: str | None = None
    perturbation: str | None = None
    perturbation_seed: int = 42
    harness: Literal["direct", "pi", "decision"] = "direct"
    repair_attempts: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.repair_attempts <= MAX_REPAIR_ATTEMPTS:
            raise ValueError(
                f"repair_attempts must be between 0 and {MAX_REPAIR_ATTEMPTS}"
            )

    @property
    def key(self) -> str:
        return (
            f"{self.model}|{self.benchmark}|{self.variant or ''}|"
            f"{self.slice_spec or ''}|{self.perturbation or ''}|"
            f"{self.perturbation_seed if self.perturbation else ''}|{self.harness}|"
            f"{self.repair_attempts}"
        )

    @property
    def benchmark_label(self) -> str:
        parts = [self.benchmark]
        if self.variant:
            parts.append(self.variant)
        if self.perturbation:
            parts.append(self.perturbation)
        return " · ".join(parts)

    @property
    def title(self) -> str:
        return f"{self.benchmark_label} · {self.model} · {self.harness_label}"

    @property
    def harness_label(self) -> str:
        label = {"pi": "Pi agent", "decision": "Decision"}.get(self.harness, "Direct")
        return f"{label} + repair" if self.repair_attempts else label

    def planned_total(self) -> int:
        if self.variant is not None:
            total = len(tasks_for_job(self))
        else:
            total = slice_task_count(self.benchmark)
        try:
            start, end = parse_slice(self.slice_spec, total)
        except SliceError:
            selected = total
        else:
            selected = max(0, end - start)
        if self.variant is not None:
            return selected
        variants = getattr(benchmark(self.benchmark), "variant_count", 1)
        return selected * variants if isinstance(variants, int) else selected


def expand_jobs(jobs: list[JobSpec], client: object) -> list[JobSpec]:
    """Expand benchmarks with independent score variants into concrete jobs."""
    expanded: list[JobSpec] = []
    for job in jobs:
        if job.variant is not None:
            expanded.append(job)
            continue
        bench = benchmark(job.benchmark)
        variants = getattr(bench, "variants", None)
        if not callable(variants):
            expanded.append(job)
            continue
        for variant in variants(client, job.model):
            expanded.append(
                JobSpec(
                    model=job.model,
                    benchmark=job.benchmark,
                    slice_spec=job.slice_spec,
                    variant=str(variant),
                    perturbation=job.perturbation,
                    perturbation_seed=job.perturbation_seed,
                    harness=job.harness,
                    repair_attempts=job.repair_attempts,
                )
            )
    return expanded


def route_decision_jobs(
    jobs: list[JobSpec],
    client: object,
    forced: Collection[str] = (),
) -> list[JobSpec]:
    """Send native decision models through /v1/systemone.

    A model is a decision model when the server says so, or when the user
    names it in ``forced`` (``--decision-models``) because a proxy in between
    hides it. Harness and repair settings are generation options, so a
    decision model's jobs drop them (``--harness both`` collapses to one job).
    A benchmark that cannot be asked as a typed question is a configuration
    error, raised before anything runs.
    """
    detect = getattr(client, "is_decision_model", None)
    if not callable(detect) and not forced:
        return jobs

    def is_decision(model: str) -> bool:
        return model in forced or (callable(detect) and bool(detect(model)))

    routed: dict[str, JobSpec] = {}
    unsupported: dict[str, None] = {}
    for job in jobs:
        if is_decision(job.model):
            if not decision.supports(job.benchmark):
                unsupported[f"{job.model} × {job.benchmark}"] = None
                continue
            job = replace(job, harness="decision", repair_attempts=0)
        routed.setdefault(job.key, job)
    if unsupported:
        raise ValueError(
            "Decision models only answer multiple-choice and yes/no benchmarks ("
            + ", ".join(decision.supported_benchmarks())
            + "). Unsupported: "
            + ", ".join(unsupported)
        )
    return list(routed.values())


def tasks_for_job(job: JobSpec) -> list[Task]:
    """Return tasks belonging to one concrete job variant."""
    bench = benchmark(job.benchmark)
    tasks = tasks_for(job.benchmark)
    select = getattr(bench, "tasks_for_variant", None)
    if job.variant is not None and callable(select):
        return list(select(tasks, job.variant))
    return tasks


def prompt_for(bench: object, task: Task, client: object, model: str) -> str:
    """Build a prompt, allowing synthetic suites to use the server tokenizer."""
    build_for = getattr(bench, "build_prompt_for", None)
    if callable(build_for):
        return str(build_for(task, client, model))
    return str(bench.build_prompt(task))


def option_count(bench: object, task: Task) -> int | None:
    """Return how many answers a multiple-choice task offers, else None.

    Tasks list their options in ``metadata["choices"]``; yes/no suites that
    keep no list declare ``answer_options`` on the benchmark instead.
    """
    choices = task.metadata.get("choices")
    if choices:
        return len(choices)
    declared = getattr(bench, "answer_options", None)
    return declared if isinstance(declared, int) and declared > 0 else None


def repair_budget(bench: object, task: Task, requested: int) -> int:
    """Repairs allowed for one task: none when it has at most two options.

    Telling a model its pick of two was wrong hands it the answer, so a
    repair there measures nothing.
    """
    count = option_count(bench, task)
    return 0 if count is not None and count <= 2 else requested


def plan_total_tasks(jobs: list[JobSpec], client: object | None = None) -> int:
    concrete = expand_jobs(jobs, client) if client is not None else jobs
    return sum(job.planned_total() for job in concrete)
