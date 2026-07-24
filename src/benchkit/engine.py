"""Benchmark execution engine shared by the TUI and the headless runner.

The engine is deliberately UI agnostic: it walks a list of jobs (one model x
one benchmark x one optional task slice), emits events as it goes, and returns
the same result dictionaries that :mod:`benchkit.report` already knows how to
serialize.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from benchkit.benchmarks import REGISTRY
from benchkit.benchmarks.base import Task
from benchkit.client import InferenceClient


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
    return len(tasks_for(key))


@dataclass(frozen=True)
class JobSpec:
    """One model run against one benchmark, optionally sliced."""

    model: str
    benchmark: str
    slice_spec: str | None = None

    @property
    def key(self) -> str:
        return f"{self.model}|{self.benchmark}|{self.slice_spec or ''}"

    @property
    def title(self) -> str:
        return f"{self.benchmark} · {self.model}"

    def planned_total(self) -> int:
        total = task_count(self.benchmark)
        try:
            start, end = parse_slice(self.slice_spec, total)
        except SliceError:
            return total
        return max(0, end - start)


@dataclass
class TaskRecord:
    """Outcome of a single task."""

    index: int
    task_id: str
    passed: bool
    tok_s: float
    response_time_s: float
    prompt: str
    response: str
    error: str = ""
    entry_point: str = ""

    @property
    def label(self) -> str:
        return (
            f"{self.task_id} ({self.entry_point})" if self.entry_point else self.task_id
        )


@dataclass
class RunStarted:
    jobs: list[JobSpec]
    total_tasks: int


@dataclass
class JobStarted:
    index: int
    job: JobSpec
    total: int
    overall_total: int


@dataclass
class TaskCompleted:
    index: int
    job: JobSpec
    record: TaskRecord
    passed: int
    completed: int


@dataclass
class JobCompleted:
    index: int
    job: JobSpec
    result: dict
    skipped: bool


@dataclass
class RunCompleted:
    results: list[dict]
    stopped: bool
    elapsed: float


@dataclass
class RunFailed:
    message: str


EngineEvent = (
    RunStarted | JobStarted | TaskCompleted | JobCompleted | RunCompleted | RunFailed
)
Sink = Callable[[EngineEvent], None]


class RunControls:
    """Thread-safe pause / skip / stop switches driven by the UI."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._skip = threading.Event()
        self._running = threading.Event()
        self._running.set()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    @property
    def paused(self) -> bool:
        return not self._running.is_set()

    def stop(self) -> None:
        self._stop.set()
        self._running.set()

    def skip_job(self) -> None:
        self._skip.set()
        self._running.set()

    def pause(self) -> None:
        self._running.clear()

    def resume(self) -> None:
        self._running.set()

    def toggle_pause(self) -> bool:
        """Flip the paused state and return ``True`` when now paused."""
        if self.paused:
            self.resume()
            return False
        self.pause()
        return True

    def take_skip(self) -> bool:
        """Consume a pending skip request."""
        if self._skip.is_set():
            self._skip.clear()
            return True
        return False

    def wait_while_paused(self) -> None:
        self._running.wait()


ERROR_GENERATION = {
    "response": "",
    "tok_s": 0.0,
    "eval_count": 0,
    "eval_duration_ns": 0,
    "response_time_s": 0.0,
    "done_reason": "error",
}


def plan_total_tasks(jobs: list[JobSpec]) -> int:
    return sum(job.planned_total() for job in jobs)


@dataclass
class Engine:
    """Runs a list of jobs, streaming events to a sink."""

    client: InferenceClient
    jobs: list[JobSpec]
    sink: Sink | None = None
    controls: RunControls = field(default_factory=RunControls)

    def emit(self, event: EngineEvent) -> None:
        if self.sink is not None:
            self.sink(event)

    def run(self) -> list[dict]:
        results: list[dict] = []
        started = time.time()
        overall_total = plan_total_tasks(self.jobs)
        self.emit(RunStarted(list(self.jobs), overall_total))

        try:
            for index, job in enumerate(self.jobs):
                if self.controls.stopped:
                    break
                result, skipped = self._run_job(index, job, overall_total)
                if result is not None:
                    results.append(result)
                    self.emit(JobCompleted(index, job, result, skipped))
                self._maybe_unload(index, job)
        except Exception as exc:  # surfaced to the UI instead of a traceback
            self.emit(RunFailed(f"{type(exc).__name__}: {exc}"))
            raise

        self.emit(
            RunCompleted(
                results, self.controls.stopped, round(time.time() - started, 1)
            )
        )
        return results

    def _maybe_unload(self, index: int, job: JobSpec) -> None:
        """Free VRAM once the last job for a model finishes."""
        upcoming = self.jobs[index + 1 :]
        if any(other.model == job.model for other in upcoming):
            return
        if len(self.jobs) <= 1:
            return
        try:
            self.client.unload_model(job.model)
        except Exception:
            pass

    def _run_job(
        self, index: int, job: JobSpec, overall_total: int
    ) -> tuple[dict | None, bool]:
        bench = benchmark(job.benchmark)
        all_tasks = tasks_for(job.benchmark)

        slice_spec = job.slice_spec
        try:
            start, end = parse_slice(slice_spec, len(all_tasks))
        except SliceError:
            start, end, slice_spec = 0, len(all_tasks), None
        tasks = all_tasks[start:end]

        self.emit(JobStarted(index, job, len(tasks), overall_total))

        passed = 0
        errors = 0
        total_tokens = 0
        total_eval_ns = 0
        total_response_time = 0.0
        records: list[TaskRecord] = []
        skipped = False
        wall_start = time.time()

        for position, task in enumerate(tasks):
            self.controls.wait_while_paused()
            if self.controls.stopped:
                break
            if self.controls.take_skip():
                skipped = True
                break

            prompt = bench.build_prompt(task)
            error = ""
            try:
                gen = self.client.generate(job.model, prompt)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                gen = dict(ERROR_GENERATION)
                errors += 1

            try:
                ok = bool(bench.evaluate(task, gen["response"]))
            except Exception as exc:
                ok = False
                error = error or f"evaluation failed: {type(exc).__name__}: {exc}"

            if ok:
                passed += 1

            total_tokens += gen["eval_count"]
            total_eval_ns += gen["eval_duration_ns"]
            total_response_time += gen["response_time_s"]

            record = TaskRecord(
                index=position,
                task_id=task.id,
                passed=ok,
                tok_s=round(gen["tok_s"], 1),
                response_time_s=round(gen["response_time_s"], 2),
                prompt=prompt,
                response=gen["response"],
                error=error,
                entry_point=str(task.metadata.get("entry_point", "")),
            )
            records.append(record)
            self.emit(TaskCompleted(index, job, record, passed, position + 1))

        if not records:
            return None, skipped

        total_time = round(time.time() - wall_start, 1)
        completed = len(records)
        tok_s = total_tokens / (total_eval_ns / 1e9) if total_eval_ns > 0 else 0.0
        score = passed / completed * 100 if completed else 0.0

        result = {
            "model": job.model,
            "benchmark": bench.name,
            "score": round(score, 1),
            "passed": passed,
            "total": completed,
            "tok_s": round(tok_s, 1),
            "avg_response_time": round(total_response_time / completed, 1),
            "total_time": total_time,
            "slice": slice_spec,
            "errors": errors,
            "tasks": [
                {
                    "task_id": record.task_id,
                    "passed": record.passed,
                    "tok_s": record.tok_s,
                    "response_time_s": record.response_time_s,
                    "prompt": record.prompt,
                    "response": record.response,
                    "error": record.error,
                    "entry_point": record.entry_point,
                }
                for record in records
            ],
        }
        return result, skipped
