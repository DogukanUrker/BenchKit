"""Benchmark execution engine shared by the TUI and the headless runner.

The engine is deliberately UI agnostic: it walks a list of jobs (one model x
one benchmark x one optional task slice), emits events as it goes, and returns
the same result dictionaries that :mod:`benchkit.report` already knows how to
serialize.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from numbers import Real
from statistics import median
from typing import Literal

import httpx

from benchkit.benchmarks import REGISTRY
from benchkit.benchmarks.base import Task
from benchkit.client import GenerationUpdate, InferenceClient
from benchkit.looping import LoopAnalyzer
from benchkit.metrics import throughput_metrics


class SliceError(ValueError):
    """Raised when a task slice specification cannot be parsed."""


class _GenerationCancelled(Exception):
    """Unwind a streaming request from its own worker thread immediately."""


class _DoomLoopKilled(Exception):
    """Unwind a generation that exceeded the configured loop threshold."""


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


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

    @property
    def key(self) -> str:
        return (
            f"{self.model}|{self.benchmark}|{self.variant or ''}|"
            f"{self.slice_spec or ''}"
        )

    @property
    def benchmark_label(self) -> str:
        return f"{self.benchmark} · {self.variant}" if self.variant else self.benchmark

    @property
    def title(self) -> str:
        return f"{self.benchmark_label} · {self.model}"

    def planned_total(self) -> int:
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
    score: float = 0.0
    error: str = ""
    entry_point: str = ""
    group: str = ""
    thinking: str = ""
    output_tokens: int = 0
    done_reason: str = ""
    timed_out: bool = False
    loop_killed: bool = False
    loop_kill_score: float = 0.0
    loop_killed_at_s: float | None = None
    trace_status: str = "unavailable"
    thinking_time_s: float = 0.0
    time_to_first_answer_s: float | None = None
    loop_state: str = "unavailable"
    loop_score: float = 0.0
    loop_source: str = "none"
    loop_detected_at_s: float | None = None
    repeated_ngram_coverage: float = 0.0
    max_window_similarity: float = 0.0
    low_novelty_windows: int = 0
    max_repeated_block: int = 0
    loop_evidence: str = "none"
    active_cycle: bool = False
    recovered_cycle: bool = False
    cycle_period_tokens: int = 0
    cycle_repetitions: int = 0
    repeated_suffix_tokens: int = 0

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
    concurrency: int = 1


@dataclass(frozen=True)
class TaskPhase:
    """Current phase of a task before its terminal result is available."""

    index: int
    job: JobSpec
    position: int
    total: int
    task_id: str
    entry_point: str
    phase: Literal["generating", "evaluating", "timed_out", "loop_killed"]
    activity: str

    @property
    def label(self) -> str:
        return (
            f"{self.task_id} ({self.entry_point})" if self.entry_point else self.task_id
        )


@dataclass(frozen=True)
class GenerationProgress:
    """Throttled live state from an in-flight model generation."""

    index: int
    job: JobSpec
    position: int
    total: int
    task_id: str
    entry_point: str
    phase: Literal["waiting", "thinking", "answering"]
    elapsed_s: float
    thinking_chars: int
    response_chars: int
    trace_status: str
    loop_state: str
    loop_score: float
    loop_source: str
    loop_kill_remaining_s: float | None
    prompt: str
    thinking: str
    response: str

    @property
    def label(self) -> str:
        return (
            f"{self.task_id} ({self.entry_point})" if self.entry_point else self.task_id
        )


@dataclass
class TaskCompleted:
    index: int
    job: JobSpec
    record: TaskRecord
    passed: int
    completed: int
    score_points: float | None = None


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
    RunStarted
    | JobStarted
    | TaskPhase
    | GenerationProgress
    | TaskCompleted
    | JobCompleted
    | RunCompleted
    | RunFailed
)
Sink = Callable[[EngineEvent], None]


class RunControls:
    """Thread-safe pause / skip / stop switches driven by the UI."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._skip = threading.Event()
        self._cancel_request = threading.Event()
        self._running = threading.Event()
        self._running.set()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    @property
    def cancel_event(self) -> threading.Event:
        """Event passed to clients to kill every active model request."""
        return self._cancel_request

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_request.is_set()

    @property
    def paused(self) -> bool:
        return not self._running.is_set()

    def stop(self) -> None:
        self._stop.set()
        self._cancel_request.set()
        self._running.set()

    def skip_job(self) -> None:
        self._skip.set()
        self._cancel_request.set()
        self._running.set()

    @property
    def skip_requested(self) -> bool:
        return self._skip.is_set()

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
            self._cancel_request.clear()
            return True
        return False

    def cancel_active(self) -> None:
        """Cancel in-flight requests while unwinding an internal job failure."""
        self._cancel_request.set()
        self._running.set()

    def wait_while_paused(self) -> None:
        self._running.wait()


ERROR_GENERATION = {
    "thinking": "",
    "response": "",
    "tok_s": 0.0,
    "eval_count": 0,
    "eval_duration_ns": 0,
    "response_time_s": 0.0,
    "done_reason": "error",
    "timed_out": False,
    "cancelled": False,
    "loop_killed": False,
    "trace_status": "unavailable",
}


@dataclass
class _GeneratedTask:
    """Generation state handed from a request worker to serial evaluation."""

    position: int
    task: Task
    prompt: str
    gen: dict
    error: str
    errors: int
    analyzer: LoopAnalyzer
    first_answer_at: float | None
    loop_detected_at: float | None


@dataclass
class _TaskOutcome:
    """One finalized task plus the raw metrics used by its job aggregate."""

    record: TaskRecord
    errors: int
    eval_count: int
    eval_duration_ns: int
    response_time_s: float


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
                )
            )
    return expanded


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


def plan_total_tasks(jobs: list[JobSpec], client: object | None = None) -> int:
    concrete = expand_jobs(jobs, client) if client is not None else jobs
    return sum(job.planned_total() for job in concrete)


def _result_metadata(job: JobSpec) -> dict:
    bench = benchmark(job.benchmark)
    metadata = getattr(bench, "result_metadata", None)
    if not callable(metadata):
        return {}
    value = metadata(job.variant)
    return dict(value) if isinstance(value, dict) else {}


def group_breakdown(records: list[TaskRecord], order: tuple[str, ...] = ()) -> list[dict]:
    """Per-group pass rates for a benchmark that labels its tasks.

    A benchmark opts in by putting a ``group`` in each task's metadata (
    LiveCodeBench uses the problem difficulty). The aggregate score alone hides
    exactly the split that matters on small models, where medium and hard
    collapse to zero while easy still separates them.
    """
    totals: dict[str, list[int]] = {}
    for record in records:
        if not record.group:
            continue
        bucket = totals.setdefault(record.group, [0, 0])
        bucket[0] += int(record.passed)
        bucket[1] += 1

    if not totals:
        return []

    ranked = sorted(
        totals,
        key=lambda name: (order.index(name) if name in order else len(order), name),
    )
    return [
        {
            "name": name,
            "passed": totals[name][0],
            "total": totals[name][1],
            "score": round(totals[name][0] / totals[name][1] * 100, 1),
        }
        for name in ranked
    ]


def _empty_result(job: JobSpec) -> dict:
    """Result shape for a job that finished without scoring a task."""
    return {
        "model": job.model,
        "benchmark": job.benchmark,
        "benchmark_label": job.benchmark_label,
        "variant": job.variant,
        "score": 0.0,
        "passed": 0,
        "total": 0,
        "tok_s": 0.0,
        "tok_s_per_stream": 0.0,
        "tok_s_aggregate": 0.0,
        "concurrency_eff": 0.0,
        "total_output_tokens": 0,
        "sum_generation_time": 0.0,
        "sum_request_time": 0.0,
        "avg_response_time": 0.0,
        "total_time": 0.0,
        "slice": job.slice_spec,
        "concurrency": 1,
        "errors": 0,
        "timeouts": 0,
        "loop_kills": 0,
        "loops": 0,
        "suspected_loops": 0,
        "loop_rate": 0.0,
        "trace_coverage": 0.0,
        "median_thinking_time": 0.0,
        "median_time_to_answer": 0.0,
        "groups": [],
        "note": "",
        "tasks": [],
        **_result_metadata(job),
    }


@dataclass
class Engine:
    """Runs a list of jobs, streaming events to a sink."""

    client: InferenceClient
    jobs: list[JobSpec]
    sink: Sink | None = None
    controls: RunControls = field(default_factory=RunControls)
    loop_kill_enabled: bool = field(
        default_factory=lambda: _env_bool("BENCHKIT_LOOP_KILL", True)
    )
    loop_kill_percent: float = field(
        default_factory=lambda: _env_float("BENCHKIT_LOOP_KILL_PERCENT", 80.0)
    )
    loop_kill_seconds: float = field(
        default_factory=lambda: _env_float("BENCHKIT_LOOP_KILL_SECONDS", 10.0)
    )
    failure: str | None = field(default=None, init=False)
    _emit_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.loop_kill_percent = min(100.0, max(0.0, self.loop_kill_percent))
        self.loop_kill_seconds = max(0.0, self.loop_kill_seconds)

    def emit(self, event: EngineEvent) -> None:
        if self.sink is not None:
            # Generation progress originates from multiple request workers.
            # Keep every sink callback atomic so front-ends can stay ordinary,
            # single-threaded state machines.
            with self._emit_lock:
                self.sink(event)

    def run(self) -> list[dict]:
        """Run every job and return the results, including a partial run.

        An unexpected failure ends the run through ``RunFailed`` rather than an
        exception, so the caller still gets the jobs that already finished and
        can write a report for them.
        """
        self.jobs = expand_jobs(self.jobs, self.client)
        results: list[dict] = []
        started = time.time()
        overall_total = plan_total_tasks(self.jobs)
        self.emit(RunStarted(list(self.jobs), overall_total))

        for index, job in enumerate(self.jobs):
            if self.controls.stopped:
                break
            try:
                result, skipped = self._run_job(index, job, overall_total)
            except Exception as exc:
                self.failure = f"{type(exc).__name__}: {exc}"
                self.emit(RunFailed(self.failure))
                break
            # Every job reports a terminal event, even an empty one, so the UI
            # never leaves a queue row running.
            self.emit(JobCompleted(index, job, result or _empty_result(job), skipped))
            if result is not None:
                results.append(result)
            self._maybe_unload(index, job)

        elapsed = (
            round(sum(result["total_time"] for result in results), 1)
            if getattr(self.client, "simulated_timing", False)
            else round(time.time() - started, 1)
        )
        self.emit(RunCompleted(results, self.controls.stopped, elapsed))
        return results

    def _maybe_unload(self, index: int, job: JobSpec) -> None:
        """Free VRAM once the last job for a model finishes."""
        upcoming = self.jobs[index + 1 :]
        if any(other.model == job.model for other in upcoming):
            return
        if len(self.jobs) <= 1:
            return
        with contextlib.suppress(Exception):
            self.client.unload_model(job.model)

    def _max_parallel_requests(self, model: str, total: int) -> int:
        """Resolve a safe worker count, keeping unknown clients serial."""
        if total <= 1:
            return 1
        discover = getattr(self.client, "max_parallel_requests", None)
        if not callable(discover):
            return 1
        try:
            capacity = discover(model)
            if isinstance(capacity, bool):
                return 1
            return min(total, max(1, int(capacity)))
        except Exception:
            # Capacity discovery is an optional optimization. A missing or
            # non-standard monitoring endpoint must never block a benchmark.
            return 1

    def _run_job(
        self, index: int, job: JobSpec, overall_total: int
    ) -> tuple[dict | None, bool]:
        bench = benchmark(job.benchmark)
        all_tasks = tasks_for_job(job)

        slice_spec = job.slice_spec
        try:
            start, end = parse_slice(slice_spec, len(all_tasks))
        except SliceError:
            start, end, slice_spec = 0, len(all_tasks), None
        tasks = all_tasks[start:end]

        concurrency = self._max_parallel_requests(job.model, len(tasks))
        wall_start = time.perf_counter()
        self.emit(
            JobStarted(
                index,
                job,
                len(tasks),
                overall_total,
                concurrency=concurrency,
            )
        )

        if not tasks:
            return None, False

        passed = 0
        score_points = 0.0
        errors = 0
        total_tokens = 0
        total_eval_ns = 0
        total_response_time = 0.0
        records_by_position: dict[int, TaskRecord] = {}
        skipped = False
        paused_time = 0.0
        next_position = 0
        futures: set[Future[_GeneratedTask]] = set()

        with ThreadPoolExecutor(
            max_workers=concurrency,
            thread_name_prefix=f"benchkit-{index}",
        ) as pool:

            def fill_workers() -> None:
                nonlocal next_position
                while (
                    len(futures) < concurrency
                    and next_position < len(tasks)
                    and not self.controls.paused
                    and not self.controls.stopped
                    and not self.controls.skip_requested
                ):
                    position = next_position
                    next_position += 1
                    future = pool.submit(
                        self._generate_task,
                        index,
                        job,
                        position,
                        len(tasks),
                        tasks[position],
                        bench,
                    )
                    futures.add(future)

            fill_workers()
            try:
                while futures or next_position < len(tasks):
                    if not futures:
                        if self.controls.stopped or self.controls.skip_requested:
                            break
                        if self.controls.paused:
                            pause_started = time.perf_counter()
                            self.controls.wait_while_paused()
                            paused_time += time.perf_counter() - pause_started
                            if self.controls.stopped or self.controls.skip_requested:
                                break
                        fill_workers()
                        if not futures:
                            continue

                    done, _ = wait(
                        tuple(futures),
                        timeout=0.05,
                        return_when=FIRST_COMPLETED,
                    )
                    if not done:
                        continue

                    generated_batch: list[_GeneratedTask] = []
                    for future in done:
                        futures.discard(future)
                        generated_batch.append(future.result())

                    # Refill request slots before evaluation so the next
                    # generations can overlap local benchmark scoring.
                    fill_workers()

                    if self.controls.stopped or self.controls.skip_requested:
                        continue

                    for generated in sorted(
                        generated_batch, key=lambda item: item.position
                    ):
                        outcome = self._finalize_task(
                            index,
                            job,
                            len(tasks),
                            bench,
                            generated,
                        )
                        if outcome is None:
                            continue
                        record = outcome.record
                        records_by_position[record.index] = record
                        passed += int(record.passed)
                        score_points += record.score
                        errors += outcome.errors
                        total_tokens += outcome.eval_count
                        total_eval_ns += outcome.eval_duration_ns
                        total_response_time += outcome.response_time_s
                        completed = len(records_by_position)
                        self.emit(
                            TaskCompleted(
                                index,
                                job,
                                record,
                                passed,
                                completed,
                                score_points,
                            )
                        )
            except Exception:
                self.controls.cancel_active()
                for future in futures:
                    future.cancel()
                raise

        if self.controls.skip_requested:
            skipped = self.controls.take_skip()

        if not records_by_position:
            return None, skipped

        records = [
            records_by_position[position] for position in sorted(records_by_position)
        ]
        measured_wall_time_s = max(0.0, time.perf_counter() - wall_start - paused_time)
        # Demo mode deliberately accelerates its fake generations. Use their
        # simulated request durations so its reports still demonstrate
        # internally consistent throughput rather than thousands of tok/s.
        wall_time_s = (
            total_response_time / concurrency
            if getattr(self.client, "simulated_timing", False)
            and total_response_time > 0
            else measured_wall_time_s
        )
        total_time = round(wall_time_s, 1)
        completed = len(records)
        throughput = throughput_metrics(
            output_tokens=total_tokens,
            generation_time_s=total_eval_ns / 1e9,
            request_time_s=total_response_time,
            wall_time_s=wall_time_s,
        )
        tok_s_per_stream = round(throughput["tok_s_per_stream"], 1)
        tok_s_aggregate = round(throughput["tok_s_aggregate"], 1)
        concurrency_eff = round(
            min(float(concurrency), throughput["concurrency_eff"]),
            2,
        )
        score = (
            sum(record.score for record in records) / completed * 100
            if completed
            else 0.0
        )
        loops = sum(record.loop_state == "looping" for record in records)
        suspected = sum(
            record.loop_state == "suspected" and not record.recovered_cycle
            for record in records
        )
        recovered = sum(record.recovered_cycle for record in records)
        timeouts = sum(record.timed_out for record in records)
        loop_kills = sum(record.loop_killed for record in records)
        traced = sum(record.trace_status != "unavailable" for record in records)
        thinking_times = [
            record.thinking_time_s for record in records if record.thinking
        ]
        answer_times = [
            record.time_to_first_answer_s
            for record in records
            if record.time_to_first_answer_s is not None
        ]

        result = {
            "model": job.model,
            "benchmark": bench.name,
            "benchmark_label": job.benchmark_label,
            "variant": job.variant,
            "score": round(score, 1),
            "passed": passed,
            "total": completed,
            # Keep ``tok_s`` as a compatibility alias for consumers of older
            # BenchKit JSON. New displays use the explicitly named metrics.
            "tok_s": tok_s_per_stream,
            "tok_s_per_stream": tok_s_per_stream,
            "tok_s_aggregate": tok_s_aggregate,
            "concurrency_eff": concurrency_eff,
            "total_output_tokens": total_tokens,
            "sum_generation_time": round(total_eval_ns / 1e9, 3),
            "sum_request_time": round(total_response_time, 3),
            "avg_response_time": round(total_response_time / completed, 1),
            "total_time": total_time,
            "slice": slice_spec,
            "concurrency": concurrency,
            "errors": errors,
            "timeouts": timeouts,
            "loop_kills": loop_kills,
            "loops": loops,
            "suspected_loops": suspected,
            "recovered_loops": recovered,
            "loop_rate": round(loops / completed * 100, 1),
            "trace_coverage": round(traced / completed * 100, 1),
            "median_thinking_time": round(median(thinking_times), 1)
            if thinking_times
            else 0.0,
            "median_time_to_answer": round(median(answer_times), 1)
            if answer_times
            else 0.0,
            # Optional, benchmark-provided context: a per-group breakdown and a
            # one-line note describing how the suite was configured.
            "groups": group_breakdown(
                records, tuple(getattr(bench, "group_order", ()))
            ),
            "note": str(getattr(bench, "report_note", "") or ""),
            "loop_kill_enabled": self.loop_kill_enabled,
            "loop_kill_percent": self.loop_kill_percent,
            "loop_kill_seconds": self.loop_kill_seconds,
            "tasks": [
                {
                    "task_id": record.task_id,
                    "passed": record.passed,
                    "score": round(record.score * 100, 1),
                    "tok_s": record.tok_s,
                    "response_time_s": record.response_time_s,
                    "prompt": record.prompt,
                    "response": record.response,
                    "error": record.error,
                    "entry_point": record.entry_point,
                    "group": record.group,
                    "thinking": record.thinking,
                    "output_tokens": record.output_tokens,
                    "done_reason": record.done_reason,
                    "timed_out": record.timed_out,
                    "loop_killed": record.loop_killed,
                    "loop_kill_score": record.loop_kill_score,
                    "loop_killed_at_s": record.loop_killed_at_s,
                    "trace_status": record.trace_status,
                    "thinking_time_s": record.thinking_time_s,
                    "time_to_first_answer_s": record.time_to_first_answer_s,
                    "loop_state": record.loop_state,
                    "loop_score": record.loop_score,
                    "loop_source": record.loop_source,
                    "loop_detected_at_s": record.loop_detected_at_s,
                    "repeated_ngram_coverage": (record.repeated_ngram_coverage),
                    "max_window_similarity": record.max_window_similarity,
                    "low_novelty_windows": record.low_novelty_windows,
                    "max_repeated_block": record.max_repeated_block,
                    "loop_evidence": record.loop_evidence,
                    "active_cycle": record.active_cycle,
                    "recovered_cycle": record.recovered_cycle,
                    "cycle_period_tokens": record.cycle_period_tokens,
                    "cycle_repetitions": record.cycle_repetitions,
                    "repeated_suffix_tokens": record.repeated_suffix_tokens,
                }
                for record in records
            ],
            **_result_metadata(job),
        }
        return result, skipped

    def _generate_task(
        self,
        index: int,
        job: JobSpec,
        position: int,
        total: int,
        task: Task,
        bench: object,
    ) -> _GeneratedTask:
        """Generate one answer in a request worker, including live analysis."""
        prompt = prompt_for(bench, task, self.client, job.model)
        error = ""
        errors = 0
        entry_point = str(task.metadata.get("entry_point", ""))
        self.emit(
            TaskPhase(
                index=index,
                job=job,
                position=position + 1,
                total=total,
                task_id=task.id,
                entry_point=entry_point,
                phase="generating",
                activity="waiting for model",
            )
        )
        analyzer = LoopAnalyzer()
        last_progress_emit = 0.0
        last_progress_state = ""
        reasoning_channel_seen = False
        first_answer_at: float | None = None
        loop_detected_at: float | None = None
        loop_above_since: float | None = None
        loop_killed_at_s: float | None = None
        loop_kill_score = 0.0

        def on_progress(update: GenerationUpdate) -> None:
            nonlocal first_answer_at
            nonlocal last_progress_emit
            nonlocal last_progress_state
            nonlocal loop_above_since
            nonlocal loop_detected_at
            nonlocal loop_killed_at_s
            nonlocal loop_kill_score
            nonlocal reasoning_channel_seen

            if self.controls.cancel_requested:
                raise _GenerationCancelled

            analyzer.add(thinking=update.thinking, answer=update.response)
            reasoning_channel_seen = (
                reasoning_channel_seen or update.reasoning_channel_seen
            )
            if update.response and first_answer_at is None:
                first_answer_at = update.elapsed_s

            if analyzer.answer_chars:
                live_phase = "answering"
            elif analyzer.thinking_chars:
                live_phase = "thinking"
            else:
                live_phase = "waiting"

            trace_status = (
                "observed"
                if analyzer.thinking_chars
                else "available_empty"
                if reasoning_channel_seen
                else "unavailable"
            )
            state_key = f"{live_phase}:{trace_status}"
            now = time.monotonic()
            if (
                not update.done
                and state_key == last_progress_state
                and now - last_progress_emit < 0.25
            ):
                return
            snapshot = analyzer.snapshot()
            if snapshot.state == "looping" and loop_detected_at is None:
                loop_detected_at = update.elapsed_s
            threshold = self.loop_kill_percent / 100
            if not self.loop_kill_enabled:
                loop_above_since = None
                remaining = None
            elif snapshot.confirmed_cycle and snapshot.score >= threshold:
                if loop_above_since is None:
                    loop_above_since = now
                above_for = now - loop_above_since
                remaining = max(0.0, self.loop_kill_seconds - above_for)
                if above_for >= self.loop_kill_seconds and not update.done:
                    loop_killed_at_s = update.elapsed_s
                    loop_kill_score = snapshot.score
                    raise _DoomLoopKilled
            else:
                loop_above_since = None
                remaining = None
            last_progress_emit = now
            last_progress_state = state_key
            self.emit(
                GenerationProgress(
                    index=index,
                    job=job,
                    position=position + 1,
                    total=total,
                    task_id=task.id,
                    entry_point=entry_point,
                    phase=live_phase,
                    elapsed_s=round(update.elapsed_s, 2),
                    thinking_chars=analyzer.thinking_chars,
                    response_chars=analyzer.answer_chars,
                    trace_status=trace_status,
                    loop_state=snapshot.state,
                    loop_score=snapshot.score,
                    loop_source=snapshot.source,
                    loop_kill_remaining_s=(
                        round(remaining, 2) if remaining is not None else None
                    ),
                    prompt=prompt,
                    thinking=analyzer.thinking,
                    response=analyzer.answer,
                )
            )

        try:
            gen = self.client.generate(
                job.model,
                prompt,
                on_progress=on_progress,
                cancel_event=self.controls.cancel_event,
            )
        except _GenerationCancelled:
            gen = dict(ERROR_GENERATION)
            gen.update(done_reason="cancelled", cancelled=True)
        except _DoomLoopKilled:
            elapsed_s = loop_killed_at_s or 0.0
            thinking = analyzer.thinking
            response = analyzer.answer
            gen = dict(ERROR_GENERATION)
            gen.update(
                thinking=thinking,
                response=response,
                response_time_s=elapsed_s,
                done_reason="loop_killed",
                loop_killed=True,
                loop_kill_score=loop_kill_score,
                loop_killed_at_s=elapsed_s,
                trace_status=(
                    "observed"
                    if thinking
                    else "available_empty"
                    if reasoning_channel_seen
                    else "unavailable"
                ),
            )
        except Exception as exc:
            gen = dict(ERROR_GENERATION)
            if self.controls.cancel_requested:
                gen.update(done_reason="cancelled", cancelled=True)
            elif isinstance(exc, (TimeoutError, httpx.TimeoutException)):
                error = f"generation timed out after {self.client.timeout:g}s"
                gen.update(
                    done_reason="timeout",
                    timed_out=True,
                    response_time_s=self.client.timeout,
                )
            else:
                error = f"{type(exc).__name__}: {exc}"
            if not gen.get("cancelled"):
                errors += 1

        return _GeneratedTask(
            position=position,
            task=task,
            prompt=prompt,
            gen=gen,
            error=error,
            errors=errors,
            analyzer=analyzer,
            first_answer_at=first_answer_at,
            loop_detected_at=loop_detected_at,
        )

    def _finalize_task(
        self,
        index: int,
        job: JobSpec,
        total: int,
        bench: object,
        generated: _GeneratedTask,
    ) -> _TaskOutcome | None:
        """Evaluate a generated answer and build its deterministic task record."""
        gen = generated.gen
        if (
            self.controls.stopped
            or self.controls.skip_requested
            or gen.get("cancelled")
        ):
            return None

        position = generated.position
        task = generated.task
        prompt = generated.prompt
        error = generated.error
        errors = generated.errors
        entry_point = str(task.metadata.get("entry_point", ""))

        final_analyzer = generated.analyzer
        final_thinking = gen.get("thinking", "") or ""
        final_answer = gen.get("response", "") or ""
        if not final_analyzer.reconcile(
            thinking=final_thinking,
            answer=final_answer,
        ):
            final_analyzer = LoopAnalyzer()
            final_analyzer.add(
                thinking=final_thinking,
                answer=final_answer,
            )
        loop = final_analyzer.snapshot(final=True)
        loop_detected_at = generated.loop_detected_at
        if loop.state == "looping" and loop_detected_at is None:
            loop_detected_at = gen.get("response_time_s", 0.0)
        first_answer_at = generated.first_answer_at
        if first_answer_at is None and gen.get("response"):
            first_answer_at = gen.get("response_time_s", 0.0)

        timed_out = bool(gen.get("timed_out"))
        loop_killed = bool(gen.get("loop_killed"))
        evaluation_score = 0.0
        if loop_killed:
            ok = False
            error = (
                "doom loop killed after a confirmed active cycle stayed at or above "
                f"{self.loop_kill_percent:g}% for "
                f"{self.loop_kill_seconds:g}s"
            )
            errors += 1
            phase = "loop_killed"
            activity = "doom loop killed — skipping task"
        elif timed_out:
            ok = False
            if not error:
                error = f"generation timed out after {self.client.timeout:g}s"
                errors += 1
            phase = "timed_out"
            activity = "timed out — skipping task"
        else:
            phase = "evaluating"
            activity = getattr(bench, "evaluation_activity", "evaluating response")

        self.emit(
            TaskPhase(
                index=index,
                job=job,
                position=position + 1,
                total=total,
                task_id=task.id,
                entry_point=entry_point,
                phase=phase,
                activity=activity,
            )
        )

        if not loop_killed and not timed_out:
            try:
                evaluation = bench.evaluate(task, gen["response"])
                if isinstance(evaluation, bool):
                    evaluation_score = 1.0 if evaluation else 0.0
                elif isinstance(evaluation, Real):
                    evaluation_score = min(1.0, max(0.0, float(evaluation)))
                else:
                    evaluation_score = 1.0 if bool(evaluation) else 0.0
                ok = evaluation_score >= 1.0
            except Exception as exc:
                ok = False
                if not error:
                    error = f"evaluation failed: {type(exc).__name__}: {exc}"
                    errors += 1

        response_time_s = float(gen.get("response_time_s", 0.0))
        record = TaskRecord(
            index=position,
            task_id=task.id,
            passed=ok,
            tok_s=round(float(gen.get("tok_s", 0.0)), 1),
            response_time_s=round(response_time_s, 2),
            prompt=prompt,
            response=gen.get("response", ""),
            score=evaluation_score,
            error=error,
            entry_point=entry_point,
            group=str(task.metadata.get("group", "")),
            thinking=gen.get("thinking", ""),
            output_tokens=int(gen.get("eval_count", 0)),
            done_reason=gen.get("done_reason", ""),
            timed_out=timed_out,
            loop_killed=loop_killed,
            loop_kill_score=float(gen.get("loop_kill_score", 0.0)),
            loop_killed_at_s=gen.get("loop_killed_at_s"),
            trace_status=gen.get("trace_status", "unavailable"),
            thinking_time_s=round(
                (first_answer_at if first_answer_at is not None else response_time_s)
                if gen.get("thinking")
                else 0.0,
                2,
            ),
            time_to_first_answer_s=(
                round(first_answer_at, 2) if first_answer_at is not None else None
            ),
            loop_state=loop.state,
            loop_score=loop.score,
            loop_source=loop.source,
            loop_detected_at_s=(
                round(loop_detected_at, 2) if loop_detected_at is not None else None
            ),
            repeated_ngram_coverage=loop.repeated_ngram_coverage,
            max_window_similarity=loop.max_window_similarity,
            low_novelty_windows=loop.low_novelty_windows,
            max_repeated_block=loop.max_repeated_block,
            loop_evidence=loop.evidence,
            active_cycle=loop.active_cycle,
            recovered_cycle=loop.recovered_cycle,
            cycle_period_tokens=loop.cycle_period_tokens,
            cycle_repetitions=loop.cycle_repetitions,
            repeated_suffix_tokens=loop.repeated_suffix_tokens,
        )
        return _TaskOutcome(
            record=record,
            errors=errors,
            eval_count=int(gen.get("eval_count", 0)),
            eval_duration_ns=int(gen.get("eval_duration_ns", 0)),
            response_time_s=response_time_s,
        )
