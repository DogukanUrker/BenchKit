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
from dataclasses import dataclass, field, replace
from statistics import median
from typing import Literal

import httpx

from benchkit.benchmarks import REGISTRY
from benchkit.benchmarks.base import Task
from benchkit.client import GenerationUpdate, InferenceClient
from benchkit.evaluation import (
    EvaluationResult,
    combine_attempts,
    evaluate_response,
    standalone_repair_prompt,
)
from benchkit.looping import LoopAnalyzer
from benchkit.metrics import throughput_metrics
from benchkit.perturbations import annotate_robustness, perturb_task
from benchkit.pi_agent import PiAgentRunner
from benchkit.sandbox import cleanup_run_resources

MAX_REPAIR_ATTEMPTS = 10


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
    perturbation: str | None = None
    perturbation_seed: int = 42
    harness: Literal["direct", "pi"] = "direct"
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
        label = "Pi agent" if self.harness == "pi" else "Direct"
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
    outcome: Literal[
        "pass",
        "fail",
        "loop_killed",
        "timeout",
        "length_exceeded",
        "harness_error",
        "contaminated",
    ] = "fail"
    perturbation: dict = field(default_factory=dict)
    score: float = 0.0
    error: str = ""
    entry_point: str = ""
    thinking: str = ""
    output_tokens: int = 0
    tokens_recovered: bool = False
    done_reason: str = ""
    timed_out: bool = False
    loop_killed: bool = False
    length_exceeded: bool = False
    harness_error: bool = False
    contaminated: bool = False
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
    harness: str = "direct"
    harness_version: str = ""
    input_tokens: int = 0
    model_turns: int = 1
    tool_calls: int = 0
    tool_trace: list[dict] = field(default_factory=list)
    pi_scaffold: dict = field(default_factory=dict)
    attempts: list[dict] = field(default_factory=list)
    repair_attempts_used: int = 0
    repair_feedback: list[str] = field(default_factory=list)
    first_attempt_score: float = 0.0
    repaired: bool = False
    workspace: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return (
            f"{self.task_id} ({self.entry_point})" if self.entry_point else self.task_id
        )


def _repair_score_curve(records: list[TaskRecord], repair_attempts: int) -> list[dict]:
    """Return aggregate score and incremental gains after each repair round."""
    if not records or repair_attempts <= 0:
        return []

    def score_at(record: TaskRecord, round_number: int) -> float:
        if record.attempts:
            attempt = record.attempts[min(round_number, len(record.attempts) - 1)]
            return float(attempt.get("score", 0.0))
        if round_number == 0:
            return record.first_attempt_score * 100
        return record.score * 100

    curve: list[dict] = []
    previous_score = 0.0
    for round_number in range(repair_attempts + 1):
        scores = [score_at(record, round_number) for record in records]
        aggregate = sum(scores) / len(scores)
        previous_scores = (
            [score_at(record, round_number - 1) for record in records]
            if round_number
            else scores
        )
        curve.append(
            {
                "round": round_number,
                "score": round(aggregate, 1),
                "delta_pp": round(aggregate - previous_score, 1)
                if round_number
                else 0.0,
                "cumulative_delta_pp": round(aggregate - curve[0]["score"], 1)
                if curve
                else 0.0,
                "attempted": sum(
                    len(record.attempts) > round_number for record in records
                )
                if round_number
                else len(records),
                "new_passes": sum(
                    before < 100 <= after
                    for before, after in zip(previous_scores, scores, strict=True)
                )
                if round_number
                else sum(score >= 100 for score in scores),
            }
        )
        previous_score = aggregate
    return curve


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
    phase: Literal[
        "generating",
        "evaluating",
        "timed_out",
        "loop_killed",
        "length_exceeded",
        "harness_error",
    ]
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
    scored_total: int | None = None


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


@dataclass(frozen=True)
class ModelUnloaded:
    """A force-unload attempt for a model that finished its last job."""

    model: str
    error: str = ""


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
    | ModelUnloaded
    | RunCompleted
    | RunFailed
)
Sink = Callable[[EngineEvent], None]


class RunControls:
    """Thread-safe pause / skip / stop / force-unload switches for the UI."""

    def __init__(self, force_unload: bool = False) -> None:
        self._stop = threading.Event()
        self._skip = threading.Event()
        self._cancel_request = threading.Event()
        self._running = threading.Event()
        self._running.set()
        self._force_unload = force_unload

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

    @property
    def force_unload(self) -> bool:
        return self._force_unload

    def toggle_force_unload(self) -> bool:
        """Flip force-unload and return the new state."""
        self._force_unload = not self._force_unload
        return self._force_unload

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
    "length_exceeded": False,
    "harness_error": False,
    "trace_status": "unavailable",
}


@dataclass
class _GeneratedTask:
    """Generation state handed from a request worker to serial evaluation."""

    position: int
    task: Task
    prompt: str
    perturbation: dict
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
                    perturbation=job.perturbation,
                    perturbation_seed=job.perturbation_seed,
                    harness=job.harness,
                    repair_attempts=job.repair_attempts,
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
    value = metadata(job.variant) if callable(metadata) else None
    result = dict(value) if isinstance(value, dict) else {}
    result.update(
        harness=job.harness,
        harness_label=job.harness_label,
        repair_attempts=job.repair_attempts,
    )
    if job.perturbation:
        result.update(
            perturbation=job.perturbation,
            perturbation_seed=job.perturbation_seed,
            include_in_overall=False,
        )
    return result


def _empty_result(job: JobSpec) -> dict:
    """Result shape for a job that finished without scoring a task."""
    return {
        "model": job.model,
        "benchmark": job.benchmark,
        "benchmark_label": job.benchmark_label,
        "variant": job.variant,
        "harness": job.harness,
        "harness_label": job.harness_label,
        "score": 0.0,
        "passed": 0,
        "total": 0,
        "scored_total": 0,
        "tok_s": 0.0,
        "tok_s_per_stream": 0.0,
        "tok_s_aggregate": 0.0,
        "concurrency_eff": 0.0,
        "throughput_coverage": 0.0,
        "throughput_items": 0,
        "throughput_wall_time": 0.0,
        "total_output_tokens": 0,
        "total_input_tokens": 0,
        "model_turns": 0,
        "tool_calls": 0,
        "first_attempt_score": 0.0,
        "repair_delta_pp": 0.0,
        "repair_attempted": 0,
        "repair_successes": 0,
        "sum_generation_time": 0.0,
        "sum_request_time": 0.0,
        "avg_response_time": 0.0,
        "total_time": 0.0,
        "slice": job.slice_spec,
        "concurrency": 1,
        "errors": 0,
        "harness_errors": 0,
        "contaminated": 0,
        "contamination_by_language": {},
        "contaminated_tasks": [],
        "length_exceeded": 0,
        "timeouts": 0,
        "loop_kills": 0,
        "failures": 0,
        "loops": 0,
        "suspected_loops": 0,
        "loop_rate": 0.0,
        "trace_coverage": 0.0,
        "median_thinking_time": 0.0,
        "median_time_to_answer": 0.0,
        "tasks": [],
        **_result_metadata(job),
    }


def _harness_pair_key(result: dict) -> tuple[object, ...]:
    return (
        result.get("model"),
        result.get("benchmark"),
        result.get("variant"),
        result.get("slice"),
        result.get("perturbation"),
        result.get("perturbation_seed") if result.get("perturbation") else None,
        result.get("repair_attempts", 0),
    )


def annotate_harness_effect(results: list[dict]) -> None:
    """Attach paired direct-versus-agent deltas to Pi result rows."""
    direct = {
        _harness_pair_key(result): result
        for result in results
        if result.get("harness", "direct") == "direct"
    }
    for result in results:
        if result.get("harness") != "pi":
            continue
        baseline = direct.get(_harness_pair_key(result))
        if baseline is None:
            continue
        baseline_tasks = {task["task_id"]: task for task in baseline.get("tasks", [])}
        pairs = [
            (baseline_tasks[task["task_id"]], task)
            for task in result.get("tasks", [])
            if task["task_id"] in baseline_tasks
            and not task.get("harness_error")
            and not baseline_tasks[task["task_id"]].get("harness_error")
        ]
        if not pairs:
            continue
        total = len(pairs)
        direct_points = sum(float(before.get("score", 0)) for before, _ in pairs)
        pi_points = sum(float(after.get("score", 0)) for _, after in pairs)
        direct_first_points = sum(
            float(before.get("first_attempt_score", before.get("score", 0)))
            for before, _ in pairs
        )
        pi_first_points = sum(
            float(after.get("first_attempt_score", after.get("score", 0)))
            for _, after in pairs
        )
        gains = sum(
            not bool(before.get("passed")) and bool(after.get("passed"))
            for before, after in pairs
        )
        regressions = sum(
            bool(before.get("passed")) and not bool(after.get("passed"))
            for before, after in pairs
        )
        direct_loop_kills = sum(bool(before.get("loop_killed")) for before, _ in pairs)
        harness_loop_kills = sum(bool(after.get("loop_killed")) for _, after in pairs)
        result.update(
            harness_paired_total=total,
            direct_score=round(direct_points / total, 1),
            harness_score=round(pi_points / total, 1),
            harness_score_delta_pp=round((pi_points - direct_points) / total, 1),
            direct_first_score=round(direct_first_points / total, 1),
            harness_first_score=round(pi_first_points / total, 1),
            harness_first_score_delta_pp=round(
                (pi_first_points - direct_first_points) / total,
                1,
            ),
            harness_gains=gains,
            harness_regressions=regressions,
            direct_loop_kill_rate=round(direct_loop_kills / total * 100, 1),
            harness_loop_kill_rate=round(harness_loop_kills / total * 100, 1),
            loop_kill_delta_pp=round(
                (harness_loop_kills - direct_loop_kills) / total * 100,
                1,
            ),
        )


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
    _evaluation_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    _pi_runner: PiAgentRunner | None = field(default=None, init=False, repr=False)
    _workspace_pi_runners: dict[str, PiAgentRunner] = field(
        default_factory=dict, init=False, repr=False
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

    def _pi(
        self, bench: object | None = None, task: Task | None = None
    ) -> PiAgentRunner:
        task_image_factory = getattr(bench, "pi_image_for_task", None)
        if callable(task_image_factory) and task is not None:
            image = task_image_factory(task)
            key = f"{getattr(bench, 'name', type(bench).__name__)}:{image.image}"
            if key not in self._workspace_pi_runners:
                self._workspace_pi_runners[key] = PiAgentRunner(
                    self.client, image=image
                )
            return self._workspace_pi_runners[key]
        image_factory = getattr(bench, "pi_image", None)
        if callable(image_factory):
            key = str(getattr(bench, "name", type(bench).__name__))
            if key not in self._workspace_pi_runners:
                self._workspace_pi_runners[key] = PiAgentRunner(
                    self.client, image=image_factory()
                )
            return self._workspace_pi_runners[key]
        if self._pi_runner is None:
            if getattr(self.client, "provider", None) == "demo":
                raise RuntimeError("Pi harness is unavailable in demo mode")
            self._pi_runner = PiAgentRunner(self.client)
        return self._pi_runner

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

        try:
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
                self.emit(
                    JobCompleted(index, job, result or _empty_result(job), skipped)
                )
                if result is not None:
                    results.append(result)
                self._maybe_unload(index, job)
        finally:
            used_pi = self._pi_runner is not None or bool(self._workspace_pi_runners)
            if self._pi_runner is not None:
                with contextlib.suppress(Exception):
                    self._pi_runner.cleanup()
            for runner in self._workspace_pi_runners.values():
                with contextlib.suppress(Exception):
                    runner.cleanup()
            if used_pi:
                # Removes the run's shared builder, its build cache, and any
                # image, container, network, or volume still carrying the run
                # label. Scoped by label, never a global prune.
                with contextlib.suppress(Exception):
                    cleanup_run_resources()

        annotate_robustness(results)
        annotate_harness_effect(results)
        elapsed = (
            round(sum(result["total_time"] for result in results), 1)
            if getattr(self.client, "simulated_timing", False)
            else round(time.time() - started, 1)
        )
        self.emit(RunCompleted(results, self.controls.stopped, elapsed))
        return results

    def _maybe_unload(self, index: int, job: JobSpec) -> None:
        """Free VRAM once the last job for a model finishes.

        Jobs run strictly one model at a time and ``_run_job`` only returns
        after every in-flight request has drained, so by the time this runs the
        model's slots are idle. When force-unload is on, the model is evicted
        here (and the unload request completes) before the next model's job
        submits its first request.
        """
        upcoming = self.jobs[index + 1 :]
        if any(other.model == job.model for other in upcoming):
            return
        if len(self.jobs) <= 1:
            return
        if self.controls.force_unload:
            error = ""
            try:
                self.client.force_unload_model(job.model)
            except Exception as exc:
                error = str(exc)
            self.emit(ModelUnloaded(job.model, error))
        else:
            with contextlib.suppress(Exception):
                self.client.unload_model(job.model)

    def _max_parallel_requests(self, job: JobSpec, total: int) -> int:
        """Resolve a safe worker count, keeping unknown clients serial."""
        if total <= 1:
            return 1
        discover = getattr(self.client, "max_parallel_requests", None)
        if not callable(discover):
            return 1
        try:
            capacity = discover(job.model)
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
        if getattr(bench, "workspace_task", False) and job.harness != "pi":
            raise ValueError(
                f"{job.benchmark} requires the Pi harness; use --harness pi"
            )
        all_tasks = tasks_for_job(job)

        slice_spec = job.slice_spec
        try:
            start, end = parse_slice(slice_spec, len(all_tasks))
        except SliceError:
            start, end, slice_spec = 0, len(all_tasks), None
        tasks = all_tasks[start:end]

        concurrency = self._max_parallel_requests(job, len(tasks))
        self.emit(
            JobStarted(
                index,
                job,
                len(tasks),
                overall_total,
                concurrency=concurrency,
            )
        )

        if job.harness == "pi" and tasks:
            first = tasks[0]
            self.emit(
                TaskPhase(
                    index=index,
                    job=job,
                    position=1,
                    total=len(tasks),
                    task_id=first.id,
                    entry_point=str(first.metadata.get("entry_point", "")),
                    phase="generating",
                    activity="preparing pinned Pi sandbox image",
                )
            )
            for task in tasks:
                self._pi(bench, task).prepare()

        if not tasks:
            return None, False

        # Image resolution is run setup, not task execution. Per-task Pi
        # container startup and every agent/tool turn remain in the timing.
        wall_start = time.perf_counter()
        passed = 0
        score_points = 0.0
        scored_total = 0
        errors = 0
        total_tokens = 0
        total_response_time = 0.0
        throughput_tokens = 0
        throughput_eval_ns = 0
        throughput_response_time = 0.0
        throughput_items = 0
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
                        scored_total += int(
                            not record.harness_error and not record.contaminated
                        )
                        errors += outcome.errors
                        total_tokens += outcome.eval_count
                        total_response_time += outcome.response_time_s
                        complete_terminal_metrics = not (
                            record.length_exceeded or record.timed_out
                        ) or (
                            outcome.eval_count > 0
                            and outcome.eval_duration_ns > 0
                            and outcome.response_time_s > 0
                        )
                        if (
                            not (record.loop_killed or record.harness_error)
                            and complete_terminal_metrics
                        ):
                            throughput_tokens += outcome.eval_count
                            throughput_eval_ns += outcome.eval_duration_ns
                            throughput_response_time += outcome.response_time_s
                            throughput_items += 1
                        completed = len(records_by_position)
                        self.emit(
                            TaskCompleted(
                                index,
                                job,
                                record,
                                passed,
                                completed,
                                score_points,
                                scored_total,
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
        scored_records = [
            record
            for record in records
            if not record.harness_error and not record.contaminated
        ]
        scored_total = len(scored_records)
        throughput_coverage = (
            throughput_response_time / total_response_time
            if total_response_time > 0
            else throughput_items / completed
            if completed
            else 0.0
        )
        throughput_wall_time = wall_time_s * throughput_coverage
        throughput = throughput_metrics(
            output_tokens=throughput_tokens,
            generation_time_s=throughput_eval_ns / 1e9,
            request_time_s=throughput_response_time,
            wall_time_s=throughput_wall_time,
        )
        tok_s_per_stream = round(throughput["tok_s_per_stream"], 1)
        tok_s_aggregate = round(throughput["tok_s_aggregate"], 1)
        concurrency_eff = round(
            min(float(concurrency), throughput["concurrency_eff"]),
            2,
        )
        score = (
            sum(record.score for record in scored_records) / scored_total * 100
            if scored_total
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
        length_exceeded = sum(record.length_exceeded for record in records)
        harness_errors = sum(record.harness_error for record in records)
        contaminated_records = [record for record in records if record.contaminated]
        failures = sum(record.outcome == "fail" for record in records)
        traced = sum(record.trace_status != "unavailable" for record in records)
        latency_records = [
            record
            for record in records
            if not (
                record.loop_killed
                or record.timed_out
                or record.length_exceeded
                or record.harness_error
            )
            and record.time_to_first_answer_s is not None
        ]
        thinking_times = [record.thinking_time_s for record in latency_records]
        answer_times = [
            float(record.time_to_first_answer_s) for record in latency_records
        ]
        pi_scaffold = next(
            (record.pi_scaffold for record in records if record.pi_scaffold), {}
        )
        pi_metadata = {f"pi_{key}": value for key, value in pi_scaffold.items()}
        agentic = [
            record.workspace.get("agentic_metrics", {})
            for record in records
            if isinstance(record.workspace.get("agentic_metrics"), dict)
        ]
        schema_valid = sum(
            int(item.get("tool_schema_valid_calls", 0)) for item in agentic
        )
        schema_invalid = sum(
            int(item.get("tool_schema_invalid_calls", 0)) for item in agentic
        )
        errored_calls = sum(int(item.get("errored_tool_calls", 0)) for item in agentic)
        recoveries = sum(int(item.get("post_error_recoveries", 0)) for item in agentic)
        redundant_calls = sum(
            int(item.get("redundant_tool_calls", 0)) for item in agentic
        )
        destructive_actions = sum(
            int(item.get("destructive_action_count", 0)) for item in agentic
        )
        observed_calls = schema_valid + schema_invalid
        solved_records = [record for record in records if record.passed]
        repair_score_curve = _repair_score_curve(scored_records, job.repair_attempts)
        repair_round_columns = {
            key: value
            for point in repair_score_curve[1:]
            for key, value in {
                f"repair_round_{point['round']}_score": point["score"],
                f"repair_round_{point['round']}_delta_pp": point["delta_pp"],
                f"repair_round_{point['round']}_cumulative_delta_pp": point[
                    "cumulative_delta_pp"
                ],
                f"repair_round_{point['round']}_attempted": point["attempted"],
                f"repair_round_{point['round']}_new_passes": point["new_passes"],
            }.items()
        }

        result = {
            "model": job.model,
            "benchmark": bench.name,
            "benchmark_label": job.benchmark_label,
            "variant": job.variant,
            "harness": job.harness,
            "harness_label": job.harness_label,
            "harness_version": (
                next(
                    (
                        record.harness_version
                        for record in records
                        if record.harness_version
                    ),
                    "",
                )
            ),
            "score": round(score, 1),
            "passed": passed,
            "total": completed,
            "scored_total": scored_total,
            # Keep ``tok_s`` as a compatibility alias for consumers of older
            # BenchKit JSON. New displays use the explicitly named metrics.
            "tok_s": tok_s_per_stream,
            "tok_s_per_stream": tok_s_per_stream,
            "tok_s_aggregate": tok_s_aggregate,
            "concurrency_eff": concurrency_eff,
            "throughput_coverage": round(throughput_coverage * 100, 1),
            "throughput_items": throughput_items,
            "throughput_wall_time": round(throughput_wall_time, 3),
            "total_output_tokens": total_tokens,
            "throughput_output_tokens": throughput_tokens,
            "total_input_tokens": sum(record.input_tokens for record in records),
            "model_turns": sum(record.model_turns for record in records),
            "tool_calls": sum(record.tool_calls for record in records),
            "tool_schema_valid_calls": schema_valid,
            "tool_schema_invalid_calls": schema_invalid,
            "tool_schema_validity_rate": round(schema_valid / observed_calls * 100, 1)
            if observed_calls
            else None,
            "errored_tool_calls": errored_calls,
            "post_error_recoveries": recoveries,
            "post_error_recovery_rate": round(recoveries / errored_calls * 100, 1)
            if errored_calls
            else None,
            "redundant_tool_calls": redundant_calls,
            "redundant_action_rate": round(redundant_calls / observed_calls * 100, 1)
            if observed_calls
            else None,
            "destructive_action_count": destructive_actions,
            "destructive_action_rate": round(
                destructive_actions / observed_calls * 100, 1
            )
            if observed_calls
            else None,
            "avg_turns_to_solve": round(
                sum(record.model_turns for record in solved_records)
                / len(solved_records),
                1,
            )
            if solved_records
            else None,
            "avg_tokens_to_solve": round(
                sum(
                    record.input_tokens + record.output_tokens
                    for record in solved_records
                )
                / len(solved_records),
                1,
            )
            if solved_records
            else None,
            "first_attempt_score": round(
                sum(record.first_attempt_score for record in scored_records)
                / scored_total
                * 100,
                1,
            )
            if scored_total
            else 0.0,
            "repair_delta_pp": round(
                score
                - sum(record.first_attempt_score for record in scored_records)
                / scored_total
                * 100,
                1,
            )
            if scored_total
            else 0.0,
            "repair_attempted": sum(
                record.repair_attempts_used > 0 for record in records
            ),
            "repair_turns": sum(record.repair_attempts_used for record in records),
            "repair_successes": sum(record.repaired for record in records),
            "repair_score_curve": repair_score_curve,
            **repair_round_columns,
            "sum_generation_time": round(throughput_eval_ns / 1e9, 3),
            "sum_request_time": round(throughput_response_time, 3),
            "avg_response_time": round(throughput_response_time / throughput_items, 1)
            if throughput_items
            else 0.0,
            "total_time": total_time,
            "slice": slice_spec,
            "concurrency": concurrency,
            "errors": errors,
            "harness_errors": harness_errors,
            "contaminated": len(contaminated_records),
            "contamination_by_language": {
                language: sum(
                    record.workspace.get("language") == language
                    for record in contaminated_records
                )
                for language in sorted(
                    {
                        str(record.workspace.get("language"))
                        for record in contaminated_records
                        if record.workspace.get("language")
                    }
                )
            },
            "contaminated_tasks": [
                {
                    "task_id": record.task_id,
                    "language": record.workspace.get("language", ""),
                    "guard_hits": record.workspace.get("answer_key_guard_hits", []),
                }
                for record in contaminated_records
            ],
            "length_exceeded": length_exceeded,
            "failures": failures,
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
            "loop_kill_enabled": self.loop_kill_enabled,
            "loop_kill_percent": self.loop_kill_percent,
            "loop_kill_seconds": self.loop_kill_seconds,
            "tasks": [
                {
                    "task_id": record.task_id,
                    "passed": record.passed,
                    "outcome": record.outcome,
                    "score": round(record.score * 100, 1),
                    "tok_s": record.tok_s,
                    "response_time_s": record.response_time_s,
                    "prompt": record.prompt,
                    "response": record.response,
                    "perturbation": record.perturbation or None,
                    "error": record.error,
                    "entry_point": record.entry_point,
                    "thinking": record.thinking,
                    "output_tokens": record.output_tokens,
                    "tokens_recovered": record.tokens_recovered,
                    "done_reason": record.done_reason,
                    "timed_out": record.timed_out,
                    "loop_killed": record.loop_killed,
                    "length_exceeded": record.length_exceeded,
                    "harness_error": record.harness_error,
                    "contaminated": record.contaminated,
                    "contamination_verdict": record.workspace.get(
                        "contamination_verdict", "CLEAR"
                    ),
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
                    "harness": record.harness,
                    "harness_version": record.harness_version,
                    "input_tokens": record.input_tokens,
                    "model_turns": record.model_turns,
                    "tool_calls": record.tool_calls,
                    "tool_trace": record.tool_trace,
                    "attempts": record.attempts,
                    "repair_attempts_used": record.repair_attempts_used,
                    "repair_feedback": record.repair_feedback,
                    "first_attempt_score": round(record.first_attempt_score * 100, 1),
                    "repaired": record.repaired,
                    "workspace": record.workspace or None,
                    "agentic_metrics": record.workspace.get("agentic_metrics") or None,
                }
                for record in records
            ],
            **pi_metadata,
            **_result_metadata(job),
        }
        task_statistics = getattr(bench, "task_statistics", None)
        if callable(task_statistics):
            result["task_statistics"] = task_statistics(records)
        return result, skipped

    def _verify_response(
        self,
        bench: object,
        task: Task,
        response: str,
    ) -> EvaluationResult:
        """Run evaluators serially and convert infrastructure errors to data."""
        with self._evaluation_lock:
            try:
                return evaluate_response(bench, task, response)
            except Exception as exc:
                return EvaluationResult(
                    score=0.0,
                    error=f"evaluation failed: {type(exc).__name__}: {exc}",
                )

    def _generate_direct_with_repairs(
        self,
        job: JobSpec,
        prompt: str,
        verifier: Callable[[str], EvaluationResult],
        on_progress: Callable[[GenerationUpdate], None],
    ) -> dict:
        """Run stateless direct calls with an explicit feedback transcript."""
        attempts: list[tuple[dict, EvaluationResult]] = []
        feedback_sent: list[str] = []
        request_prompt = prompt
        elapsed_offset = 0.0

        for attempt in range(job.repair_attempts + 1):

            def forward(
                update: GenerationUpdate,
                *,
                number: int = attempt,
                offset: float = elapsed_offset,
            ) -> None:
                on_progress(
                    replace(
                        update,
                        elapsed_s=offset + update.elapsed_s,
                        attempt=number,
                    )
                )

            gen = self.client.generate(
                job.model,
                request_prompt,
                on_progress=forward,
                cancel_event=self.controls.cancel_event,
            )
            terminal = bool(
                gen.get("cancelled") or gen.get("timed_out") or gen.get("loop_killed")
            )
            evaluation = (
                EvaluationResult(score=0.0)
                if terminal
                else verifier(str(gen.get("response") or ""))
            )
            attempts.append((gen, evaluation))
            if terminal or evaluation.passed or evaluation.error:
                break
            if attempt >= job.repair_attempts:
                break

            feedback_sent.append(evaluation.feedback)
            request_prompt = standalone_repair_prompt(
                prompt,
                str(gen.get("response") or ""),
                evaluation.feedback,
                attempt + 1,
                job.repair_attempts,
            )
            elapsed_offset += float(gen.get("response_time_s") or 0.0)

        return combine_attempts(attempts, feedback_sent)

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
        case = perturb_task(
            job.benchmark,
            task,
            job.perturbation,
            job.perturbation_seed,
        )
        task = case.evaluation_task
        prompt = prompt_for(bench, case.prompt_task, self.client, job.model)
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
        current_attempt = 0
        streamed_thinking_parts: list[str] = []
        streamed_response_parts: list[str] = []

        def on_progress(update: GenerationUpdate) -> None:
            nonlocal analyzer
            nonlocal current_attempt
            nonlocal first_answer_at
            nonlocal last_progress_emit
            nonlocal last_progress_state
            nonlocal loop_above_since
            nonlocal loop_detected_at
            nonlocal loop_killed_at_s
            nonlocal loop_kill_score
            nonlocal reasoning_channel_seen
            nonlocal streamed_response_parts
            nonlocal streamed_thinking_parts

            if self.controls.cancel_requested:
                raise _GenerationCancelled

            if update.attempt != current_attempt:
                current_attempt = update.attempt
                analyzer = LoopAnalyzer()
                reasoning_channel_seen = False
                loop_above_since = None
                loop_detected_at = None
                last_progress_emit = 0.0
                last_progress_state = ""
                streamed_thinking_parts = []
                streamed_response_parts = []
                self.emit(
                    TaskPhase(
                        index=index,
                        job=job,
                        position=position + 1,
                        total=total,
                        task_id=task.id,
                        entry_point=entry_point,
                        phase="generating",
                        activity=(
                            f"running verifier-feedback repair {current_attempt}"
                        ),
                    )
                )

            streamed_thinking_parts.append(update.thinking)
            streamed_response_parts.append(update.response)
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

        request_started = time.perf_counter()
        try:
            generator = self._pi(bench, task) if job.harness == "pi" else self.client
            workspace = bool(getattr(bench, "workspace_task", False))
            workspace_setup = None
            workspace_verifier = None
            repair_prompt_builder = None
            if workspace:

                def workspace_setup(environment):
                    return bench.prepare_workspace(task, environment)

                def workspace_verifier(environment, tool_trace):
                    return bench.verify_workspace(task, environment, tool_trace)

                candidate_builder = getattr(bench, "build_repair_prompt", None)
                if callable(candidate_builder):
                    repair_prompt_builder = candidate_builder

            if job.repair_attempts:

                def verifier(response: str) -> EvaluationResult:
                    return self._verify_response(bench, task, response)

                if job.harness == "pi":
                    gen = generator.generate(
                        job.model,
                        prompt,
                        on_progress=on_progress,
                        cancel_event=self.controls.cancel_event,
                        verifier=verifier,
                        repair_attempts=job.repair_attempts,
                        workspace_setup=workspace_setup,
                        workspace_verifier=workspace_verifier,
                        repair_prompt_builder=repair_prompt_builder,
                    )
                else:
                    gen = self._generate_direct_with_repairs(
                        job,
                        prompt,
                        verifier,
                        on_progress,
                    )
            else:
                gen = generator.generate(
                    job.model,
                    prompt,
                    on_progress=on_progress,
                    cancel_event=self.controls.cancel_event,
                    **(
                        {
                            "workspace_setup": workspace_setup,
                            "workspace_verifier": workspace_verifier,
                        }
                        if workspace
                        else {}
                    ),
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
                gen.update(
                    harness_error=True,
                    response_time_s=time.perf_counter() - request_started,
                )
            if not gen.get("cancelled"):
                errors += 1

        if (
            gen.get("loop_killed") or gen.get("timed_out") or gen.get("length_exceeded")
        ) and not int(gen.get("eval_count") or 0):
            token_count = self._count_streamed_tokens(
                job.model,
                streamed_thinking_parts,
                streamed_response_parts,
            )
            if token_count is not None:
                response_time_s = float(gen.get("response_time_s") or 0.0)
                gen.update(
                    eval_count=token_count,
                    tokens_recovered=True,
                    tok_s=(
                        token_count / response_time_s if response_time_s > 0 else 0.0
                    ),
                )

        return _GeneratedTask(
            position=position,
            task=task,
            prompt=prompt,
            perturbation=case.details,
            gen=gen,
            error=error,
            errors=errors,
            analyzer=analyzer,
            first_answer_at=first_answer_at,
            loop_detected_at=loop_detected_at,
        )

    def _count_streamed_tokens(
        self,
        model: str,
        thinking_parts: list[str],
        response_parts: list[str],
    ) -> int | None:
        """Count terminal stream text when the final usage payload never arrived."""
        tokenize = getattr(self.client, "tokenize", None)
        if not callable(tokenize):
            return None
        total = 0
        observed = False
        for content in ("".join(thinking_parts), "".join(response_parts)):
            if not content:
                continue
            try:
                tokens = tokenize(model, content, add_special=False)
            except (TypeError, ValueError):
                return None
            if tokens is None:
                return None
            total += len(tokens)
            observed = True
        return total if observed else None

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
        length_exceeded = bool(gen.get("length_exceeded"))
        harness_error = bool(gen.get("harness_error"))
        terminal_workspace_evaluation = bool(gen.get("workspace_evaluated")) and (
            "evaluation_score" in gen
        )
        evaluation_score = 0.0
        evaluation_details = dict(gen.get("evaluation_details") or {})
        contaminated = evaluation_details.get("contamination_verdict") == "CONTAMINATED"
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
        elif timed_out and not terminal_workspace_evaluation:
            ok = False
            if not error:
                timeout_s = float(gen.get("timeout_s") or self.client.timeout)
                error = f"generation timed out after {timeout_s:g}s"
                errors += 1
            phase = "timed_out"
            activity = "timed out — skipping task"
        elif length_exceeded and not terminal_workspace_evaluation:
            ok = False
            if not error:
                error = "generation reached the model output-token limit"
                errors += 1
            phase = "length_exceeded"
            activity = "output-token limit reached — scoring task as failed"
        elif harness_error:
            ok = False
            phase = "harness_error"
            activity = "harness error — excluding task from score"
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

        if (
            not loop_killed
            and (not timed_out or terminal_workspace_evaluation)
            and (not length_exceeded or terminal_workspace_evaluation)
            and not harness_error
        ):
            if "evaluation_score" in gen:
                evaluation_score = min(
                    1.0, max(0.0, float(gen.get("evaluation_score") or 0.0))
                )
                evaluation_error = str(gen.get("evaluation_error") or "")
                ok = evaluation_score >= 1.0 and not evaluation_error
                if evaluation_error and not error:
                    error = evaluation_error
                    errors += 1
                    harness_error = True
            else:
                evaluation = self._verify_response(bench, task, gen["response"])
                evaluation_score = evaluation.score
                ok = evaluation.passed
                # Verifiers that record artifacts (rendered pages, screenshots,
                # workspace state) return them here; the repair path carries
                # them on the generation instead.
                evaluation_details = dict(evaluation.details)
                if evaluation.error and not error:
                    error = evaluation.error
                    errors += 1
                    harness_error = True
            if not ok and not error and gen.get("evaluation_error"):
                ok = False
                error = str(gen["evaluation_error"])
                errors += 1
                harness_error = True

        if contaminated:
            ok = False
            evaluation_score = 0.0

        outcome = (
            "harness_error"
            if harness_error
            else "contaminated"
            if contaminated
            else "pass"
            if ok
            else "length_exceeded"
            if length_exceeded
            else "loop_killed"
            if loop_killed
            else "timeout"
            if timed_out
            else "fail"
        )

        response_time_s = float(gen.get("response_time_s", 0.0))
        record = TaskRecord(
            index=position,
            task_id=task.id,
            passed=ok,
            tok_s=round(float(gen.get("tok_s", 0.0)), 1),
            response_time_s=round(response_time_s, 2),
            prompt=prompt,
            response=gen.get("response", ""),
            outcome=outcome,
            perturbation=generated.perturbation,
            score=evaluation_score,
            error=error,
            entry_point=entry_point,
            thinking=gen.get("thinking", ""),
            output_tokens=int(gen.get("eval_count", 0)),
            tokens_recovered=bool(gen.get("tokens_recovered")),
            done_reason=gen.get("done_reason", ""),
            timed_out=timed_out,
            loop_killed=loop_killed,
            length_exceeded=length_exceeded,
            harness_error=harness_error,
            contaminated=contaminated,
            loop_kill_score=float(gen.get("loop_kill_score", 0.0)),
            loop_killed_at_s=gen.get("loop_killed_at_s"),
            trace_status=gen.get("trace_status", "unavailable"),
            thinking_time_s=round(
                min(
                    first_answer_at if first_answer_at is not None else response_time_s,
                    response_time_s,
                )
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
            harness=str(gen.get("harness") or job.harness),
            harness_version=str(gen.get("harness_version") or ""),
            input_tokens=int(gen.get("input_tokens") or 0),
            model_turns=int(gen.get("model_turns") or 1),
            tool_calls=int(gen.get("tool_calls") or 0),
            tool_trace=list(gen.get("tool_trace") or []),
            pi_scaffold=dict(gen.get("pi_scaffold") or {}),
            attempts=list(gen.get("attempts") or []),
            repair_attempts_used=int(gen.get("repair_attempts_used") or 0),
            repair_feedback=list(gen.get("repair_feedback") or []),
            first_attempt_score=float(gen.get("first_attempt_score", evaluation_score)),
            repaired=bool(gen.get("repaired")),
            workspace=evaluation_details,
        )
        return _TaskOutcome(
            record=record,
            errors=errors,
            eval_count=int(gen.get("eval_count", 0)),
            eval_duration_ns=int(gen.get("eval_duration_ns", 0)),
            response_time_s=response_time_s,
        )
