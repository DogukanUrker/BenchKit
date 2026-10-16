"""Events the engine emits while it runs.

Pure data shared by the TUI and the headless runner. ``JobSpec`` and
``TaskRecord`` are only needed for annotations, so this module imports nothing
at runtime.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from benchkit.jobs import JobSpec
    from benchkit.results import TaskRecord


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
