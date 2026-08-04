"""Headless run rendering for `benchkit --headless`.

The TUI is the primary interface; this module keeps a scriptable, log-friendly
path for CI and remote shells.

Two renderings share one event sink:

* An interactive terminal gets a live dashboard - the active job with its own
  bar and counters, the streaming task activity, the last few finished tasks
  and an overall bar with an ETA. Notable events (errors, timeouts, loop kills,
  finished jobs) are printed above the dashboard so they stay in scrollback.
* A pipe, a file or a CI log gets plain timestamped lines instead, because a
  redrawing dashboard is worthless once the escape codes are stripped.
"""

from __future__ import annotations

import signal
import time
from collections import deque
from dataclasses import dataclass, field
from types import FrameType

from rich.console import Console, ConsoleOptions, RenderResult
from rich.live import Live
from rich.text import Text

from benchkit.client import InferenceClient
from benchkit.engine import (
    Engine,
    GenerationProgress,
    JobCompleted,
    JobSpec,
    JobStarted,
    RunCompleted,
    RunControls,
    RunFailed,
    RunStarted,
    TaskCompleted,
    TaskPhase,
    TaskRecord,
    slice_label,
)

RECENT_TASKS = 5


def _fmt_time(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds >= 3600:
        return f"{seconds // 3600}h {seconds % 3600 // 60}m"
    if seconds >= 60:
        return f"{seconds // 60}m {seconds % 60}s"
    return f"{seconds}s"


def _plural(count: int, word: str) -> str:
    return f"{count:,} {word}{'' if count == 1 else 's'}"


def _score_style(score: float) -> str:
    if score >= 80:
        return "green"
    if score < 50:
        return "red"
    return "white"


@dataclass(frozen=True)
class _Glyphs:
    """Drawing characters, degraded for terminals without UTF-8."""

    spinner: tuple[str, ...]
    full: str
    partial: tuple[str, ...]
    empty: str
    passed: str
    failed: str
    error: str
    dot: str

    @classmethod
    def for_console(cls, console: Console) -> "_Glyphs":
        encoding = (console.encoding or "").lower().replace("-", "")
        if encoding.startswith("utf") and not console.legacy_windows:
            return cls(
                spinner=("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"),
                full="█",
                partial=("▏", "▎", "▍", "▌", "▋", "▊", "▉"),
                empty="░",
                passed="✓",
                failed="✗",
                error="!",
                dot="·",
            )
        return cls(
            spinner=("|", "/", "-", "\\"),
            full="#",
            partial=(),
            empty=".",
            passed="+",
            failed="x",
            error="!",
            dot="-",
        )

    def frame(self) -> str:
        return self.spinner[int(time.monotonic() * 10) % len(self.spinner)]


def _bar(
    fraction: float,
    width: int,
    glyphs: _Glyphs,
    *,
    style: str = "white",
    empty_style: str = "bright_black",
) -> Text:
    """Render a progress bar, using partial blocks for sub-cell precision."""
    width = max(4, width)
    exact = min(1.0, max(0.0, fraction)) * width
    filled = int(exact)
    bar = Text(glyphs.full * filled, style=style)
    remainder = exact - filled
    if glyphs.partial and filled < width and remainder > 1 / (len(glyphs.partial) + 1):
        index = min(len(glyphs.partial) - 1, int(remainder * len(glyphs.partial)))
        bar.append(glyphs.partial[index], style=style)
        filled += 1
    bar.append(glyphs.empty * (width - filled), style=empty_style)
    return bar


def _spread(left: Text, right: Text, width: int) -> Text:
    """Left/right justify two runs of text on one line."""
    room = max(0, width - right.cell_len - 1)
    if left.cell_len > room:
        left = left.copy()
        left.truncate(room, overflow="ellipsis")
    line = left.copy()
    line.append(" " * max(1, width - left.cell_len - right.cell_len))
    line.append_text(right)
    return line


@dataclass
class _LiveStats:
    """Outcome counters for one headless job."""

    passed: int = 0
    failed: int = 0
    errors: int = 0
    loops: int = 0

    def add(self, record: TaskRecord) -> None:
        if record.passed:
            self.passed += 1
        elif record.error:
            self.errors += 1
        else:
            self.failed += 1
        self.loops += int(record.loop_state == "looping")

    def fields(self, *, live_looping: bool = False) -> dict[str, int]:
        return {
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "loops": self.loops + int(live_looping),
        }

    @property
    def completed(self) -> int:
        return self.passed + self.failed + self.errors

    @property
    def score(self) -> float | None:
        return self.passed / self.completed * 100 if self.completed else None


def _score_text(score: float | None, width: int = 6) -> Text:
    """Right aligned live score, dimmed until the first task lands."""
    if score is None:
        return Text("--.-%".rjust(width), style="dim")
    return Text(f"{score:.1f}%".rjust(width), style=f"bold {_score_style(score)}")


def _counters(stats: _LiveStats, *, live_looping: bool = False) -> Text:
    """`PASS 9  FAIL 2  ERR 1  LOOP 0`, with zero counts dimmed."""
    fields = stats.fields(live_looping=live_looping)
    text = Text()
    for label, key, style in (
        ("PASS", "passed", "green"),
        ("FAIL", "failed", "red"),
        ("ERR", "errors", "yellow"),
        ("LOOP", "loops", "magenta"),
    ):
        value = fields[key]
        if text:
            text.append("  ")
        text.append(label, style=f"bold {style}" if value else "dim")
        text.append(f" {value}", style="" if value else "dim")
    return text


@dataclass
class _JobView:
    """Live state of the job the engine is currently running."""

    index: int
    job: JobSpec
    total: int
    completed: int = 0
    stats: _LiveStats = field(default_factory=_LiveStats)
    activity: Text = field(default_factory=Text)
    live_looping: bool = False


@dataclass
class _RunState:
    """Everything the dashboard needs, updated from the engine sink."""

    jobs: list[JobSpec] = field(default_factory=list)
    overall_total: int = 0
    overall_done: int = 0
    started: float = field(default_factory=time.monotonic)
    current: _JobView | None = None
    recent: deque[Text] = field(default_factory=lambda: deque(maxlen=RECENT_TASKS))
    stopping: bool = False

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    @property
    def eta(self) -> float | None:
        if not self.overall_done or self.overall_done >= self.overall_total:
            return None
        rate = self.elapsed / self.overall_done
        return rate * (self.overall_total - self.overall_done)


class _Dashboard:
    """Renderable redrawn by `rich.live.Live` on every refresh."""

    def __init__(self, state: _RunState, glyphs: _Glyphs) -> None:
        self.state = state
        self.glyphs = glyphs

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        width = max(40, options.max_width)
        state = self.state
        view = state.current

        yield Text()
        if view is not None:
            yield from self._job_lines(view, width)
            for line in state.recent:
                yield line
            if state.recent:
                yield Text()
        yield self._overall_line(width)

    def _job_lines(self, view: _JobView, width: int) -> RenderResult:
        glyphs = self.glyphs
        title = Text()
        title.append(view.job.benchmark, style="bold")
        title.append(f" {glyphs.dot} ", style="dim")
        title.append(view.job.model)
        meta = Text(
            f"job {view.index + 1}/{len(self.state.jobs)} {glyphs.dot} "
            f"{slice_label(view.job.slice_spec)}",
            style="dim",
        )
        yield _spread(title, meta, width)

        counters = _counters(view.stats, live_looping=view.live_looping)
        progress = Text(f"{view.completed}/{view.total}".rjust(9), style="dim")
        score = _score_text(view.stats.score)
        tail = Text.assemble(progress, "  ", score, "  ", counters)
        bar_width = max(8, width - tail.cell_len - 2)
        yield Text.assemble(
            _bar(view.completed / max(view.total, 1), bar_width, glyphs), "  ", tail
        )

        activity = Text()
        activity.append(glyphs.frame(), style="cyan")
        activity.append(" ")
        activity.append_text(view.activity)
        activity.truncate(width, overflow="ellipsis")
        yield activity
        yield Text()

    def _overall_line(self, width: int) -> Text:
        state = self.state
        glyphs = self.glyphs
        percent = state.overall_done / max(state.overall_total, 1)
        parts = [f"{state.overall_done}/{state.overall_total} tasks"]
        parts.append(f"{percent * 100:.0f}%")
        parts.append(_fmt_time(state.elapsed))
        eta = state.eta
        if eta is not None and not state.stopping:
            parts.append(f"~{_fmt_time(eta)} left")
        if state.stopping:
            parts.append("stopping")
        label = Text("overall  ", style="dim")
        tail = Text(f"  {f' {glyphs.dot} '.join(parts)}", style="dim")
        bar_width = max(8, width - label.cell_len - tail.cell_len)
        return Text.assemble(
            label,
            _bar(
                percent,
                bar_width,
                glyphs,
                style="bright_black",
                empty_style="bright_black",
            ),
            tail,
        )


class _Reporter:
    """Turns engine events into either a live dashboard or plain log lines."""

    def __init__(self, console: Console, *, verbose: bool) -> None:
        self.console = console
        self.verbose = verbose
        self.glyphs = _Glyphs.for_console(console)
        self.plain = not console.is_terminal
        self.state = _RunState()
        self._live: Live | None = None

    def __enter__(self) -> "_Reporter":
        if not self.plain:
            self._live = Live(
                _Dashboard(self.state, self.glyphs),
                console=self.console,
                refresh_per_second=10,
                transient=True,
            )
            self._live.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._live is not None:
            self._live.__exit__(*exc)
            self._live = None

    def log(self, renderable: object) -> None:
        """Print above the dashboard, where it stays in scrollback."""
        # Log lines are read by machines as often as by people, so a piped run
        # keeps one event on one line instead of wrapping at 80 columns.
        self.console.print(renderable, soft_wrap=self.plain)

    def _stamp(self) -> str:
        return time.strftime("%H:%M:%S")

    # -- events ---------------------------------------------------------

    def __call__(self, event: object) -> None:
        if isinstance(event, RunStarted):
            self.on_run_started(event)
        elif isinstance(event, JobStarted):
            self.on_job_started(event)
        elif isinstance(event, TaskPhase):
            self.on_task_phase(event)
        elif isinstance(event, GenerationProgress):
            self.on_generation(event)
        elif isinstance(event, TaskCompleted):
            self.on_task_completed(event)
        elif isinstance(event, JobCompleted):
            self.on_job_completed(event)
        elif isinstance(event, RunFailed):
            self.log(Text.assemble(("Run failed: ", "bold red"), event.message))
        elif isinstance(event, RunCompleted):
            self.on_run_completed(event)

    def on_run_started(self, event: RunStarted) -> None:
        self.state.jobs = list(event.jobs)
        self.state.overall_total = event.total_tasks
        self.state.started = time.monotonic()
        models = len({job.model for job in event.jobs})
        benchmarks = len({job.benchmark for job in event.jobs})
        self.log(
            Text.assemble(
                (_plural(len(event.jobs), "job"), "bold"),
                (
                    f" {self.glyphs.dot} {_plural(models, 'model')}"
                    f" {self.glyphs.dot} {_plural(benchmarks, 'benchmark')}"
                    f" {self.glyphs.dot} {_plural(event.total_tasks, 'task')}",
                    "dim",
                ),
            )
        )

    def on_job_started(self, event: JobStarted) -> None:
        self.state.current = _JobView(
            index=event.index, job=event.job, total=max(event.total, 1)
        )
        self.state.recent.clear()
        if self.plain:
            self.log(
                Text(
                    f"[{self._stamp()}] job {event.index + 1}/{len(self.state.jobs)}"
                    f" {event.job.benchmark} on {event.job.model}"
                    f" {self.glyphs.dot} {event.total} tasks"
                    f" {self.glyphs.dot} {slice_label(event.job.slice_spec)}",
                    style="dim",
                )
            )

    def _activity(self, label: str, detail: Text) -> None:
        view = self.state.current
        if view is None:
            return
        text = Text()
        text.append(label, style="bold")
        text.append(f" {self.glyphs.dot} ", style="dim")
        text.append_text(detail)
        view.activity = text

    def on_task_phase(self, event: TaskPhase) -> None:
        view = self.state.current
        if view is None:
            return
        view.live_looping = False
        style = "yellow" if event.phase in {"timed_out", "loop_killed"} else "dim"
        self._activity(event.label, Text(event.activity, style=style))

    def on_generation(self, event: GenerationProgress) -> None:
        view = self.state.current
        if view is None:
            return
        view.live_looping = event.loop_state == "looping"
        detail = Text(event.phase, style="dim")
        detail.append(
            f" {self.glyphs.dot} "
            f"{event.thinking_chars + event.response_chars:,} chars"
            f" {self.glyphs.dot} {event.elapsed_s:.0f}s",
            style="dim",
        )
        if event.loop_state == "looping":
            detail.append(f" {self.glyphs.dot} ", style="dim")
            detail.append("LOOPING", style="bold magenta")
        elif event.loop_state == "suspected":
            detail.append(f" {self.glyphs.dot} loop suspected", style="magenta")
        if event.loop_kill_remaining_s is not None:
            detail.append(
                f" {self.glyphs.dot} kill in {event.loop_kill_remaining_s:.1f}s",
                style="bold red",
            )
        self._activity(event.label, detail)

    def _task_line(self, record: TaskRecord, position: int, total: int) -> Text:
        glyphs = self.glyphs
        if record.passed:
            mark, style, word = glyphs.passed, "green", "PASS"
        elif record.loop_killed:
            mark, style, word = glyphs.error, "magenta", "LOOP KILLED"
        elif record.timed_out:
            mark, style, word = glyphs.error, "yellow", "TIMEOUT"
        elif record.error:
            mark, style, word = glyphs.error, "yellow", "ERROR"
        else:
            mark, style, word = glyphs.failed, "red", "FAIL"

        line = Text("  ")
        line.append(mark, style=style)
        line.append(" ")
        line.append(f"{position}/{total}".rjust(len(str(total)) * 2 + 1), style="dim")
        line.append("  ")
        line.append(word.ljust(11), style=f"bold {style}")
        line.append(record.label)
        line.append(
            f"  {record.response_time_s:.1f}s {glyphs.dot} {record.tok_s:.0f} tok/s",
            style="dim",
        )
        if record.recovered_cycle:
            line.append(f" {glyphs.dot} loop recovered", style="yellow")
        elif record.loop_state == "looping" and not record.loop_killed:
            line.append(f" {glyphs.dot} looped", style="magenta")
        if record.error:
            line.append(f"  {record.error}", style="dim")
        return line

    def on_task_completed(self, event: TaskCompleted) -> None:
        view = self.state.current
        record = event.record
        self.state.overall_done += 1
        if view is not None:
            view.stats.add(record)
            view.completed = event.completed
            view.live_looping = False
        total = view.total if view is not None else event.completed
        line = self._task_line(record, event.completed, total)
        if self.plain:
            stamped = Text()
            stamped.append(f"[{self._stamp()}]", style="dim")
            self.log(stamped.append_text(line))
        else:
            self.state.recent.append(line)
            # Anything that is not an ordinary miss is worth keeping in
            # scrollback, since the dashboard window scrolls it away.
            if record.error or record.timed_out or record.loop_killed:
                self.log(line)
        if self.verbose:
            self._verbose_task(record)

    def _verbose_task(self, record: TaskRecord) -> None:
        console = self.console
        console.print()
        console.print(Text(record.label, style="bold"))
        console.print(Text("Prompt", style="dim"))
        console.print(record.prompt, markup=False, highlight=False)
        if record.thinking:
            console.print(Text("Thinking", style="dim"))
            console.print(record.thinking, markup=False, highlight=False)
        console.print(Text("Response", style="dim"))
        console.print(record.response, markup=False, highlight=False)
        console.print()

    def on_job_completed(self, event: JobCompleted) -> None:
        result = event.result
        view = self.state.current
        self.state.current = None
        glyphs = self.glyphs
        stamp = f"[{self._stamp()}] " if self.plain else ""

        if not result["total"]:
            self.log(
                Text.assemble(
                    (f"{stamp}{glyphs.error} ", "yellow"),
                    ("skipped".ljust(8), "bold yellow"),
                    (f"{result['benchmark']} on {result['model']}", "dim"),
                )
            )
            return

        stats = view.stats if view is not None else _LiveStats()
        partial = self.state.stopping or result["total"] < (
            view.total if view is not None else result["total"]
        )
        line = Text()
        line.append(
            f"{stamp}{glyphs.error if partial else glyphs.passed} ", style="dim"
        )
        line.append(result["benchmark"], style="bold")
        line.append(f" {glyphs.dot} ", style="dim")
        line.append(result["model"])
        line.append("  ")
        line.append_text(_score_text(result["score"]))
        line.append("  ")
        line.append_text(_counters(stats))
        line.append(
            f"  {_plural(result['total'], 'task')}"
            f"{' (partial)' if partial else ''}"
            f" {glyphs.dot} {result['tok_s']:.1f} tok/s"
            f" {glyphs.dot} {result.get('loop_kills', 0)} killed"
            f" {glyphs.dot} {_fmt_time(result['total_time'])}",
            style="dim",
        )
        self.log(line)

    def on_run_completed(self, event: RunCompleted) -> None:
        if not event.results:
            return
        glyphs = self.glyphs
        word = "Stopped" if event.stopped else "Finished"
        self.log(
            Text(
                f"{word} {_plural(len(event.results), 'job')}"
                f" {glyphs.dot} {_plural(self.state.overall_done, 'task')}"
                f" {glyphs.dot} {_fmt_time(event.elapsed)}",
                style="dim",
            )
        )


def run(
    client: InferenceClient,
    jobs: list[JobSpec],
    console: Console,
    verbose: bool = False,
) -> tuple[list[dict], str | None]:
    """Run every job, printing progress to `console`.

    Returns the results plus the failure message when the run ended early.
    Ctrl+C asks the engine to stop after the current task, so a partial run
    still reports and still writes its artifacts; a second Ctrl+C aborts.
    """
    controls = RunControls()
    reporter = _Reporter(console, verbose=verbose)
    engine = Engine(client=client, jobs=jobs, sink=reporter, controls=controls)

    def on_interrupt(signum: int, frame: FrameType | None) -> None:
        if controls.stopped:
            raise KeyboardInterrupt
        reporter.state.stopping = True
        controls.stop()
        reporter.log(
            Text("Interrupted, stopping after the current task…", style="bold yellow")
        )

    previous = None
    try:
        previous = signal.signal(signal.SIGINT, on_interrupt)
    except ValueError:  # not on the main thread
        pass

    try:
        with reporter:
            results = engine.run()
    except KeyboardInterrupt:
        console.print(Text("Aborted.", style="bold yellow"))
        return [], "interrupted"
    finally:
        if previous is not None:
            signal.signal(signal.SIGINT, previous)

    return results, engine.failure
