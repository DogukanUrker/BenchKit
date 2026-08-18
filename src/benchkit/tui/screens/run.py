"""Live run screen: progress, streaming task results and run controls."""

from __future__ import annotations

import time
from pathlib import Path

from rich.text import Text
from textual import events, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.message import Message
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, ProgressBar, Static

from benchkit.engine import (
    Engine,
    GenerationProgress,
    JobCompleted,
    JobSpec,
    JobStarted,
    ModelUnloaded,
    RunCompleted,
    RunControls,
    RunFailed,
    TaskCompleted,
    TaskPhase,
    TaskRecord,
    benchmark,
    plan_total_tasks,
)
from benchkit.report import save
from benchkit.tui.formatting import (
    bar,
    fmt_count,
    fmt_duration,
    result_color,
    score_color,
    throughput_stat,
)
from benchkit.tui.screens.modals import TaskDetailScreen
from benchkit.tui.widgets import SectionTitle, StatCard, apply_compact


class EngineMessage(Message):
    """An engine event forwarded from the worker thread."""

    def __init__(self, event: object) -> None:
        super().__init__()
        self.event = event


def _variant_cap_note(jobs: list[JobSpec]) -> str:
    """Describe context variants omitted by served-context discovery."""
    groups: dict[tuple[str, str], list[str]] = {}
    for job in jobs:
        if job.variant is not None:
            groups.setdefault((job.model, job.benchmark), []).append(job.variant)

    notes: list[str] = []
    for (model, benchmark_key), variants in groups.items():
        expected = tuple(
            str(value)
            for value in getattr(benchmark(benchmark_key), "variant_labels", ())
        )
        if not expected or set(expected).issubset(variants):
            continue
        highest = next(
            (value for value in reversed(expected) if value in variants),
            "none",
        )
        notes.append(f"{model} {benchmark_key.upper()} ≤{highest}")
    return ", ".join(notes)


def _include_in_overall(job: JobSpec) -> bool:
    """Match the engine/report rule for model-level score aggregation."""
    return job.perturbation is None and getattr(
        benchmark(job.benchmark), "include_in_overall", True
    )


class RunScreen(Screen[None]):
    """Everything that happens while benchmarks are running."""

    BINDINGS = [
        Binding("p", "pause", "Pause"),
        Binding("k", "skip", "Skip job"),
        Binding("x", "stop", "Stop run"),
        Binding("u", "toggle_unload", "Force unload"),
        Binding("f", "toggle_failures", "Failures only"),
        Binding("t", "toggle_follow", "Follow"),
        Binding("escape", "stop", "Stop run", show=False),
    ]

    def __init__(self, jobs: list[JobSpec], *, force_unload: bool = False) -> None:
        super().__init__()
        self.jobs = jobs
        self.force_unload = force_unload
        self.controls = RunControls(force_unload=force_unload)
        self.records: dict[int, list[TaskRecord]] = {}
        self.current_index = -1
        self.current_total = 0
        self.current_passed = 0
        self.current_score_points = 0.0
        self.current_completed = 0
        self.current_scored_total = 0
        self.latency_sum = 0.0
        self.tok_s_sum = 0.0
        self.output_tokens_sum = 0
        self.job_started_at = time.monotonic()
        self.overall_total = 0
        self.overall_completed = 0
        self.overall_score_points = 0.0
        self.overall_scored_tasks = 0
        self.started_at = time.monotonic()
        self.only_failures = False
        self.follow = True
        self.finished = False
        self.current_loops = 0
        self.current_suspected = 0
        self.current_recovered = 0
        self.current_concurrency = 1
        self.completed_generated_chars = 0
        self.live_progress: dict[str, GenerationProgress] = {}
        self.live_tasks: dict[str, TaskPhase] = {}
        self.generating_tasks: set[str] = set()
        self.live_rows: set[str] = set()
        self.live_modal: TaskDetailScreen | None = None
        self.live_modal_key: str | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="run-root"):
            with Horizontal(id="run-head"):
                yield Static("Starting…", id="run-title")
                yield Static("QUEUED", id="run-state")
            with Vertical(id="progress-block"):
                yield Static("", id="job-progress-label")
                yield ProgressBar(total=100, show_eta=False, id="job-progress")
                yield Static("", id="overall-progress-label")
                yield ProgressBar(total=100, show_eta=False, id="overall-progress")
            with Horizontal(id="stat-row"):
                yield StatCard("Accuracy", "--", id="stat-accuracy")
                yield StatCard("Passed", "0", id="stat-passed")
                yield StatCard("Failed", "0", id="stat-failed")
                yield StatCard("Loops", "0", id="stat-loops")
                yield StatCard("Live", "--", id="stat-live")
                yield StatCard("Throughput", "--", id="stat-speed")
                yield StatCard("Elapsed", "0s", id="stat-elapsed")
            with Horizontal(id="run-body"):
                with Vertical(classes="pane", id="queue-pane"):
                    yield SectionTitle("Run queue", id="queue-title")
                    yield DataTable(id="queue", cursor_type="row", zebra_stripes=True)
                with Vertical(classes="pane", id="task-pane"):
                    yield SectionTitle("Tasks", "newest last", id="task-title")
                    yield DataTable(id="tasks", cursor_type="row", zebra_stripes=True)
        yield Footer()

    def on_mount(self) -> None:
        self.overall_total = plan_total_tasks(self.jobs)
        subtitle = f"{len(self.jobs)} run(s) · {fmt_count(self.overall_total)} tasks"
        if cap_note := _variant_cap_note(self.jobs):
            subtitle += f" · served cap: {cap_note}"
        if self.force_unload:
            subtitle += " · force unload"
        self.sub_title = subtitle

        queue = self.query_one("#queue", DataTable)
        queue.add_column("#", key="index", width=3)
        queue.add_column("Model", key="model")
        queue.add_column("Harness", key="harness", width=18)
        queue.add_column("Benchmark", key="benchmark")
        queue.add_column("Progress", key="progress", width=10)
        queue.add_column("Score", key="score", width=7)
        queue.add_column("Status", key="status", width=8)
        for index, job in enumerate(self.jobs):
            queue.add_row(
                str(index + 1),
                job.model,
                job.harness_label,
                job.benchmark_label,
                Text("—", style="dim"),
                Text("—", style="dim"),
                Text("queued", style="dim"),
                key=str(index),
            )

        tasks = self.query_one("#tasks", DataTable)
        tasks.add_column("#", key="index", width=4)
        tasks.add_column("Task", key="task", width=28)
        tasks.add_column("Result", key="result", width=9)
        tasks.add_column("Loop", key="loop", width=9)
        tasks.add_column("Latency", key="latency", width=8)
        tasks.add_column("Stream tok/s", key="tok_s", width=12)

        self.query_one("#overall-progress", ProgressBar).update(
            total=max(self.overall_total, 1), progress=0
        )
        self._set_state("RUNNING", "running")
        self.set_interval(0.5, self._tick)
        self._start_engine()

    @property
    def dark(self) -> bool:
        return self.app.current_theme.dark

    def on_resize(self, event: events.Resize) -> None:
        apply_compact(self, event.size.height)

    def on_unmount(self) -> None:
        self.controls.stop()

    # Engine -----------------------------------------------------------

    @work(thread=True, exclusive=True, group="engine")
    def _start_engine(self) -> None:
        engine = Engine(
            client=self.app.client,
            jobs=self.jobs,
            sink=lambda event: self.post_message(EngineMessage(event)),
            controls=self.controls,
        )
        try:
            results = engine.run()
        except Exception as exc:
            self.app.call_from_thread(
                self.notify, f"Run failed: {exc}", severity="error", timeout=10
            )
            self.app.call_from_thread(self._finished, [], None)
            return

        output: Path | None = None
        if results:
            self.app.call_from_thread(self._saving)
            try:
                output = save(
                    results,
                    provider=getattr(self.app.client, "label", ""),
                    host=getattr(self.app.client, "host", ""),
                )
            except Exception as exc:
                self.app.call_from_thread(
                    self.notify, f"Could not save reports: {exc}", severity="error"
                )
        self.app.call_from_thread(self._finished, results, output)

    def on_engine_message(self, message: EngineMessage) -> None:
        event = message.event
        if isinstance(event, JobStarted):
            self._job_started(event)
        elif isinstance(event, TaskPhase):
            self._task_phase(event)
        elif isinstance(event, GenerationProgress):
            self._generation_progress(event)
        elif isinstance(event, TaskCompleted):
            self._task_completed(event)
        elif isinstance(event, JobCompleted):
            self._job_completed(event)
        elif isinstance(event, RunCompleted):
            self._run_completed(event)
        elif isinstance(event, ModelUnloaded):
            if event.error:
                self.notify(
                    f"Force-unload failed for {event.model}: {event.error}",
                    severity="error",
                    timeout=6,
                )
            else:
                self.notify(f"Force-unloading {event.model}", timeout=3)
        elif isinstance(event, RunFailed):
            self.notify(event.message, severity="error", timeout=10)

    def _job_started(self, event: JobStarted) -> None:
        self.current_index = event.index
        self.current_total = event.total
        self.current_passed = 0
        self.current_score_points = 0.0
        self.current_completed = 0
        self.current_scored_total = 0
        self.latency_sum = 0.0
        self.tok_s_sum = 0.0
        self.output_tokens_sum = 0
        self.job_started_at = time.monotonic()
        self.current_loops = 0
        self.current_suspected = 0
        self.current_recovered = 0
        self.current_concurrency = event.concurrency
        self.completed_generated_chars = 0
        self.live_progress.clear()
        self.live_tasks.clear()
        self.generating_tasks.clear()
        self.live_rows.clear()
        self.records[event.index] = []

        job = event.job
        title = Content.from_markup(
            "[b]$bench[/b]  [dim]on[/dim]  $model  [cyan]$harness[/cyan]"
            "[dim]$slice$parallel[/dim]",
            bench=job.benchmark_label,
            model=job.model,
            harness=job.harness_label,
            slice=f"   slice {job.slice_spec}" if job.slice_spec else "",
            parallel=f"   · {event.concurrency} concurrent",
        )
        self.query_one("#run-title", Static).update(title)

        self.query_one("#job-progress", ProgressBar).update(
            total=max(event.total, 1), progress=0
        )
        self.query_one("#tasks", DataTable).clear()
        self._update_queue_row(event.index, status="running", style="bold")
        self._update_labels()

    @staticmethod
    def _live_key(event: TaskPhase | GenerationProgress) -> str:
        return f"{event.index}:{event.position - 1}"

    def _generated_chars(self) -> int:
        """Monotonic job total across finished and currently streaming tasks."""
        return self.completed_generated_chars + sum(
            progress.thinking_chars + progress.response_chars
            for progress in self.live_progress.values()
        )

    def _update_live_summary(self) -> None:
        """Render aggregate concurrency without jumping between task names."""
        if self.current_index < 0:
            return
        active = len(self.generating_tasks)
        generated_chars = self._generated_chars()
        self.query_one("#job-progress-label", Static).update(
            f"Job {self.current_index + 1}/{len(self.jobs)} · "
            f"{fmt_count(self.current_completed)}/{fmt_count(self.current_total)} "
            f"complete · {fmt_count(active)} active · "
            f"{fmt_count(generated_chars)} chars generated"
        )
        live_value = (
            f"{active} ACTIVE"
            if active
            else "FINALIZING"
            if self.live_tasks
            else "DONE"
            if self.current_completed
            else "STARTING"
        )
        self.query_one("#stat-live", StatCard).set_state(
            live_value,
            f"{fmt_count(generated_chars)} chars generated this job",
        )

    def _update_live_state(self) -> None:
        """Summarize all live requests instead of displaying the last event."""
        if self.controls.stopped:
            self._set_state("STOPPED", "done")
            return
        if self.controls.skip_requested:
            self._set_state("REQUESTS CANCELLED", "paused")
            return
        if self.controls.paused:
            self._set_state("PAUSED", "paused")
            return

        progress = list(self.live_progress.values())
        active_progress = [
            self.live_progress[key]
            for key in self.generating_tasks
            if key in self.live_progress
        ]
        kill_remaining = [
            item.loop_kill_remaining_s
            for item in progress
            if item.loop_kill_remaining_s is not None
        ]
        looping = sum(item.loop_state == "looping" for item in progress)
        suspected = sum(item.loop_state == "suspected" for item in progress)
        if kill_remaining:
            self._set_state(
                f"LOOPING · KILL IN {min(kill_remaining):.1f}s",
                "looping",
            )
        elif looping:
            self._set_state(
                f"{fmt_count(looping)} LOOPING",
                "looping",
            )
        elif suspected:
            self._set_state(
                f"{fmt_count(suspected)} LOOP SUSPECTED",
                "paused",
            )
        elif self.generating_tasks:
            if self.current_concurrency > 1:
                self._set_state(
                    f"{fmt_count(len(self.generating_tasks))} ACTIVE",
                    "running",
                )
            else:
                phase = active_progress[0].phase if active_progress else "waiting"
                self._set_state(
                    {
                        "thinking": "THINKING",
                        "answering": "ANSWERING",
                    }.get(phase, "WAITING FOR MODEL"),
                    "running",
                )
        elif self.live_tasks:
            self._set_state("EVALUATING", "evaluating")
        else:
            self._set_state("RUNNING", "running")

    def _add_live_row(self, event: TaskPhase) -> None:
        key = self._live_key(event)
        if key in self.live_rows:
            return
        table = self.query_one("#tasks", DataTable)
        table.add_row(
            str(event.position),
            event.label,
            Text("WAIT", style="dim"),
            Text("observing", style="dim"),
            "0s",
            "—",
            key=key,
        )
        self.live_rows.add(key)
        if self.follow:
            table.scroll_end(animate=False)

    def _task_phase(self, event: TaskPhase) -> None:
        key = self._live_key(event)
        if event.phase == "generating":
            self.live_tasks[key] = event
            self.generating_tasks.add(key)
            self._add_live_row(event)
        elif event.phase == "evaluating":
            self.generating_tasks.discard(key)
            if key in self.live_rows:
                self.query_one("#tasks", DataTable).update_cell(
                    key, "result", Text("EVAL", style="bold")
                )
        elif event.phase == "loop_killed":
            self.generating_tasks.discard(key)
        else:
            self.generating_tasks.discard(key)

        self._update_live_state()
        self._update_live_summary()

    def _generation_progress(self, event: GenerationProgress) -> None:
        key = self._live_key(event)
        self.live_progress[key] = event
        self.generating_tasks.add(key)
        phase = self.live_tasks.get(key)
        if phase is not None:
            self._add_live_row(phase)
        self._update_live_state()
        self._update_live_summary()
        live_loops = self.current_loops + sum(
            progress.loop_state == "looping" for progress in self.live_progress.values()
        )
        live_suspected = self.current_suspected + sum(
            progress.loop_state == "suspected"
            for progress in self.live_progress.values()
        )
        kill_remaining = [
            progress.loop_kill_remaining_s
            for progress in self.live_progress.values()
            if progress.loop_kill_remaining_s is not None
        ]
        self.query_one("#stat-loops", StatCard).set_state(
            str(live_loops),
            (
                f"kill in {min(kill_remaining):.1f}s"
                if kill_remaining
                else f"{live_suspected} suspect · {self.current_recovered} recovered"
            ),
        )

        if key in self.live_rows:
            table = self.query_one("#tasks", DataTable)
            table.update_cell(
                key,
                "result",
                Text(event.phase[:7].upper(), style="bold"),
            )
            table.update_cell(
                key,
                "loop",
                _loop_cell(event.loop_state, event.loop_source),
            )
            table.update_cell(
                key,
                "latency",
                fmt_duration(event.elapsed_s),
            )

        if (
            self.live_modal is not None
            and self.live_modal.is_mounted
            and self.live_modal_key == key
        ):
            self.live_modal.update_live(_progress_task_data(event))

    def _task_completed(self, event: TaskCompleted) -> None:
        record = event.record
        key = f"{event.index}:{record.index}"
        self.records.setdefault(event.index, []).append(record)
        self.current_completed = event.completed
        self.current_passed = event.passed
        self.current_score_points = (
            event.score_points
            if event.score_points is not None
            else float(event.passed)
        )
        self.current_scored_total = (
            event.scored_total
            if event.scored_total is not None
            else event.completed - int(record.harness_error)
        )
        self.overall_completed += 1
        if _include_in_overall(event.job) and not record.harness_error:
            self.overall_score_points += (
                record.score if record.score > 0 or not record.passed else 1.0
            )
            self.overall_scored_tasks += 1
        self.latency_sum += record.response_time_s
        self.tok_s_sum += record.tok_s
        self.output_tokens_sum += record.output_tokens
        self.current_loops += int(record.loop_state == "looping")
        self.current_suspected += int(
            record.loop_state == "suspected" and not record.recovered_cycle
        )
        self.current_recovered += int(record.recovered_cycle)

        self._add_task_row(event.index, record)
        last_progress = self.live_progress.pop(key, None)
        streamed_chars = (
            last_progress.thinking_chars + last_progress.response_chars
            if last_progress is not None
            else 0
        )
        self.completed_generated_chars += max(
            streamed_chars,
            len(record.thinking) + len(record.response),
        )
        self.live_tasks.pop(key, None)
        self.generating_tasks.discard(key)
        self.live_rows.discard(key)
        live_loops = sum(
            progress.loop_state == "looping" for progress in self.live_progress.values()
        )
        live_suspected = sum(
            progress.loop_state == "suspected"
            for progress in self.live_progress.values()
        )
        self.query_one("#stat-loops", StatCard).set_state(
            str(self.current_loops + live_loops),
            f"{self.current_suspected + live_suspected} suspect · "
            f"{self.current_recovered} recovered",
        )
        if (
            self.live_modal is not None
            and self.live_modal.is_mounted
            and self.live_modal_key == key
        ):
            self.live_modal.update_live(_record_task_data(record), finished=True)
            self.live_modal = None
            self.live_modal_key = None
        self.query_one("#job-progress", ProgressBar).update(progress=event.completed)
        self.query_one("#overall-progress", ProgressBar).update(
            progress=self.overall_completed
        )
        self._update_live_state()
        score = (
            self.current_score_points / self.current_scored_total * 100
            if self.current_scored_total
            else 0.0
        )
        self._update_queue_row(
            event.index,
            progress=bar(event.completed / max(self.current_total, 1), 10),
            score=Text(f"{score:.1f}%", style=f"bold {score_color(score, self.dark)}"),
        )
        self._update_labels()
        self._update_stats()

    def _job_completed(self, event: JobCompleted) -> None:
        result = event.result
        if not result["total"]:
            self._update_queue_row(
                event.index,
                progress=bar(0.0, 10),
                score=Text("—", style="dim"),
                status="skipped",
                style="dim",
            )
            return

        score = result["score"]
        self._update_queue_row(
            event.index,
            progress=bar(1.0, 10),
            score=Text(f"{score:.1f}%", style=f"bold {score_color(score, self.dark)}"),
            status="skipped" if event.skipped else "done",
            style="dim" if event.skipped else "",
        )

    def _run_completed(self, event: RunCompleted) -> None:
        self._set_state("STOPPED" if event.stopped else "COMPLETE", "done")

    def _saving(self) -> None:
        self._set_state("SAVING", "done")

    def _finished(self, results: list[dict], output: Path | None) -> None:
        self.finished = True
        if not results:
            self.notify("Run stopped before any task finished.", timeout=4)
            self.app.pop_screen()
            return
        self.app.show_results(results, output)

    # Table helpers ----------------------------------------------------

    def _add_task_row(self, job_index: int, record: TaskRecord) -> None:
        table = self.query_one("#tasks", DataTable)
        key = f"{job_index}:{record.index}"
        if self.only_failures and record.passed:
            if key in self.live_rows:
                table.remove_row(key)
            return
        if key in self.live_rows:
            table.update_cell(key, "result", _result_cell(record, self.dark))
            table.update_cell(
                key,
                "loop",
                _loop_cell(
                    record.loop_state,
                    record.loop_source,
                    recovered=record.recovered_cycle,
                ),
            )
            table.update_cell(key, "latency", fmt_duration(record.response_time_s))
            table.update_cell(
                key,
                "tok_s",
                f"{record.tok_s:.1f}" if record.tok_s else "—",
            )
        else:
            table.add_row(
                str(record.index + 1),
                record.label,
                _result_cell(record, self.dark),
                _loop_cell(
                    record.loop_state,
                    record.loop_source,
                    recovered=record.recovered_cycle,
                ),
                fmt_duration(record.response_time_s),
                f"{record.tok_s:.1f}" if record.tok_s else "—",
                key=key,
            )
        if self.follow:
            table.scroll_end(animate=False)

    def _rebuild_tasks(self) -> None:
        table = self.query_one("#tasks", DataTable)
        table.clear()
        self.live_rows.clear()
        for record in sorted(
            self.records.get(self.current_index, []), key=lambda item: item.index
        ):
            if self.only_failures and record.passed:
                continue
            table.add_row(
                str(record.index + 1),
                record.label,
                _result_cell(record, self.dark),
                _loop_cell(
                    record.loop_state,
                    record.loop_source,
                    recovered=record.recovered_cycle,
                ),
                fmt_duration(record.response_time_s),
                f"{record.tok_s:.1f}" if record.tok_s else "—",
                key=f"{self.current_index}:{record.index}",
            )
        for key, phase in sorted(
            self.live_tasks.items(), key=lambda item: item[1].position
        ):
            self._add_live_row(phase)
            progress = self.live_progress.get(key)
            if progress is None:
                continue
            table.update_cell(
                key,
                "result",
                Text(progress.phase[:7].upper(), style="bold"),
            )
            table.update_cell(
                key,
                "loop",
                _loop_cell(progress.loop_state, progress.loop_source),
            )
            table.update_cell(key, "latency", fmt_duration(progress.elapsed_s))
        if self.follow:
            table.scroll_end(animate=False)

    def _update_queue_row(
        self,
        index: int,
        *,
        progress: str | None = None,
        score: Text | None = None,
        status: str | None = None,
        style: str = "",
    ) -> None:
        queue = self.query_one("#queue", DataTable)
        key = str(index)
        if progress is not None:
            queue.update_cell(key, "progress", Text(progress, style="dim"))
        if score is not None:
            queue.update_cell(key, "score", score)
        if status is not None:
            queue.update_cell(key, "status", Text(status, style=style or ""))

    def _update_labels(self) -> None:
        overall_label = self.query_one("#overall-progress-label", Static)
        self._update_live_summary()
        overall_label.update(
            f"Overall · {fmt_count(self.overall_completed)}/"
            f"{fmt_count(self.overall_total)} tasks"
        )

    def _update_stats(self) -> None:
        completed = max(self.current_completed, 1)
        scored = max(self.current_scored_total, 1)
        measured_elapsed = max(time.monotonic() - self.job_started_at, 0.001)
        elapsed = (
            max(self.latency_sum / self.current_concurrency, 0.001)
            if getattr(self.app.client, "simulated_timing", False)
            and self.latency_sum > 0
            else measured_elapsed
        )
        score = self.current_score_points / scored * 100
        failed = self.current_completed - self.current_passed
        accuracy = self.query_one("#stat-accuracy", StatCard)
        current_job = self.jobs[self.current_index]
        include_current = _include_in_overall(current_job)
        if not include_current:
            overall_hint = (
                "paired perturbation · not averaged"
                if current_job.perturbation
                else "per-context · not averaged"
            )
        elif self.overall_scored_tasks:
            overall = self.overall_score_points / self.overall_scored_tasks * 100
            overall_hint = f"{overall:.1f}% overall"
        else:
            overall_hint = "— overall"
        accuracy.set_state(f"{score:.1f}%", overall_hint)
        self.query_one("#stat-passed", StatCard).set_state(str(self.current_passed))
        self.query_one("#stat-failed", StatCard).set_state(str(failed))
        speed = self.query_one("#stat-speed", StatCard)
        stream_speed = self.tok_s_sum / completed
        speed.set_state(
            *throughput_stat(
                concurrency=self.current_concurrency,
                aggregate=self.output_tokens_sum / elapsed,
                stream=stream_speed,
                effective=min(
                    self.current_concurrency,
                    self.latency_sum / elapsed,
                ),
                precision=0,
            )
        )

    def _tick(self) -> None:
        elapsed = time.monotonic() - self.started_at
        self.query_one("#stat-elapsed", StatCard).set_state(fmt_duration(elapsed))

    def _set_state(self, label: str, css_class: str) -> None:
        state = self.query_one("#run-state", Static)
        state.set_classes(f"state {css_class}")
        state.update(label)

    # Events -----------------------------------------------------------

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id != "tasks":
            return
        selected_key = str(event.row_key.value)
        progress = self.live_progress.get(selected_key)
        if progress is not None:
            modal = TaskDetailScreen(
                progress.job.title,
                _progress_task_data(progress),
                live=True,
            )
            self.live_modal = modal
            self.live_modal_key = selected_key
            self.app.push_screen(modal, self._live_modal_closed)
            return
        job_index, _, position = selected_key.partition(":")
        records = self.records.get(int(job_index), [])
        record = next((r for r in records if r.index == int(position)), None)
        if record is None:
            return
        job = self.jobs[int(job_index)]
        self.app.push_screen(TaskDetailScreen(job.title, _record_task_data(record)))

    def _live_modal_closed(self, _result: None) -> None:
        self.live_modal = None
        self.live_modal_key = None

    # Actions ----------------------------------------------------------

    def action_pause(self) -> None:
        if self.finished:
            return
        paused = self.controls.toggle_pause()
        self._set_state(
            "PAUSED" if paused else "RUNNING", "paused" if paused else "running"
        )
        self.notify(
            "Paused after active requests finish" if paused else "Resumed",
            timeout=2,
        )

    def action_skip(self) -> None:
        if self.finished:
            return
        self.controls.skip_job()
        self._set_state("REQUEST CANCELLED", "paused")
        self.notify("Active requests cancelled — moving to the next job.", timeout=2)

    def action_stop(self) -> None:
        if self.finished:
            return
        self.controls.stop()
        self._set_state("STOPPED", "done")
        self.notify("Run stopped. Active requests cancelled.", timeout=2)

    def action_toggle_failures(self) -> None:
        self.only_failures = not self.only_failures
        self.query_one("#task-title", SectionTitle).set_detail(
            "failures only" if self.only_failures else "newest last"
        )
        self._rebuild_tasks()

    def action_toggle_follow(self) -> None:
        self.follow = not self.follow
        self.notify(f"Auto-scroll {'on' if self.follow else 'off'}", timeout=2)

    def action_toggle_unload(self) -> None:
        if self.finished:
            return
        enabled = self.controls.toggle_force_unload()
        self.notify(
            f"Force unload {'on' if enabled else 'off'} "
            "— applies at the next model switch",
            timeout=3,
        )


def _result_cell(record: TaskRecord, dark: bool = True) -> Text:
    label = (
        "LOOP KILL"
        if record.loop_killed
        else "TIMEOUT"
        if record.timed_out
        else "LENGTH"
        if record.length_exceeded
        else "ERROR"
        if record.error
        else "PASS"
        if record.passed
        else f"{record.score:.0%}"
        if record.score > 0
        else "FAIL"
    )
    color = (
        score_color(record.score * 100, dark)
        if not record.passed and record.score > 0 and not record.error
        else result_color(record.passed, bool(record.error), dark)
    )
    return Text(label, style=f"bold {color}")


def _loop_label(state: str, source: str) -> str:
    if state == "looping":
        return "OUTPUT LOOP" if source == "answer" else "LOOPING"
    if state == "suspected":
        return "SUSPECT"
    if state == "clear":
        return "NO TRACE" if source == "answer" else "CLEAR"
    if state == "observing":
        return "WATCHING"
    return "NO TRACE"


def _loop_cell(state: str, source: str, *, recovered: bool = False) -> Text:
    label = "RECOVERED" if recovered else _loop_label(state, source)
    style = (
        "bold yellow"
        if recovered
        else "bold red"
        if state == "looping"
        else "bold yellow"
        if state == "suspected"
        else "green"
        if state == "clear" and source != "answer"
        else "dim"
    )
    return Text(label, style=style)


def _progress_task_data(event: GenerationProgress) -> dict:
    return {
        "task_id": event.label,
        "passed": None,
        "prompt": event.prompt,
        "thinking": event.thinking,
        "response": event.response,
        "error": "",
        "tok_s": 0.0,
        "response_time_s": event.elapsed_s,
        "live_phase": event.phase,
        "trace_status": event.trace_status,
        "loop_state": event.loop_state,
        "loop_score": event.loop_score,
        "loop_source": event.loop_source,
        "loop_kill_remaining_s": event.loop_kill_remaining_s,
        "thinking_chars": event.thinking_chars,
        "response_chars": event.response_chars,
    }


def _record_task_data(record: TaskRecord) -> dict:
    return {
        "task_id": record.label,
        "passed": record.passed,
        "score": record.score * 100,
        "prompt": record.prompt,
        "thinking": record.thinking,
        "response": record.response,
        "error": record.error,
        "tok_s": record.tok_s,
        "response_time_s": record.response_time_s,
        "output_tokens": record.output_tokens,
        "done_reason": record.done_reason,
        "timed_out": record.timed_out,
        "loop_killed": record.loop_killed,
        "length_exceeded": record.length_exceeded,
        "loop_kill_score": record.loop_kill_score,
        "loop_killed_at_s": record.loop_killed_at_s,
        "trace_status": record.trace_status,
        "thinking_time_s": record.thinking_time_s,
        "time_to_first_answer_s": record.time_to_first_answer_s,
        "loop_state": record.loop_state,
        "loop_score": record.loop_score,
        "loop_source": record.loop_source,
        "loop_detected_at_s": record.loop_detected_at_s,
        "repeated_ngram_coverage": record.repeated_ngram_coverage,
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
