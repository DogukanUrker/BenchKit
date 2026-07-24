"""Live run screen: progress, streaming task results and run controls."""

from __future__ import annotations

import time
from pathlib import Path

from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.message import Message
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, ProgressBar, Static

from benchkit.engine import (
    Engine,
    JobCompleted,
    JobSpec,
    JobStarted,
    RunCompleted,
    RunControls,
    RunFailed,
    TaskCompleted,
    TaskRecord,
    plan_total_tasks,
)
from benchkit.report import save
from benchkit.tui.formatting import (
    bar,
    fmt_count,
    fmt_duration,
    result_color,
    score_color,
)
from benchkit.tui.screens.modals import ConfirmScreen, TaskDetailScreen
from benchkit.tui.widgets import SectionTitle, StatCard, apply_compact


class EngineMessage(Message):
    """An engine event forwarded from the worker thread."""

    def __init__(self, event: object) -> None:
        super().__init__()
        self.event = event


class RunScreen(Screen[None]):
    """Everything that happens while benchmarks are running."""

    BINDINGS = [
        Binding("p", "pause", "Pause"),
        Binding("k", "skip", "Skip job"),
        Binding("x", "stop", "Stop run"),
        Binding("f", "toggle_failures", "Failures only"),
        Binding("t", "toggle_follow", "Follow"),
        Binding("escape", "stop", "Stop run", show=False),
    ]

    def __init__(self, jobs: list[JobSpec]) -> None:
        super().__init__()
        self.jobs = jobs
        self.controls = RunControls()
        self.records: dict[int, list[TaskRecord]] = {}
        self.current_index = -1
        self.current_total = 0
        self.current_passed = 0
        self.current_completed = 0
        self.latency_sum = 0.0
        self.tok_s_sum = 0.0
        self.overall_total = 0
        self.overall_completed = 0
        self.overall_passed = 0
        self.started_at = time.monotonic()
        self.only_failures = False
        self.follow = True
        self.finished = False

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
                yield StatCard("Speed", "--", id="stat-speed")
                yield StatCard("Elapsed", "0s", id="stat-elapsed")
                yield StatCard("ETA", "--", id="stat-eta")
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
        self.sub_title = (
            f"{len(self.jobs)} run(s) · {fmt_count(self.overall_total)} tasks"
        )

        queue = self.query_one("#queue", DataTable)
        queue.add_column("#", key="index", width=3)
        queue.add_column("Model", key="model")
        queue.add_column("Benchmark", key="benchmark")
        queue.add_column("Progress", key="progress", width=10)
        queue.add_column("Score", key="score", width=7)
        queue.add_column("Status", key="status", width=8)
        for index, job in enumerate(self.jobs):
            queue.add_row(
                str(index + 1),
                job.model,
                job.benchmark,
                Text("—", style="dim"),
                Text("—", style="dim"),
                Text("queued", style="dim"),
                key=str(index),
            )

        tasks = self.query_one("#tasks", DataTable)
        tasks.add_column("#", key="index", width=4)
        tasks.add_column("Task", key="task", width=28)
        tasks.add_column("Result", key="result", width=7)
        tasks.add_column("Latency", key="latency", width=8)
        tasks.add_column("tok/s", key="tok_s", width=7)

        self.query_one("#overall-progress", ProgressBar).update(
            total=max(self.overall_total, 1), progress=0
        )
        self._set_state("RUNNING", "running")
        self.set_interval(0.5, self._tick)
        self._start_engine()

    @property
    def dark(self) -> bool:
        return self.app.current_theme.dark

    def on_resize(self, event) -> None:
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
        except Exception:
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
        elif isinstance(event, TaskCompleted):
            self._task_completed(event)
        elif isinstance(event, JobCompleted):
            self._job_completed(event)
        elif isinstance(event, RunCompleted):
            self._run_completed(event)
        elif isinstance(event, RunFailed):
            self.notify(event.message, severity="error", timeout=10)

    def _job_started(self, event: JobStarted) -> None:
        self.current_index = event.index
        self.current_total = event.total
        self.current_passed = 0
        self.current_completed = 0
        self.latency_sum = 0.0
        self.tok_s_sum = 0.0
        self.records[event.index] = []

        job = event.job
        title = Content.from_markup(
            "[b]$bench[/b][dim]  on  [/dim]$model[dim]$slice[/dim]",
            bench=job.benchmark,
            model=job.model,
            slice=f"   slice {job.slice_spec}" if job.slice_spec else "",
        )
        self.query_one("#run-title", Static).update(title)

        self.query_one("#job-progress", ProgressBar).update(
            total=max(event.total, 1), progress=0
        )
        self.query_one("#tasks", DataTable).clear()
        self._update_queue_row(event.index, status="running", style="bold")
        self._update_labels()

    def _task_completed(self, event: TaskCompleted) -> None:
        record = event.record
        self.records.setdefault(event.index, []).append(record)
        self.current_completed = event.completed
        self.current_passed = event.passed
        self.overall_completed += 1
        if record.passed:
            self.overall_passed += 1
        self.latency_sum += record.response_time_s
        self.tok_s_sum += record.tok_s

        self._add_task_row(event.index, record)
        self.query_one("#job-progress", ProgressBar).update(progress=event.completed)
        self.query_one("#overall-progress", ProgressBar).update(
            progress=self.overall_completed
        )
        score = self.current_passed / self.current_completed * 100
        self._update_queue_row(
            event.index,
            progress=bar(event.completed / max(self.current_total, 1), 10),
            score=Text(f"{score:.1f}%", style=f"bold {score_color(score, self.dark)}"),
        )
        self._update_labels()
        self._update_stats()

    def _job_completed(self, event: JobCompleted) -> None:
        result = event.result
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
        if self.only_failures and record.passed:
            return
        table = self.query_one("#tasks", DataTable)
        table.add_row(
            str(record.index + 1),
            record.label,
            _result_cell(record, self.dark),
            fmt_duration(record.response_time_s),
            f"{record.tok_s:.1f}" if record.tok_s else "—",
            key=f"{job_index}:{record.index}",
        )
        if self.follow:
            table.scroll_end(animate=False)

    def _rebuild_tasks(self) -> None:
        table = self.query_one("#tasks", DataTable)
        table.clear()
        for record in self.records.get(self.current_index, []):
            if self.only_failures and record.passed:
                continue
            table.add_row(
                str(record.index + 1),
                record.label,
                _result_cell(record, self.dark),
                fmt_duration(record.response_time_s),
                f"{record.tok_s:.1f}" if record.tok_s else "—",
                key=f"{self.current_index}:{record.index}",
            )
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
        job_label = self.query_one("#job-progress-label", Static)
        overall_label = self.query_one("#overall-progress-label", Static)
        position = self.current_index + 1
        job_label.update(
            f"Job {position}/{len(self.jobs)} · "
            f"{fmt_count(self.current_completed)}/{fmt_count(self.current_total)} tasks"
        )
        overall_label.update(
            f"Overall · {fmt_count(self.overall_completed)}/"
            f"{fmt_count(self.overall_total)} tasks"
        )

    def _update_stats(self) -> None:
        completed = max(self.current_completed, 1)
        score = self.current_passed / completed * 100
        failed = self.current_completed - self.current_passed
        accuracy = self.query_one("#stat-accuracy", StatCard)
        accuracy.set_state(
            f"{score:.1f}%",
            f"{self.overall_passed}/{self.overall_completed} overall",
        )
        self.query_one("#stat-passed", StatCard).set_state(str(self.current_passed))
        self.query_one("#stat-failed", StatCard).set_state(str(failed))
        self.query_one("#stat-speed", StatCard).set_state(
            f"{self.tok_s_sum / completed:.0f} tok/s",
            f"{self.latency_sum / completed:.1f}s avg",
        )

    def _tick(self) -> None:
        elapsed = time.monotonic() - self.started_at
        self.query_one("#stat-elapsed", StatCard).set_state(fmt_duration(elapsed))
        eta = self.query_one("#stat-eta", StatCard)
        if self.overall_completed and not self.finished:
            per_task = elapsed / self.overall_completed
            remaining = max(0, self.overall_total - self.overall_completed)
            eta.set_state(fmt_duration(per_task * remaining), f"{per_task:.1f}s / task")
        elif self.finished:
            eta.set_state("done")

    def _set_state(self, label: str, css_class: str) -> None:
        state = self.query_one("#run-state", Static)
        state.set_classes(f"state {css_class}")
        state.update(label)

    # Events -----------------------------------------------------------

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id != "tasks":
            return
        job_index, _, position = str(event.row_key.value).partition(":")
        records = self.records.get(int(job_index), [])
        record = next((r for r in records if r.index == int(position)), None)
        if record is None:
            return
        job = self.jobs[int(job_index)]
        self.app.push_screen(
            TaskDetailScreen(
                job.title,
                {
                    "task_id": record.label,
                    "passed": record.passed,
                    "prompt": record.prompt,
                    "response": record.response,
                    "error": record.error,
                    "tok_s": record.tok_s,
                    "response_time_s": record.response_time_s,
                },
            )
        )

    # Actions ----------------------------------------------------------

    def action_pause(self) -> None:
        if self.finished:
            return
        paused = self.controls.toggle_pause()
        self._set_state(
            "PAUSED" if paused else "RUNNING", "paused" if paused else "running"
        )
        self.notify("Paused after the current task" if paused else "Resumed", timeout=2)

    def action_skip(self) -> None:
        if self.finished:
            return
        self.controls.skip_job()
        self.notify("Skipping to the next run…", timeout=2)

    def action_stop(self) -> None:
        if self.finished:
            return

        def confirm(answer: bool | None) -> None:
            if answer:
                self.controls.stop()
                self._set_state("STOPPING", "paused")
                self.notify("Stopping — finishing the current task.", timeout=4)

        self.app.push_screen(
            ConfirmScreen(
                "Stop this run?",
                "Finished tasks are kept and reports are still written.",
                confirm="Stop run",
            ),
            confirm,
        )

    def action_toggle_failures(self) -> None:
        self.only_failures = not self.only_failures
        self.query_one("#task-title", SectionTitle).set_detail(
            "failures only" if self.only_failures else "newest last"
        )
        self._rebuild_tasks()

    def action_toggle_follow(self) -> None:
        self.follow = not self.follow
        self.notify(f"Auto-scroll {'on' if self.follow else 'off'}", timeout=2)


def _result_cell(record: TaskRecord, dark: bool = True) -> Text:
    label = "ERROR" if record.error else ("PASS" if record.passed else "FAIL")
    color = result_color(record.passed, bool(record.error), dark)
    return Text(label, style=f"bold {color}")
