"""Per-run drill-down: every task of a single model x benchmark."""

from __future__ import annotations

from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Input

from benchkit.engine import slice_label
from benchkit.tui.formatting import (
    fmt_count,
    fmt_duration,
    result_color,
    score_color,
)
from benchkit.tui.screens.modals import TaskDetailScreen
from benchkit.tui.widgets import SectionTitle, StatCard, apply_compact

FILTERS = ["all", "failed", "passed"]


class JobDetailScreen(Screen[None]):
    """Task-level view of one benchmark run."""

    BINDINGS = [
        Binding("f", "cycle_filter", "Filter"),
        Binding("slash", "focus_search", "Search"),
        Binding("escape", "close", "Back"),
    ]

    def __init__(self, result: dict) -> None:
        super().__init__()
        self.result = result
        self.filter_index = 0

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="detail-root"):
            with Horizontal(id="stat-row"):
                yield StatCard("Score", "--", id="stat-score")
                yield StatCard("Passed", "--", id="stat-passed")
                yield StatCard("Loop rate", "--", id="stat-loops")
                yield StatCard("Speed", "--", id="stat-speed")
                yield StatCard("Avg latency", "--", id="stat-latency")
                yield StatCard("Duration", "--", id="stat-duration")
            with Vertical(classes="pane", id="detail-pane"):
                yield SectionTitle("Tasks", "enter to open", id="detail-title")
                yield Input(placeholder="Search task id or response…", id="task-search")
                yield DataTable(id="task-table", cursor_type="row", zebra_stripes=True)
        yield Footer()

    def on_mount(self) -> None:
        result = self.result
        self.title = f"{result['benchmark']} · {result['model']}"
        self.sub_title = f"{result['passed']}/{result['total']} passed"

        table = self.query_one("#task-table", DataTable)
        table.add_column("#", key="index", width=5)
        table.add_column("Task", key="task")
        table.add_column("Result", key="result", width=8)
        table.add_column("Loop", key="loop", width=9)
        table.add_column("Think", key="thinking", width=8)
        table.add_column("Latency", key="latency", width=9)
        table.add_column("tok/s", key="tok_s", width=8)

        score = result["score"]
        self.query_one("#stat-score", StatCard).set_state(
            f"{score:.1f}%", slice_label(result.get("slice"))
        )
        self.query_one("#stat-passed", StatCard).set_state(
            f"{result['passed']}/{result['total']}",
            f"{result.get('errors', 0)} error(s)",
        )
        self.query_one("#stat-loops", StatCard).set_state(
            f"{result.get('loop_rate', 0):.1f}%",
            f"{result.get('loops', 0)} loop · "
            f"{result.get('suspected_loops', 0)} suspect · "
            f"{result.get('recovered_loops', 0)} recovered · "
            f"{result.get('loop_kills', 0)} killed",
        )
        self.query_one("#stat-speed", StatCard).set_state(
            f"{result['tok_s']:.1f} tok/s"
        )
        self.query_one("#stat-latency", StatCard).set_state(
            f"{result['avg_response_time']}s"
        )
        self.query_one("#stat-duration", StatCard).set_state(
            fmt_duration(result["total_time"])
        )
        self._restyle_score()

        self._fill()
        self.watch(self.app, "theme", self._theme_changed, init=False)
        table.focus()

    def _restyle_score(self) -> None:
        color = score_color(self.result["score"], self.app.current_theme.dark)
        self.query_one("#stat-score", StatCard).styles.color = color

    def _theme_changed(self, _theme: str) -> None:
        self._restyle_score()
        self._fill()

    def on_resize(self, event: events.Resize) -> None:
        apply_compact(self, event.size.height)

    # Rendering --------------------------------------------------------

    def _visible_tasks(self) -> list[tuple[int, dict]]:
        mode = FILTERS[self.filter_index]
        needle = self.query_one("#task-search", Input).value.strip().lower()
        rows = []
        for index, task in enumerate(self.result["tasks"]):
            if mode == "failed" and task["passed"]:
                continue
            if mode == "passed" and not task["passed"]:
                continue
            if (
                needle
                and needle not in task["task_id"].lower()
                and needle not in (task.get("response", "").lower())
                and needle not in (task.get("thinking", "").lower())
            ):
                continue
            rows.append((index, task))
        return rows

    def _fill(self) -> None:
        table = self.query_one("#task-table", DataTable)
        table.clear()
        rows = self._visible_tasks()
        for index, task in rows:
            entry = task.get("entry_point")
            label = f"{task['task_id']} ({entry})" if entry else task["task_id"]
            table.add_row(
                str(index + 1),
                label,
                _result_cell(task, self.app.current_theme.dark),
                _loop_cell(task),
                (
                    fmt_duration(task.get("thinking_time_s", 0))
                    if task.get("thinking")
                    else "—"
                ),
                fmt_duration(task.get("response_time_s", 0)),
                f"{task.get('tok_s', 0):.1f}",
                key=str(index),
            )
        self.query_one("#detail-title", SectionTitle).set_detail(
            f"{FILTERS[self.filter_index]} · {fmt_count(len(rows))} shown · enter to open"
        )

    # Events -----------------------------------------------------------

    def on_input_changed(self) -> None:
        self._fill()

    def on_input_submitted(self) -> None:
        self.query_one("#task-table", DataTable).focus()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        index = int(str(event.row_key.value))
        task = self.result["tasks"][index]
        title = f"{self.result['benchmark']} · {self.result['model']}"
        self.app.push_screen(TaskDetailScreen(title, task))

    # Actions ----------------------------------------------------------

    def action_cycle_filter(self) -> None:
        self.filter_index = (self.filter_index + 1) % len(FILTERS)
        self._fill()

    def action_focus_search(self) -> None:
        self.query_one("#task-search", Input).focus()

    def action_close(self) -> None:
        self.app.pop_screen()


def _result_cell(task: dict, dark: bool = True) -> Text:
    error = bool(task.get("error"))
    passed = bool(task.get("passed"))
    label = (
        "LOOPKILL"
        if task.get("loop_killed")
        else "TIMEOUT"
        if task.get("timed_out")
        else "ERROR"
        if error
        else "PASS"
        if passed
        else "FAIL"
    )
    return Text(label, style=f"bold {result_color(passed, error, dark)}")


def _loop_cell(task: dict) -> Text:
    state = task.get("loop_state", "unavailable")
    source = task.get("loop_source", "none")
    label = (
        "RECOVERED"
        if task.get("recovered_cycle")
        else "OUTPUT"
        if state == "looping" and source == "answer"
        else "LOOPING"
        if state == "looping"
        else "SUSPECT"
        if state == "suspected"
        else "NO TRACE"
        if state == "clear" and source == "answer"
        else "CLEAR"
        if state == "clear"
        else "NO TRACE"
    )
    style = (
        "bold yellow"
        if task.get("recovered_cycle")
        else "bold red"
        if state == "looping"
        else "bold yellow"
        if state == "suspected"
        else "green"
        if state == "clear" and source != "answer"
        else "dim"
    )
    return Text(label, style=style)
