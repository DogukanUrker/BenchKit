"""Modal dialogs: task limits, task inspection and confirmations."""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Input,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)

from benchkit.engine import SliceError, parse_slice, slice_label
from benchkit.template_check import TemplateReport
from benchkit.tui.formatting import fmt_count, fmt_duration


class LimitScreen(ModalScreen[str | None]):
    """Ask for a per-benchmark task slice."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+x", "clear", "Run all tasks"),
    ]

    def __init__(self, benchmark: str, current: str, total: int) -> None:
        super().__init__()
        self.benchmark = benchmark
        self.current = current
        self.total = total

    def compose(self) -> ComposeResult:
        with Vertical(id="limit-dialog", classes="dialog"):
            yield Static(f"Task limit · {self.benchmark}", classes="card-title")
            yield Static(f"{fmt_count(self.total)} tasks available", classes="hint")
            yield Input(value=self.current, placeholder="all tasks", id="limit-input")
            yield Static(
                "20 → first 20    -20 → last 20    40-80 → tasks 40 through 80",
                classes="hint",
            )
            yield Static("", id="limit-preview", classes="hint")
            with Horizontal(classes="dialog-actions"):
                yield Button("Apply", id="apply", variant="primary")
                yield Button("Run all", id="clear")
                yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        self.query_one("#limit-input", Input).focus()
        self._preview(self.current)

    def on_input_changed(self, event: Input.Changed) -> None:
        self._preview(event.value)

    def on_input_submitted(self) -> None:
        self._apply()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "apply":
            self._apply()
        elif event.button.id == "clear":
            self.action_clear()
        else:
            self.action_cancel()

    def _preview(self, value: str) -> None:
        preview = self.query_one("#limit-preview", Static)
        spec = value.strip()
        if not spec:
            preview.update("Runs every task.")
            return
        try:
            start, end = parse_slice(spec, self.total or 1)
        except SliceError:
            preview.update("[b]Invalid[/b] — use N, -N or A-B.")
            return
        preview.update(f"{slice_label(spec)} → {fmt_count(max(0, end - start))} tasks")

    def _apply(self) -> None:
        spec = self.query_one("#limit-input", Input).value.strip()
        if spec:
            try:
                parse_slice(spec, self.total or 1)
            except SliceError:
                self.notify("Invalid slice — use N, -N or A-B.", severity="error")
                return
        self.dismiss(spec or None)

    def action_clear(self) -> None:
        self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(self.current or None)


class TaskDetailScreen(ModalScreen[None]):
    """Prompt, response and error for a single task."""

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("q", "close", "Close", show=False),
        Binding("c", "copy", "Copy response"),
        Binding("y", "copy_thinking", "Copy thinking"),
    ]

    def __init__(self, context: str, task: dict, *, live: bool = False) -> None:
        super().__init__()
        self.context = context
        self.task_data = task
        self.live = live

    def _header(self) -> Content:
        live_phase = self.task_data.get("live_phase") if self.live else None
        passed = self.task_data.get("passed")
        error = self.task_data.get("error")
        if live_phase:
            status, style = str(live_phase).upper(), "b blue"
        elif self.task_data.get("loop_killed"):
            status, style = "LOOP KILLED", "b red"
        elif self.task_data.get("timed_out"):
            status, style = "TIMEOUT", "b yellow"
        elif error:
            status, style = "ERROR", "b yellow"
        elif passed:
            status, style = "PASS", "b green"
        elif float(self.task_data.get("score", 0)) > 0:
            status = f"PARTIAL {float(self.task_data['score']):.0f}%"
            style = "b yellow"
        else:
            status, style = "FAIL", "b red"
        return Content.from_markup(
            "[b]$task[/b]  [$style]$status[/$style]   [dim]$context[/dim]",
            task=self.task_data.get("task_id", ""),
            status=status,
            style=style,
            context=self.context,
        )

    def _metadata(self) -> str:
        loop_state = self.task_data.get("loop_state", "unavailable")
        loop_label = (
            "recovered" if self.task_data.get("recovered_cycle") else loop_state
        )
        loop_score = float(self.task_data.get("loop_score", 0.0))
        trace = self.task_data.get("trace_status", "unavailable")
        pieces = [
            fmt_duration(self.task_data.get("response_time_s", 0)),
            f"{self.task_data.get('tok_s', 0)} stream tok/s",
            f"trace {trace}",
            f"loop {loop_label} ({loop_score:.0%})",
        ]
        if float(self.task_data.get("score", 0)) > 0 and not self.task_data.get(
            "passed"
        ):
            pieces.insert(0, f"{float(self.task_data['score']):.0f}% task score")
        if self.task_data.get("error"):
            pieces.append(self.task_data["error"])
        elif self.task_data.get("loop_kill_remaining_s") is not None:
            pieces.append(f"kill in {self.task_data['loop_kill_remaining_s']:.1f}s")
        return " · ".join(pieces)

    def compose(self) -> ComposeResult:
        with Vertical(id="task-dialog", classes="dialog"):
            yield Static(self._header(), id="task-header")
            yield Static(
                self._metadata(),
                classes="hint",
                id="task-meta",
            )
            with TabbedContent(initial="thinking-tab" if self.live else "response-tab"):
                with TabPane("Thinking", id="thinking-tab"):
                    yield TextArea(
                        self.task_data.get("thinking", "")
                        or "(reasoning trace unavailable or empty)",
                        read_only=True,
                        soft_wrap=True,
                        id="thinking-text",
                    )
                with TabPane("Response", id="response-tab"):
                    yield TextArea(
                        self.task_data.get("response", "") or "(empty response)",
                        read_only=True,
                        soft_wrap=True,
                        id="response-text",
                    )
                with TabPane("Prompt", id="prompt-tab"):
                    yield TextArea(
                        self.task_data.get("prompt", ""),
                        read_only=True,
                        soft_wrap=True,
                        id="prompt-text",
                    )
        yield Footer()

    def update_live(self, task: dict, *, finished: bool = False) -> None:
        """Refresh an open task inspector from a streamed snapshot."""
        self.task_data.update(task)
        if finished:
            self.live = False
            self.task_data.pop("live_phase", None)
        if not self.is_mounted:
            return
        try:
            self.query_one("#task-header", Static).update(self._header())
            self.query_one("#task-meta", Static).update(self._metadata())
            _update_live_text(
                self.query_one("#thinking-text", TextArea),
                self.task_data.get("thinking", ""),
                "(reasoning trace unavailable or empty)",
            )
            _update_live_text(
                self.query_one("#response-text", TextArea),
                self.task_data.get("response", ""),
                "(empty response)",
            )
        except NoMatches:
            # A final stream update may race with the modal being dismissed.
            return

    def action_close(self) -> None:
        self.dismiss(None)

    def action_copy(self) -> None:
        self.app.copy_to_clipboard(self.task_data.get("response", ""))
        self.notify("Response copied to clipboard", timeout=2)

    def action_copy_thinking(self) -> None:
        self.app.copy_to_clipboard(self.task_data.get("thinking", ""))
        self.notify("Thinking copied to clipboard", timeout=2)


def _update_live_text(area: TextArea, text: str, empty: str) -> None:
    """Append a live snapshot while preserving the reader's scroll position."""
    value = text or empty
    if value == area.text:
        return

    follow_tail = area.is_vertical_scroll_end
    scroll_x, scroll_y = area.scroll_offset
    if area.text != empty and value.startswith(area.text):
        area.insert(value[len(area.text) :], area.document.end)
        area.history.clear()
    else:
        # The analyzer retains a bounded tail, so very long streams may
        # eventually drop an old prefix and require a full replacement.
        area.load_text(value)

    if follow_tail:
        area.scroll_end(animate=False, x_axis=False)
    else:
        area.call_after_refresh(
            lambda: area.scroll_to(
                x=scroll_x,
                y=scroll_y,
                animate=False,
                force=True,
            )
        )


class TemplateCheckScreen(ModalScreen[None]):
    """Compact results for the setup screen's model template check."""

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("q", "close", "Close", show=False),
    ]

    def __init__(self, reports: list[TemplateReport]) -> None:
        super().__init__()
        self.reports = reports

    def compose(self) -> ComposeResult:
        with Vertical(id="template-check-dialog", classes="dialog"):
            yield Static("Chat template checks", classes="card-title")
            yield Static(
                "Reference-free sanity checks using the server's rendered prompt.",
                classes="hint",
            )
            yield DataTable(id="template-check-table", cursor_type="row")
            with Horizontal(classes="dialog-actions"):
                yield Button("Close", id="close", variant="primary")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#template-check-table", DataTable)
        table.add_columns("Model", "Status", "Details")
        labels = {
            "pass": Text("PASS", style="green"),
            "warn": Text("WARN", style="yellow"),
            "fail": Text("FAIL", style="red"),
            "unavailable": Text("N/A", style="dim"),
        }
        for report in self.reports:
            table.add_row(report.model, labels[report.status], report.detail)
        table.focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close":
            self.action_close()

    def action_close(self) -> None:
        self.dismiss(None)


class ConfirmScreen(ModalScreen[bool]):
    """Yes/no confirmation."""

    BINDINGS = [
        Binding("escape", "no", "Cancel"),
        Binding("y", "yes", "Yes"),
        Binding("n", "no", "No"),
    ]

    def __init__(
        self, question: str, detail: str = "", confirm: str = "Confirm"
    ) -> None:
        super().__init__()
        self.question = question
        self.detail = detail
        self.confirm_label = confirm

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-dialog", classes="dialog"):
            yield Static(self.question, classes="card-title")
            if self.detail:
                yield Static(self.detail, classes="hint")
            with Horizontal(classes="dialog-actions"):
                yield Button(self.confirm_label, id="yes", variant="error")
                yield Button("Cancel", id="no", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")

    def action_yes(self) -> None:
        self.dismiss(True)

    def action_no(self) -> None:
        self.dismiss(False)
