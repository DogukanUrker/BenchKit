"""Modal dialogs: task limits, task inspection and confirmations."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Footer,
    Input,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)

from benchkit.engine import SliceError, parse_slice, slice_label
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
    ]

    def __init__(self, context: str, task: dict) -> None:
        super().__init__()
        self.context = context
        self.task_data = task

    def compose(self) -> ComposeResult:
        passed = self.task_data.get("passed")
        error = self.task_data.get("error")
        if error:
            status, style = "ERROR", "b yellow"
        elif passed:
            status, style = "PASS", "b green"
        else:
            status, style = "FAIL", "b red"
        header = Content.from_markup(
            "[b]$task[/b]  [$style]$status[/$style]   [dim]$context[/dim]",
            task=self.task_data.get("task_id", ""),
            status=status,
            style=style,
            context=self.context,
        )

        with Vertical(id="task-dialog", classes="dialog"):
            yield Static(header)
            yield Static(
                f"{fmt_duration(self.task_data.get('response_time_s', 0))} · "
                f"{self.task_data.get('tok_s', 0)} tok/s"
                + (
                    f" · {self.task_data['error']}"
                    if self.task_data.get("error")
                    else ""
                ),
                classes="hint",
            )
            with TabbedContent(initial="response-tab"):
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

    def action_close(self) -> None:
        self.dismiss(None)

    def action_copy(self) -> None:
        self.app.copy_to_clipboard(self.task_data.get("response", ""))
        self.notify("Response copied to clipboard", timeout=2)


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
