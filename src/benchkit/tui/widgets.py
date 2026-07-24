"""Reusable presentational widgets for the BenchKit TUI."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.content import Content
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static


class StatCard(Widget):
    """A label above a large value, used for the live run statistics."""

    DEFAULT_CSS = """
    StatCard {
        width: 1fr;
        height: 4;
        padding: 0 1;
        border-left: outer $panel;
        background: $surface;
    }
    StatCard .stat-label {
        color: $text-muted;
        text-style: none;
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }
    StatCard .stat-value {
        text-style: bold;
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }
    StatCard .stat-hint {
        color: $text-muted;
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }
    """

    value: reactive[str] = reactive("--")
    hint: reactive[str] = reactive("")

    def __init__(
        self,
        label: str,
        value: str = "--",
        hint: str = "",
        *,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(id=id, classes=classes)
        self.label = label
        self.set_reactive(StatCard.value, value)
        self.set_reactive(StatCard.hint, hint)

    def compose(self) -> ComposeResult:
        yield Static(self.label, classes="stat-label")
        yield Static(self.value, classes="stat-value", id="value")
        yield Static(self.hint, classes="stat-hint", id="hint")

    def watch_value(self, value: str) -> None:
        if self.is_mounted:
            self.query_one("#value", Static).update(value)

    def watch_hint(self, hint: str) -> None:
        if self.is_mounted:
            self.query_one("#hint", Static).update(hint)

    def set_state(self, value: str, hint: str = "") -> None:
        self.value = value
        self.hint = hint


class Banner(Static):
    """The BenchKit wordmark used on the connection screen."""

    ART = r"""
██████╗ ███████╗███╗   ██╗ ██████╗██╗  ██╗██╗████████╗
██╔══██╗██╔════╝████╗  ██║██╔════╝██║  ██║██║╚══██╔══╝
██████╔╝█████╗  ██╔██╗ ██║██║     ███████║██║   ██║
██╔══██╗██╔══╝  ██║╚██╗██║██║     ██╔══██║██║   ██║
██████╔╝███████╗██║ ╚████║╚██████╗██║  ██║██║   ██║
╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝╚═╝   ╚═╝
""".strip("\n")

    def __init__(self, **kwargs) -> None:
        super().__init__(self.ART, **kwargs)


class SectionTitle(Static):
    """A pane heading with an optional right-aligned counter."""

    def __init__(self, title: str, detail: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self._title = title
        self._detail = detail
        self.update(self._render())

    def _render(self) -> Content:
        if self._detail:
            return Content.from_markup(
                "[b]$title[/b]  [dim]$detail[/dim]",
                title=self._title,
                detail=self._detail,
            )
        return Content.from_markup("[b]$title[/b]", title=self._title)

    def set_detail(self, detail: str) -> None:
        self._detail = detail
        self.update(self._render())


def apply_compact(screen, height: int) -> None:
    """Tighten the stat cards and progress block on short terminals."""
    screen.set_class(height < 32, "compact")
