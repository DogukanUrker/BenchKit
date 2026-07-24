"""The BenchKit terminal application."""

from __future__ import annotations

from pathlib import Path

from textual.app import App
from textual.binding import Binding

from benchkit.engine import JobSpec
from benchkit.tui.screens.connect import ConnectScreen
from benchkit.tui.screens.help import HelpScreen
from benchkit.tui.screens.results import ResultsScreen
from benchkit.tui.screens.run import RunScreen
from benchkit.tui.screens.setup import SetupScreen
from benchkit.tui.theme import (
    ALTERNATE_THEME,
    DEFAULT_THEME,
    THEMES,
    VARIABLE_DEFAULTS,
)


class BenchKitApp(App[None]):
    """Full-screen benchmark cockpit: connect, configure, run, inspect."""

    CSS_PATH = "app.tcss"
    TITLE = "BenchKit"
    SUB_TITLE = "benchmark your local LLMs"
    ENABLE_COMMAND_PALETTE = True

    BINDINGS = [
        Binding("question_mark", "help", "Help"),
        Binding("f1", "help", "Help", show=False),
        Binding("ctrl+q", "quit", "Quit", priority=True, show=True),
        Binding("f2", "toggle_theme", "Theme", show=False),
    ]

    def __init__(self, *, demo: bool = False, host: str | None = None) -> None:
        super().__init__()
        for theme in THEMES:
            self.register_theme(theme)
        self.theme = DEFAULT_THEME
        self.demo = demo
        self.host_override = host
        self.client = None
        self.models: list[dict] = []
        self.last_output: Path | None = None

    def get_theme_variable_defaults(self) -> dict[str, str]:
        return dict(VARIABLE_DEFAULTS)

    def on_mount(self) -> None:
        self.push_screen(ConnectScreen())

    # Navigation -------------------------------------------------------

    def show_setup(self, client, models: list[dict]) -> None:
        self.client = client
        self.models = models
        self.push_screen(SetupScreen())

    def start_run(self, jobs: list[JobSpec]) -> None:
        self.push_screen(RunScreen(jobs))

    def show_results(self, results: list[dict], output: Path | None) -> None:
        self.last_output = output
        self.switch_screen(ResultsScreen(results, output))

    def back_to_setup(self) -> None:
        """Return to the configuration screen from anywhere downstream."""
        while len(self.screen_stack) > 2 and not isinstance(self.screen, SetupScreen):
            self.pop_screen()

    # Actions ----------------------------------------------------------

    def action_help(self) -> None:
        if not isinstance(self.screen, HelpScreen):
            self.push_screen(HelpScreen())

    def action_toggle_theme(self) -> None:
        self.theme = ALTERNATE_THEME if self.theme == DEFAULT_THEME else DEFAULT_THEME
        self.notify(f"Theme: {self.theme}", timeout=2)


def run_tui(*, demo: bool = False, host: str | None = None) -> None:
    BenchKitApp(demo=demo, host=host).run()
