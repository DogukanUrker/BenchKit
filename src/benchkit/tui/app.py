"""The BenchKit terminal application."""

from __future__ import annotations

from pathlib import Path

from textual.app import App
from textual.binding import Binding
from textual.theme import Theme

from benchkit.engine import JobSpec
from benchkit.tui.screens.connect import ConnectScreen
from benchkit.tui.screens.help import HelpScreen
from benchkit.tui.screens.results import ResultsScreen
from benchkit.tui.screens.run import RunScreen
from benchkit.tui.screens.setup import SetupScreen

BENCHKIT_DARK = Theme(
    name="benchkit",
    primary="#7aa2f7",
    secondary="#bb9af7",
    accent="#2ac3de",
    success="#9ece6a",
    warning="#e0af68",
    error="#f7768e",
    foreground="#c0caf5",
    background="#11141c",
    surface="#171b24",
    panel="#1f2430",
    dark=True,
    variables={
        "block-cursor-text-style": "bold",
        "footer-key-foreground": "#2ac3de",
        "input-selection-background": "#7aa2f7 35%",
        "border": "#2b3244",
    },
)

BENCHKIT_LIGHT = Theme(
    name="benchkit-light",
    primary="#3d5afe",
    secondary="#7c4dff",
    accent="#00838f",
    success="#2e7d32",
    warning="#a06000",
    error="#c62828",
    foreground="#1a1f2b",
    background="#f4f5f8",
    surface="#ffffff",
    panel="#e8eaf0",
    dark=False,
    variables={
        "block-cursor-text-style": "bold",
        "footer-key-foreground": "#00838f",
    },
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
        self.demo = demo
        self.host_override = host
        self.client = None
        self.models: list[dict] = []
        self.last_output: Path | None = None

    def on_mount(self) -> None:
        self.register_theme(BENCHKIT_DARK)
        self.register_theme(BENCHKIT_LIGHT)
        self.theme = "benchkit"
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
        self.theme = "benchkit-light" if self.theme == "benchkit" else "benchkit"
        self.notify(f"Theme: {self.theme}", timeout=2)


def run_tui(*, demo: bool = False, host: str | None = None) -> None:
    BenchKitApp(demo=demo, host=host).run()
