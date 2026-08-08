"""Keyboard reference."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Footer, Markdown, Static

HELP = """
### Everywhere

| Key | Action |
| --- | --- |
| `?` / `F1` | This help |
| `Ctrl+P` | Command palette |
| `F2` | Dogi light / dark theme |
| `Ctrl+Q` | Quit |
| `Tab` / `Shift+Tab` | Move between panes |

### Connect

| Key | Action |
| --- | --- |
| `Ctrl+R` | Connect / retry |
| `F3` | Demo mode (no server needed) |
| `Enter` | Connect from any field |

### Setup

| Key | Action |
| --- | --- |
| `Space` / `Enter` | Toggle the highlighted row |
| `a` / `n` / `i` | Select all / clear / invert the focused list |
| `l` | Task limit for the highlighted benchmark |
| `p` | Toggle the paired choice-order run |
| `v` | Check chat templates (selected models, or all) |
| `/` | Jump to the filter box |
| `s` / `F5` | Start the run |
| `Esc` | Back to the connection screen |

Limits accept `20` (first 20), `-20` (last 20) and `40-80` (a range).
Choice-order runs each selected task clean and with deterministically shuffled options.

### Run

| Key | Action |
| --- | --- |
| `p` | Pause / resume between tasks |
| `k` | Skip the current model × benchmark |
| `x` | Stop the run and keep what finished |
| `f` | Show failures only |
| `t` | Follow the newest task (auto-scroll) |
| `Enter` | Inspect the highlighted task |

### Results

| Key | Action |
| --- | --- |
| `Enter` | Drill into a run's tasks |
| `s` / `m` / `t` | Sort by score / model / duration |
| `c` | Copy the results folder path |
| `r` | Configure another run |
| `Esc` | Back to setup |
"""


class HelpScreen(ModalScreen[None]):
    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("question_mark", "close", "Close", show=False),
        Binding("q", "close", "Close", show=False),
    ]

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="help-dialog", classes="dialog"):
            yield Static("Keyboard reference", classes="card-title")
            yield Markdown(HELP)
        yield Footer()

    def action_close(self) -> None:
        self.dismiss(None)
