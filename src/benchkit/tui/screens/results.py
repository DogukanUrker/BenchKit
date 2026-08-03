"""Results screen: sortable summary plus drill-down into every task."""

from __future__ import annotations

from pathlib import Path

from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static

from benchkit.tui.formatting import bar, fmt_count, fmt_duration, score_color
from benchkit.tui.screens.detail import JobDetailScreen
from benchkit.tui.widgets import SectionTitle, StatCard, apply_compact

SORT_KEYS = {
    "model": lambda r: (r["model"], r["benchmark"]),
    "benchmark": lambda r: (r["benchmark"], -r["score"]),
    "score": lambda r: -r["score"],
    "passed": lambda r: -r["passed"],
    "errors": lambda r: -r.get("errors", 0),
    "loops": lambda r: -r.get("loop_rate", 0),
    "tok_s": lambda r: -r["tok_s"],
    "avg": lambda r: r["avg_response_time"],
    "time": lambda r: r["total_time"],
}


class ResultsScreen(Screen[None]):
    """What the run produced, and where it was written."""

    BINDINGS = [
        Binding("s", "sort('score')", "Sort score"),
        Binding("m", "sort('model')", "Sort model"),
        Binding("t", "sort('time')", "Sort time"),
        Binding("r", "again", "New run"),
        Binding("c", "copy_path", "Copy path"),
        Binding("escape", "again", "Back", show=False),
    ]

    def __init__(self, results: list[dict], output: Path | None) -> None:
        super().__init__()
        self.results = list(results)
        self.output = output
        self.sort_key = "score"

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="results-root"):
            with Horizontal(id="stat-row"):
                yield StatCard("Best score", "--", id="stat-best")
                yield StatCard("Runs", "0", id="stat-runs")
                yield StatCard("Tasks", "0", id="stat-tasks")
                yield StatCard("Loops", "0", id="stat-loops")
                yield StatCard("Fastest", "--", id="stat-fastest")
                yield StatCard("Total time", "--", id="stat-time")
            with Vertical(classes="pane", id="results-pane"):
                yield SectionTitle(
                    "Results",
                    "enter to inspect · click a header to sort",
                    id="results-title",
                )
                yield DataTable(id="summary", cursor_type="row", zebra_stripes=True)
            yield Static("", id="output-path", classes="hint")
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = "run complete"
        table = self.query_one("#summary", DataTable)
        table.add_column("Model", key="model")
        table.add_column("Benchmark", key="benchmark")
        table.add_column("Score", key="score", width=8)
        table.add_column("", key="bar", width=14)
        table.add_column("Passed", key="passed", width=10)
        table.add_column("Loops", key="loops", width=8)
        table.add_column("Errors", key="errors", width=7)
        table.add_column("tok/s", key="tok_s", width=8)
        table.add_column("Avg", key="avg", width=8)
        table.add_column("Time", key="time", width=9)

        self._fill_table()
        self._fill_stats()
        self.watch(self.app, "theme", self._theme_changed, init=False)

        if self.output is not None:
            path = self.output.resolve()
            self.query_one("#output-path", Static).update(
                f"Reports → {path}  ·  results.json · csv · md · html"
            )
        else:
            self.query_one("#output-path", Static).update(
                "Nothing was written to disk."
            )
        table.focus()

    def on_resize(self, event: events.Resize) -> None:
        apply_compact(self, event.size.height)

    def _theme_changed(self, _theme: str) -> None:
        # Table cell colours are baked in, so redraw them for the new palette.
        self._fill_table()

    # Rendering --------------------------------------------------------

    def _sorted(self) -> list[tuple[int, dict]]:
        key = SORT_KEYS.get(self.sort_key, SORT_KEYS["score"])
        return sorted(enumerate(self.results), key=lambda pair: key(pair[1]))

    def _fill_table(self) -> None:
        table = self.query_one("#summary", DataTable)
        table.clear()
        dark = self.app.current_theme.dark
        for index, result in self._sorted():
            score = result["score"]
            color = score_color(score, dark)
            table.add_row(
                result["model"],
                result["benchmark"]
                + (f" [{result['slice']}]" if result.get("slice") else ""),
                Text(f"{score:.1f}%", style=f"bold {color}"),
                Text(bar(score / 100, 12), style=color),
                f"{result['passed']}/{result['total']}",
                (
                    f"{result.get('loop_rate', 0):.1f}%"
                    if result.get("loops", 0)
                    else "—"
                ),
                str(result.get("errors", 0)) if result.get("errors") else "—",
                f"{result['tok_s']:.1f}",
                f"{result['avg_response_time']}s",
                fmt_duration(result["total_time"]),
                key=str(index),
            )

    def _fill_stats(self) -> None:
        if not self.results:
            return
        best = max(self.results, key=lambda r: r["score"])
        fastest = max(self.results, key=lambda r: r["tok_s"])
        tasks = sum(r["total"] for r in self.results)
        total_time = sum(r["total_time"] for r in self.results)
        loops = sum(r.get("loops", 0) for r in self.results)
        loop_kills = sum(r.get("loop_kills", 0) for r in self.results)

        self.query_one("#stat-best", StatCard).set_state(
            f"{best['score']:.1f}%", f"{best['model']} · {best['benchmark']}"
        )
        self.query_one("#stat-runs", StatCard).set_state(str(len(self.results)))
        self.query_one("#stat-tasks", StatCard).set_state(fmt_count(tasks))
        self.query_one("#stat-loops", StatCard).set_state(
            str(loops),
            (f"{loops / tasks * 100:.1f}% · {loop_kills} killed" if tasks else ""),
        )
        self.query_one("#stat-fastest", StatCard).set_state(
            f"{fastest['tok_s']:.0f} tok/s", fastest["model"]
        )
        self.query_one("#stat-time", StatCard).set_state(fmt_duration(total_time))

    # Events -----------------------------------------------------------

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        index = int(str(event.row_key.value))
        self.app.push_screen(JobDetailScreen(self.results[index]))

    def on_data_table_header_selected(self, event: DataTable.HeaderSelected) -> None:
        key = str(event.column_key.value)
        if key in SORT_KEYS:
            self.action_sort(key)

    # Actions ----------------------------------------------------------

    def action_sort(self, key: str) -> None:
        self.sort_key = key
        self._fill_table()
        self.query_one("#results-title", SectionTitle).set_detail(
            f"sorted by {key} · enter to inspect"
        )

    def action_again(self) -> None:
        self.app.back_to_setup()

    def action_copy_path(self) -> None:
        if self.output is None:
            return
        path = str(self.output.resolve())
        self.app.copy_to_clipboard(path)
        self.notify(f"Copied {path}", timeout=3)
