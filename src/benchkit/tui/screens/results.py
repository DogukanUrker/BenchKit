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

from benchkit.metrics import aggregate_tok_s, effective_concurrency, stream_tok_s
from benchkit.tui.formatting import (
    bar,
    fmt_count,
    fmt_duration,
    score_color,
    throughput_stat,
)
from benchkit.tui.screens.detail import JobDetailScreen
from benchkit.tui.widgets import SectionTitle, StatCard, apply_compact


def _benchmark_label(result: dict) -> str:
    return str(result.get("benchmark_label") or result["benchmark"])


def _is_parallel(result: dict) -> bool:
    return result.get("concurrency", 1) > 1


def _display_tok_s(result: dict) -> float:
    return aggregate_tok_s(result) if _is_parallel(result) else stream_tok_s(result)


def _delta_cell(result: dict) -> Text | str:
    value = result.get("score_delta_pp")
    if value is None:
        return "—"
    delta = float(value)
    style = "green" if delta > 0 else "red" if delta < 0 else "dim"
    return Text(f"{delta:+.1f}", style=style)


def _harness_delta_cell(result: dict) -> Text | str:
    value = result.get("harness_delta_pp")
    if value is None:
        return "—"
    delta = float(value)
    style = "green" if delta > 0 else "red" if delta < 0 else "dim"
    return Text(f"{delta:+.1f}", style=style)


def _repair_delta_cell(result: dict) -> Text | str:
    if not result.get("repair_attempts"):
        return "—"
    delta = float(result.get("repair_delta_pp", 0))
    style = "green" if delta > 0 else "red" if delta < 0 else "dim"
    return Text(f"{delta:+.1f}", style=style)


SORT_KEYS = {
    "model": lambda r: (r["model"], r["benchmark"]),
    "benchmark": lambda r: (_benchmark_label(r), -r["score"]),
    "harness": lambda r: (r.get("harness", "direct"), -r["score"]),
    "harness_delta": lambda r: -float(r.get("harness_delta_pp", 0)),
    "repair_delta": lambda r: -float(r.get("repair_delta_pp", 0)),
    "score": lambda r: -r["score"],
    "delta": lambda r: -float(r.get("score_delta_pp", 0)),
    "regressions": lambda r: -int(r.get("regressions", 0)),
    "passed": lambda r: -r["passed"],
    "errors": lambda r: -r.get("errors", 0),
    "loops": lambda r: -r.get("loop_rate", 0),
    "parallel": lambda r: -r.get("concurrency", 1),
    "aggregate": lambda r: -aggregate_tok_s(r),
    "stream": lambda r: -stream_tok_s(r),
    "effective": lambda r: -effective_concurrency(r),
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
        self.has_parallel = any(_is_parallel(result) for result in self.results)
        self.has_perturbations = any(
            result.get("perturbation") for result in self.results
        )
        self.has_harness_pairs = any(
            result.get("harness_delta_pp") is not None for result in self.results
        )
        self.has_repairs = any(
            result.get("repair_attempts", 0) for result in self.results
        )
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
                yield StatCard("Peak throughput", "--", id="stat-fastest")
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
        table.add_column("Harness", key="harness", width=18)
        table.add_column("Benchmark", key="benchmark")
        table.add_column("Score", key="score", width=8)
        table.add_column("", key="bar", width=14)
        if self.has_harness_pairs:
            table.add_column("Harness Δ", key="harness_delta", width=10)
        if self.has_repairs:
            table.add_column("Repair Δ", key="repair_delta", width=9)
            table.add_column("Fixed", key="repair_fixed", width=8)
        if self.has_perturbations:
            table.add_column("Δ pp", key="delta", width=7)
            table.add_column("Regress", key="regressions", width=9)
        table.add_column("Passed", key="passed", width=10)
        table.add_column("Loops", key="loops", width=8)
        table.add_column("Errors", key="errors", width=7)
        if self.has_parallel:
            table.add_column("Parallel", key="parallel", width=8)
            table.add_column("Agg tok/s", key="aggregate", width=10)
            table.add_column("Stream", key="stream", width=8)
            table.add_column("Eff", key="effective", width=7)
        else:
            table.add_column("Tok/s", key="stream", width=9)
        table.add_column("Avg", key="avg", width=8)
        table.add_column("Wall", key="time", width=9)

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
            cells: list[object] = [
                result["model"],
                result.get("harness_label", "Direct"),
                _benchmark_label(result)
                + (f" [{result['slice']}]" if result.get("slice") else ""),
                Text(f"{score:.1f}%", style=f"bold {color}"),
                Text(bar(score / 100, 12), style=color),
            ]
            if self.has_harness_pairs:
                cells.append(_harness_delta_cell(result))
            if self.has_repairs:
                cells.extend(
                    [
                        _repair_delta_cell(result),
                        (
                            f"{result.get('repair_successes', 0)}/"
                            f"{result.get('repair_attempted', 0)}"
                            if result.get("repair_attempts")
                            else "—"
                        ),
                    ]
                )
            if self.has_perturbations:
                cells.extend(
                    [
                        _delta_cell(result),
                        (
                            f"{result['regressions']}/{result['paired_total']}"
                            if result.get("paired_total") is not None
                            else "—"
                        ),
                    ]
                )
            cells.extend(
                [
                    f"{result['passed']}/{result['total']}",
                    (
                        f"{result.get('loop_rate', 0):.1f}%"
                        if result.get("loops", 0)
                        else "—"
                    ),
                    str(result.get("errors", 0)) if result.get("errors") else "—",
                ]
            )
            if self.has_parallel:
                parallel = _is_parallel(result)
                cells.extend(
                    [
                        str(result.get("concurrency", 1)),
                        f"{aggregate_tok_s(result):.1f}" if parallel else "—",
                        f"{stream_tok_s(result):.1f}",
                        f"{effective_concurrency(result):.2f}x" if parallel else "—",
                    ]
                )
            else:
                cells.append(f"{stream_tok_s(result):.1f}")
            cells.extend(
                [
                    f"{result['avg_response_time']}s",
                    fmt_duration(result["total_time"]),
                ]
            )
            table.add_row(
                *cells,
                key=str(index),
            )

    def _fill_stats(self) -> None:
        if not self.results:
            return
        overall_results = [
            result for result in self.results if result.get("include_in_overall", True)
        ]
        best = max(overall_results or self.results, key=lambda r: r["score"])
        fastest = max(self.results, key=_display_tok_s)
        tasks = sum(r["total"] for r in self.results)
        total_time = sum(r["total_time"] for r in self.results)
        loops = sum(r.get("loops", 0) for r in self.results)
        loop_kills = sum(r.get("loop_kills", 0) for r in self.results)

        self.query_one("#stat-best", StatCard).set_state(
            f"{best['score']:.1f}%",
            f"{best['model']} · {best.get('harness_label', 'Direct')} · "
            f"{_benchmark_label(best)}",
        )
        self.query_one("#stat-runs", StatCard).set_state(str(len(self.results)))
        self.query_one("#stat-tasks", StatCard).set_state(fmt_count(tasks))
        self.query_one("#stat-loops", StatCard).set_state(
            str(loops),
            (f"{loops / tasks * 100:.1f}% · {loop_kills} killed" if tasks else ""),
        )
        fastest_card = self.query_one("#stat-fastest", StatCard)
        value, hint = throughput_stat(
            concurrency=fastest.get("concurrency", 1),
            aggregate=aggregate_tok_s(fastest),
            stream=stream_tok_s(fastest),
            effective=effective_concurrency(fastest),
            precision=0,
        )
        fastest_card.set_state(value, f"{fastest['model']} · {hint}")
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
