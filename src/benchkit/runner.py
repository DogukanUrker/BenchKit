"""Headless run rendering for `benchkit --headless`.

The TUI is the primary interface; this module keeps a scriptable, log-friendly
path for CI and remote shells.
"""

from __future__ import annotations

from dataclasses import dataclass

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    Task,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Column
from rich.text import Text

from benchkit.client import InferenceClient
from benchkit.engine import (
    Engine,
    GenerationProgress,
    JobCompleted,
    JobSpec,
    JobStarted,
    RunControls,
    RunFailed,
    TaskCompleted,
    TaskPhase,
    TaskRecord,
    slice_label,
)


def _fmt_time(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds >= 60:
        return f"{seconds // 60}m {seconds % 60}s"
    return f"{seconds}s"


def _score_style(score: float) -> str:
    if score >= 80:
        return "green"
    if score < 50:
        return "red"
    return "white"


@dataclass
class _LiveStats:
    """Outcome counters for one headless job."""

    passed: int = 0
    failed: int = 0
    errors: int = 0
    loops: int = 0

    def add(self, record: TaskRecord) -> None:
        if record.passed:
            self.passed += 1
        elif record.error:
            self.errors += 1
        else:
            self.failed += 1
        self.loops += int(record.loop_state == "looping")

    def fields(self, *, live_looping: bool = False) -> dict[str, int]:
        return {
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "loops": self.loops + int(live_looping),
        }


class _LiveScoreColumn(ProgressColumn):
    """Compact live score and outcome counts shown beside the progress bar."""

    def get_table_column(self) -> Column:
        return Column(no_wrap=True)

    def render(self, task: Task) -> Text:
        passed = int(task.fields.get("passed", 0))
        failed = int(task.fields.get("failed", 0))
        errors = int(task.fields.get("errors", 0))
        loops = int(task.fields.get("loops", 0))
        completed = passed + failed + errors

        if completed:
            score = passed / completed * 100
            score_text = Text(f"{score:5.1f}%", style=f"bold {_score_style(score)}")
        else:
            score_text = Text("  --.-%", style="dim")

        score_text.append(" ")
        score_text.append("PASS", style="bold green")
        score_text.append(f" {passed} ")
        score_text.append("FAIL", style="bold red")
        score_text.append(f" {failed} ")
        score_text.append("ERR", style="bold yellow")
        score_text.append(f" {errors} ")
        score_text.append("LOOP", style="bold magenta")
        score_text.append(f" {loops}")
        return score_text


def run(
    client: InferenceClient,
    jobs: list[JobSpec],
    console: Console,
    verbose: bool = False,
) -> tuple[list[dict], str | None]:
    """Run every job, printing progress to `console`.

    Returns the results plus the failure message when the run ended early.
    """
    progress = Progress(
        TextColumn(
            "[dim]{task.description}",
            table_column=Column(
                ratio=1,
                min_width=8,
                no_wrap=True,
                overflow="ellipsis",
            ),
        ),
        BarColumn(
            bar_width=12,
            style="bright_black",
            complete_style="white",
            finished_style="white",
        ),
        MofNCompleteColumn(),
        _LiveScoreColumn(),
        TimeElapsedColumn(),
        console=console,
        expand=True,
        transient=True,
    )
    bars: dict[int, int] = {}
    live_stats: dict[int, _LiveStats] = {}

    def sink(event: object) -> None:
        if isinstance(event, JobStarted):
            console.print()
            console.print(
                f"[bold]{event.job.benchmark}[/bold] [dim]on[/dim] "
                f"[white]{event.job.model}[/white] "
                f"[dim]· {event.total} tasks · {slice_label(event.job.slice_spec)}[/dim]"
            )
            stats = _LiveStats()
            live_stats[event.index] = stats
            bars[event.index] = progress.add_task(
                event.job.benchmark,
                total=max(event.total, 1),
                **stats.fields(),
            )
        elif isinstance(event, TaskPhase):
            progress.update(
                bars[event.index],
                description=(
                    f"{event.job.benchmark} · {event.label} · {event.activity}"
                ),
            )
        elif isinstance(event, GenerationProgress):
            loop = (
                " · LOOPING"
                if event.loop_state == "looping"
                else " · loop suspected"
                if event.loop_state == "suspected"
                else ""
            )
            kill = (
                f" · kill in {event.loop_kill_remaining_s:.1f}s"
                if event.loop_kill_remaining_s is not None
                else ""
            )
            progress.update(
                bars[event.index],
                **live_stats[event.index].fields(
                    live_looping=event.loop_state == "looping"
                ),
                description=(
                    f"{event.job.benchmark} · {event.label} · {event.phase} · "
                    f"{event.thinking_chars + event.response_chars:,} chars"
                    f"{loop}{kill}"
                ),
            )
        elif isinstance(event, TaskCompleted):
            stats = live_stats[event.index]
            stats.add(event.record)
            progress.update(
                bars[event.index],
                completed=event.completed,
                **stats.fields(),
            )
            if verbose:
                record = event.record
                status = (
                    "[bold red]LOOP KILLED[/bold red]"
                    if record.loop_killed
                    else "[bold yellow]TIMEOUT[/bold yellow]"
                    if record.timed_out
                    else "[bold green]PASS[/bold green]"
                    if record.passed
                    else "[bold yellow]ERROR[/bold yellow]"
                    if record.error
                    else "[bold red]FAIL[/bold red]"
                )
                console.print(f"[bold]{record.label}[/bold] [dim]·[/dim] {status}")
                console.print("[dim]Prompt[/dim]")
                console.print(record.prompt, markup=False, highlight=False)
                if record.thinking:
                    console.print("[dim]Thinking[/dim]")
                    console.print(record.thinking, markup=False, highlight=False)
                console.print("[dim]Response[/dim]")
                console.print(record.response, markup=False, highlight=False)
                console.print()
        elif isinstance(event, RunFailed):
            console.print(f"[red]Run failed:[/red] {event.message}")
        elif isinstance(event, JobCompleted):
            result = event.result
            if not result["total"]:
                console.print(
                    f"[yellow]Skipped[/yellow] [bold]{result['benchmark']}[/bold] "
                    f"[dim]on[/dim] [white]{result['model']}[/white]"
                )
                return
            stats = live_stats[event.index]
            style = _score_style(result["score"])
            console.print(
                f"[dim]Finished[/dim] [bold]{result['benchmark']}[/bold] "
                f"[dim]on[/dim] [white]{result['model']}[/white]  "
                f"[bold {style}]{result['score']:.1f}%[/bold {style}]  "
                f"[green]PASS {stats.passed}[/green] · "
                f"[red]FAIL {stats.failed}[/red] · "
                f"[yellow]ERR {stats.errors}[/yellow] · "
                f"[magenta]LOOP {stats.loops}[/magenta] "
                f"[dim]· {result['total']} tasks · {result['tok_s']:.1f} tok/s · "
                f"{result.get('loop_kills', 0)} killed · "
                f"{_fmt_time(result['total_time'])}[/dim]"
            )

    engine = Engine(client=client, jobs=jobs, sink=sink, controls=RunControls())
    with progress:
        results = engine.run()
    return results, engine.failure
