"""Headless run rendering for `benchkit --headless`.

The TUI is the primary interface; this module keeps a scriptable, log-friendly
path for CI and remote shells.
"""

from __future__ import annotations

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

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
        TextColumn("[dim]{task.description}"),
        BarColumn(
            bar_width=None,
            style="bright_black",
            complete_style="white",
            finished_style="white",
        ),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        expand=True,
        transient=True,
    )
    bars: dict[int, int] = {}

    def sink(event: object) -> None:
        if isinstance(event, JobStarted):
            console.print()
            console.print(
                f"[bold]{event.job.benchmark}[/bold] [dim]on[/dim] "
                f"[white]{event.job.model}[/white] "
                f"[dim]· {event.total} tasks · {slice_label(event.job.slice_spec)}[/dim]"
            )
            bars[event.index] = progress.add_task(
                event.job.benchmark, total=max(event.total, 1)
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
                description=(
                    f"{event.job.benchmark} · {event.label} · {event.phase} · "
                    f"{event.thinking_chars + event.response_chars:,} chars"
                    f"{loop}{kill}"
                ),
            )
        elif isinstance(event, TaskCompleted):
            progress.update(bars[event.index], completed=event.completed)
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
            style = _score_style(result["score"])
            console.print(
                f"[dim]Finished[/dim] [bold]{result['benchmark']}[/bold] "
                f"[dim]on[/dim] [white]{result['model']}[/white]  "
                f"[bold {style}]{result['score']:.1f}%[/bold {style}] "
                f"[dim]{result['passed']}/{result['total']} · "
                f"{result['tok_s']:.1f} tok/s · {result.get('loops', 0)} loops · "
                f"{result.get('loop_kills', 0)} killed · "
                f"{_fmt_time(result['total_time'])}[/dim]"
            )

    engine = Engine(client=client, jobs=jobs, sink=sink, controls=RunControls())
    with progress:
        results = engine.run()
    return results, engine.failure
