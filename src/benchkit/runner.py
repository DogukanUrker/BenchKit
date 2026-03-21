"""Run benchmarks across models and collect results."""

import time

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TimeElapsedColumn,
)

from benchkit.ollama import generate, unload_model


def _fmt_time(s: float) -> str:
    s = int(round(s))
    if s >= 60:
        return f"{s // 60}m {s % 60}s"
    return f"{s}s"


def _score_style(score: float) -> str:
    if score >= 80:
        return "green"
    if score < 50:
        return "red"
    return "white"


def _task_label(total: int, overall: int, slice_spec: str | None) -> str:
    if slice_spec is None:
        return f"{total} tasks"
    return f"{total}/{overall} tasks · slice {slice_spec}"


def _apply_slice(tasks: list, spec: str | None) -> list:
    """Apply a slice spec to a task list and return the sliced result.

    Raises ValueError for unparseable specs so the caller can warn and fall back.
    """
    if spec is None:
        return tasks
    if spec.startswith("-"):
        # Last N tasks
        n = int(spec)
        return tasks[n:]
    elif "-" in spec:
        # Range: start-end (0-indexed)
        start, end = spec.split("-", 1)
        return tasks[int(start) : int(end)]
    else:
        # First N tasks
        return tasks[: int(spec)]


def run(
    host: str,
    models: list[str],
    benchmarks: list[tuple],
    console: Console,
    verbose: bool = False,
) -> list[dict]:
    results = []

    for model in models:
        for bench, slice_spec in benchmarks:
            all_tasks = bench.load_tasks()
            try:
                tasks = _apply_slice(all_tasks, slice_spec)
            except ValueError:
                console.print(
                    f"[yellow]Invalid slice[/yellow] [white]{slice_spec}[/white] "
                    f"[dim]Running all tasks instead.[/dim]"
                )
                tasks = all_tasks
                slice_spec = None

            console.print()
            console.print(
                f"[bold]{bench.name}[/bold] [dim]on[/dim] [white]{model}[/white]"
            )
            console.print(
                f"[dim]{_task_label(len(tasks), len(all_tasks), slice_spec)}[/dim]"
            )

            passed = 0
            total_tokens = 0
            total_eval_ns = 0
            total_response_time = 0.0
            task_details = []

            wall_start = time.time()

            with Progress(
                BarColumn(
                    bar_width=None,
                    style="bright_black",
                    complete_style="white",
                    finished_style="white",
                    pulse_style="bright_black",
                ),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console,
                expand=True,
            ) as progress:
                bar = progress.add_task("", total=len(tasks))

                for task in tasks:
                    prompt = bench.build_prompt(task)
                    gen = generate(host, model, prompt)

                    ok = bench.evaluate(task, gen["response"])
                    if ok:
                        passed += 1

                    if verbose:
                        entry = task.metadata.get("entry_point")
                        label = f"{task.id} ({entry})" if entry else task.id
                        status = (
                            "[bold green]PASS[/bold green]"
                            if ok
                            else "[bold red]FAIL[/bold red]"
                        )
                        console.print(f"[bold]{label}[/bold] [dim]·[/dim] {status}")
                        console.print("[dim]Prompt[/dim]", highlight=False)
                        console.print(prompt, markup=False, highlight=False)
                        console.print("[dim]Response[/dim]", highlight=False)
                        console.print(gen["response"], markup=False, highlight=False)
                        console.print()

                    total_tokens += gen["eval_count"]
                    total_eval_ns += gen["eval_duration_ns"]
                    total_response_time += gen["response_time_s"]

                    task_details.append(
                        {
                            "task_id": task.id,
                            "passed": ok,
                            "tok_s": gen["tok_s"],
                            "response_time_s": gen["response_time_s"],
                            "prompt": prompt,
                            "response": gen["response"],
                        }
                    )

                    progress.advance(bar)

            total_time = round(time.time() - wall_start, 1)
            tok_s = total_tokens / (total_eval_ns / 1e9) if total_eval_ns > 0 else 0.0
            n = len(tasks)
            score = passed / n * 100 if n > 0 else 0.0
            avg_response_time = round(total_response_time / n, 1) if n > 0 else 0.0
            score_style = _score_style(score)

            results.append(
                {
                    "model": model,
                    "benchmark": bench.name,
                    "score": round(score, 1),
                    "passed": passed,
                    "total": n,
                    "tok_s": round(tok_s, 1),
                    "avg_response_time": avg_response_time,
                    "total_time": total_time,
                    "slice": slice_spec,
                    "tasks": task_details,
                }
            )

            console.print(
                f"[dim]Finished[/dim] "
                f"[bold]{bench.name}[/bold] [dim]on[/dim] "
                f"[white]{model}[/white]  "
                f"[bold {score_style}]{score:.1f}%[/bold {score_style}] "
                f"[dim]{passed}/{n} · {tok_s:.1f} tok/s · {_fmt_time(total_time)}[/dim]"
            )

        if len(models) > 1:
            unload_model(host, model)

    return results
