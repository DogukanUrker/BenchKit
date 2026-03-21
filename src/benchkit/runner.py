"""Run benchmarks across models and collect results."""

import time

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

from benchkit.ollama import generate, unload_model


def _fmt_time(s: float) -> str:
    s = int(round(s))
    if s >= 60:
        return f"{s // 60}m {s % 60}s"
    return f"{s}s"


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
                    f"[yellow]⚠ Invalid slice '{slice_spec}', running all tasks.[/yellow]"
                )
                tasks = all_tasks
                slice_spec = None

            if slice_spec is not None:
                console.print(
                    f"\n⏳ Running [bold]{bench.name}[/bold] on [cyan]{model}[/cyan] "
                    f"({len(tasks)}/{len(all_tasks)} tasks)"
                )
            else:
                console.print(
                    f"\n⏳ Running [bold]{bench.name}[/bold] on [cyan]{model}[/cyan] "
                    f"({len(tasks)} tasks)"
                )

            passed = 0
            total_tokens = 0
            total_eval_ns = 0
            total_response_time = 0.0
            task_details = []

            wall_start = time.time()

            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                console=console,
            ) as progress:
                bar = progress.add_task("Evaluating", total=len(tasks))

                for task in tasks:
                    prompt = bench.build_prompt(task)
                    gen = generate(host, model, prompt)

                    ok = bench.evaluate(task, gen["response"])
                    if ok:
                        passed += 1

                    if verbose:
                        entry = task.metadata.get("entry_point")
                        label = f"{task.id} ({entry})" if entry else task.id
                        status = "[green]PASS[/green]" if ok else "[red]FAIL[/red]"
                        console.rule(f"[bold]{label}[/bold] ── {status}")
                        console.print("PROMPT:", highlight=False)
                        console.print(prompt, markup=False, highlight=False)
                        console.print("\nRESPONSE:", highlight=False)
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
                f"  ✅ {passed}/{n} ({score:.1f}%) | "
                f"{tok_s:.1f} tok/s | {_fmt_time(total_time)}"
            )

        if len(models) > 1:
            unload_model(host, model)

    return results
