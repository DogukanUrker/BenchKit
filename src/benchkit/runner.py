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


def run(
    host: str,
    models: list[str],
    benchmarks: list,
    console: Console,
) -> list[dict]:
    results = []

    for model in models:
        for bench in benchmarks:
            tasks = bench.load_tasks()
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

                    total_tokens += gen["eval_count"]
                    total_eval_ns += gen["eval_duration_ns"]
                    total_response_time += gen["response_time_s"]

                    task_details.append(
                        {
                            "task_id": task.id,
                            "passed": ok,
                            "tok_s": gen["tok_s"],
                            "response_time_s": gen["response_time_s"],
                        }
                    )

                    progress.advance(bar)

            total_time = round(time.time() - wall_start, 1)
            tok_s = total_tokens / (total_eval_ns / 1e9) if total_eval_ns > 0 else 0.0
            score = passed / len(tasks) * 100
            avg_response_time = (
                round(total_response_time / len(tasks), 1) if tasks else 0.0
            )

            results.append(
                {
                    "model": model,
                    "benchmark": bench.name,
                    "score": round(score, 1),
                    "passed": passed,
                    "total": len(tasks),
                    "tok_s": round(tok_s, 1),
                    "avg_response_time": avg_response_time,
                    "total_time": total_time,
                    "tasks": task_details,
                }
            )

            console.print(
                f"  ✅ {passed}/{len(tasks)} ({score:.1f}%) | "
                f"{tok_s:.1f} tok/s | {_fmt_time(total_time)}"
            )

        if len(models) > 1:
            unload_model(host, model)

    return results
