"""Run benchmarks across models and collect results."""

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

from benchkit.ollama import generate, unload_model


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
            task_details = []

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

                    task_details.append(
                        {
                            "task_id": task.id,
                            "passed": ok,
                            "tok_s": gen["tok_s"],
                        }
                    )

                    progress.advance(bar)

            tok_s = total_tokens / (total_eval_ns / 1e9) if total_eval_ns > 0 else 0.0
            score = passed / len(tasks) * 100

            results.append(
                {
                    "model": model,
                    "benchmark": bench.name,
                    "score": round(score, 1),
                    "passed": passed,
                    "total": len(tasks),
                    "tok_s": round(tok_s, 1),
                    "tasks": task_details,
                }
            )

            console.print(
                f"  ✅ {passed}/{len(tasks)} ({score:.1f}%) | {tok_s:.1f} tok/s"
            )

        if len(models) > 1:
            unload_model(host, model)

    return results
