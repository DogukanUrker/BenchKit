"""BenchKit entry point.

Running `benchkit` opens the full-screen terminal app. `--headless` keeps a
scriptable path that prints to stdout instead.
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from benchkit.benchmarks import REGISTRY
from benchkit.client import InferenceClient
from benchkit.engine import JobSpec, SliceError, parse_slice, task_count
from benchkit.report import save
from benchkit.runner import run
from benchkit.tui import run_tui

console = Console(highlight=False)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="benchkit",
        description="Benchmark local LLMs with real evaluation suites.",
    )
    parser.add_argument("--host", help="Inference host, overrides BENCHKIT_HOST")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run against a built-in offline model server (no inference server needed)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Skip the TUI and run from flags, printing progress to stdout",
    )
    parser.add_argument(
        "--models",
        default="",
        help="Headless: comma separated model names, or 'all'",
    )
    parser.add_argument(
        "--benchmarks",
        default="",
        help="Headless: comma separated benchmarks, e.g. humaneval:20,gsm8k:-50",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the available benchmarks and exit",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Headless: print every prompt and response",
    )
    return parser.parse_args(argv)


def _fmt_time(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds >= 60:
        return f"{seconds // 60}m {seconds % 60}s"
    return f"{seconds}s"


def _score_text(score: float) -> Text:
    if score >= 80:
        style = "bold green"
    elif score < 50:
        style = "bold red"
    else:
        style = "bold"
    return Text(f"{score:.1f}%", style=style)


def _list_benchmarks() -> None:
    table = Table(box=box.MINIMAL, border_style="dim", header_style="bold")
    table.add_column("Benchmark")
    table.add_column("Tasks", justify="right")
    for key in REGISTRY:
        try:
            count = f"{task_count(key):,}"
        except Exception:
            count = "?"
        table.add_row(key, count)
    console.print(table)


def _client(args: argparse.Namespace):
    if args.demo:
        from benchkit.demo import DemoClient

        return DemoClient()

    client = InferenceClient.from_env()
    if args.host:
        client.host = args.host.rstrip("/")
    return client


def _headless_jobs(args: argparse.Namespace, available: list[str]) -> list[JobSpec]:
    if args.models.strip().lower() in {"all", "*"}:
        models = available
    else:
        models = [name.strip() for name in args.models.split(",") if name.strip()]

    unknown = [name for name in models if name not in available]
    if unknown:
        console.print(f"[red]Unknown model(s):[/red] {', '.join(unknown)}")
        console.print(f"[dim]Available: {', '.join(available)}[/dim]")
        sys.exit(1)
    if not models:
        console.print("[red]--models is required with --headless[/red]")
        sys.exit(1)

    specs: list[tuple[str, str | None]] = []
    for part in args.benchmarks.split(","):
        part = part.strip()
        if not part:
            continue
        key, _, slice_spec = part.partition(":")
        key = key.strip()
        slice_spec = slice_spec.strip() or None
        if key not in REGISTRY:
            console.print(f"[red]Unknown benchmark:[/red] {key}")
            console.print(f"[dim]Available: {', '.join(REGISTRY)}[/dim]")
            sys.exit(1)
        try:
            total = task_count(key)
        except Exception as exc:
            console.print(f"[red]Could not load {key}:[/red] {exc}")
            sys.exit(1)
        try:
            parse_slice(slice_spec, total)
        except SliceError as exc:
            console.print(f"[red]{exc}[/red]")
            sys.exit(1)
        specs.append((key, slice_spec))

    if not specs:
        console.print("[red]--benchmarks is required with --headless[/red]")
        sys.exit(1)

    return [JobSpec(model, key, spec) for model in models for key, spec in specs]


def _headless(args: argparse.Namespace) -> None:
    try:
        client = _client(args)
    except ValueError as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        sys.exit(1)

    console.print(f"[bold]BenchKit[/bold] [dim]{client.host}[/dim]")

    try:
        models = client.list_models()
    except Exception as exc:
        console.print(f"[red]Connection failed:[/red] {exc}")
        sys.exit(1)

    jobs = _headless_jobs(args, [model["name"] for model in models])
    if args.demo:
        client.prime(sorted({job.benchmark for job in jobs}))

    results, failure = run(client, jobs, console, args.verbose)
    if not results:
        console.print("[yellow]No results.[/yellow]")
        sys.exit(1 if failure else 0)

    table = Table(
        box=box.MINIMAL,
        border_style="dim",
        header_style="bold",
        pad_edge=False,
        padding=(0, 1),
    )
    table.add_column("Model")
    table.add_column("Benchmark")
    table.add_column("Score", justify="right")
    table.add_column("Passed", justify="right")
    table.add_column("tok/s", justify="right", style="dim")
    table.add_column("Avg Time", justify="right", style="dim")
    table.add_column("Total", justify="right", style="dim")

    for result in results:
        table.add_row(
            result["model"],
            result["benchmark"],
            _score_text(result["score"]),
            f"{result['passed']}/{result['total']}",
            str(result["tok_s"]),
            f"{result['avg_response_time']}s",
            _fmt_time(result["total_time"]),
        )

    console.print()
    console.print(table)

    out = save(results, provider=getattr(client, "label", ""), host=client.host)
    console.print(f"[dim]Saved:[/dim] [white]{out}[/white]")
    if failure:
        sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    if args.list:
        _list_benchmarks()
        return

    if args.headless:
        _headless(args)
        return

    run_tui(demo=args.demo, host=args.host)


if __name__ == "__main__":
    main()
