"""BenchKit CLI - interactive benchmark runner."""

import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from benchkit.benchmarks import REGISTRY
from benchkit.ollama import get_host, list_models
from benchkit.report import save
from benchkit.runner import run

console = Console()


def _fmt_time(s: float) -> str:
    s = int(round(s))
    if s >= 60:
        return f"{s // 60}m {s % 60}s"
    return f"{s}s"


def _pick(
    prompt: str, options: list[str], descriptions: list[str] | None = None
) -> list[int]:
    """Prompt user to pick one or more options. Returns selected indices."""
    console.print("  [bold cyan]0.[/bold cyan] All", highlight=False)
    for i, opt in enumerate(options, 1):
        desc = f"  ({descriptions[i - 1]})" if descriptions else ""
        console.print(
            f"  [bold cyan]{i}.[/bold cyan] [white]{opt}[/white]{desc}", highlight=False
        )

    raw = console.input(f"\n{prompt} (comma-separated): ").strip()
    indices = []
    for part in raw.split(","):
        part = part.strip()
        if part == "0":
            return list(range(len(options)))
        if part.isdigit() and 1 <= int(part) <= len(options):
            indices.append(int(part) - 1)

    return indices


def _pick_benchmarks(
    options: list[str], descriptions: list[str] | None = None
) -> list[tuple[int, str | None]]:
    """Prompt user to pick benchmarks with optional slice specs.

    Returns list of (zero-based index, slice_spec) tuples where slice_spec
    is None (all tasks) or a string like "10", "-10", "5-15".
    """
    console.print("  [bold cyan]0.[/bold cyan] All", highlight=False)
    for i, opt in enumerate(options, 1):
        desc = f"  ({descriptions[i - 1]})" if descriptions else ""
        console.print(
            f"  [bold cyan]{i}.[/bold cyan] [white]{opt}[/white]{desc}", highlight=False
        )

    raw = console.input(
        "\nSelect benchmarks (comma-separated, e.g. 1,2:10,3:-5): "
    ).strip()

    picked: list[tuple[int, str | None]] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue

        if ":" in part:
            num_str, slice_spec = part.split(":", 1)
            num_str = num_str.strip()
            slice_spec = slice_spec.strip() or None
        else:
            num_str = part
            slice_spec = None

        if num_str == "0":
            return [(i, None) for i in range(len(options))]

        if num_str.isdigit() and 1 <= int(num_str) <= len(options):
            picked.append((int(num_str) - 1, slice_spec))

    return picked


def main() -> None:
    load_dotenv()
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    console.print("\n[bold blue]BenchKit[/bold blue] - Benchmark your local LLMs\n")

    # Connect to Ollama
    host = get_host()
    console.print(f"Connecting to [cyan]{host}[/cyan]")

    try:
        models = list_models(host)
    except Exception as e:
        console.print(f"[red]✗ Cannot reach Ollama at {host}[/red]")
        console.print(f"  {e}")
        console.print("\n  Set OLLAMA_HOST or start Ollama first.")
        sys.exit(1)

    if not models:
        console.print("[red]No models found. Pull some models first.[/red]")
        sys.exit(1)

    # Model selection
    console.print(f"\n[green]✓[/green] Found {len(models)} model(s)\n")
    model_names = [m["name"] for m in models]
    model_sizes = [
        f"[cyan]{m.get('size', 0) / (1024**3):.1f}[/cyan] GB" for m in models
    ]
    picked = _pick("Select models", model_names, model_sizes)
    if not picked:
        console.print("[red]No models selected.[/red]")
        sys.exit(1)
    selected_models = [model_names[i] for i in picked]

    # Benchmark selection
    console.print()
    bench_names = list(REGISTRY.keys())
    bench_descs = []
    for name in bench_names:
        b = REGISTRY[name]()
        bench_descs.append(f"{len(b.load_tasks())} tasks")
    picked = _pick_benchmarks(bench_names, bench_descs)
    if not picked:
        console.print("[red]No benchmarks selected.[/red]")
        sys.exit(1)
    selected_benchmarks = [
        (REGISTRY[bench_names[i]](), slice_spec) for i, slice_spec in picked
    ]

    # Run
    results = run(host, selected_models, selected_benchmarks, console, verbose)

    # Summary table
    console.print()
    table = Table(title="Results")
    table.add_column("Model", style="cyan")
    table.add_column("Benchmark")
    table.add_column("Score", justify="right", style="green")
    table.add_column("Passed", justify="right")
    table.add_column("tok/s", justify="right", style="yellow")
    table.add_column("Avg Resp", justify="right", style="yellow")
    table.add_column("Total Time", justify="right")

    for r in results:
        table.add_row(
            r["model"],
            r["benchmark"],
            f"{r['score']}%",
            f"{r['passed']}/{r['total']}",
            str(r["tok_s"]),
            f"{r['avg_response_time']}s",
            _fmt_time(r["total_time"]),
        )

    console.print(table)

    # Save
    out = save(results)
    console.print(f"\n📁 Results saved to [cyan]{out}[/cyan]")


if __name__ == "__main__":
    main()
