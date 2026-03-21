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


def main() -> None:
    load_dotenv()

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
    picked = _pick("Select benchmarks", bench_names, bench_descs)
    if not picked:
        console.print("[red]No benchmarks selected.[/red]")
        sys.exit(1)
    selected_benchmarks = [REGISTRY[bench_names[i]]() for i in picked]

    # Run
    results = run(host, selected_models, selected_benchmarks, console)

    # Summary table
    console.print()
    table = Table(title="Results")
    table.add_column("Model", style="cyan")
    table.add_column("Benchmark")
    table.add_column("Score", justify="right", style="green")
    table.add_column("Passed", justify="right")
    table.add_column("tok/s", justify="right", style="yellow")

    for r in results:
        table.add_row(
            r["model"],
            r["benchmark"],
            f"{r['score']}%",
            f"{r['passed']}/{r['total']}",
            str(r["tok_s"]),
        )

    console.print(table)

    # Save
    out = save(results)
    console.print(f"\n📁 Results saved to [cyan]{out}[/cyan]")


if __name__ == "__main__":
    main()
