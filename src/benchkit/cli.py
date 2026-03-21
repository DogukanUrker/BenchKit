"""BenchKit CLI - interactive benchmark runner."""

import sys

from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from benchkit.benchmarks import REGISTRY
from benchkit.ollama import get_host, list_models
from benchkit.report import save
from benchkit.runner import run

console = Console(highlight=False)


def _fmt_time(s: float) -> str:
    s = int(round(s))
    if s >= 60:
        return f"{s // 60}m {s % 60}s"
    return f"{s}s"


def _section(title: str, subtitle: str | None = None) -> None:
    console.print()
    console.print(title)
    if subtitle:
        console.print(f"[dim]{subtitle}[/dim]")


def _options_table(options: list[str], descriptions: list[str] | None = None) -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(justify="right", style="dim", no_wrap=True)
    table.add_column(style="white")

    if descriptions:
        table.add_column(style="dim")

    table.add_row("0.", "All", *([] if not descriptions else [""]))
    for i, opt in enumerate(options, 1):
        if descriptions:
            table.add_row(f"{i}.", opt, descriptions[i - 1])
        else:
            table.add_row(f"{i}.", opt)

    return table


def _score_text(score: float) -> Text:
    if score >= 80:
        style = "bold green"
    elif score < 50:
        style = "bold red"
    else:
        style = "bold"
    return Text(f"{score:.1f}%", style=style)


def _pick(
    prompt: str, options: list[str], descriptions: list[str] | None = None
) -> list[int]:
    """Prompt user to pick one or more options. Returns selected indices."""
    console.print(_options_table(options, descriptions))
    console.print("[dim]Comma-separated. Use 0 to select all.[/dim]")
    raw = console.input(f"{prompt}: ").strip()
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
    console.print(_options_table(options, descriptions))
    console.print("[dim]Examples: 1,2:10,3:-5,4:5-15[/dim]")
    raw = console.input("Select benchmarks: ").strip()

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

    console.print()
    console.print("[bold]BenchKit[/bold]")
    console.print("[dim]Benchmark your local LLMs[/dim]")

    # Connect to Ollama
    host = get_host()
    _section("Session")
    console.print(f"[dim]Host  {host}[/dim]")
    if verbose:
        console.print("[dim]Verbose output enabled[/dim]")

    try:
        models = list_models(host)
    except Exception as e:
        console.print("[red]Connection failed[/red]")
        console.print(f"[dim]{e}[/dim]")
        console.print("[dim]Set OLLAMA_HOST or start Ollama first.[/dim]")
        sys.exit(1)

    if not models:
        console.print("[red]No models found[/red]")
        console.print("[dim]Pull a model in Ollama first.[/dim]")
        sys.exit(1)

    # Model selection
    console.print(f"[dim]{len(models)} model(s) available[/dim]")
    _section("Models")
    model_names = [m["name"] for m in models]
    model_sizes = [f"{m.get('size', 0) / (1024**3):.1f} GB" for m in models]
    picked = _pick("Select models", model_names, model_sizes)
    if not picked:
        console.print("[red]No models selected[/red]")
        sys.exit(1)
    selected_models = [model_names[i] for i in picked]
    console.print(f"[dim]Selected {len(selected_models)} model(s)[/dim]")

    # Benchmark selection
    _section("Benchmarks")
    bench_names = list(REGISTRY.keys())
    bench_descs = []
    for name in bench_names:
        b = REGISTRY[name]()
        bench_descs.append(f"{len(b.load_tasks())} tasks")
    picked = _pick_benchmarks(bench_names, bench_descs)
    if not picked:
        console.print("[red]No benchmarks selected[/red]")
        sys.exit(1)
    selected_benchmarks = [
        (REGISTRY[bench_names[i]](), slice_spec) for i, slice_spec in picked
    ]
    console.print(f"[dim]Selected {len(selected_benchmarks)} benchmark run(s)[/dim]")

    # Run
    _section("Run")
    results = run(host, selected_models, selected_benchmarks, console, verbose)

    # Summary table
    _section("Results")
    table = Table(
        box=box.MINIMAL,
        border_style="dim",
        header_style="bold",
        pad_edge=False,
        padding=(0, 1),
    )
    table.add_column("Model", style="white")
    table.add_column("Benchmark")
    table.add_column("Score", justify="right")
    table.add_column("Passed", justify="right")
    table.add_column("tok/s", justify="right", style="dim")
    table.add_column("Avg Time", justify="right", style="dim")
    table.add_column("Total", justify="right", style="dim")

    for r in results:
        table.add_row(
            r["model"],
            r["benchmark"],
            _score_text(r["score"]),
            f"{r['passed']}/{r['total']}",
            str(r["tok_s"]),
            f"{r['avg_response_time']}s",
            _fmt_time(r["total_time"]),
        )

    console.print(table)

    # Save
    out = save(results)
    console.print(f"[dim]Saved:[/dim] [white]{out}[/white]")


if __name__ == "__main__":
    main()
