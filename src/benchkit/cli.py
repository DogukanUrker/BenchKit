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

from benchkit.benchmarks import REGISTRY, all_tags, keys_for_tags, tags_for
from benchkit.client import InferenceClient
from benchkit.engine import (
    JobSpec,
    SliceError,
    expand_jobs,
    parse_slice,
    slice_task_count,
    task_count,
)
from benchkit.metrics import aggregate_tok_s, effective_concurrency, stream_tok_s
from benchkit.perf import DEFAULT_DEPTHS, PerfConfig, parse_depths, run_profile
from benchkit.perf_report import save_profile
from benchkit.perturbations import PERTURBATIONS, perturbations_for
from benchkit.report import save
from benchkit.runner import run
from benchkit.tui import run_tui

console = Console(highlight=False)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="benchkit",
        description="Benchmark local LLMs with real evaluation suites.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("perf",),
        help="Run a dedicated model performance profile",
    )
    parser.add_argument(
        "perf_model",
        nargs="?",
        metavar="MODEL",
        help="Model name for `benchkit perf MODEL`",
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
        "--tag",
        default="",
        help=(
            "Select benchmarks by tag, e.g. --tag code or --tag mcq,-saturated. "
            "Combines with --benchmarks; a per-benchmark slice can be appended "
            "with --benchmarks. Tags: " + " ".join(all_tags())
        ),
    )
    parser.add_argument(
        "--perturbation",
        choices=PERTURBATIONS,
        help=("Headless: add a paired perturbed run alongside each clean baseline"),
    )
    parser.add_argument(
        "--perturbation-seed",
        type=int,
        default=42,
        metavar="SEED",
        help="Headless: deterministic perturbation seed (default: 42)",
    )
    parser.add_argument(
        "--harness",
        choices=("direct", "pi", "both"),
        default="direct",
        help=(
            "Execution harness: raw API, latest stock Pi in Docker, or paired "
            "direct + Pi runs (default: direct)"
        ),
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
    parser.add_argument(
        "--depths",
        default=",".join(
            "minimal" if value == 0 else f"{value // 1024}k" for value in DEFAULT_DEPTHS
        ),
        help="Perf: comma-separated context depths (default: minimal,4k,16k,32k,64k)",
    )
    parser.add_argument(
        "--gen",
        type=int,
        default=128,
        metavar="TOKENS",
        help="Perf: generated tokens per measured run (default: 128)",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=5,
        help="Perf: measured repetitions per depth (default: 5)",
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=1,
        help="Perf: discarded warmups per depth (default: 1)",
    )
    parser.add_argument(
        "--config-note",
        default="",
        help="Perf: optional hardware or server configuration note",
    )
    return parser.parse_args(argv)


def _fmt_time(seconds: float) -> str:
    seconds = round(seconds)
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


def _outcome_counts(result: dict) -> tuple[int, int, int]:
    """Return pass, ordinary failure and error-task counts for a result."""
    tasks = result.get("tasks", [])
    failed = sum(not task.get("passed") and not task.get("error") for task in tasks)
    errors = sum(not task.get("passed") and bool(task.get("error")) for task in tasks)
    return int(result["passed"]), failed, errors


def _split_tags(value: str) -> tuple[list[str], list[str]]:
    """Split a --tag value into required and excluded tags."""
    include: list[str] = []
    exclude: list[str] = []
    for term in value.replace(" ", ",").split(","):
        term = term.strip().lower()
        if not term:
            continue
        if term.startswith("-"):
            exclude.append(term[1:])
        else:
            include.append(term)
    return include, exclude


def _keys_for_tag_option(value: str) -> list[str]:
    """Resolve a --tag value to benchmark keys, or exit with a clear error."""
    include, exclude = _split_tags(value)
    known = set(all_tags())
    unknown = [tag for tag in include + exclude if tag not in known]
    if unknown:
        console.print(f"[red]Unknown tag(s):[/red] {', '.join(unknown)}")
        console.print(f"[dim]Available: {', '.join(sorted(known))}[/dim]")
        sys.exit(1)

    keys = keys_for_tags(include, exclude)
    if not keys:
        console.print(f"[red]No benchmark matches --tag {value}[/red]")
        sys.exit(1)
    return keys


def _list_benchmarks(tag: str = "") -> None:
    keys = _keys_for_tag_option(tag) if tag.strip() else list(REGISTRY)
    table = Table(box=box.MINIMAL, border_style="dim", header_style="bold")
    table.add_column("Benchmark")
    table.add_column("Tasks", justify="right")
    table.add_column("Tags")
    table.add_column("Perturbations")
    table.add_column("Notes")
    for key in keys:
        try:
            declared_label = getattr(REGISTRY[key], "list_task_count", "")
            count = declared_label or f"{task_count(key):,}"
        except Exception:
            count = "?"
        note = getattr(REGISTRY[key], "list_note", "")
        table.add_row(
            key,
            count,
            f"[dim]{' '.join(tags_for(key))}[/dim]",
            f"[cyan]{' '.join(perturbations_for(key))}[/cyan]",
            f"[yellow]{note}[/yellow]" if note else "",
        )
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

    requested = [part for part in args.benchmarks.split(",") if part.strip()]
    if args.tag.strip():
        tagged = _keys_for_tag_option(args.tag)
        explicit = {part.partition(":")[0].strip() for part in requested}
        requested += [key for key in tagged if key not in explicit]

    specs: list[tuple[str, str | None]] = []
    for part in requested:
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
            total = slice_task_count(key)
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
        console.print("[red]--benchmarks or --tag is required with --headless[/red]")
        sys.exit(1)

    if args.perturbation:
        unsupported = [
            key for key, _ in specs if args.perturbation not in perturbations_for(key)
        ]
        if unsupported:
            unique = list(dict.fromkeys(unsupported))
            console.print(
                f"[red]{args.perturbation} is unsupported for:[/red] "
                + ", ".join(unique)
            )
            supported = [
                key for key in REGISTRY if args.perturbation in perturbations_for(key)
            ]
            console.print(f"[dim]Supported: {', '.join(supported)}[/dim]")
            sys.exit(1)

    harnesses = ("direct", "pi") if args.harness == "both" else (args.harness,)
    jobs: list[JobSpec] = []
    for model in models:
        for key, spec in specs:
            for harness in harnesses:
                jobs.append(JobSpec(model, key, spec, harness=harness))
                if args.perturbation:
                    jobs.append(
                        JobSpec(
                            model,
                            key,
                            spec,
                            perturbation=args.perturbation,
                            perturbation_seed=args.perturbation_seed,
                            harness=harness,
                        )
                    )
    return jobs


def _headless(args: argparse.Namespace) -> None:
    if args.demo and args.harness != "direct":
        console.print("[red]Pi harness is unavailable in demo mode.[/red]")
        sys.exit(1)
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

    jobs = expand_jobs(
        _headless_jobs(args, [model["name"] for model in models]),
        client,
    )
    if not jobs:
        console.print(
            "[red]No supported RULER context bucket fits the selected model(s).[/red]"
        )
        sys.exit(1)
    if args.demo:
        perturbations = sorted(
            {
                (job.perturbation, job.perturbation_seed)
                for job in jobs
                if job.perturbation
            }
        )
        client.prime(
            sorted({job.benchmark for job in jobs}),
            perturbations=perturbations,
        )

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
    # Fold long model and benchmark names instead of ellipsizing them: the
    # summary is the part of a headless run people copy into a report.
    table.add_column("Model", overflow="fold")
    table.add_column("Harness")
    table.add_column("Benchmark", overflow="fold")
    table.add_column("Score", justify="right")
    has_harness_pairs = any(
        result.get("harness_delta_pp") is not None for result in results
    )
    if has_harness_pairs:
        table.add_column("Harness Δ", justify="right")
    has_perturbations = any(result.get("perturbation") for result in results)
    if has_perturbations:
        table.add_column("Delta", justify="right")
        table.add_column("Regress", justify="right")
    table.add_column("P/F/E", justify="right")
    table.add_column("Loops/Killed", justify="right")
    has_parallel = any(result.get("concurrency", 1) > 1 for result in results)
    if has_parallel:
        table.add_column("Parallel", justify="right", style="dim")
        table.add_column("Agg tok/s", justify="right")
        table.add_column("Stream tok/s", justify="right", style="dim")
        table.add_column("Eff", justify="right", style="dim")
    else:
        table.add_column("Tok/s", justify="right")
    table.add_column("Avg Time", justify="right", style="dim")
    table.add_column("Wall", justify="right", style="dim")

    for result in results:
        passed, failed, errors = _outcome_counts(result)
        parallel = result.get("concurrency", 1) > 1
        row = [
            result["model"],
            result.get("harness_label", "Direct"),
            result.get("benchmark_label", result["benchmark"]),
            _score_text(result["score"]),
        ]
        if has_harness_pairs:
            harness_delta = result.get("harness_delta_pp")
            row.append(
                f"{float(harness_delta):+.1f}pp" if harness_delta is not None else "—"
            )
        if has_perturbations:
            delta = result.get("score_delta_pp")
            row.extend(
                [
                    f"{float(delta):+.1f}pp" if delta is not None else "—",
                    (
                        f"{result.get('regressions', 0)}/"
                        f"{result.get('paired_total', 0)}"
                        if result.get("perturbation")
                        else "—"
                    ),
                ]
            )
        row.extend(
            [
                f"{passed}/{failed}/{errors}",
                f"{result.get('loops', 0)}/{result.get('loop_kills', 0)}",
            ]
        )
        if has_parallel:
            row.extend(
                [
                    str(result.get("concurrency", 1)),
                    f"{aggregate_tok_s(result):.1f}" if parallel else "—",
                    f"{stream_tok_s(result):.1f}",
                    f"{effective_concurrency(result):.2f}x" if parallel else "—",
                ]
            )
        else:
            row.append(f"{stream_tok_s(result):.1f}")
        row.extend(
            [
                f"{result['avg_response_time']}s",
                _fmt_time(result["total_time"]),
            ]
        )
        table.add_row(
            *row,
        )

    console.print()
    console.print(table)

    out = save(results, provider=getattr(client, "label", ""), host=client.host)
    console.print(f"[dim]Saved:[/dim] [white]{out}[/white]")
    if failure:
        sys.exit(1)


def _perf(args: argparse.Namespace) -> None:
    if args.demo:
        console.print("[red]Performance profiling is unavailable in demo mode.[/red]")
        sys.exit(1)
    if not args.perf_model:
        console.print("[red]A model name is required:[/red] benchkit perf MODEL")
        sys.exit(1)
    if args.gen < 1 or args.reps < 1 or args.warmups < 0:
        console.print(
            "[red]Perf settings require --gen and --reps above zero, "
            "and --warmups at least zero.[/red]"
        )
        sys.exit(1)
    try:
        depths = parse_depths(args.depths)
        client = _client(args)
    except ValueError as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        sys.exit(1)

    console.print(f"[bold]BenchKit perf[/bold] [dim]{client.host}[/dim]")
    try:
        models = client.list_models()
    except Exception as exc:
        console.print(f"[red]Connection failed:[/red] {exc}")
        sys.exit(1)

    model_meta = next(
        (model for model in models if model.get("name") == args.perf_model),
        None,
    )
    if model_meta is None:
        console.print(f"[red]Unknown model:[/red] {args.perf_model}")
        console.print(
            "[dim]Available: "
            + ", ".join(str(model.get("name")) for model in models)
            + "[/dim]"
        )
        sys.exit(1)
    if client.provider != "openai":
        console.print(
            "[red]Perf v1 requires an OpenAI-compatible llama.cpp/llama-swap "
            "endpoint.[/red]"
        )
        sys.exit(1)

    model_status = str(model_meta.get("status") or "").strip().lower()
    if model_status == "unloaded":
        console.print(
            "[yellow]Model status: unloaded[/yellow] "
            "[dim]· the first request may measure a cold start[/dim]"
        )
    elif model_status == "loaded":
        console.print(
            "[green]Model status: loaded[/green] "
            "[dim]· the model is already resident, so this is a warm test[/dim]"
        )
    elif model_status:
        console.print(
            f"[dim]Model status: {model_status} · cold/warm state is not known[/dim]"
        )
    else:
        console.print(
            "[dim]Model status: unavailable · cold/warm state is not known[/dim]"
        )

    config = PerfConfig(
        model=args.perf_model,
        depths=depths,
        gen_tokens=args.gen,
        reps=args.reps,
        warmups=args.warmups,
        config_note=args.config_note,
    )
    try:
        profile = run_profile(
            client,
            config,
            model_meta=model_meta,
            progress=lambda message: console.print(f"[dim]· {message}[/dim]"),
        )
    except Exception as exc:
        console.print(f"[red]Performance profile failed:[/red] {exc}")
        sys.exit(1)

    table = Table(
        box=box.MINIMAL,
        border_style="dim",
        header_style="bold",
        pad_edge=False,
    )
    table.add_column("Depth")
    table.add_column("Input", justify="right")
    table.add_column("PP tok/s", justify="right")
    table.add_column("TG tok/s", justify="right")
    table.add_column("TTFT", justify="right")
    table.add_column("Wall", justify="right")
    table.add_column("Overhead", justify="right")
    table.add_column("Runs", justify="right")
    for case in profile["cases"]:
        inaccurate = not case.get("depth_within_tolerance", True)
        depth = case["depth_label"]
        if inaccurate:
            depth = f"[yellow]{depth}![/yellow]"
        table.add_row(
            depth,
            f"{case['actual_input_tokens']:,}" if case["actual_input_tokens"] else "—",
            (f"{case['pp_tps']['median']:.1f}" if case.get("pp_reliable") else "n/a"),
            f"{case['tg_tps']['median']:.1f}",
            f"{case['ttft_s']['median']:.3f}s",
            f"{case['wall_time_s']['median']:.2f}s",
            (
                f"{case['overhead_s']['median']:.3f}s"
                if case.get("overhead_available")
                else "—"
            ),
            f"{case['successful_reps']}/{args.reps}",
        )
    console.print()
    console.print(table)

    sweep = profile.get("sweep_summary") or {}
    console.print(
        "[dim]Sweep median · "
        f"TG {float(sweep.get('tg_tps_median') or 0):.1f} tok/s · "
        f"PP {float(sweep.get('pp_tps_median') or 0):.1f} tok/s"
        "[/dim]"
    )

    calibrated = [
        case for case in profile["cases"] if case.get("depth_method") == "calibrated"
    ]
    inaccurate = [
        case
        for case in profile["cases"]
        if not case.get("depth_within_tolerance", True)
    ]
    estimated = [
        case for case in profile["cases"] if case.get("depth_method") == "estimated"
    ]
    if calibrated:
        console.print(
            "[dim]Depth method: calibrated · native tokenizer is not exposed "
            "by the server; server-reported input tokens were used.[/dim]"
        )
    if inaccurate:
        labels = ", ".join(case["depth_label"] for case in inaccurate)
        console.print(
            f"[bold yellow]! Depth tolerance not reached for: {labels}. "
            "Use the Actual Input column when comparing results.[/bold yellow]"
        )
    if estimated:
        labels = ", ".join(case["depth_label"] for case in estimated)
        console.print(
            f"[bold yellow]Depth calibration unavailable for: {labels}. "
            "Use the Actual Input column.[/bold yellow]"
        )

    drift = profile.get("drift_check") or {}
    if drift.get("error"):
        console.print(f"[yellow]Drift check failed:[/yellow] {drift['error']}")
    elif drift:
        message = (
            f"Drift check · {drift['depth_label']} TG "
            f"{drift['initial_tg_tps']:.1f} → {drift['final_tg_tps']:.1f} tok/s "
            f"({drift['change']:+.1%})"
            + ("" if drift.get("stable") else " · possible warmup/drift anomaly")
        )
        console.print(
            f"[dim]{message}[/dim]"
            if drift.get("stable")
            else f"[bold yellow]{message}[/bold yellow]"
        )

    output = save_profile(profile)
    console.print(
        f"[dim]Saved:[/dim] [white]{output}[/white] "
        "[dim]· perf.json · csv · md · html[/dim]"
    )
    console.print(
        "[dim]Tip: Run the ruler benchmark to verify how much of the configured "
        "context window is actually usable.[/dim]"
    )


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    if args.list:
        _list_benchmarks(args.tag)
        return

    if args.command == "perf":
        _perf(args)
        return

    if args.headless:
        _headless(args)
        return

    run_tui(demo=args.demo, host=args.host)


if __name__ == "__main__":
    main()
