"""Save benchmark results to disk."""

import csv
import json
import os
from datetime import datetime
from importlib.resources import files
from pathlib import Path

from benchkit import artifacts
from benchkit.report_markdown import render_markdown


def _safe_json(data: object) -> str:
    """Serialize data for an HTML script element without allowing tag breakout."""
    return (
        json.dumps(data, ensure_ascii=False)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def render_html(
    results: list[dict],
    generated_at: str,
    provider: str = "",
    host: str = "",
    hardware: str = "",
) -> str:
    """Render the packaged HTML template with embedded report data."""
    payload = _safe_json(
        {
            "generated_at": generated_at,
            "provider": provider,
            "host": host,
            "hardware": hardware,
            "results": results,
        }
    )
    template = (
        files("benchkit").joinpath("templates/report.html").read_text(encoding="utf-8")
    )
    return template.replace("__BENCHKIT_REPORT_DATA__", payload)


def arena_results(results: list[dict]) -> list[dict]:
    """Result rows scored by rendering, whether or not a page ever loaded.

    Selection is by suite, not by outcome: a run whose tasks all timed out
    still gets a gallery, and it still shows those tasks.
    """
    return [
        result
        for result in results
        if result.get("scoring") in {"render-only", "build-metrics"}
        or any(
            isinstance(task.get("workspace"), dict)
            and (
                task["workspace"].get("render_status")
                or task["workspace"].get("build_status")
            )
            for task in result.get("tasks") or []
        )
    ]


def render_arena_html(
    results: list[dict],
    generated_at: str,
    provider: str = "",
    host: str = "",
    hardware: str = "",
) -> str:
    """Render the screenshot gallery for the render-scored suites.

    The gallery lives beside the screenshots and generated pages it links to,
    so a preview opens the model's own page in a new tab instead of a picture
    of it.
    """
    payload = _safe_json(
        {
            "generated_at": generated_at,
            "provider": provider,
            "host": host,
            "hardware": hardware,
            "results": results,
        }
    )
    template = (
        files("benchkit").joinpath("templates/arena.html").read_text(encoding="utf-8")
    )
    return template.replace("__BENCHKIT_ARENA_DATA__", payload)


def save(
    results: list[dict],
    provider: str = "",
    host: str = "",
    hardware: str | None = None,
) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = Path("results") / ts
    out.mkdir(parents=True, exist_ok=True)
    hardware = (
        hardware if hardware is not None else os.environ.get("BENCHKIT_HARDWARE", "")
    )

    # Screenshots and generated pages staged during the run move in here first,
    # so every path written below is relative to the report directory.
    artifacts.collect(out, results)

    # Full JSON (includes per-task details)
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary CSV (one row per model×benchmark, no tasks column)
    summary = [{k: v for k, v in r.items() if k != "tasks"} for r in results]
    if summary:
        with open(out / "results.csv", "w", newline="") as f:
            fieldnames = list(
                dict.fromkeys(key for result in summary for key in result)
            )
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary)

    # Markdown table and task-level details.
    with open(out / "results.md", "w") as f:
        f.write(render_markdown(results, ts))

    with open(out / "results.html", "w") as f:
        f.write(render_html(results, ts, provider, host, hardware))

    # Rendered suites get their own gallery: big previews that open the
    # generated page itself, next to the pass rate.
    if gallery := arena_results(results):
        with open(out / "arena.html", "w") as f:
            f.write(render_arena_html(gallery, ts, provider, host, hardware))

    artifacts.cleanup()
    return out
