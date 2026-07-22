"""Save benchmark results to disk."""

import csv
import json
import os
from datetime import datetime
from importlib.resources import files
from pathlib import Path


def _fmt_time(s: float) -> str:
    s = int(round(s))
    if s >= 60:
        return f"{s // 60}m {s % 60}s"
    return f"{s}s"


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
        files("benchkit")
        .joinpath("templates/report.html")
        .read_text(encoding="utf-8")
    )
    return template.replace("__BENCHKIT_REPORT_DATA__", payload)


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
        hardware
        if hardware is not None
        else os.environ.get("BENCHKIT_HARDWARE", "")
    )

    # Full JSON (includes per-task details)
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary CSV (one row per model×benchmark, no tasks column)
    summary = [{k: v for k, v in r.items() if k != "tasks"} for r in results]
    if summary:
        with open(out / "results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary[0].keys())
            writer.writeheader()
            writer.writerows(summary)

    # Markdown table and task-level details.
    with open(out / "results.md", "w") as f:
        f.write("# BenchKit Results\n\n")
        f.write(f"**Date:** {ts}\n\n")
        f.write(
            "| Model | Benchmark | Score | Passed | Total | tok/s | Avg Resp | Total Time |\n"
        )
        f.write(
            "|-------|-----------|-------|--------|-------|-------|----------|------------|\n"
        )
        for result in results:
            f.write(
                f"| {result['model']} | {result['benchmark']} "
                f"| {result['score']}% | {result['passed']} "
                f"| {result['total']} | {result['tok_s']} "
                f"| {result['avg_response_time']}s "
                f"| {_fmt_time(result['total_time'])} |\n"
            )

        f.write("\n---\n\n")
        for result in results:
            f.write(f"## {result['model']} / {result['benchmark']}\n\n")
            for task in result["tasks"]:
                task_id = task["task_id"]
                entry = task.get("entry_point")
                label = f"{task_id} ({entry})" if entry else task_id
                status = "✅ PASS" if task["passed"] else "❌ FAIL"
                f.write(f"### {label} — {status}\n\n")
                f.write("**Prompt:**\n\n")
                f.write(f"~~~\n{task['prompt'].rstrip()}\n~~~\n\n")
                f.write("**Response:**\n\n")
                f.write(f"~~~\n{task['response'].rstrip()}\n~~~\n\n")
                f.write("---\n\n")

    with open(out / "results.html", "w") as f:
        f.write(render_html(results, ts, provider, host, hardware))

    return out
