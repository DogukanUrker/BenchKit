"""Save benchmark results to disk."""

import csv
import json
import os
from datetime import datetime
from importlib.resources import files
from pathlib import Path

from benchkit.metrics import aggregate_tok_s, effective_concurrency, stream_tok_s


def _fmt_time(s: float) -> str:
    s = round(s)
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
        files("benchkit").joinpath("templates/report.html").read_text(encoding="utf-8")
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
        hardware if hardware is not None else os.environ.get("BENCHKIT_HARDWARE", "")
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
            "| Model | Benchmark | Parallel | Score | Passed | Total | Loops | Killed | Trace | Agg tok/s | Stream tok/s | Effective | Avg Resp | Wall Time |\n"
        )
        f.write(
            "|-------|-----------|----------|-------|--------|-------|-------|--------|-------|-----------|--------------|-----------|----------|-----------|\n"
        )
        for result in results:
            f.write(
                f"| {result['model']} | {result['benchmark']} "
                f"| {result.get('concurrency', 1)} | {result['score']}% "
                f"| {result['passed']} "
                f"| {result['total']} | {result.get('loop_rate', 0)}% "
                f"| {result.get('loop_kills', 0)} "
                f"| {result.get('trace_coverage', 0)}% "
                f"| {aggregate_tok_s(result):.1f} "
                f"| {stream_tok_s(result):.1f} "
                f"| {effective_concurrency(result):.2f}x "
                f"| {result['avg_response_time']}s "
                f"| {_fmt_time(result['total_time'])} |\n"
            )

        f.write(
            "\n**Throughput:** Agg tok/s is total output tokens divided by job "
            "wall time. Stream tok/s uses summed server decode time. Effective "
            "concurrency is summed request time divided by wall time.\n"
        )

        f.write("\n---\n\n")
        for result in results:
            f.write(f"## {result['model']} / {result['benchmark']}\n\n")
            for task in result["tasks"]:
                task_id = task["task_id"]
                entry = task.get("entry_point")
                label = f"{task_id} ({entry})" if entry else task_id
                status = (
                    "🛑 LOOP KILLED"
                    if task.get("loop_killed")
                    else "⏱️ TIMEOUT"
                    if task.get("timed_out")
                    else "⚠️ ERROR"
                    if task.get("error")
                    else "✅ PASS"
                    if task["passed"]
                    else "❌ FAIL"
                )
                loop = (
                    "RECOVERED"
                    if task.get("recovered_cycle")
                    else task.get("loop_state", "unavailable").upper()
                )
                if (
                    loop == "CLEAR"
                    and task.get("loop_source") == "answer"
                    and task.get("trace_status") == "unavailable"
                ):
                    loop = "NO TRACE"
                f.write(f"### {label} — {status} · {loop}\n\n")
                f.write(
                    f"**Generation:** {task.get('output_tokens', 0)} tokens · "
                    f"{task.get('tok_s', 0)} stream tok/s · "
                    f"{task.get('response_time_s', 0)}s · "
                    f"trace {task.get('trace_status', 'unavailable')} · "
                    f"loop score {task.get('loop_score', 0):.1%}\n\n"
                )
                if task.get("error"):
                    f.write(f"**Error:** {task['error']}\n\n")
                f.write("**Prompt:**\n\n")
                f.write(f"~~~\n{task['prompt'].rstrip()}\n~~~\n\n")
                if task.get("thinking"):
                    f.write("**Thinking:**\n\n")
                    f.write(f"~~~\n{task['thinking'].rstrip()}\n~~~\n\n")
                f.write("**Response:**\n\n")
                f.write(f"~~~\n{task['response'].rstrip()}\n~~~\n\n")
                f.write("---\n\n")

    with open(out / "results.html", "w") as f:
        f.write(render_html(results, ts, provider, host, hardware))

    return out
