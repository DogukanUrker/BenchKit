"""Save benchmark results to disk."""

import csv
import json
from datetime import datetime
from pathlib import Path


def _fmt_time(s: float) -> str:
    s = int(round(s))
    if s >= 60:
        return f"{s // 60}m {s % 60}s"
    return f"{s}s"


def save(results: list[dict]) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = Path("results") / ts
    out.mkdir(parents=True, exist_ok=True)

    # Full JSON (includes per-task details)
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary CSV (one row per model×benchmark, no tasks column)
    summary = [{k: v for k, v in r.items() if k != "tasks"} for r in results]
    if summary:
        with open(out / "results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=summary[0].keys())
            w.writeheader()
            w.writerows(summary)

    # Markdown table
    with open(out / "results.md", "w") as f:
        f.write("# BenchKit Results\n\n")
        f.write(f"**Date:** {ts}\n\n")
        f.write(
            "| Model | Benchmark | Score | Passed | Total | tok/s | Avg Resp | Total Time |\n"
        )
        f.write(
            "|-------|-----------|-------|--------|-------|-------|----------|------------|\n"
        )
        for r in results:
            f.write(
                f"| {r['model']} | {r['benchmark']} "
                f"| {r['score']}% | {r['passed']} "
                f"| {r['total']} | {r['tok_s']} "
                f"| {r['avg_response_time']}s "
                f"| {_fmt_time(r['total_time'])} |\n"
            )

    return out
