"""Save benchmark results to disk."""

import csv
import json
from datetime import datetime
from pathlib import Path


def save(results: list[dict]) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = Path("results") / ts
    out.mkdir(parents=True, exist_ok=True)

    # Full JSON (includes per-task details)
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary CSV (one row per model×benchmark)
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
        f.write("| Model | Benchmark | Score | Passed | Total | tok/s |\n")
        f.write("|-------|-----------|-------|--------|-------|-------|\n")
        for r in results:
            f.write(
                f"| {r['model']} | {r['benchmark']} "
                f"| {r['score']}% | {r['passed']} "
                f"| {r['total']} | {r['tok_s']} |\n"
            )

    return out
