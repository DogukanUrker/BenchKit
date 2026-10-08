"""Build `src/benchkit/datasets/banking77.jsonl` from the upstream release.

BANKING77 is published by PolyAI as CSV, so the dataset file is generated
rather than hand-written:

    uv run python scripts/build_banking77.py

Each test row keeps its upstream position as a namespaced `task_id`, `text`
becomes `question` and `category` becomes `answer`. Upstream sorts rows by
intent, and BenchKit slices are contiguous, so rows are interleaved round-robin
across intents: `banking77:77` covers every intent once. The 77 intents stay in
the upstream `categories.json` order, which is the order every prompt lists
them in.
"""

from __future__ import annotations

import csv
import io
import itertools
import json
import urllib.request
from pathlib import Path

BASE = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
    "master/banking_data/"
)
DATASETS = Path(__file__).parent.parent / "src" / "benchkit" / "datasets"
OUTPUT = DATASETS / "banking77.jsonl"
INTENTS = DATASETS / "banking77_intents.json"


def _fetch(name: str) -> str:
    with urllib.request.urlopen(BASE + name, timeout=60) as response:
        return response.read().decode("utf-8")


def main() -> None:
    intents = json.loads(_fetch("categories.json"))
    if len(intents) != 77 or len(set(intents)) != 77:
        raise SystemExit(f"expected 77 distinct intents, got {len(intents)}")

    rows = list(csv.DictReader(io.StringIO(_fetch("test.csv"))))
    unknown = {row["category"] for row in rows} - set(intents)
    if unknown:
        raise SystemExit(f"rows use intents outside categories.json: {unknown}")

    by_intent: dict[str, list[tuple[int, dict]]] = {name: [] for name in intents}
    for index, row in enumerate(rows):
        by_intent[row["category"]].append((index, row))
    interleaved = [
        entry
        for group in itertools.zip_longest(*by_intent.values())
        for entry in group
        if entry is not None
    ]

    INTENTS.write_text(json.dumps(intents, indent=2) + "\n", encoding="utf-8")
    with open(OUTPUT, "w", encoding="utf-8") as out:
        for index, row in interleaved:
            out.write(
                json.dumps(
                    {
                        "task_id": f"BANKING77/{index}",
                        "question": row["text"].strip(),
                        "answer": row["category"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"wrote {len(rows)} messages and {len(intents)} intents to {DATASETS}")


if __name__ == "__main__":
    main()
