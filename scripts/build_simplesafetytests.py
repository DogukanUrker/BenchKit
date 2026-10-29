"""Build `src/benchkit/datasets/simplesafetytests.jsonl` from the upstream CSV.

SimpleSafetyTests is published by its authors as one CSV on GitHub, so the
dataset file is generated rather than hand-written:

    uv run python scripts/build_simplesafetytests.py

The commit is pinned, so a rebuild reproduces the bundled file byte for byte.
Each row keeps its upstream `id` as a namespaced `task_id`, `prompts_final`
becomes `prompt` and `category` becomes `prompt_type`; the harm area is kept as
written. Rows stay in upstream order (grouped by harm area); the benchmark
spreads them at load time so any slice covers all five areas.
"""

from __future__ import annotations

import csv
import io
import json
import urllib.request
from collections import Counter
from pathlib import Path

COMMIT = "d7aee9a9422a5a5488f478fd79c2479c891c0f3b"
URL = (
    f"https://raw.githubusercontent.com/bertiev/SimpleSafetyTests/{COMMIT}/"
    "SimpleSafetyTests%20-%20test%20cases.csv"
)
OUTPUT = (
    Path(__file__).parent.parent
    / "src"
    / "benchkit"
    / "datasets"
    / "simplesafetytests.jsonl"
)


def main() -> None:
    with urllib.request.urlopen(URL, timeout=60) as response:
        text = response.read().decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text)))

    areas = Counter(row["harm_area"] for row in rows)
    if len(rows) != 100 or len(areas) != 5 or set(areas.values()) != {20}:
        raise SystemExit(f"expected 100 rows, 20 in each of 5 areas: {areas}")

    with open(OUTPUT, "w", encoding="utf-8") as out:
        for row in rows:
            record = {
                "task_id": f"SimpleSafetyTests/{row['id']}",
                "prompt": row["prompts_final"].strip(),
                "harm_area": row["harm_area"],
                "prompt_type": row["category"],
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
