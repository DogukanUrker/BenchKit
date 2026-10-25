"""Build `src/benchkit/datasets/cruxeval.jsonl` from the upstream release.

CRUXEval ships one JSONL file of 800 short Python functions, each with one
input and the output that running the function on it produces:

    uv run python scripts/build_cruxeval.py

The file is pinned to an upstream commit. Only the schema changes: `id`
(`sample_N`) becomes a namespaced `task_id` and `code` becomes `function`;
`input` and `output` stay the upstream Python source text. Every output must
parse with `ast.literal_eval`, because that is how `cruxeval-o` scores it.
"""

from __future__ import annotations

import ast
import json
import urllib.request
from pathlib import Path

COMMIT = "190faf16d175b5847b0af05d937872b1fb395942"
URL = (
    "https://raw.githubusercontent.com/facebookresearch/cruxeval/"
    f"{COMMIT}/data/cruxeval.jsonl"
)
OUTPUT = Path(__file__).parent.parent / "src" / "benchkit" / "datasets"
OUTPUT /= "cruxeval.jsonl"


def main() -> None:
    with urllib.request.urlopen(URL, timeout=60) as response:
        lines = response.read().decode("utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    if len(rows) != 800:
        raise SystemExit(f"expected 800 functions, got {len(rows)}")

    with open(OUTPUT, "w", encoding="utf-8") as out:
        for row in rows:
            ast.literal_eval(row["output"])
            index = int(row["id"].removeprefix("sample_"))
            out.write(
                json.dumps(
                    {
                        "task_id": f"CRUXEval/{index}",
                        "function": row["code"],
                        "input": row["input"],
                        "output": row["output"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"wrote {len(rows)} functions to {OUTPUT}")


if __name__ == "__main__":
    main()
