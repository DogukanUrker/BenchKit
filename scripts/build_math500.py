"""Build `src/benchkit/datasets/math500.jsonl` from the upstream release.

MATH-500 is the 500-problem test split OpenAI published with "Let's Verify
Step by Step", drawn from Hendrycks et al.'s MATH test set:

    uv run python scripts/build_math500.py

The file is pinned to a commit of `openai/prm800k` and checked against the
sha256 of its Git LFS object. It is the same set as Hugging Face's
`HuggingFaceH4/MATH-500`. Only the schema changes: rows get a namespaced
`task_id` in upstream order, and the worked `solution` is dropped to keep the
bundled datasets small. Before writing, every reference answer must equal the
last `\\boxed{}` of its own solution under BenchKit's grader, so a grader
change that breaks a reference fails here rather than in a run.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

from benchkit.benchmarks.math500 import answers_match, last_boxed

COMMIT = "7ecc794703b2877f63226f2477a49b34f9b25163"
URL = (
    "https://media.githubusercontent.com/media/openai/prm800k/"
    f"{COMMIT}/prm800k/math_splits/test.jsonl"
)
SHA256 = "35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132"
OUTPUT = Path(__file__).parent.parent / "src" / "benchkit" / "datasets"
OUTPUT /= "math500.jsonl"


def main() -> None:
    with urllib.request.urlopen(URL, timeout=60) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"sha256 mismatch: expected {SHA256}, got {digest}")
    lines = payload.decode("utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    if len(rows) != 500:
        raise SystemExit(f"expected 500 problems, got {len(rows)}")

    for row in rows:
        boxed = last_boxed(row["solution"])
        if boxed is None or answers_match(boxed, row["answer"]) is None:
            raise SystemExit(
                f"{row['unique_id']}: answer {row['answer']!r} does not match "
                f"its solution's boxed {boxed!r}"
            )

    with open(OUTPUT, "w", encoding="utf-8") as out:
        for index, row in enumerate(rows):
            out.write(
                json.dumps(
                    {
                        "task_id": f"MATH500/{index}",
                        "problem": row["problem"],
                        "answer": row["answer"],
                        "subject": row["subject"],
                        "level": int(row["level"]),
                        "unique_id": row["unique_id"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"wrote {len(rows)} problems to {OUTPUT}")


if __name__ == "__main__":
    main()
