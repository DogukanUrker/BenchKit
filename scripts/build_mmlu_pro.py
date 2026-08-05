"""Build `src/benchkit/datasets/mmlu_pro.jsonl` from the upstream release.

The MMLU-Pro test split is distributed on Hugging Face, so the dataset file is
generated rather than hand-written:

    uv run --with datasets python scripts/build_mmlu_pro.py

Only the schema changes: `question_id` becomes a namespaced `task_id`,
`options` becomes `choices`, and the answer stays the upstream letter. Padding
options (`N/A`) left by the authors' pruning pass are dropped, which is what
makes some questions carry fewer than ten choices.
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset

LETTERS = "ABCDEFGHIJ"
PADDING = {"n/a", "na", ""}
OUTPUT = (
    Path(__file__).parent.parent / "src" / "benchkit" / "datasets" / "mmlu_pro.jsonl"
)


def _choices(options: list[str]) -> list[str]:
    """Upstream options with the trailing `N/A` padding removed."""
    trimmed = list(options)
    while trimmed and str(trimmed[-1]).strip().lower() in PADDING:
        trimmed.pop()
    return trimmed


def main() -> None:
    rows = load_dataset("TIGER-Lab/MMLU-Pro", split="test")

    written = 0
    with open(OUTPUT, "w", encoding="utf-8") as out:
        for row in rows:
            choices = _choices(row["options"])
            answer = row["answer"]
            index = row["answer_index"]

            if index >= len(choices) or LETTERS[index] != answer:
                raise SystemExit(
                    f"question {row['question_id']}: answer {answer!r} does not "
                    f"match option {index} of {len(choices)}"
                )

            out.write(
                json.dumps(
                    {
                        "task_id": f"MMLU-Pro/{row['category']}/{row['question_id']}",
                        "question": row["question"],
                        "choices": choices,
                        "labels": list(LETTERS[: len(choices)]),
                        "answer": answer,
                        "category": row["category"],
                        "src": row["src"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1

    print(f"wrote {written} questions to {OUTPUT}")


if __name__ == "__main__":
    main()
