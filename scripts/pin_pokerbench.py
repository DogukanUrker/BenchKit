"""Pin the PokerBench test splits in `benchmarks/downloads.py`.

PokerBench is read straight from the authors' Hugging Face release, so there
is nothing to rebuild; a new revision only needs new checksums:

    uv run python scripts/pin_pokerbench.py [REVISION]

With no revision the current `main` is resolved to its commit. The four test
files are downloaded at that commit, joined and validated exactly as the
benchmark loads them, and the `POKERBENCH_REVISION` / `POKERBENCH` block to
paste into `downloads.py` is printed with each file's sha256 and size.
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
from collections import Counter
from pathlib import Path

import httpx

from benchkit.benchmarks.downloads import POKERBENCH
from benchkit.benchmarks.pokerbench import SPLITS, build_split, verb

REPO = "RZ412/PokerBench"


def main() -> None:
    revision = sys.argv[1] if len(sys.argv) > 1 else None
    with httpx.Client(follow_redirects=True, timeout=120) as client:
        if revision is None:
            info = client.get(f"https://huggingface.co/api/datasets/{REPO}")
            info.raise_for_status()
            revision = info.json()["sha"]
        base = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}"
        with tempfile.TemporaryDirectory() as tmp:
            pins = {}
            for key, remote in POKERBENCH.items():
                filename = Path(remote.name).name
                data = client.get(f"{base}/{filename}")
                data.raise_for_status()
                path = Path(tmp) / filename
                path.write_bytes(data.content)
                pins[key] = (
                    filename,
                    hashlib.sha256(data.content).hexdigest(),
                    len(data.content),
                    path,
                )

            tasks = []
            for split in SPLITS:
                tasks += build_split(
                    split, pins[f"{split}_json"][3], pins[f"{split}_csv"][3]
                )

    streets = Counter(task.metadata["street"] for task in tasks)
    options = Counter(len(task.metadata["choices"]) for task in tasks)
    answers = Counter(verb(task.metadata["answer"]) for task in tasks)
    print(f"{len(tasks)} tasks; streets {dict(streets)}")
    print(f"options per spot {dict(sorted(options.items()))}")
    print(f"solver actions {dict(answers)}\n")

    print(f'POKERBENCH_REVISION = "{revision}"')
    for key, (filename, digest, size, _path) in pins.items():
        print(f'    "{key}": RemoteFile(')
        print(f'        name="pokerbench/{filename}",')
        print(f'        url=f"{{_POKERBENCH}}/{filename}",')
        print(f'        sha256="{digest}",')
        print(f"        size={size:_},")
        print("    ),")


if __name__ == "__main__":
    main()
