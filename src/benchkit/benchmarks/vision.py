"""Shared plumbing for image benchmarks.

Vision datasets are too large to bundle, so each suite downloads a pinned
Hugging Face revision on first use, either one parquet file with embedded
images (``load_rows``) or a question manifest plus one file per image
(``load_files``). Rows and images are cached under ``~/.cache/benchkit/vision``
(override with ``BENCHKIT_VISION_CACHE``). The rows file is written last, so a
cache directory without it is an interrupted download and is resumed.

Tasks carry local image paths in ``metadata["images"]``; the engine hands them
to the client, which sends them inline as base64.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import httpx
import pyarrow.parquet as pq

from benchkit.benchmarks.base import Task

HUB = "https://huggingface.co/datasets"
_ATTEMPTS = 4
_DOWNLOAD_WORKERS = 8
_LOCK = threading.Lock()
# Image bytes carry no reliable name, so sniff the format for the media type.
_MAGIC = ((b"\x89PNG", ".png"), (b"\xff\xd8", ".jpg"), (b"GIF8", ".gif"))


def cache_root() -> Path:
    configured = os.environ.get("BENCHKIT_VISION_CACHE")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "benchkit" / "vision"


def _http_client() -> httpx.Client:
    # One pooled client per download: a fresh connection per image means a
    # DNS lookup per image, which flaky home resolvers drop under load.
    limits = httpx.Limits(max_connections=_DOWNLOAD_WORKERS)
    return httpx.Client(follow_redirects=True, timeout=60, limits=limits)


def _download_file(client: httpx.Client, url: str, target: Path) -> None:
    partial = target.with_name(target.name + ".part")
    for attempt in range(_ATTEMPTS):
        try:
            with client.stream("GET", url) as r:
                r.raise_for_status()
                with open(partial, "wb") as f:
                    for chunk in r.iter_bytes():
                        f.write(chunk)
            partial.replace(target)
            return
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            retryable = not isinstance(exc, httpx.HTTPStatusError) or (
                exc.response.status_code in {429, 500, 502, 503, 504}
            )
            if not retryable or attempt == _ATTEMPTS - 1:
                raise RuntimeError(f"could not download {url}: {exc}") from exc
            time.sleep(2**attempt)


def _suffix(data: bytes) -> str:
    for magic, suffix in _MAGIC:
        if data.startswith(magic):
            return suffix
    return ".webp" if data[8:12] == b"WEBP" else ".png"


def load_rows(
    name: str,
    dataset: str,
    parquet: str,
    revision: str,
    convert: Callable[[int, dict], dict],
) -> list[dict]:
    """Return cached rows for one parquet file, downloading it once.

    ``convert(row_idx, row)`` maps an upstream row (without its image) to the
    JSON stored in the cache. Each stored row gains an absolute ``image`` path.
    """
    return _cached(
        name,
        revision,
        lambda root: _from_parquet(root, _url(dataset, revision, parquet), convert),
    )


def load_files(
    name: str,
    dataset: str,
    manifest: str,
    revision: str,
    parse: Callable[[bytes], list[dict]],
) -> list[dict]:
    """Return cached rows for a manifest that names one image file per row.

    ``parse(manifest_bytes)`` returns the rows to store; each row's ``image``
    is a path inside the dataset repo and is replaced by the cached copy.
    """

    def build(root: Path) -> list[dict]:
        source = root / "manifest"
        with _http_client() as client:
            _download_file(client, _url(dataset, revision, manifest), source)
            rows = parse(source.read_bytes())

            def fetch(row: dict) -> dict:
                target = root / "images" / row["image"]
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    url = _url(dataset, revision, row["image"])
                    _download_file(client, url, target)
                return {**row, "image": str(target.relative_to(root))}

            with ThreadPoolExecutor(_DOWNLOAD_WORKERS) as pool:
                stored = list(pool.map(fetch, rows))
        source.unlink()
        return stored

    return _cached(name, revision, build)


def _url(dataset: str, revision: str, path: str) -> str:
    return f"{HUB}/{dataset}/resolve/{revision}/{quote(path)}"


def _cached(
    name: str, revision: str, build: Callable[[Path], list[dict]]
) -> list[dict]:
    root = cache_root() / name / revision[:12]
    rows_file = root / "rows.jsonl"
    with _LOCK:
        if not rows_file.exists():
            root.mkdir(parents=True, exist_ok=True)
            stored = build(root)
            partial = rows_file.with_name(rows_file.name + ".part")
            with open(partial, "w", encoding="utf-8") as f:
                for entry in stored:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            partial.replace(rows_file)
    with open(rows_file, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    for row in rows:
        row["image"] = str(root / row["image"])
    return rows


def _from_parquet(
    root: Path, url: str, convert: Callable[[int, dict], dict]
) -> list[dict]:
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    source = root / "source.parquet"
    if not source.exists():
        with _http_client() as client:
            _download_file(client, url, source)

    stored = []
    for index, row in enumerate(pq.read_table(source).to_pylist()):
        data = row.pop("image")["bytes"]
        target = images / f"{index}{_suffix(data)}"
        target.write_bytes(data)
        entry = convert(index, row)
        entry["image"] = str(target.relative_to(root))
        stored.append(entry)
    # The images now hold everything the parquet did.
    source.unlink()
    return stored


def spread(tasks: Iterable[Task], key: Callable[[Task], str]) -> list[Task]:
    """Order tasks so any prefix slice keeps every group's share.

    Upstream rows are grouped by category, and BenchKit slices are contiguous,
    so ``ocrbench:100`` would otherwise test one category only. Each task is
    placed at its fractional position inside its own group; ties keep the
    upstream order.
    """
    groups: dict[str, list[Task]] = defaultdict(list)
    for task in tasks:
        groups[key(task)].append(task)
    placed = [
        ((i + 0.5) / len(members), order, task)
        for order, members in enumerate(groups.values())
        for i, task in enumerate(members)
    ]
    placed.sort(key=lambda item: (item[0], item[1]))
    return [task for _, _, task in placed]


def category_scores(records: list[object], categories: dict[str, str]) -> dict:
    """Percent correct per category over the scored records."""
    totals: dict[str, list[float]] = defaultdict(list)
    for record in records:
        category = categories.get(record.task_id)
        if category is not None:
            totals[category].append(float(getattr(record, "score", 0.0)))
    return {
        category: round(100 * sum(scores) / len(scores), 1)
        for category, scores in sorted(totals.items())
    }
