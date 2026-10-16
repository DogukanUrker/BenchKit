"""Datasets that are fetched on demand instead of shipping in the wheel.

Small suites are bundled so ``--demo`` and smoke runs work offline. Anything
large lives in a cache and is described by a :class:`Dataset` on the benchmark
class, which lets ``benchkit datasets`` and the TUI list what is installed and
install what is missing without loading any tasks.

Plain files such as ``mmlu.jsonl`` are :class:`RemoteFile` downloads: a pinned
URL plus a sha256, verified before the file is moved into place, so a partial
or tampered download never reaches a run. They are cached under
``~/.cache/benchkit/datasets`` (override with ``BENCHKIT_DATASET_CACHE``).
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx

Progress = Callable[[int, int | None], None]
"""Called with ``(bytes_done, bytes_total)`` while a download streams."""

_ATTEMPTS = 4
_LOCK = threading.Lock()
# Pinned commit of this repository that still carried the bundled copies. A
# commit pin plus a checksum keeps every install byte-identical to the file
# earlier results were scored against.
_PIN = "5789a5e2dcbb056abcbf32f810a1ed01de1cdad6"
_RAW = f"https://raw.githubusercontent.com/DogukanUrker/BenchKit/{_PIN}"


def cache_root() -> Path:
    configured = os.environ.get("BENCHKIT_DATASET_CACHE")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "benchkit" / "datasets"


def fmt_bytes(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


@dataclass(frozen=True)
class Dataset:
    """What a benchmark needs on disk, and how to get it.

    ``install`` is ``None`` for data BenchKit cannot fetch itself; ``hint``
    then says how to provide it.
    """

    summary: str
    size: int | None
    location: Callable[[], Path]
    is_installed: Callable[[], bool]
    install: Callable[[Progress | None], object] | None
    hint: str = ""


@dataclass(frozen=True)
class RemoteFile:
    """One checksummed file fetched into the dataset cache on first use."""

    name: str
    url: str
    sha256: str
    size: int

    @property
    def path(self) -> Path:
        return cache_root() / self.name

    def installed(self) -> bool:
        path = self.path
        return path.is_file() and path.stat().st_size == self.size

    def ensure(self, progress: Progress | None = None) -> Path:
        """Return the cached path, downloading and verifying it if needed."""
        with _LOCK:
            if not self.installed():
                self._download(progress)
        return self.path

    def source(self, summary: str) -> Dataset:
        return Dataset(
            summary=summary,
            size=self.size,
            location=lambda: self.path,
            is_installed=self.installed,
            install=self.ensure,
        )

    def _download(self, progress: Progress | None) -> None:
        target = self.path
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        for attempt in range(_ATTEMPTS):
            try:
                digest = hashlib.sha256()
                done = 0
                with httpx.stream(
                    "GET", self.url, follow_redirects=True, timeout=60
                ) as r:
                    r.raise_for_status()
                    with open(partial, "wb") as f:
                        for chunk in r.iter_bytes():
                            f.write(chunk)
                            digest.update(chunk)
                            done += len(chunk)
                            if progress:
                                progress(done, self.size)
                break
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                retryable = not isinstance(exc, httpx.HTTPStatusError) or (
                    exc.response.status_code in {429, 500, 502, 503, 504}
                )
                if not retryable or attempt == _ATTEMPTS - 1:
                    partial.unlink(missing_ok=True)
                    raise DatasetUnavailable(self, str(exc)) from exc
                time.sleep(2**attempt)

        if digest.hexdigest() != self.sha256:
            partial.unlink(missing_ok=True)
            raise DatasetUnavailable(
                self, f"checksum mismatch (got {digest.hexdigest()[:12]}…)"
            )
        partial.replace(target)


class DatasetUnavailable(RuntimeError):
    """A dataset is not cached and could not be fetched."""

    def __init__(self, remote: RemoteFile, reason: str) -> None:
        super().__init__(
            f"{remote.name} is not installed and could not be downloaded "
            f"({reason}). Run `benchkit datasets install` once you are online, "
            f"or place the file at {remote.path} (sha256 {remote.sha256})."
        )


MMLU = RemoteFile(
    name="mmlu.jsonl",
    url=f"{_RAW}/src/benchkit/datasets/mmlu.jsonl",
    sha256="e3e9beb9b3f6b342b99ef59ead76a03c7553fb86705ce17ce95e58e51c1cc37d",
    size=8_563_175,
)

MMLU_PRO = RemoteFile(
    name="mmlu_pro.jsonl",
    url=f"{_RAW}/src/benchkit/datasets/mmlu_pro.jsonl",
    sha256="92ec532d9e2bae342ec7256005c20b6e5defbed76504280df775fb77068f077e",
    size=10_260_787,
)
