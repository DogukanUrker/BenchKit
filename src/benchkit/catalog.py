"""Which benchmark datasets are installed, and installing the missing ones.

Shared by ``benchkit datasets`` and the TUI's datasets dialog. A benchmark
whose class has no ``data_source`` ships its data in the wheel and is always
installed. Nothing here loads tasks, so listing never triggers a download.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from benchkit.benchmarks import REGISTRY
from benchkit.benchmarks.downloads import Dataset, Progress, fmt_bytes


@dataclass(frozen=True)
class DatasetStatus:
    benchmark: str
    kind: Literal["bundled", "download", "manual"]
    installed: bool
    summary: str
    size: int | None
    location: str
    hint: str = ""
    approximate: bool = False

    @property
    def installable(self) -> bool:
        return self.kind == "download" and not self.installed


def source(key: str) -> Dataset | None:
    return getattr(REGISTRY[key], "data_source", None)


def status(key: str) -> DatasetStatus:
    data = source(key)
    if data is None:
        return DatasetStatus(key, "bundled", True, "bundled with BenchKit", None, "")
    try:
        installed = bool(data.is_installed())
    except Exception:
        installed = False
    return DatasetStatus(
        benchmark=key,
        kind="download" if data.install is not None else "manual",
        installed=installed,
        summary=data.summary,
        size=data.size,
        location=str(data.location()),
        hint=data.hint,
        approximate=data.approximate,
    )


def size_label(entry: DatasetStatus) -> str:
    """What the data costs on disk, or what installing it will download.

    Installed data is measured, so the figure is right whatever upstream
    shipped. Missing data shows its declared download size with a ``↓``, or
    ``?`` when the suite declares none. Measuring walks the cache, so this is
    kept out of ``status`` and only runs for listings.
    """
    if entry.kind == "bundled":
        path = _bundled_file(entry.benchmark)
        return fmt_bytes(path.stat().st_size) if path else "in code"
    if entry.installed:
        return fmt_bytes(_disk_usage(Path(entry.location)))
    if entry.kind == "manual":
        return ""
    if entry.size is None:
        return "?"
    return f"↓ {'~' if entry.approximate else ''}{fmt_bytes(entry.size)}"


def _bundled_file(key: str) -> Path | None:
    module = importlib.import_module(REGISTRY[key].__module__)
    dataset = getattr(module, "DATASET", None)
    if dataset is None or isinstance(dataset, str):
        return None
    path = Path(str(dataset))
    return path if path.is_file() else None


def _disk_usage(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def statuses(keys: Iterable[str] | None = None) -> list[DatasetStatus]:
    return [status(key) for key in (REGISTRY if keys is None else keys)]


def missing(keys: Iterable[str] | None = None) -> list[str]:
    """Benchmarks whose data is absent but can be fetched automatically.

    Suites sharing one source (ruler and ruler-full) are listed once.
    """
    seen: set[int] = set()
    keys_out = []
    for entry in statuses(keys):
        identity = id(source(entry.benchmark))
        if entry.installable and identity not in seen:
            seen.add(identity)
            keys_out.append(entry.benchmark)
    return keys_out


def install(key: str, progress: Progress | None = None) -> None:
    """Fetch one benchmark's data. Raises with an actionable message."""
    data = source(key)
    if data is None:
        return
    if data.install is None:
        raise RuntimeError(f"{key} cannot be installed automatically: {data.hint}")
    data.install(progress)
