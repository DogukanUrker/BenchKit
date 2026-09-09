"""Run-scoped staging for per-task binary artifacts.

Benchmarks produce screenshots and generated pages while a run is still in
flight, but the results directory only exists once :func:`benchkit.report.save`
stamps it. Artifacts are therefore written to a staging directory keyed by the
process, and collected into the results directory at save time with the paths
in each task record rewritten to be relative to the report.
"""

from __future__ import annotations

import contextlib
import re
import shutil
import uuid
from pathlib import Path

STAGING_ROOT = Path("results") / ".artifacts"
_RUN_ID = uuid.uuid4().hex[:12]
_COUNTER = 0

# Artifact keys in a task's ``workspace`` payload, and where they are collected.
COLLECTED = (
    ("screenshot", "screenshots", ".png"),
    ("screenshot_thumbnail", "screenshots", ".jpg"),
    # mc-arena photographs one build from three fixed cameras; the plain
    # ``screenshot`` above is the three of them side by side.
    ("screenshot_iso", "screenshots", "-iso.png"),
    ("screenshot_side", "screenshots", "-side.png"),
    ("screenshot_top", "screenshots", "-top.png"),
    ("page_html", "pages", ".html"),
    ("response_text", "pages", ".txt"),
    ("script_py", "builds", ".py"),
    ("blocks_json", "builds", ".json"),
)


def staging_root() -> Path:
    return STAGING_ROOT / _RUN_ID


def task_dir(prefix: str) -> Path:
    """Create a fresh directory for one evaluation's artifacts."""
    global _COUNTER
    _COUNTER += 1
    directory = staging_root() / f"{_slug(prefix)}-{_COUNTER:04d}"
    directory.mkdir(parents=True, exist_ok=True)
    # Absolute, so a path recorded mid-run still resolves at report time.
    return directory.resolve()


def _slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-")
    return text[:80] or "artifact"


def collect(out: Path, results: list[dict]) -> None:
    """Move staged artifacts into the report and relativize their paths."""
    root = staging_root().resolve()
    for result in results:
        model = _slug(str(result.get("model") or "model"))
        for task in result.get("tasks") or []:
            workspace = task.get("workspace")
            if not isinstance(workspace, dict):
                continue
            for key, subdir, suffix in COLLECTED:
                source = workspace.get(key)
                if not isinstance(source, str) or not source:
                    continue
                path = Path(source)
                if not path.is_absolute() or not path.is_file():
                    workspace.pop(key, None)
                    continue
                if root not in path.resolve().parents:
                    continue
                name = f"{model}__{_slug(str(task.get('task_id') or 'task'))}{suffix}"
                target = out / subdir / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                workspace[key] = f"{subdir}/{name}"


def cleanup() -> None:
    """Remove this process's staging directory, and the root when it empties."""
    shutil.rmtree(staging_root(), ignore_errors=True)
    with contextlib.suppress(OSError):
        STAGING_ROOT.rmdir()
