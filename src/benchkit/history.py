"""Read historical BenchKit reports and serve the local history dashboard."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import posixpath
import threading
import webbrowser
from collections.abc import Iterable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

# Run directories are served under this prefix, one numbered slot per results
# root, so two roots holding the same timestamped run stay distinct.
ARTIFACT_PREFIX = "/files"

# Pages a run wrote itself, and that the dashboard links to directly.
RUN_PAGES = {"gallery": "arena.html", "report": "results.html"}

# The mc-arena viewer is served from the installed package rather than copied
# into every run directory: it is 3.5 MB of pinned bundle that never varies,
# and it only works over HTTP anyway (its mesher runs in a web worker, which
# browsers refuse to start from a file:// page).
VIEWER_PREFIX = "/viewer"


def _safe_json(data: object) -> str:
    """Serialize data for an HTML script element without allowing tag breakout."""
    return (
        json.dumps(data, ensure_ascii=False)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def _number(value: object) -> float | int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return int(parsed) if parsed.is_integer() else parsed


def _schema_generation(result: dict[str, Any]) -> str:
    if "trace_coverage" in result or "first_attempt_score" in result:
        return "current"
    if "harness" in result or "concurrency" in result:
        return "enhanced"
    return "legacy"


def _task_diagnostic(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": str(task.get("task_id") or ""),
        "score": _number(task.get("score")),
        "passed": task.get("passed") if isinstance(task.get("passed"), bool) else None,
        "outcome": str(task.get("outcome") or ""),
        "loop_state": str(task.get("loop_state") or ""),
        "loop_killed": task.get("loop_killed")
        if isinstance(task.get("loop_killed"), bool)
        else None,
        "loop_score": _number(task.get("loop_score")),
        "loop_source": str(task.get("loop_source") or ""),
        "timed_out": task.get("timed_out")
        if isinstance(task.get("timed_out"), bool)
        else None,
        "length_exceeded": task.get("length_exceeded")
        if isinstance(task.get("length_exceeded"), bool)
        else None,
        "harness_error": task.get("harness_error")
        if isinstance(task.get("harness_error"), bool)
        else None,
        "repaired": task.get("repaired")
        if isinstance(task.get("repaired"), bool)
        else None,
        "repair_attempts_used": _number(task.get("repair_attempts_used")),
        "tok_s": _number(task.get("tok_s")),
        "response_time_s": _number(task.get("response_time_s")),
    }


def _run_pages(directory: Path | None, base: str) -> dict[str, str]:
    """URLs for the pages a run actually wrote, skipping the ones it did not."""
    if directory is None or not base:
        return {}
    return {
        name: f"{base}/{filename}"
        for name, filename in RUN_PAGES.items()
        if (directory / filename).is_file()
    }


def _benchmark_row(
    result: dict[str, Any],
    *,
    run_id: str,
    sources: list[str],
    digest: str,
    index: int,
    pages: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    model = result.get("model")
    benchmark = result.get("benchmark")
    score = _number(result.get("score"))
    if not model or not benchmark or score is None:
        return None
    tasks = result.get("tasks")
    return {
        "id": f"{digest[:12]}-{index}",
        "run_id": run_id,
        "sources": sources,
        "pages": dict(pages or {}),
        "model": str(model),
        "benchmark": str(benchmark),
        "benchmark_label": str(result.get("benchmark_label") or benchmark),
        "score": score,
        "passed": _number(result.get("passed")),
        "scored_total": _number(result.get("scored_total") or result.get("total")),
        "total": _number(result.get("total")),
        "slice": result.get("slice"),
        "variant": result.get("variant"),
        "perturbation": result.get("perturbation"),
        "harness": str(result.get("harness") or "legacy"),
        "harness_label": str(result.get("harness_label") or "Legacy / direct"),
        "harness_version": str(result.get("harness_version") or ""),
        "schema_generation": _schema_generation(result),
        "first_attempt_score": _number(result.get("first_attempt_score")),
        "repair_attempted": _number(result.get("repair_attempted")),
        "repair_successes": _number(result.get("repair_successes")),
        "repair_delta_pp": _number(result.get("repair_delta_pp")),
        "loops": _number(result.get("loops")),
        "suspected_loops": _number(result.get("suspected_loops")),
        "recovered_loops": _number(result.get("recovered_loops")),
        "loop_kills": _number(result.get("loop_kills")),
        "loop_rate": _number(result.get("loop_rate")),
        "loop_kill_enabled": result.get("loop_kill_enabled")
        if isinstance(result.get("loop_kill_enabled"), bool)
        else None,
        "errors": _number(result.get("errors")),
        "timeouts": _number(result.get("timeouts")),
        "length_exceeded": _number(result.get("length_exceeded")),
        "harness_errors": _number(result.get("harness_errors")),
        "tok_s": _number(result.get("tok_s")),
        "tok_s_aggregate": _number(result.get("tok_s_aggregate")),
        "tok_s_per_stream": _number(result.get("tok_s_per_stream")),
        "avg_response_time": _number(result.get("avg_response_time")),
        "total_time": _number(result.get("total_time")),
        "concurrency": _number(result.get("concurrency")),
        "trace_coverage": _number(result.get("trace_coverage")),
        "task_diagnostics": [
            _task_diagnostic(task) for task in tasks if isinstance(task, dict)
        ]
        if isinstance(tasks, list)
        else [],
    }


def _perf_rows(
    profile: dict[str, Any], *, run_id: str, sources: list[str], digest: str
) -> list[dict[str, Any]]:
    model = profile.get("model") or (profile.get("settings") or {}).get("model")
    cases = profile.get("cases")
    if not model or not isinstance(cases, list):
        return []
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            continue
        rows.append(
            {
                "id": f"{digest[:12]}-{index}",
                "run_id": run_id,
                "sources": sources,
                "model": str(model),
                "version": str(profile.get("version") or ""),
                "provider": str(profile.get("provider") or ""),
                "host": str(profile.get("host") or ""),
                "depth": _number(case.get("actual_input_tokens") or case.get("depth")),
                "depth_label": str(case.get("depth_label") or case.get("depth") or ""),
                "repetitions": _number(
                    case.get("successful_reps") or len(case.get("runs") or [])
                ),
                "pp_tps": _number((case.get("pp_tps") or {}).get("median")),
                "tg_tps": _number((case.get("tg_tps") or {}).get("median")),
                "ttft_s": _number((case.get("ttft_s") or {}).get("median")),
                "wall_time_s": _number((case.get("wall_time_s") or {}).get("median")),
            }
        )
    return rows


def build_archive(results_dirs: Iterable[str | Path]) -> dict[str, Any]:
    """Read and normalize all supported reports from one or more directories."""
    roots = [Path(value).expanduser().resolve() for value in results_dirs]
    warnings: list[dict[str, str]] = []
    discovered = 0
    reports: dict[tuple[str, str], dict[str, Any]] = {}

    for root_index, root in enumerate(roots):
        if not root.is_dir():
            warnings.append(
                {
                    "kind": "missing_directory",
                    "path": str(root),
                    "message": "Results directory does not exist or is not a directory.",
                }
            )
            continue
        paths = sorted(root.glob("*/results.json")) + sorted(root.glob("*/perf.json"))
        for report_path in paths:
            discovered += 1
            report_type = "benchmark" if report_path.name == "results.json" else "perf"
            try:
                raw = report_path.read_bytes()
                payload = json.loads(raw)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                warnings.append(
                    {
                        "kind": "invalid_report",
                        "path": str(report_path),
                        "message": str(exc),
                    }
                )
                continue
            digest = hashlib.sha256(raw).hexdigest()
            key = (report_type, digest)
            location = str(report_path)
            if key in reports:
                reports[key]["sources"].append(location)
                continue
            reports[key] = {
                "type": report_type,
                "digest": digest,
                "run_id": report_path.parent.name,
                "sources": [location],
                "payload": payload,
                "directory": report_path.parent,
                "base": f"{ARTIFACT_PREFIX}/{root_index}/{report_path.parent.name}",
            }

    benchmark_rows: list[dict[str, Any]] = []
    performance_rows: list[dict[str, Any]] = []
    for report in reports.values():
        payload = report["payload"]
        if report["type"] == "benchmark":
            if not isinstance(payload, list):
                warnings.append(
                    {
                        "kind": "unsupported_schema",
                        "path": report["sources"][0],
                        "message": "Expected a JSON array for results.json.",
                    }
                )
                continue
            for index, result in enumerate(payload):
                if not isinstance(result, dict):
                    continue
                row = _benchmark_row(
                    result,
                    run_id=report["run_id"],
                    sources=report["sources"],
                    digest=report["digest"],
                    index=index,
                    pages=_run_pages(report["directory"], report["base"]),
                )
                if row is not None:
                    benchmark_rows.append(row)
        elif isinstance(payload, dict):
            performance_rows.extend(
                _perf_rows(
                    payload,
                    run_id=report["run_id"],
                    sources=report["sources"],
                    digest=report["digest"],
                )
            )
        else:
            warnings.append(
                {
                    "kind": "unsupported_schema",
                    "path": report["sources"][0],
                    "message": "Expected a JSON object for perf.json.",
                }
            )

    benchmark_rows.sort(key=lambda row: (row["run_id"], row["id"]))
    performance_rows.sort(key=lambda row: (row["run_id"], row["id"]))
    return {
        "roots": [str(root) for root in roots],
        "counts": {
            "discovered_reports": discovered,
            "unique_reports": len(reports),
            "benchmark_results": len(benchmark_rows),
            "performance_points": len(performance_rows),
            "models": len({row["model"] for row in benchmark_rows}),
        },
        "benchmark_rows": benchmark_rows,
        "performance_rows": performance_rows,
        "warnings": warnings,
    }


def render_history_html(archive: dict[str, Any]) -> str:
    """Render the packaged history dashboard with embedded archive data."""
    template = (
        files("benchkit").joinpath("templates/history.html").read_text(encoding="utf-8")
    )
    return template.replace("__BENCHKIT_HISTORY_DATA__", _safe_json(archive))


def resolve_artifact(roots: tuple[Path, ...], path: str) -> Path | None:
    """Map an artifact URL onto a file inside one results root, or nothing.

    Everything about this is deliberately narrow: the URL names a root by
    index rather than by path, the result is resolved and then checked to be
    inside that root, and only regular files are served. A request that walks
    out of the root, names an unknown root, or lands on a directory gets
    nothing back.
    """
    relative = unquote(urlsplit(path).path)
    if not relative.startswith(f"{ARTIFACT_PREFIX}/"):
        return None
    parts = [part for part in relative[len(ARTIFACT_PREFIX) + 1 :].split("/") if part]
    if len(parts) < 2 or not parts[0].isdigit():
        return None
    index = int(parts[0])
    if index >= len(roots):
        return None

    root = roots[index].resolve()
    try:
        candidate = (root / posixpath.join(*parts[1:])).resolve()
    except (OSError, ValueError):
        return None
    if candidate != root and root not in candidate.parents:
        return None
    return candidate if candidate.is_file() else None


def resolve_viewer_asset(path: str) -> Path | None:
    """Map a viewer URL onto one file of the vendored prismarine-viewer build."""
    relative = unquote(urlsplit(path).path)
    if not relative.startswith(f"{VIEWER_PREFIX}/"):
        return None
    name = relative[len(VIEWER_PREFIX) + 1 :]
    # One flat directory of known assets: no sub-paths, so nothing to escape.
    if not name or "/" in name or name in {".", ".."}:
        return None
    root = Path(str(files("benchkit").joinpath("mc_viewer/dist")))
    candidate = root / name
    return candidate if candidate.is_file() else None


def _artifact_headers(relative_parts: tuple[str, ...]) -> list[tuple[str, str]]:
    """Response headers for one served artifact.

    Pages under ``pages/`` were written by the model being benchmarked, not by
    BenchKit. Serving them from the dashboard's own origin would let them read
    everything else this server exposes, so they are handed an opaque origin
    with a sandbox policy: the scene still runs, it just cannot reach back.
    """
    # parts are (root index, run directory, ...path inside the run).
    headers = [("X-Content-Type-Options", "nosniff")]
    if len(relative_parts) > 2 and relative_parts[2] == "pages":
        headers.append(("Content-Security-Policy", "sandbox allow-scripts"))
    return headers


def create_history_server(
    results_dirs: Iterable[str | Path], port: int = 0
) -> ThreadingHTTPServer:
    """Create a localhost-only server that rescans results on every page load."""
    roots = tuple(Path(value).expanduser().resolve() for value in results_dirs)

    class HistoryHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if urlsplit(self.path).path in {"/", "/index.html"}:
                self._send(
                    render_history_html(build_archive(roots)).encode(),
                    "text/html; charset=utf-8",
                )
                return

            asset = resolve_viewer_asset(self.path)
            if asset is not None:
                # The viewer gunzips blockStates itself, so the .gz goes out
                # as opaque bytes: declaring Content-Encoding would have the
                # browser unwrap it first, and the page would then try to
                # decompress plain JSON.
                kind, _ = mimetypes.guess_type(asset.name)
                self._send(
                    asset.read_bytes(),
                    "application/octet-stream"
                    if asset.suffix == ".gz"
                    else kind or "application/octet-stream",
                    extra=[("X-Content-Type-Options", "nosniff")],
                )
                return

            artifact = resolve_artifact(roots, self.path)
            if artifact is None:
                self.send_error(404)
                return
            try:
                body = artifact.read_bytes()
            except OSError:
                self.send_error(404)
                return
            kind, _ = mimetypes.guess_type(artifact.name)
            parts = tuple(
                part
                for part in unquote(urlsplit(self.path).path)[
                    len(ARTIFACT_PREFIX) + 1 :
                ].split("/")
                if part
            )
            self._send(
                body,
                kind or "application/octet-stream",
                extra=_artifact_headers(parts),
            )

        def _send(
            self,
            body: bytes,
            content_type: str,
            extra: list[tuple[str, str]] | None = None,
        ) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for name, value in extra or []:
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    return ThreadingHTTPServer(("127.0.0.1", port), HistoryHandler)


def serve_history(
    results_dirs: Iterable[str | Path], *, port: int = 0, open_browser: bool = True
) -> None:
    """Serve the history dashboard until interrupted."""
    server = create_history_server(results_dirs, port)
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"BenchKit history: {url}")
    print("Press Ctrl-C to stop.")
    if open_browser:
        threading.Timer(0.1, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
