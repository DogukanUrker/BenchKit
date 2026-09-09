"""Fixed-camera Minecraft build rendering for the mc-arena suite.

The renderer is prismarine-viewer, pinned to one Minecraft version and built
into ``mc_viewer/dist`` (see ``mc_viewer/build/README.md``). This module is the
Python half: it serves that bundle over a loopback HTTP server, hands one build
to the page, and photographs the result from three frozen camera positions.

Two builds are only comparable if they were photographed the same way, so the
volume, the camera rig, the lighting and the image size all live in the page's
own source and never vary per model or per run.

The page here is BenchKit's own code, not the model's. A model contributes a
list of block positions and nothing executable - its script already ran, under
Docker, in :func:`benchkit.sandbox.run_python_script`.
"""

from __future__ import annotations

import atexit
import contextlib
import functools
import http.server
import json
import shutil
import socketserver
import tempfile
import threading
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

from benchkit.browser import chromium_executable, launch_args

# Frozen alongside the camera rig in ``mc_viewer/build/entry.js``.
VIEWS = ("iso", "side", "top")
BUILD_SIZE = 32
DEFAULT_TIMEOUT_S = 120.0


@dataclass
class MCRenderResult:
    """What one build's three photographs came back with."""

    rendered: bool = False
    views: dict[str, str] = field(default_factory=dict)
    contact_sheet: str = ""
    # Blocks the renderer placed, and ones its block table did not recognise.
    placed: int = 0
    unresolved: int = 0
    page_errors: list[str] = field(default_factory=list)
    console_errors: list[str] = field(default_factory=list)
    # Set when this machine could not render at all (no browser, no WebGL).
    # mc-arena scores the block list, not the picture, so this is a missing
    # illustration rather than a failed task.
    error: str = ""

    @property
    def diagnostics(self) -> str:
        lines = [f"[uncaught] {message}" for message in self.page_errors]
        lines += [f"[console] {message}" for message in self.console_errors]
        if self.error:
            lines.append(f"[renderer] {self.error}")
        return "\n".join(lines)


def dist_dir() -> Path:
    """The vendored prismarine-viewer bundle shipped inside the package."""
    return Path(str(files("benchkit").joinpath("mc_viewer/dist")))


def viewer_version() -> dict:
    """The pinned Minecraft version and prismarine-viewer commit."""
    try:
        return json.loads((dist_dir() / "version.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    """The viewer's own asset requests are not run output."""

    def log_message(self, *_args: object) -> None:
        return

    def do_GET(self) -> None:
        # Chromium asks every page for a favicon; a 404 for one would be
        # recorded as a console error against the build.
        if self.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        super().do_GET()


@dataclass
class _Site:
    """One loopback web server over a copy of the viewer bundle.

    prismarine-viewer meshes chunks in a web worker, and Chromium refuses to
    start a worker from a ``file://`` page, so the bundle is served over HTTP
    on 127.0.0.1 instead. Nothing outside this machine can reach it.
    """

    root: Path
    port: int
    server: socketserver.TCPServer

    def url(self, build: str) -> str:
        return f"http://127.0.0.1:{self.port}/viewer.html?build={build}"


_SITE: _Site | None = None
_SITE_LOCK = threading.Lock()
_BUILD_COUNTER = 0


def _site() -> _Site:
    """Start this process's viewer server on first use."""
    global _SITE
    with _SITE_LOCK:
        if _SITE is not None:
            return _SITE
        root = Path(tempfile.mkdtemp(prefix="benchkit-mc-viewer-"))
        for asset in dist_dir().iterdir():
            if asset.is_file():
                shutil.copyfile(asset, root / asset.name)
        handler = functools.partial(_QuietHandler, directory=str(root))
        server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler)
        server.daemon_threads = True
        threading.Thread(
            target=server.serve_forever,
            name="benchkit-mc-viewer",
            daemon=True,
        ).start()
        _SITE = _Site(root=root, port=server.server_address[1], server=server)
        atexit.register(shutdown)
        return _SITE


def shutdown() -> None:
    """Stop the viewer server and remove its copy of the bundle."""
    global _SITE
    with _SITE_LOCK:
        site, _SITE = _SITE, None
    if site is None:
        return
    site.server.shutdown()
    site.server.server_close()
    shutil.rmtree(site.root, ignore_errors=True)


def _stage_build(blocks: list[dict]) -> str:
    """Write one build next to the viewer and return the name to fetch it by."""
    global _BUILD_COUNTER
    site = _site()
    with _SITE_LOCK:
        _BUILD_COUNTER += 1
        name = f"build-{_BUILD_COUNTER:05d}.json"
    (site.root / name).write_text(json.dumps(blocks), encoding="utf-8")
    return name


def _render(
    blocks: list[dict],
    out_dir: Path,
    timeout_s: float,
    result: MCRenderResult,
) -> None:
    """Photograph one build and fill ``result`` in place."""
    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright
    except ImportError:
        result.error = (
            "playwright is not installed; install the browser extra "
            "(uv sync --extra browser) and run 'playwright install chromium'"
        )
        return

    site = _site()
    staged = _stage_build(blocks)
    url = site.url(staged)
    timeout_ms = timeout_s * 1000
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                args=launch_args(),
                executable_path=chromium_executable(),
            )
            try:
                context = browser.new_context(
                    # Wide enough for the three views side by side.
                    viewport={"width": 2000, "height": 760},
                    accept_downloads=False,
                    service_workers="block",
                )
                context.set_default_timeout(timeout_ms)
                page = context.new_page()
                page.on(
                    "console",
                    lambda message: (
                        result.console_errors.append(message.text)
                        if message.type == "error"
                        else None
                    ),
                )
                page.on(
                    "pageerror", lambda error: result.page_errors.append(str(error))
                )
                page.goto(url, wait_until="load", timeout=timeout_ms)
                page.wait_for_function(
                    "window.__mcarena !== undefined", timeout=timeout_ms
                )
                state = page.evaluate("window.__mcarena") or {}
                if not state.get("ok"):
                    result.error = str(state.get("error") or "the viewer did not run")
                    return

                result.placed = int(state.get("placed") or 0)
                result.unresolved = int(state.get("unresolved") or 0)
                out_dir.mkdir(parents=True, exist_ok=True)
                for view in VIEWS:
                    shot = out_dir / f"{view}.png"
                    page.locator(f"#view-{view}").screenshot(
                        path=str(shot), timeout=timeout_ms
                    )
                    result.views[view] = str(shot)
                sheet = out_dir / "views.png"
                page.locator("#views").screenshot(path=str(sheet), timeout=timeout_ms)
                result.contact_sheet = str(sheet)
                result.rendered = True
            finally:
                browser.close()
    except PlaywrightError as exc:
        message = str(exc).strip().splitlines()[0] if str(exc).strip() else repr(exc)
        if "executable doesn't exist" in message.lower():
            result.error = (
                "chromium is not installed for playwright (run "
                "'playwright install chromium', or set BENCHKIT_RENDER_CHROMIUM "
                f"to an existing build): {message}"
            )
        else:
            result.error = message
    except Exception as exc:  # pragma: no cover - defensive
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        # The viewer has read it by now, and a long run would otherwise leave
        # one of these behind per task until the process exits.
        with contextlib.suppress(OSError):
            (site.root / staged).unlink()


def render_build(
    blocks: list[dict],
    out_dir: Path,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> MCRenderResult:
    """Render one block list from three fixed cameras into ``out_dir``."""
    result = MCRenderResult()
    # Playwright's sync API refuses to run in a thread with a live asyncio
    # loop, which is exactly where the TUI evaluates tasks.
    worker = threading.Thread(
        target=_render,
        args=(blocks, out_dir, timeout_s, result),
        name="benchkit-mc-render",
        daemon=True,
    )
    worker.start()
    worker.join(timeout=timeout_s + 60.0)
    if worker.is_alive():
        result.rendered = False
        result.error = f"the build did not render within its {timeout_s:g}s budget"
    return result
