"""Headless Chromium rendering for untrusted, model-generated HTML pages.

The renderer is deliberately hostile to the page it loads: every request is
denied unless its host is on a small module-CDN allowlist, downloads and
service workers are off, and every navigation, settle window and screenshot
runs under an explicit deadline. A page that hangs, crashes or wanders off the
allowlist fails the render instead of stalling the run.
"""

from __future__ import annotations

import os
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

DEFAULT_VIEWPORT = (1280, 800)
DEFAULT_SETTLE_S = 4.0
DEFAULT_TIMEOUT_S = 30.0
THUMBNAIL_QUALITY = 70

# Chromium error codes that mean the request never reached a server: no DNS,
# no route, no proxy, no TLS. Anything else - an abort, a 404, a page-level
# fetch the model got wrong - says nothing about the machine.
CONNECTION_ERRORS = (
    "ERR_NAME_NOT_RESOLVED",
    "ERR_NAME_RESOLUTION_FAILED",
    "ERR_INTERNET_DISCONNECTED",
    "ERR_NETWORK_CHANGED",
    "ERR_CONNECTION_",
    "ERR_ADDRESS_UNREACHABLE",
    "ERR_PROXY_CONNECTION_FAILED",
    "ERR_TUNNEL_CONNECTION_FAILED",
    "ERR_TIMED_OUT",
    "ERR_SSL_",
    "ERR_CERT_",
)

# Hosts that ship the ES module builds a self-contained three.js page can
# realistically import. Everything else - telemetry, analytics, arbitrary
# fetches from generated code - is aborted before it leaves the browser.
DEFAULT_ALLOWED_HOSTS = (
    "cdn.jsdelivr.net",
    "cdnjs.cloudflare.com",
    "esm.sh",
    "threejs.org",
    "unpkg.com",
)

# Recorded before any page script runs, so a page cannot hide a missing canvas
# or a swallowed rejection by monkey-patching the same APIs afterwards.
_INSTRUMENTATION = """
(() => {
  const state = { frames: 0, contexts: [], rejections: [] };
  Object.defineProperty(window, "__benchkit", { value: state });
  const raf = window.requestAnimationFrame.bind(window);
  window.requestAnimationFrame = callback => raf(time => {
    state.frames += 1;
    return callback(time);
  });
  const getContext = HTMLCanvasElement.prototype.getContext;
  HTMLCanvasElement.prototype.getContext = function (type, ...rest) {
    const context = getContext.call(this, type, ...rest);
    if (context) state.contexts.push(String(type));
    return context;
  };
  window.addEventListener("unhandledrejection", event => {
    state.rejections.push(String((event.reason && event.reason.stack) || event.reason));
  });
})();
"""

_PAGE_STATE = """
() => {
  const state = window.__benchkit || { frames: 0, contexts: [], rejections: [] };
  const canvases = [...document.querySelectorAll("canvas")].map(canvas => ({
    width: canvas.width,
    height: canvas.height,
    client_width: canvas.clientWidth,
    client_height: canvas.clientHeight,
  }));
  return {
    frames: state.frames,
    contexts: [...new Set(state.contexts)],
    rejections: state.rejections,
    canvases,
  };
}
"""


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def allowed_hosts() -> tuple[str, ...]:
    """Hosts the rendered page may reach, or none in strict offline mode."""
    if _env_bool("BENCHKIT_RENDER_OFFLINE", False):
        return ()
    raw = os.environ.get("BENCHKIT_RENDER_ALLOWED_HOSTS")
    if raw is None:
        return DEFAULT_ALLOWED_HOSTS
    return tuple(host.strip().lower() for host in raw.split(",") if host.strip())


def _host_allowed(url: str, hosts: tuple[str, ...]) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    if not host:
        return False
    return any(host == allowed or host.endswith(f".{allowed}") for allowed in hosts)


@dataclass
class RenderResult:
    """Everything one page load told us about the generated file."""

    rendered: bool = False
    console_errors: list[str] = field(default_factory=list)
    page_errors: list[str] = field(default_factory=list)
    blocked_requests: list[str] = field(default_factory=list)
    failed_requests: list[str] = field(default_factory=list)
    # Requests BenchKit allowed that never reached a server at all.
    unreachable: list[str] = field(default_factory=list)
    canvases: list[dict] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)
    frames: int = 0
    screenshot: str = ""
    thumbnail: str = ""
    # Set when BenchKit itself could not render (no Playwright, no browser).
    # Callers report this as a harness error rather than a failed page.
    error: str = ""

    @property
    def diagnostics(self) -> str:
        """Console and network output, in the order a developer would read it."""
        lines: list[str] = []
        for message in self.page_errors:
            lines.append(f"[uncaught] {message}")
        for message in self.console_errors:
            lines.append(f"[console] {message}")
        for url in self.blocked_requests:
            lines.append(f"[blocked] {url}")
        for url in self.failed_requests:
            lines.append(f"[network] {url}")
        if not self.canvases:
            lines.append("[page] no <canvas> element was present after the wait")
        elif not any(
            canvas.get("width") and canvas.get("height") for canvas in self.canvases
        ):
            lines.append("[page] the <canvas> element has zero drawing-buffer size")
        if self.canvases and not self.contexts:
            lines.append("[page] no rendering context was acquired on any canvas")
        return "\n".join(lines)


def _launch_args() -> list[str]:
    args = [
        # SwiftShader keeps WebGL working on headless machines without a GPU.
        "--use-gl=angle",
        "--use-angle=swiftshader",
        "--enable-unsafe-swiftshader",
        "--disable-dev-shm-usage",
        "--mute-audio",
    ]
    if _env_bool("BENCHKIT_RENDER_NO_SANDBOX", False):
        args.append("--no-sandbox")
    return args


def _render(
    page_path: Path,
    screenshot: Path,
    thumbnail: Path,
    viewport: tuple[int, int],
    settle_s: float,
    timeout_s: float,
    result: RenderResult,
) -> None:
    """Load one file:// page and fill ``result`` in place."""
    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright
    except ImportError:
        result.error = (
            "playwright is not installed; install the browser extra "
            "(uv sync --extra browser) and run 'playwright install chromium'"
        )
        return

    hosts = allowed_hosts()
    page_url = page_path.resolve().as_uri()
    timeout_ms = timeout_s * 1000

    def guard(route, request) -> None:
        """Deny anything the page asks for beyond itself and the allowlist."""
        url = request.url
        if url == page_url or url.startswith(("data:", "blob:", "about:")):
            route.continue_()
            return
        if _host_allowed(url, hosts):
            route.continue_()
            return
        result.blocked_requests.append(url)
        route.abort()

    executable = os.environ.get("BENCHKIT_RENDER_CHROMIUM") or None
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                args=_launch_args(),
                executable_path=executable,
            )
            try:
                context = browser.new_context(
                    viewport={"width": viewport[0], "height": viewport[1]},
                    accept_downloads=False,
                    service_workers="block",
                )
                context.set_default_timeout(timeout_ms)
                context.add_init_script(_INSTRUMENTATION)
                context.route("**/*", guard)
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

                def on_failed(request) -> None:
                    if request.url in result.blocked_requests:
                        return
                    reason = request.failure or ""
                    result.failed_requests.append(
                        f"{request.url} ({reason})" if reason else request.url
                    )
                    if any(code in reason for code in CONNECTION_ERRORS):
                        result.unreachable.append(request.url)

                page.on("requestfailed", on_failed)
                page.goto(page_url, wait_until="load", timeout=timeout_ms)
                page.wait_for_timeout(settle_s * 1000)
                state = page.evaluate(_PAGE_STATE)
                result.frames = int(state.get("frames") or 0)
                result.contexts = [str(item) for item in state.get("contexts") or []]
                result.canvases = list(state.get("canvases") or [])
                result.page_errors.extend(
                    str(item) for item in state.get("rejections") or []
                )
                page.screenshot(path=str(screenshot), type="png", timeout=timeout_ms)
                thumbnail.write_bytes(
                    page.screenshot(
                        type="jpeg",
                        quality=THUMBNAIL_QUALITY,
                        timeout=timeout_ms,
                    )
                )
                result.screenshot = str(screenshot)
                result.thumbnail = str(thumbnail)
            finally:
                browser.close()
    except PlaywrightError as exc:
        message = str(exc).strip().splitlines()[0] if str(exc).strip() else repr(exc)
        if "executable doesn't exist" in message.lower():
            result.error = (
                "chromium is not installed for playwright (run "
                f"'playwright install chromium', or set BENCHKIT_RENDER_CHROMIUM "
                f"to an existing build): {message}"
            )
        elif screenshot.exists():
            # The page loaded and was captured; the failure is the page's.
            result.page_errors.append(message)
        else:
            result.page_errors.append(f"page load failed: {message}")
    except Exception as exc:  # pragma: no cover - defensive
        result.error = f"{type(exc).__name__}: {exc}"


def render_page(
    page_path: Path,
    screenshot: Path,
    thumbnail: Path,
    *,
    viewport: tuple[int, int] = DEFAULT_VIEWPORT,
    settle_s: float | None = None,
    timeout_s: float | None = None,
) -> RenderResult:
    """Render one HTML file and report whether the page came up clean.

    The page is judged rendered when it raised no uncaught exception, logged no
    console error, reached no blocked host, and ended with a sized canvas that
    actually acquired a drawing context.
    """
    settle_s = DEFAULT_SETTLE_S if settle_s is None else settle_s
    timeout_s = DEFAULT_TIMEOUT_S if timeout_s is None else timeout_s
    result = RenderResult()

    # Playwright's sync API refuses to run inside a thread with a live asyncio
    # loop, which is exactly where the TUI evaluates tasks, so every render
    # gets a clean thread of its own.
    worker = threading.Thread(
        target=_render,
        args=(
            page_path,
            screenshot,
            thumbnail,
            viewport,
            settle_s,
            timeout_s,
            result,
        ),
        name="benchkit-render",
        daemon=True,
    )
    worker.start()
    # Navigation, settle and two screenshots each carry their own deadline;
    # this is the backstop for a browser that stops answering entirely.
    worker.join(timeout=(timeout_s * 3) + settle_s + 30.0)
    if worker.is_alive():
        result.error = (
            f"headless render did not finish within its {timeout_s:g}s budget"
        )
        return result

    # The verdict is what the page achieved: no uncaught exception, and a sized
    # canvas that acquired a drawing context. Console output and blocked hosts
    # are reported and fed back for repair, but a scene that draws is a scene
    # that rendered - a stray console.error or an analytics ping it never
    # needed does not undo that.
    result.rendered = bool(
        not result.error
        and not result.page_errors
        and result.contexts
        and any(
            int(canvas.get("width") or 0) > 0 and int(canvas.get("height") or 0) > 0
            for canvas in result.canvases
        )
    )
    return result


# A scene that needs nothing but a working headless WebGL stack. When this
# cannot render, no model output can either, and the fault is the machine's.
PROBE_PAGE = """<!doctype html><html><head><style>
html,body{margin:0;height:100%;overflow:hidden}canvas{display:block}
</style></head><body><canvas id="probe"></canvas><script>
const canvas = document.getElementById("probe");
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;
const gl = canvas.getContext("webgl2") || canvas.getContext("webgl");
if (!gl) throw new Error("no WebGL context: this browser has no working GPU or software rasterizer");
gl.clearColor(0.1, 0.5, 0.4, 1);
gl.clear(gl.COLOR_BUFFER_BIT);
requestAnimationFrame(() => {});
</script></body></html>
"""

_PROBE_LOCK = threading.Lock()
_PROBE: tuple[bool, str] | None = None


def probe_environment(force: bool = False) -> tuple[bool, str]:
    """Check once whether this machine can render WebGL at all.

    Returns ``(ok, detail)``. The result is cached for the process: it answers
    a question about the machine, not about any one page.
    """
    global _PROBE
    with _PROBE_LOCK:
        if _PROBE is not None and not force:
            return _PROBE

        with tempfile.TemporaryDirectory(prefix="benchkit-render-probe-") as directory:
            root = Path(directory)
            page = root / "probe.html"
            page.write_text(PROBE_PAGE, encoding="utf-8")
            outcome = render_page(
                page,
                root / "probe.png",
                root / "probe.jpg",
                settle_s=0.5,
                timeout_s=render_timeout_s(),
            )

        if outcome.error:
            answer = (False, outcome.error)
        elif outcome.rendered:
            answer = (
                True,
                f"headless WebGL is available ({', '.join(outcome.contexts)})",
            )
        else:
            answer = (
                False,
                "headless Chromium could not render a plain WebGL canvas: "
                + (outcome.diagnostics.replace("\n", " · ") or "no diagnostics"),
            )
        _PROBE = answer
        return answer


def render_settle_s() -> float:
    return _env_float("BENCHKIT_RENDER_SETTLE", DEFAULT_SETTLE_S)


def render_timeout_s() -> float:
    return _env_float("BENCHKIT_RENDER_TIMEOUT", DEFAULT_TIMEOUT_S)
