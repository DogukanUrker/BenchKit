"""Connection screen: pick a host/provider and discover models."""

from __future__ import annotations

import os

from textual import events, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Label, Select, Static

from benchkit.client import DEFAULT_HOST, InferenceClient
from benchkit.demo import DemoClient
from benchkit.tui.widgets import Banner

PROVIDERS = [
    ("auto detect", "auto"),
    ("OpenAI compatible", "openai"),
    ("Ollama", "ollama"),
]

HINT = (
    "Point BenchKit at any OpenAI-compatible server (llama.cpp, llama-swap, vLLM, "
    "LM Studio) or a native Ollama host."
)


class ConnectScreen(Screen[None]):
    """First screen: configure the endpoint and connect."""

    BINDINGS = [
        Binding("ctrl+r", "connect", "Connect"),
        Binding("f5", "connect", "Connect", show=False),
        Binding("f3", "demo", "Demo mode"),
        Binding("escape", "app.quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with VerticalScroll(id="connect-root"):
            yield Banner(id="banner")
            yield Static(
                "Benchmark your local LLMs — not vibes, actual scores.", id="tagline"
            )
            with Vertical(id="connect-card"):
                yield Static("Connection", classes="card-title")
                with Vertical(id="connect-form"):
                    with Horizontal(classes="field"):
                        yield Label("Host", classes="field-label")
                        yield Input(placeholder=DEFAULT_HOST, id="host", compact=True)
                    with Horizontal(classes="field"):
                        yield Label("Provider", classes="field-label")
                        yield Select(
                            PROVIDERS,
                            value="auto",
                            allow_blank=False,
                            id="provider",
                            compact=True,
                        )
                    with Horizontal(classes="field"):
                        yield Label("API key", classes="field-label")
                        yield Input(
                            placeholder="optional",
                            password=True,
                            id="api-key",
                            compact=True,
                        )
                    with Horizontal(classes="field"):
                        yield Label("Timeout", classes="field-label")
                        yield Input(placeholder="300", id="timeout", compact=True)
                with Horizontal(id="connect-actions"):
                    yield Button("Connect", id="connect", variant="primary")
                    yield Button("Quit", id="quit", variant="error")
                yield Static(HINT, id="connect-status")
        yield Footer()

    def on_resize(self, event: events.Resize) -> None:
        # The wordmark is 62 cells wide, so it goes first on a small terminal.
        size = event.size
        self.set_class(size.height < 30 or size.width < 66, "compact")

    def on_mount(self) -> None:
        defaults = _env_defaults()
        host = self.app.host_override or defaults["host"]
        self.query_one("#host", Input).value = host
        self.query_one("#provider", Select).value = defaults["provider"]
        self.query_one("#api-key", Input).value = defaults["api_key"]
        self.query_one("#timeout", Input).value = defaults["timeout"]

        if self.app.demo:
            self.set_timer(0.1, self.action_demo)
        else:
            self.set_timer(0.1, self.action_connect)

    # Actions ----------------------------------------------------------

    def action_connect(self) -> None:
        host = self.query_one("#host", Input).value.strip() or DEFAULT_HOST
        provider = self.query_one("#provider", Select).value
        api_key = self.query_one("#api-key", Input).value.strip() or None
        raw_timeout = self.query_one("#timeout", Input).value.strip() or "300"
        try:
            timeout = float(raw_timeout)
        except ValueError:
            self._status(f"Timeout must be a number, got {raw_timeout!r}", "error")
            return

        self._busy(True)
        self._status(f"Connecting to {host} …", "working")
        self._connect(host.rstrip("/"), str(provider), api_key, timeout)

    def action_demo(self) -> None:
        client = DemoClient()
        self._status("Demo mode — offline models, no server required.", "ok")
        self.app.show_setup(client, client.list_models())

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "connect":
            self.action_connect()
        elif event.button.id == "quit":
            self.app.exit()

    def on_input_submitted(self) -> None:
        self.action_connect()

    # Worker -----------------------------------------------------------

    @work(thread=True, exclusive=True, group="connect")
    def _connect(
        self, host: str, provider: str, api_key: str | None, timeout: float
    ) -> None:
        try:
            client = InferenceClient(host, provider, api_key, timeout)
            models = client.list_models()
        except Exception as exc:
            self.app.call_from_thread(self._failed, f"{type(exc).__name__}: {exc}")
            return
        self.app.call_from_thread(self._connected, client, models)

    def _connected(self, client: InferenceClient, models: list[dict]) -> None:
        self._busy(False)
        if not models:
            self._status(
                f"Connected to {client.host} but the server reports no models.", "error"
            )
            return
        self._status(f"{len(models)} model(s) via {client.label}", "ok")
        self.app.show_setup(client, models)

    def _failed(self, message: str) -> None:
        self._busy(False)
        self._status(
            f"{message}\nCheck the host and that the server is running, "
            "or press F3 for demo mode.",
            "error",
        )

    # Helpers ----------------------------------------------------------

    def _busy(self, busy: bool) -> None:
        self.query_one("#connect", Button).disabled = busy
        self.query_one("#connect-card").set_class(busy, "busy")

    def _status(self, message: str, kind: str = "") -> None:
        status = self.query_one("#connect-status", Static)
        status.set_classes(f"status {kind}" if kind else "status")
        status.update(message)
        status.scroll_visible()


def _env_defaults() -> dict[str, str]:
    provider = os.environ.get("BENCHKIT_PROVIDER", "auto").lower()
    if provider not in {"auto", "openai", "ollama"}:
        provider = "auto"
    host = (
        os.environ.get("BENCHKIT_HOST")
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OLLAMA_HOST")
        or DEFAULT_HOST
    )
    api_key = (
        os.environ.get("BENCHKIT_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    )
    return {
        "host": host.rstrip("/"),
        "provider": provider,
        "api_key": api_key,
        "timeout": os.environ.get("BENCHKIT_TIMEOUT", "300"),
    }
