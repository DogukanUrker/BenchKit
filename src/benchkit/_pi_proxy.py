"""Restricted inference proxy used by the Pi task sandbox.

The Pi container lives on an internal Docker network with no direct egress.
This small process runs in a separate container attached to both that network
and Docker's bridge network, forwards only OpenAI-compatible inference routes,
and keeps the upstream API key out of the agent's filesystem and environment.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

ALLOWED_PATHS = ("/v1/chat/completions", "/v1/models")
MAX_BODY_BYTES = 32 * 1024 * 1024
SCAFFOLD_PATH = "/tmp/benchkit_pi_scaffold.json"
HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


def _message_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(
        str(part.get("text") or "")
        for part in content
        if isinstance(part, dict) and part.get("type") in {"text", "input_text"}
    )


def _capture_scaffold(request_data: dict) -> None:
    """Persist the exact scaffold sent by Pi on the first model request."""
    if os.path.exists(SCAFFOLD_PATH):
        return
    messages = request_data.get("messages") or []
    system_prompt = "\n\n".join(
        _message_text(message.get("content"))
        for message in messages
        if isinstance(message, dict) and message.get("role") in {"system", "developer"}
    )
    tools = request_data.get("tools") or []
    tool_names = [
        str(tool.get("function", {}).get("name") or "")
        for tool in tools
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
    ]
    tool_names = [name for name in tool_names if name]
    tool_schema = json.dumps(tools, ensure_ascii=False, sort_keys=True)
    payload = {
        "system_prompt": system_prompt,
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
        "system_prompt_chars": len(system_prompt),
        "tools_available": tool_names,
        "tool_schema_sha256": hashlib.sha256(tool_schema.encode()).hexdigest(),
    }
    for field in ("max_completion_tokens", "max_tokens"):
        value = request_data.get(field)
        if isinstance(value, int) and value > 0:
            payload["max_output_tokens"] = value
            payload["max_output_tokens_field"] = field
            break
    try:
        with open(SCAFFOLD_PATH, "x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
    except FileExistsError:
        pass


def _upstream_target(path: str) -> tuple[str, str, int, str]:
    base = os.environ["BENCHKIT_UPSTREAM"].rstrip("/")
    parsed = urlsplit(base)
    request = urlsplit(path)
    suffix = request.path[3:] if request.path.startswith("/v1") else request.path
    prefix = parsed.path.rstrip("/")
    target = f"{prefix}{suffix}" or "/"
    query = "&".join(part for part in (parsed.query, request.query) if part)
    if query:
        target += f"?{query}"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return parsed.scheme, parsed.hostname or "", port, target


def _models_payload() -> bytes:
    model = os.environ["BENCHKIT_MODEL"]
    return json.dumps(
        {"object": "list", "data": [{"id": model, "object": "model"}]}
    ).encode()


class ProxyHandler(BaseHTTPRequestHandler):
    """Stream a deliberately tiny subset of an OpenAI-compatible API."""

    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        if self.path == "/health":
            self._plain(200, b"ok")
            return
        self._forward()

    def do_POST(self) -> None:
        self._forward()

    def _forward(self) -> None:
        route = self.path.split("?", 1)[0]
        if route not in ALLOWED_PATHS:
            self._plain(404, b"route unavailable")
            return
        if route == "/v1/models":
            if self.command != "GET":
                self._plain(405, b"method unavailable")
                return
            self._json(200, _models_payload())
            return
        if self.command != "POST":
            self._plain(405, b"method unavailable")
            return

        length = int(self.headers.get("content-length", "0") or 0)
        if length > MAX_BODY_BYTES:
            self._plain(413, b"request too large")
            return
        body = self.rfile.read(length) if length else None
        try:
            request_data = json.loads(body or b"{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._plain(400, b"invalid JSON")
            return
        _capture_scaffold(request_data)
        if request_data.get("model") != os.environ["BENCHKIT_MODEL"]:
            self._plain(403, b"model unavailable")
            return

        scheme, host, port, target = _upstream_target(self.path)
        connection_type = (
            http.client.HTTPSConnection
            if scheme == "https"
            else http.client.HTTPConnection
        )
        connection = connection_type(host, port, timeout=600)
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in HOP_BY_HOP | {"host", "authorization"}
        }
        if api_key := os.environ.get("BENCHKIT_UPSTREAM_API_KEY"):
            headers["Authorization"] = f"Bearer {api_key}"

        response_started = False
        try:
            connection.request(self.command, target, body=body, headers=headers)
            response = connection.getresponse()
            self.send_response(response.status, response.reason)
            for key, value in response.getheaders():
                if key.lower() not in HOP_BY_HOP | {"content-length"}:
                    self.send_header(key, value)
            self.send_header("Connection", "close")
            self.end_headers()
            response_started = True
            while chunk := response.read(64 * 1024):
                self.wfile.write(chunk)
                self.wfile.flush()
        except (OSError, http.client.HTTPException) as exc:
            if not response_started and not self.wfile.closed:
                self._plain(502, f"upstream error: {exc}".encode())
        finally:
            connection.close()
            self.close_connection = True

    def _plain(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def _json(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def log_message(self, _format: str, *_args: object) -> None:
        return


def main() -> None:
    server = ThreadingHTTPServer(("0.0.0.0", 8080), ProxyHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
