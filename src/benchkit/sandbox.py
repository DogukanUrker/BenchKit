"""Persistent Docker environments for unmodified coding-agent harnesses."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from benchkit.client import InferenceClient, _openai_base

PI_IMAGE = "benchkit-pi:latest"
PI_PACKAGE = "@earendil-works/pi-coding-agent@latest"
_PROXY_SOURCE = Path(__file__).with_name("_pi_proxy.py")

PI_DOCKERFILE = f"""\
FROM node:24-bookworm-slim

RUN apt-get update \\
    && apt-get install -y --no-install-recommends bash ca-certificates git python3 ripgrep \\
    && rm -rf /var/lib/apt/lists/*
RUN npm install -g {PI_PACKAGE}

COPY inference_proxy.py /opt/benchkit/inference_proxy.py
RUN mkdir -p /workspace /home/node/.pi/agent \\
    && chown -R node:node /workspace /home/node/.pi

USER node
WORKDIR /workspace
ENV PI_OFFLINE=1 PI_TELEMETRY=0
CMD ["sleep", "infinity"]
"""


class SandboxError(RuntimeError):
    """The Docker-backed agent environment could not be prepared or used."""


def _docker_binary() -> str:
    requested = os.environ.get("BENCHKIT_DOCKER", "docker")
    resolved = shutil.which(requested)
    if resolved is None:
        raise SandboxError(
            "Pi harness requires Docker; install Docker and ensure `docker` is on PATH"
        )
    return resolved


def _tail(text: str, limit: int = 4000) -> str:
    text = text.strip()
    return text[-limit:] if len(text) > limit else text


def _run(
    args: list[str],
    *,
    input_text: str | None = None,
    timeout: float | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            args,
            input=input_text,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise SandboxError(f"Docker command timed out: {args[1]}") from exc
    except OSError as exc:
        raise SandboxError(f"Could not run Docker: {exc}") from exc
    if check and completed.returncode:
        detail = _tail(completed.stderr or completed.stdout) or "unknown Docker error"
        raise SandboxError(f"Docker {args[1]} failed: {detail}")
    return completed


def _docker_upstream(url: str) -> str:
    """Make a host-local inference URL reachable from the proxy container."""
    parsed = urlsplit(_openai_base(url))
    host = parsed.hostname or ""
    if host not in {"localhost", "127.0.0.1", "::1"}:
        return urlunsplit(parsed)
    replacement = "host.docker.internal"
    if parsed.port:
        replacement += f":{parsed.port}"
    return urlunsplit((parsed.scheme, replacement, parsed.path, parsed.query, ""))


@dataclass
class LatestPiImage:
    """Build the stock Pi image once per run, resolving npm's latest release."""

    docker: str = field(default_factory=_docker_binary)
    image: str = PI_IMAGE
    version: str = ""
    _ready: bool = field(default=False, init=False, repr=False)
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def prepare(self) -> str:
        if self._ready:
            return self.version
        with self._lock:
            if self._ready:
                return self.version
            _run(
                [self.docker, "version", "--format", "{{.Server.Version}}"],
                timeout=10,
            )
            with tempfile.TemporaryDirectory(prefix="benchkit-pi-build-") as directory:
                context = Path(directory)
                (context / "Dockerfile").write_text(PI_DOCKERFILE, encoding="utf-8")
                (context / "inference_proxy.py").write_text(
                    _PROXY_SOURCE.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
                _run(
                    [
                        self.docker,
                        "build",
                        "--pull",
                        "--no-cache",
                        "--tag",
                        self.image,
                        str(context),
                    ],
                    timeout=900,
                )
            result = _run(
                [self.docker, "run", "--rm", self.image, "pi", "--version"],
                timeout=30,
            )
            self.version = result.stdout.strip() or "latest"
            self._ready = True
            return self.version

    def cleanup(self) -> None:
        """Remove the transient Pi image after its benchmark run."""
        if not self._ready:
            return
        _run(
            [self.docker, "image", "rm", "--force", self.image],
            timeout=60,
        )
        self._ready = False
        self.version = ""


@dataclass
class DockerTaskEnvironment:
    """One isolated, persistent workspace shared by every native Pi tool call."""

    client: InferenceClient
    model: str
    image: LatestPiImage
    docker: str = field(default_factory=_docker_binary)
    task_name: str = field(default_factory=lambda: f"benchkit-{uuid.uuid4().hex[:12]}")
    network_name: str = field(init=False)
    proxy_name: str = field(init=False)
    container_name: str = field(init=False)
    _started: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.network_name = f"{self.task_name}-net"
        self.proxy_name = f"{self.task_name}-proxy"
        self.container_name = f"{self.task_name}-agent"

    def __enter__(self) -> DockerTaskEnvironment:
        self.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.stop()

    def start(self) -> None:
        if self._started:
            return
        if not self.image.version:
            raise SandboxError("Pi image must be prepared before starting a task")
        try:
            _run([self.docker, "network", "create", "--internal", self.network_name])
            proxy_args = [
                self.docker,
                "run",
                "--detach",
                "--name",
                self.proxy_name,
                "--network",
                self.network_name,
                "--network-alias",
                "inference",
                "--add-host",
                "host.docker.internal:host-gateway",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--pids-limit",
                "64",
                "--memory",
                "256m",
                "--env",
                f"BENCHKIT_UPSTREAM={_docker_upstream(self.client.host)}",
                "--env",
                f"BENCHKIT_MODEL={self.model}",
            ]
            if self.client.api_key:
                proxy_args.extend(
                    ["--env", f"BENCHKIT_UPSTREAM_API_KEY={self.client.api_key}"]
                )
            proxy_args.extend(
                [self.image.image, "python3", "/opt/benchkit/inference_proxy.py"]
            )
            _run(proxy_args)
            _run([self.docker, "network", "connect", "bridge", self.proxy_name])

            memory = os.environ.get("BENCHKIT_SANDBOX_MEMORY", "2g")
            cpus = os.environ.get("BENCHKIT_SANDBOX_CPUS", "2")
            pids = os.environ.get("BENCHKIT_SANDBOX_PIDS", "256")
            _run(
                [
                    self.docker,
                    "run",
                    "--detach",
                    "--name",
                    self.container_name,
                    "--network",
                    self.network_name,
                    "--cap-drop",
                    "ALL",
                    "--security-opt",
                    "no-new-privileges",
                    "--pids-limit",
                    pids,
                    "--memory",
                    memory,
                    "--cpus",
                    cpus,
                    self.image.image,
                ]
            )
            self._write_model_config()
            self._started = True
        except Exception:
            self.stop()
            raise

    def _write_model_config(self) -> None:
        context_length = None
        discover = getattr(self.client, "context_length", None)
        if callable(discover):
            context_length = discover(self.model)
        model: dict[str, object] = {"id": self.model, "name": self.model}
        if isinstance(context_length, int) and context_length > 0:
            model["contextWindow"] = context_length

        provider: dict[str, object] = {
            "baseUrl": "http://inference:8080/v1",
            "api": "openai-completions",
            "apiKey": "benchkit-sandbox",
            "models": [model],
        }
        if self.client.provider == "ollama":
            provider["compat"] = {
                "supportsDeveloperRole": False,
                "supportsReasoningEffort": False,
            }
        payload = json.dumps({"providers": {"benchkit": provider}})
        _run(
            [
                self.docker,
                "exec",
                "--interactive",
                self.container_name,
                "tee",
                "/home/node/.pi/agent/models.json",
            ],
            input_text=payload,
        )

    def start_pi(self) -> subprocess.Popen[str]:
        if not self._started:
            raise SandboxError("task environment is not running")
        return subprocess.Popen(
            [
                self.docker,
                "exec",
                "--interactive",
                "--workdir",
                "/workspace",
                self.container_name,
                "pi",
                "--mode",
                "rpc",
                "--no-session",
                "--provider",
                "benchkit",
                "--model",
                self.model,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def exec(
        self,
        command: list[str],
        *,
        input_text: str | None = None,
        timeout: float | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        return _run(
            [self.docker, "exec", self.container_name, *command],
            input_text=input_text,
            timeout=timeout,
            check=check,
        )

    def upload(self, source: Path, destination: str) -> None:
        _run([self.docker, "cp", str(source), f"{self.container_name}:{destination}"])

    def download(self, source: str, destination: Path) -> None:
        _run([self.docker, "cp", f"{self.container_name}:{source}", str(destination)])

    def stop(self) -> None:
        for name in (self.container_name, self.proxy_name):
            _run(
                [self.docker, "rm", "--force", name],
                timeout=30,
                check=False,
            )
        _run(
            [self.docker, "network", "rm", self.network_name],
            timeout=30,
            check=False,
        )
        self._started = False
