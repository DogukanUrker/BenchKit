"""Persistent Docker environments for unmodified coding-agent harnesses."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit, urlunsplit

from benchkit.client import InferenceClient, _openai_base

_PI_IMAGE_SESSION = uuid.uuid4().hex[:12]
PI_IMAGE = f"benchkit-pi:{_PI_IMAGE_SESSION}"
AIDER_PI_IMAGE = f"benchkit-pi-aider-polyglot:{_PI_IMAGE_SESSION}"
GIT_SURGERY_PI_IMAGE = f"benchkit-pi-git-surgery:{_PI_IMAGE_SESSION}"
PI_PACKAGE = "@earendil-works/pi-coding-agent@0.84.2"
PI_VERSION = "0.84.2"
AIDER_POLYGLOT_COMMIT = "7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f"
_PROXY_SOURCE = Path(__file__).with_name("_pi_proxy.py")
_ANSWER_KEY_GUARD_SOURCE = Path(__file__).with_name("answer_key_guard.ts")
_PI_PACKAGE_ROOT = Path(__file__).with_name("pi_package")
_PATCHEVAL_IMAGE_SESSION = uuid.uuid4().hex[:12]
_BUILDKIT_DRIVER_IMAGE = "moby/buildkit:buildx-stable-1"

_PI_INSTALL = """\
COPY pi-package/package.json pi-package/package-lock.json /opt/benchkit/pi/
RUN cd /opt/benchkit/pi \\
    && npm ci --omit=dev \\
    && ln -s /opt/benchkit/pi/node_modules/.bin/pi /usr/local/bin/pi
"""

PI_DOCKERFILE = f"""\
FROM node:24-bookworm-slim

RUN apt-get update \\
    && apt-get install -y --no-install-recommends bash ca-certificates git python3 ripgrep \\
    && rm -rf /var/lib/apt/lists/*
{_PI_INSTALL}

COPY inference_proxy.py /opt/benchkit/inference_proxy.py
COPY answer_key_guard.ts /opt/benchkit/answer_key_guard.ts
RUN mkdir -p /workspace /home/node/.pi/agent \\
    && chown -R node:node /workspace /home/node/.pi

USER node
WORKDIR /workspace
ENV PI_OFFLINE=1 PI_TELEMETRY=0
CMD ["sleep", "infinity"]
"""

AIDER_PI_DOCKERFILE = f"""\
FROM node:24-bookworm

RUN apt-get update \\
    && apt-get install -y --no-install-recommends \\
       bash ca-certificates cmake curl g++ git libboost-date-time-dev \\
       openjdk-17-jdk \\
       python3 ripgrep unzip \\
    && rm -rf /var/lib/apt/lists/*
{_PI_INSTALL}
ARG TARGETARCH
RUN curl -fsSL "https://go.dev/dl/go1.21.5.linux-${{TARGETARCH}}.tar.gz" \\
      -o /tmp/go.tar.gz \\
    && tar -C /usr/local -xzf /tmp/go.tar.gz \\
    && rm /tmp/go.tar.gz
ENV PATH="/usr/local/go/bin:$PATH" RUSTUP_HOME="/opt/rustup" \\
    CARGO_HOME="/opt/cargo-home"
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \\
    | sh -s -- -y --no-modify-path
RUN mkdir -p /opt/javascript-deps \\
    && cd /opt/javascript-deps \\
    && npm init -y \\
    && npm install jest@29.7.0 @babel/core@7.25.2 \\
       @exercism/babel-preset-javascript@0.2.1 \\
       @exercism/eslint-config-javascript@0.6.0 @types/jest@29.5.12 \\
       @types/node@20.12.12 babel-jest@29.6.4 core-js@3.37.1 eslint@8.49.0
RUN curl -fsSL https://services.gradle.org/distributions/gradle-8.10.2-bin.zip \\
      -o /tmp/gradle.zip \\
    && unzip -q /tmp/gradle.zip -d /opt \\
    && rm /tmp/gradle.zip
ENV PATH="/opt/gradle-8.10.2/bin:/opt/javascript-deps/node_modules/.bin:/opt/cargo-home/bin:$PATH" \\
    NODE_PATH="/opt/javascript-deps/node_modules" \\
    GOPATH="/opt/go" GRADLE_USER_HOME="/opt/gradle-cache"
RUN git clone https://github.com/Aider-AI/polyglot-benchmark.git \\
      /opt/aider-polyglot \\
    && git -C /opt/aider-polyglot checkout {AIDER_POLYGLOT_COMMIT} \\
    && find /opt/aider-polyglot -name Cargo.toml -execdir cargo fetch \\; \\
    && find /opt/aider-polyglot/rust/exercises/practice \\
       -name Cargo-example.toml -exec sh -c '\
         for manifest do \
           cache_dir=$(mktemp -d); \
           mkdir "$cache_dir/src"; \
           cp "$manifest" "$cache_dir/Cargo.toml"; \
           cp "$(dirname "$manifest")/example.rs" "$cache_dir/src/lib.rs"; \
           cargo fetch --manifest-path "$cache_dir/Cargo.toml"; \
           rm -rf "$cache_dir"; \
         done' sh {{}} + \\
    && find /opt/aider-polyglot/go/exercises/practice -name go.mod \\
       -execdir go mod download \\; \\
    && find /opt/aider-polyglot/java/exercises/practice -name build.gradle \\
       -execdir gradle test --test-dry-run --no-daemon \\;
RUN find /opt/aider-polyglot -depth \\
      \\( -name .meta -o -name .approaches \\
         -o -iname '*example*' -o -iname '*reference*' -o -iname '*proof*' \\) \\
      -exec rm -rf {{}} + \\
    && rm -rf /opt/aider-polyglot/.git

COPY inference_proxy.py /opt/benchkit/inference_proxy.py
COPY answer_key_guard.ts /opt/benchkit/answer_key_guard.ts
RUN mkdir -p /workspace /home/node/.pi/agent /opt/go \\
    && chown -R node:node /workspace /home/node/.pi /opt/cargo-home \\
       /opt/rustup /opt/go /opt/gradle-cache \\
    && chmod -R a+rX /opt/aider-polyglot

USER node
WORKDIR /workspace
ENV PI_OFFLINE=1 PI_TELEMETRY=0 CARGO_NET_OFFLINE=true
CMD ["sleep", "infinity"]
"""

GIT_SURGERY_PI_DOCKERFILE = f"""\
FROM node:24-bookworm-slim

ARG GIT_DEBIAN_VERSION=1:2.39.5-0+deb12u3
RUN apt-get update \\
    && apt-get install -y --no-install-recommends \\
       bash ca-certificates git=${{GIT_DEBIAN_VERSION}} python3 ripgrep \\
    && test "$(git --version)" = "git version 2.39.5" \\
    && rm -rf /var/lib/apt/lists/*
{_PI_INSTALL}

COPY inference_proxy.py /opt/benchkit/inference_proxy.py
COPY answer_key_guard.ts /opt/benchkit/answer_key_guard.ts
COPY git-surgery /opt/git-surgery
RUN chmod +x /opt/git-surgery/*/setup.sh /opt/git-surgery/*/verify.sh \\
    && mkdir -p /workspace /home/node/.pi/agent \\
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


def cleanup_owned_resources(docker: str, owner_id: str) -> None:
    """Remove every container and network belonging to one Pi runner."""
    label = f"benchkit.owner={owner_id}"
    errors: list[str] = []

    def listed(command: list[str], kind: str) -> list[str]:
        try:
            result = _run(command, timeout=30, check=False)
        except Exception as exc:
            errors.append(f"could not inspect owned {kind}: {exc}")
            return []
        if result.returncode:
            detail = _tail(result.stderr or result.stdout) or "unknown Docker error"
            errors.append(f"could not inspect owned {kind}: {detail}")
            return []
        return result.stdout.split()

    container_query = [
        docker,
        "ps",
        "--all",
        "--quiet",
        "--filter",
        f"label={label}",
    ]
    network_query = [
        docker,
        "network",
        "ls",
        "--quiet",
        "--filter",
        f"label={label}",
    ]
    containers = listed(container_query, "containers")
    if containers:
        with contextlib.suppress(Exception):
            _run([docker, "rm", "--force", *containers], timeout=30, check=False)
    networks = listed(network_query, "networks")
    if networks:
        with contextlib.suppress(Exception):
            _run([docker, "network", "rm", *networks], timeout=30, check=False)

    remaining_containers = listed(container_query, "containers after cleanup")
    remaining_networks = listed(network_query, "networks after cleanup")
    if remaining_containers:
        errors.append("owned containers remain: " + ", ".join(remaining_containers))
    if remaining_networks:
        errors.append("owned networks remain: " + ", ".join(remaining_networks))
    if errors:
        raise SandboxError("Pi Docker cleanup failed: " + "; ".join(errors))


def _verify_absent(
    command: list[str], description: str, errors: list[str], *, timeout: int = 30
) -> None:
    """Record an error unless an exact Docker resource is confirmed absent."""
    try:
        result = _run(command, timeout=timeout, check=False)
    except Exception as exc:
        errors.append(f"could not verify {description}: {exc}")
        return
    if result.returncode == 0:
        errors.append(f"{description} remains")


@dataclass
class LatestPiImage:
    """Build the reproducibly pinned stock Pi image once per run."""

    docker: str = field(default_factory=_docker_binary)
    image: str = PI_IMAGE
    dockerfile: str = PI_DOCKERFILE
    no_cache: bool = True
    transient: bool = True
    pids_limit: int = 256
    build_assets: Path | None = None
    build_files: tuple[tuple[Path, str], ...] = ()
    ephemeral_buildx: bool = True
    always_cleanup_image: bool = True
    resource_scope: str = "pi"
    answer_key_guard: bool = False
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
                (context / "Dockerfile").write_text(self.dockerfile, encoding="utf-8")
                (context / "inference_proxy.py").write_text(
                    _PROXY_SOURCE.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
                (context / "answer_key_guard.ts").write_text(
                    _ANSWER_KEY_GUARD_SOURCE.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
                shutil.copytree(_PI_PACKAGE_ROOT, context / "pi-package")
                if self.build_assets is not None:
                    shutil.copytree(self.build_assets, context / "git-surgery")
                allowed = {
                    "Dockerfile",
                    "inference_proxy.py",
                    "answer_key_guard.ts",
                    "pi-package",
                    "git-surgery",
                }
                for source, destination in self.build_files:
                    target = PurePosixPath(destination)
                    if (
                        target.is_absolute()
                        or len(target.parts) != 1
                        or target.name in allowed
                        or source.is_symlink()
                        or not source.is_file()
                    ):
                        raise SandboxError("invalid explicit Pi build-context file")
                    shutil.copyfile(source, context / target.name)
                    allowed.add(target.name)
                dockerignore = ["*", *sorted(f"!{item}" for item in allowed)]
                dockerignore.extend(("!pi-package/**", "!git-surgery/**"))
                (context / ".dockerignore").write_text(
                    "\n".join(dockerignore) + "\n", encoding="utf-8"
                )
                try:
                    if self.ephemeral_buildx:
                        _run(
                            [
                                self.docker,
                                "image",
                                "rm",
                                "--force",
                                self.image,
                            ],
                            timeout=60,
                            check=False,
                        )
                        _build_with_ephemeral_buildx(
                            self.docker,
                            self.image,
                            context,
                            self.resource_scope,
                            no_cache=self.no_cache,
                        )
                    else:
                        command = [self.docker, "build", "--pull"]
                        if self.no_cache:
                            command.append("--no-cache")
                        command.extend(["--tag", self.image, str(context)])
                        _run(command, timeout=1800)
                    if self.ephemeral_buildx:
                        smoke = (
                            f"benchkit-{self.resource_scope}-smoke-"
                            f"{uuid.uuid4().hex[:16]}"
                        )
                        try:
                            result = _run(
                                [
                                    self.docker,
                                    "run",
                                    "--name",
                                    smoke,
                                    "--rm",
                                    self.image,
                                    "pi",
                                    "--version",
                                ],
                                timeout=30,
                            )
                        finally:
                            with contextlib.suppress(Exception):
                                _run(
                                    [self.docker, "rm", "--force", smoke],
                                    timeout=30,
                                    check=False,
                                )
                    else:
                        result = _run(
                            [
                                self.docker,
                                "run",
                                "--rm",
                                self.image,
                                "pi",
                                "--version",
                            ],
                            timeout=30,
                        )
                    self.version = result.stdout.strip() or "latest"
                    if self.version != PI_VERSION:
                        raise SandboxError(
                            f"Pi image reported {self.version!r}; "
                            f"expected {PI_VERSION!r}"
                        )
                except BaseException:
                    if self.transient:
                        with contextlib.suppress(Exception):
                            _run(
                                [
                                    self.docker,
                                    "image",
                                    "rm",
                                    "--force",
                                    self.image,
                                ],
                                timeout=60,
                                check=False,
                            )
                    self.version = ""
                    raise
            self._ready = True
            return self.version

    def cleanup(self) -> None:
        """Remove the transient Pi image after its benchmark run."""
        if not self.transient or (not self._ready and not self.always_cleanup_image):
            return
        try:
            _run(
                [self.docker, "image", "rm", "--force", self.image],
                timeout=60,
                check=not self.always_cleanup_image,
            )
            if self.always_cleanup_image:
                errors: list[str] = []
                _verify_absent(
                    [self.docker, "image", "inspect", self.image],
                    f"transient image {self.image}",
                    errors,
                )
                if errors:
                    raise SandboxError("Pi Docker cleanup failed: " + "; ".join(errors))
        finally:
            self._ready = False
            self.version = ""


def _build_with_ephemeral_buildx(
    docker: str,
    image: str,
    context: Path,
    resource_scope: str,
    *,
    no_cache: bool = True,
) -> None:
    """Build without shared cache, then remove the private buildkit session."""
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", resource_scope):
        raise SandboxError("invalid Pi Docker resource scope")
    builder = f"benchkit-{resource_scope}-build-{uuid.uuid4().hex[:16]}"
    buildkit_container = f"buildx_buildkit_{builder}0"
    cache_volume = f"buildx_buildkit_{builder}0_state"
    driver_was_present = bool(
        _run(
            [
                docker,
                "image",
                "inspect",
                "--format",
                "{{.Id}}",
                _BUILDKIT_DRIVER_IMAGE,
            ],
            timeout=30,
            check=False,
        ).stdout.strip()
    )
    try:
        _run(
            [
                docker,
                "buildx",
                "create",
                "--name",
                builder,
                "--driver",
                "docker-container",
                "--driver-opt",
                f"image={_BUILDKIT_DRIVER_IMAGE}",
            ],
            timeout=60,
        )
        build = [docker, "buildx", "build", "--builder", builder, "--pull"]
        if no_cache:
            build.append("--no-cache")
        build.extend(["--load", "--tag", image, str(context)])
        _run(build, timeout=1800)
    finally:
        cleanup_commands = (
            ([docker, "buildx", "rm", "--force", builder], 60),
            ([docker, "rm", "--force", buildkit_container], 30),
            ([docker, "volume", "rm", "--force", cache_volume], 30),
        )
        for command, timeout in cleanup_commands:
            with contextlib.suppress(Exception):
                _run(command, timeout=timeout, check=False)
        if not driver_was_present:
            with contextlib.suppress(Exception):
                _run(
                    [docker, "image", "rm", "--force", _BUILDKIT_DRIVER_IMAGE],
                    timeout=60,
                    check=False,
                )
        errors: list[str] = []
        _verify_absent(
            [docker, "buildx", "inspect", builder],
            f"Buildx builder {builder}",
            errors,
        )
        _verify_absent(
            [docker, "container", "inspect", buildkit_container],
            f"BuildKit container {buildkit_container}",
            errors,
        )
        _verify_absent(
            [docker, "volume", "inspect", cache_volume],
            f"BuildKit volume {cache_volume}",
            errors,
        )
        if not driver_was_present:
            _verify_absent(
                [docker, "image", "inspect", _BUILDKIT_DRIVER_IMAGE],
                f"BuildKit driver image {_BUILDKIT_DRIVER_IMAGE}",
                errors,
            )
        if errors:
            raise SandboxError("Pi Docker cleanup failed: " + "; ".join(errors))


def aider_pi_image() -> LatestPiImage:
    """Return an image definition containing every Aider Polyglot toolchain."""
    return LatestPiImage(
        image=AIDER_PI_IMAGE,
        dockerfile=AIDER_PI_DOCKERFILE,
        no_cache=False,
        transient=True,
        # cpp/bank-account creates 1,000 simultaneous std::threads. Linux
        # accounts threads against Docker's PID cgroup, so the generic Pi
        # sandbox limit of 256 makes the official test suite impossible.
        pids_limit=2048,
        answer_key_guard=True,
        resource_scope="aider-polyglot",
    )


def git_surgery_pi_image() -> LatestPiImage:
    """Return the stock Pi image with pinned Git and Git Surgery assets."""
    return LatestPiImage(
        image=GIT_SURGERY_PI_IMAGE,
        dockerfile=GIT_SURGERY_PI_DOCKERFILE,
        no_cache=False,
        transient=True,
        build_assets=Path(__file__).with_name("git_surgery"),
        resource_scope="git-surgery",
    )


@dataclass(frozen=True)
class PatchEvalRuntimeRecipe:
    """Trusted build instructions shipped with one validated PatchEval task."""

    base_image: str
    sync_command: tuple[str, ...]
    bootstrap_command: tuple[str, ...] = ()
    environment: tuple[str, ...] = ()
    schema_version: int = 1


def _regular_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Pi package build assets must not contain symlinks")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_regular_file_sha256(path)))
    return digest.hexdigest()


def _docker_run(command: tuple[str, ...]) -> str:
    return f"RUN {json.dumps(list(command), separators=(',', ':'))}\n"


def _patcheval_dockerfile(recipe: PatchEvalRuntimeRecipe) -> str:
    bootstrap = (
        _docker_run(recipe.bootstrap_command) if recipe.bootstrap_command else ""
    )
    environment = "".join(
        f"ENV {key}={json.dumps(value)}\n"
        for key, value in (item.split("=", 1) for item in recipe.environment)
    )
    return f"""\
FROM node:24-bookworm-slim AS benchkit-node
FROM {recipe.base_image}

USER root
COPY --from=benchkit-node /usr/local/ /usr/local/
RUN apt-get update \\
    && apt-get install -y --no-install-recommends \\
       bash ca-certificates coreutils git passwd ripgrep tar \\
    && rm -rf /var/lib/apt/lists/* \\
    && (getent group node >/dev/null || groupadd --gid 1000 node) \\
    && (id --user node >/dev/null 2>&1 || \\
        useradd --uid 1000 --gid node --create-home --shell /bin/bash node)
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_LINK_MODE=copy
WORKDIR /opt/project
COPY parent-source.tar /tmp/patcheval-parent.tar
RUN tar -xf /tmp/patcheval-parent.tar -C /opt/project \\
    && rm /tmp/patcheval-parent.tar
{_docker_run(recipe.sync_command)}{bootstrap}RUN rm -rf /opt/project \\
    && test -d /opt/venv \\
    && chmod -R a+rwX /opt/venv
{_PI_INSTALL}
COPY inference_proxy.py /opt/benchkit/inference_proxy.py
COPY answer_key_guard.ts /opt/benchkit/answer_key_guard.ts
RUN mkdir -p /workspace /home/node/.pi/agent \\
    && chown -R node:node /workspace /home/node/.pi

{environment}USER node
WORKDIR /workspace
ENV PATH=/opt/venv/bin:$PATH HOME=/home/node USER=node LOGNAME=node \\
    PI_OFFLINE=1 PI_TELEMETRY=0
CMD ["sleep", "infinity"]
"""


def patcheval_pi_image(
    source_archive: Path,
    source_sha256: str,
    recipe: PatchEvalRuntimeRecipe,
) -> LatestPiImage:
    """Build one task runtime locally without retaining buildkit state."""
    if source_archive.is_symlink() or not source_archive.is_file():
        raise ValueError("PatchEval source archive must be a regular file")
    if not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
        raise ValueError("PatchEval source_sha256 must be a lowercase SHA-256")
    if _regular_file_sha256(source_archive) != source_sha256:
        raise ValueError("PatchEval source archive checksum mismatch")
    dockerfile = _patcheval_dockerfile(recipe)
    recipe_json = json.dumps(
        {
            "schema_version": recipe.schema_version,
            "base_image": recipe.base_image,
            "sync_command": list(recipe.sync_command),
            "bootstrap_command": list(recipe.bootstrap_command),
            "environment": list(recipe.environment),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    identity = hashlib.sha256()
    for value in (
        dockerfile,
        source_sha256,
        recipe_json,
        _tree_sha256(_PI_PACKAGE_ROOT),
        _regular_file_sha256(_PROXY_SOURCE),
        _regular_file_sha256(_ANSWER_KEY_GUARD_SOURCE),
    ):
        identity.update(value.encode())
        identity.update(b"\0")
    image_id = identity.hexdigest()[:16]
    return LatestPiImage(
        image=f"benchkit-pi-patcheval:{_PATCHEVAL_IMAGE_SESSION}-{image_id}",
        dockerfile=dockerfile,
        no_cache=True,
        transient=True,
        build_files=((source_archive, "parent-source.tar"),),
        ephemeral_buildx=True,
        always_cleanup_image=True,
        resource_scope="patcheval",
    )


@dataclass
class DockerTaskEnvironment:
    """One isolated, persistent workspace shared by every native Pi tool call."""

    client: InferenceClient
    model: str
    image: LatestPiImage
    docker: str = field(default_factory=_docker_binary)
    owner_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    workdir: str = "/workspace"
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
            resource_labels = [
                "--label",
                "benchkit.managed=true",
                "--label",
                f"benchkit.owner={self.owner_id}",
            ]
            _run(
                [
                    self.docker,
                    "network",
                    "create",
                    "--internal",
                    *resource_labels,
                    self.network_name,
                ]
            )
            proxy_args = [
                self.docker,
                "run",
                "--detach",
                "--name",
                self.proxy_name,
                *resource_labels,
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
                "--env",
                f"BENCHKIT_UPSTREAM_TIMEOUT={self.client.timeout}",
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
            pids = os.environ.get("BENCHKIT_SANDBOX_PIDS", str(self.image.pids_limit))
            _run(
                [
                    self.docker,
                    "run",
                    "--detach",
                    "--name",
                    self.container_name,
                    *resource_labels,
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
        command = [
            self.docker,
            "exec",
            "--interactive",
            "--workdir",
            self.workdir,
            self.container_name,
            "pi",
            "--mode",
            "rpc",
            "--no-session",
            "--provider",
            "benchkit",
            "--model",
            self.model,
        ]
        if self.image.answer_key_guard:
            command.extend(["--extension", "/opt/benchkit/answer_key_guard.ts"])
        return subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def pi_scaffold(self) -> dict:
        """Read scaffold metadata captured from Pi's first inference request."""
        result = _run(
            [
                self.docker,
                "exec",
                self.proxy_name,
                "cat",
                "/tmp/benchkit_pi_scaffold.json",
            ],
            timeout=10,
            check=False,
        )
        if result.returncode:
            return {}
        try:
            payload = json.loads(result.stdout)
        except (json.JSONDecodeError, TypeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def exec(
        self,
        command: list[str],
        *,
        workdir: str | None = None,
        input_text: str | None = None,
        timeout: float | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        return _run(
            [
                self.docker,
                "exec",
                *(["--workdir", workdir] if workdir is not None else []),
                self.container_name,
                *command,
            ],
            input_text=input_text,
            timeout=timeout,
            check=check,
        )

    def upload(self, source: Path, destination: str) -> None:
        _run([self.docker, "cp", str(source), f"{self.container_name}:{destination}"])

    def download(self, source: str, destination: Path) -> None:
        _run([self.docker, "cp", f"{self.container_name}:{source}", str(destination)])

    def stop(self) -> None:
        try:
            commands = [
                [self.docker, "rm", "--force", self.container_name],
                [self.docker, "rm", "--force", self.proxy_name],
                [self.docker, "network", "rm", self.network_name],
            ]
            for command in commands:
                with contextlib.suppress(Exception):
                    _run(command, timeout=30, check=False)
        finally:
            self._started = False
