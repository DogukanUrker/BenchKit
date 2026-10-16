"""What each sandbox image contains.

The Dockerfiles for the Pi harness (stock, Aider Polyglot toolchains, Git
Surgery, PatchEval task runtimes) and the mc-arena script runner. Nothing here
talks to Docker: :mod:`benchkit.sandbox` builds, runs and removes the images.
Every tool and base image is pinned, so a change here changes what a run
measures.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

AIDER_POLYGLOT_COMMIT = "7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f"
_UV_CACHE_DIR = "/root/.cache/uv"


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
    && git config --system protocol.file.allow always \\
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


# mc-arena runs one untrusted, model-written Python script per task. The script
# only has to print JSON, so the image is a stock Python plus uv: the suite's
# answer contract is a PEP 723 script, and uv is what runs those.
MC_ARENA_UV_VERSION = "0.8.17"
MC_ARENA_DEFAULT_BASE = "python:3.12-slim"


def _mc_arena_base_image() -> str:
    """The Python base image, overridable for hosts behind a registry mirror."""
    return (
        os.environ.get("BENCHKIT_MC_ARENA_BASE_IMAGE") or MC_ARENA_DEFAULT_BASE
    ).strip()


def mc_arena_dockerfile() -> str:
    return f"""\
FROM {_mc_arena_base_image()}

RUN pip install --no-cache-dir uv=={MC_ARENA_UV_VERSION} \\
    && useradd --create-home --uid 1000 runner \\
    && mkdir -p /work && chown runner:runner /work

USER runner
WORKDIR /work
ENV HOME=/home/runner UV_NO_PROGRESS=1 UV_OFFLINE=1
CMD ["sleep", "infinity"]
"""


@dataclass(frozen=True)
class PatchEvalRuntimeRecipe:
    """Trusted build instructions shipped with one validated PatchEval task."""

    base_image: str
    sync_command: tuple[str, ...]
    bootstrap_command: tuple[str, ...] = ()
    environment: tuple[str, ...] = ()
    schema_version: int = 1


def _docker_run_cached(command: tuple[str, ...], uv_cache_id: str) -> str:
    """Run a build command with the run's shared, private uv cache mounted."""
    mount = f"--mount=type=cache,target={_UV_CACHE_DIR},id={uv_cache_id},sharing=locked"
    return f"RUN {mount} {json.dumps(list(command), separators=(',', ':'))}\n"


def patcheval_dockerfile(recipe: PatchEvalRuntimeRecipe, uv_cache_id: str) -> str:
    """Generate the three-stage task runtime.

    The stages are split by what actually changes between tasks:

    1. ``benchkit-pi-assets`` depends only on the base image and the pinned Pi
       assets, so the run's shared builder builds it once.
    2. ``benchkit-runtime`` adds the parent source and the frozen sync and
       bootstrap commands, and reuses the run's uv cache.
    3. the final stage only adds task environment and the agent user.
    """
    bootstrap = (
        _docker_run_cached(recipe.bootstrap_command, uv_cache_id)
        if recipe.bootstrap_command
        else ""
    )
    environment = "".join(
        f"ENV {key}={json.dumps(value)}\n"
        for key, value in (item.split("=", 1) for item in recipe.environment)
    )
    return f"""\
FROM node:24-bookworm-slim AS benchkit-node

FROM {recipe.base_image} AS benchkit-pi-assets

USER root
COPY --from=benchkit-node /usr/local/ /usr/local/
RUN apt-get update \\
    && apt-get install -y --no-install-recommends \\
       bash ca-certificates coreutils git passwd ripgrep tar \\
    && rm -rf /var/lib/apt/lists/* \\
    && (getent group node >/dev/null || groupadd --gid 1000 node) \\
    && (id --user node >/dev/null 2>&1 || \\
        useradd --uid 1000 --gid node --create-home --shell /bin/bash node)
{_PI_INSTALL}
COPY inference_proxy.py /opt/benchkit/inference_proxy.py
COPY answer_key_guard.ts /opt/benchkit/answer_key_guard.ts
RUN mkdir -p /workspace /home/node/.pi/agent \\
    && chown -R node:node /workspace /home/node/.pi

FROM benchkit-pi-assets AS benchkit-runtime

ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_LINK_MODE=copy
WORKDIR /opt/project
COPY parent-source.tar /tmp/patcheval-parent.tar
RUN tar -xf /tmp/patcheval-parent.tar -C /opt/project \\
    && find /opt/project -exec touch -t 198001010000 {{}} + \\
    && rm /tmp/patcheval-parent.tar
{_docker_run_cached(recipe.sync_command, uv_cache_id)}{bootstrap}RUN rm -rf /opt/project \\
    && test -d /opt/venv \\
    && chmod -R a+rwX /opt/venv

FROM benchkit-runtime

{environment}USER node
WORKDIR /workspace
ENV PATH=/opt/venv/bin:$PATH HOME=/home/node USER=node LOGNAME=node \\
    PI_OFFLINE=1 PI_TELEMETRY=0
CMD ["sleep", "infinity"]
"""
