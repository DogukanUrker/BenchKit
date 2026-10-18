"""Layered configuration: which settings apply, and where each one came from.

Precedence, highest first: CLI flags, real environment variables,
``--env-file PATH`` (or ``BENCHKIT_ENV_FILE``), ``./.env`` in the working
directory, the user config ``$XDG_CONFIG_HOME/benchkit/.env`` (default
``~/.config/benchkit/.env``), then built-in defaults. An explicit env file
replaces ``./.env`` rather than stacking on it.

Only BenchKit's own keys are applied from any file, so running inside another
project never imports that project's secrets. Parent directories are never
searched: a ``.env`` in a shared parent such as ``/tmp`` could otherwise
redirect the host and API key.
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values, set_key, unset_key

from benchkit.client import DEFAULT_HOST

ENV_FILE_VAR = "BENCHKIT_ENV_FILE"
_PREFIXES = ("BENCHKIT_", "EVALPLUS_")
_NAMES = frozenset({"OLLAMA_HOST", "OPENAI_API_KEY", "OPENAI_BASE_URL"})
_SECRET_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET", "_PASSWORD")

# Settings `benchkit config` always lists, with the default each falls back to
# ("" when unset simply means "not used"). Internal variables the harness sets
# for its own child processes are left out.
# fmt: off
SETTINGS: tuple[tuple[str, str], ...] = (
    ("BENCHKIT_PROVIDER",             "auto"),
    ("BENCHKIT_HOST",                 DEFAULT_HOST),
    ("OPENAI_BASE_URL",               ""),
    ("OLLAMA_HOST",                   ""),
    ("BENCHKIT_API_KEY",              ""),
    ("OPENAI_API_KEY",                ""),
    ("BENCHKIT_TIMEOUT",              "300"),
    ("BENCHKIT_CONCURRENCY",          ""),
    ("BENCHKIT_HARDWARE",             ""),
    ("BENCHKIT_PERF_CONFIG",          ""),
    ("BENCHKIT_RETRIES",              "3"),
    ("BENCHKIT_CONNECT_RETRIES",      "6"),
    ("BENCHKIT_RETRY_BASE_DELAY",     "0.5"),
    ("BENCHKIT_RETRY_MAX_DELAY",      "8"),
    ("BENCHKIT_RETRY_MAX_WAIT",       "60"),
    ("BENCHKIT_LOOP_KILL",            "true"),
    ("BENCHKIT_LOOP_KILL_PERCENT",    "80"),
    ("BENCHKIT_LOOP_KILL_SECONDS",    "10"),
    ("BENCHKIT_PI_TIMEOUT",           "900"),
    ("BENCHKIT_DOCKER",               "docker"),
    ("BENCHKIT_SANDBOX_MEMORY",       "2g"),
    ("BENCHKIT_SANDBOX_CPUS",         "2"),
    ("BENCHKIT_SANDBOX_PIDS",         ""),
    ("BENCHKIT_MC_ARENA_MEMORY",      "512m"),
    ("BENCHKIT_MC_ARENA_CPUS",        "1"),
    ("BENCHKIT_MC_ARENA_PIDS",        "64"),
    ("BENCHKIT_MC_ARENA_BASE_IMAGE",  ""),
    ("BENCHKIT_MC_ARENA_IMAGE",       ""),
    ("BENCHKIT_RENDER_SETTLE",        "4"),
    ("BENCHKIT_RENDER_TIMEOUT",       "30"),
    ("BENCHKIT_RENDER_ALLOWED_HOSTS", ""),
    ("BENCHKIT_RENDER_OFFLINE",       "false"),
    ("BENCHKIT_RENDER_CHROMIUM",      ""),
    ("BENCHKIT_RENDER_NO_SANDBOX",    "false"),
    ("BENCHKIT_DATASET_CACHE",        ""),
    ("BENCHKIT_VISION_CACHE",         ""),
    ("BENCHKIT_PATCHEVAL_DATASET",    ""),
    ("BENCHKIT_AIDER_DATASET",        ""),
)
# fmt: on


class ConfigError(Exception):
    """A configuration source was named but cannot be used."""


@dataclass(frozen=True)
class EnvFile:
    """One file BenchKit considered, in precedence order."""

    path: Path
    role: str
    exists: bool
    skipped: str = ""


@dataclass
class Loaded:
    """The files considered and the origin of every BenchKit key applied."""

    files: list[EnvFile] = field(default_factory=list)
    origins: dict[str, str] = field(default_factory=dict)


_loaded = Loaded()


def is_benchkit_key(key: str) -> bool:
    return key.startswith(_PREFIXES) or key in _NAMES


def is_secret(key: str) -> bool:
    return key.endswith(_SECRET_SUFFIXES)


def mask(value: str) -> str:
    """Hide a secret, keeping the last four characters of a long one."""
    if not value:
        return ""
    if len(value) <= 8:
        return "****"
    return "****" + value[-4:]


def user_config_dir(environ: MutableMapping[str, str] | None = None) -> Path:
    environ = os.environ if environ is None else environ
    base = environ.get("XDG_CONFIG_HOME") or ""
    root = Path(base).expanduser() if base else Path("~/.config").expanduser()
    return root / "benchkit"


def user_env_path(environ: MutableMapping[str, str] | None = None) -> Path:
    return user_config_dir(environ) / ".env"


def load(
    env_file: str | None = None,
    *,
    environ: MutableMapping[str, str] | None = None,
    cwd: Path | None = None,
) -> Loaded:
    """Apply BenchKit keys from the config files into ``environ``.

    A key already in the environment is never overwritten, and a file only
    fills keys that no higher file set. Raises ``ConfigError`` when an
    explicit env file does not exist.
    """
    global _loaded
    environ = os.environ if environ is None else environ
    cwd = Path.cwd() if cwd is None else cwd

    files: list[EnvFile] = []
    project = cwd / ".env"
    explicit, role = env_file, "--env-file"
    if not explicit and environ.get(ENV_FILE_VAR):
        explicit, role = environ[ENV_FILE_VAR], ENV_FILE_VAR
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            path = cwd / path
        if not path.is_file():
            raise ConfigError(f"{role}: no such file: {path}")
        files.append(EnvFile(path, role, True))
        files.append(
            EnvFile(
                project, "project", project.is_file(), skipped=f"replaced by {role}"
            )
        )
    else:
        files.append(EnvFile(project, "project", project.is_file()))
    user = user_env_path(environ)
    files.append(EnvFile(user, "user", user.is_file()))

    origins = {key: "env" for key in environ if is_benchkit_key(key)}
    for entry in files:
        if not entry.exists or entry.skipped:
            continue
        for key, value in dotenv_values(entry.path).items():
            if value is None or key == ENV_FILE_VAR or not is_benchkit_key(key):
                continue
            if key in origins:
                continue
            environ[key] = value
            origins[key] = str(entry.path)

    _loaded = Loaded(files, origins)
    return _loaded


def loaded() -> Loaded:
    """What the last ``load`` applied (empty before the first call)."""
    return _loaded


def resolve(
    flags: dict[str, str | None] | None = None,
    *,
    environ: MutableMapping[str, str] | None = None,
) -> list[tuple[str, str, str]]:
    """Every known or set BenchKit key as ``(key, value, origin)``.

    ``value`` is masked for secrets. ``origin`` is ``flag``, ``env``, the file
    path the value came from, or ``default``.
    """
    environ = os.environ if environ is None else environ
    flags = {key: value for key, value in (flags or {}).items() if value}
    defaults = dict(SETTINGS)
    extra = sorted(
        key
        for key in environ
        if is_benchkit_key(key) and key not in defaults and key != ENV_FILE_VAR
    )
    rows = []
    for key in [*defaults, *extra]:
        if key in flags:
            value, origin = flags[key], "flag"
        elif key in environ:
            value, origin = environ[key], _loaded.origins.get(key, "env")
        else:
            value, origin = defaults.get(key, ""), "default"
        rows.append((key, mask(value) if is_secret(key) else value, origin))
    return rows


def save_user(
    values: dict[str, str | None],
    *,
    environ: MutableMapping[str, str] | None = None,
) -> Path:
    """Write ``values`` to the user config; ``None`` or ``""`` removes a key.

    The file can hold an API key, so it is created readable by its owner only.
    """
    path = user_env_path(environ)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not path.exists():
        path.touch(mode=0o600)
    present = dotenv_values(path)
    for key, value in values.items():
        if not is_benchkit_key(key):
            raise ValueError(f"not a BenchKit setting: {key}")
        if value:
            set_key(path, key, value, quote_mode="auto")
        elif key in present:
            unset_key(path, key, quote_mode="auto")
    return path


def overridden(keys: list[str]) -> dict[str, str]:
    """Keys whose current value comes from somewhere above the user config."""
    user = str(user_env_path())
    return {
        key: origin
        for key in keys
        if (origin := _loaded.origins.get(key)) is not None and origin != user
    }
