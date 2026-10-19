"""Sandboxed code execution via subprocess."""

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

BLOCKED_MODULES = {
    "subprocess",
    "shutil",
    "socket",
    "http",
    "urllib",
    "requests",
    "signal",
    "ctypes",
    "multiprocessing",
}

SANDBOX_HEADER = f"""
import builtins as _b
_orig = _b.__import__
_blocked = {BLOCKED_MODULES!r}

def _safe(name, *a, **kw):
    if name.split(".")[0] in _blocked:
        raise ImportError(f"blocked: {{name}}")
    return _orig(name, *a, **kw)

_b.__import__ = _safe
"""


def sandbox_env(home: str, *, keep_prefixes: tuple[str, ...] = ()) -> dict[str, str]:
    """A minimal environment for running model-written code.

    The import guard is easy to bypass, so the child must not inherit API keys
    or anything else from BenchKit's own environment. ``HOME`` and the temp dir
    point at ``home``; ``keep_prefixes`` keeps evaluator knobs such as
    ``EVALPLUS_``.
    """
    env = {
        "PATH": os.environ.get("PATH") or os.defpath,
        "HOME": home,
        "TMPDIR": home,
        "LANG": os.environ.get("LANG") or "C.UTF-8",
    }
    # Windows Python cannot start without these.
    for name in ("SYSTEMROOT", "COMSPEC"):
        if name in os.environ:
            env[name] = os.environ[name]
    if keep_prefixes:
        env.update(
            (key, value)
            for key, value in os.environ.items()
            if key.startswith(keep_prefixes)
        )
    return env


@dataclass(frozen=True)
class ExecutionResult:
    """Code-verifier result with answer-safe diagnostic feedback."""

    passed: bool
    feedback: str = ""


def _failure_feedback(stderr: str) -> str:
    """Describe the failure category without exposing hidden test source."""
    if "SyntaxError" in stderr:
        return "The submitted Python failed to parse with SyntaxError."
    if "IndentationError" in stderr:
        return "The submitted Python failed to parse with IndentationError."
    if "AssertionError" in stderr:
        return "The submitted Python ran, but a verifier assertion failed."
    for name in (
        "NameError",
        "TypeError",
        "ValueError",
        "IndexError",
        "KeyError",
        "AttributeError",
        "ZeroDivisionError",
        "RecursionError",
        "ImportError",
    ):
        if name in stderr:
            return f"The submitted Python failed during verification with {name}."
    return "The submitted Python exited unsuccessfully during verification."


def execute_with_feedback(
    code: str, timeout: int = 10, *, sentinel: str | None = None
) -> ExecutionResult:
    """Run code and retain only a sanitized failure category.

    With ``sentinel``, a clean exit passes only if stdout carries it, so code
    that exits early with status 0 (``os._exit(0)``) cannot fake a pass.
    """
    full = SANDBOX_HEADER + "\n" + code

    with tempfile.TemporaryDirectory(prefix="benchkit-exec-") as home:
        path = Path(home) / "solution.py"
        path.write_text(full)
        try:
            result = subprocess.run(
                [sys.executable, str(path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=home,
                env=sandbox_env(home),
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                False,
                f"The submitted Python exceeded the {timeout}s verifier timeout.",
            )
    if result.returncode == 0:
        if sentinel is not None and sentinel not in result.stdout:
            return ExecutionResult(
                False, "The submitted Python exited before verification finished."
            )
        return ExecutionResult(True)
    return ExecutionResult(False, _failure_feedback(result.stderr or ""))


def execute(code: str, timeout: int = 10) -> bool:
    """Run code in a subprocess. Returns True if exit code is 0."""
    return execute_with_feedback(code, timeout).passed
