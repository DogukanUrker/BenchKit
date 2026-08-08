"""Sandboxed code execution via subprocess."""

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


def execute_with_feedback(code: str, timeout: int = 10) -> ExecutionResult:
    """Run code and retain only a sanitized failure category."""
    full = SANDBOX_HEADER + "\n" + code

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full)
        path = Path(f.name)

    try:
        result = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return ExecutionResult(True)
        return ExecutionResult(False, _failure_feedback(result.stderr or ""))
    except subprocess.TimeoutExpired:
        return ExecutionResult(
            False,
            f"The submitted Python exceeded the {timeout}s verifier timeout.",
        )
    finally:
        path.unlink(missing_ok=True)


def execute(code: str, timeout: int = 10) -> bool:
    """Run code in a subprocess. Returns True if exit code is 0."""
    return execute_with_feedback(code, timeout).passed
