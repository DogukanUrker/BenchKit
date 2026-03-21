"""Sandboxed code execution via subprocess."""

import subprocess
import sys
import tempfile
from pathlib import Path

BLOCKED_MODULES = {
    "subprocess", "shutil", "socket", "http", "urllib",
    "requests", "signal", "ctypes", "multiprocessing",
}

SANDBOX_HEADER = """
import builtins as _b
_orig = _b.__import__
_blocked = {blocked}

def _safe(name, *a, **kw):
    if name.split(".")[0] in _blocked:
        raise ImportError(f"blocked: {{name}}")
    return _orig(name, *a, **kw)

_b.__import__ = _safe
""".format(blocked=repr(BLOCKED_MODULES))


def execute(code: str, timeout: int = 10) -> bool:
    """Run code in a subprocess. Returns True if exit code is 0."""
    full = SANDBOX_HEADER + "\n" + code

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(full)
        path = Path(f.name)

    try:
        result = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            timeout=timeout,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        path.unlink(missing_ok=True)
