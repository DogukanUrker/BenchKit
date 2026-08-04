"""Retry policy for transient inference-server failures.

Inference servers behind proxies or model swappers routinely answer with
``502``/``503`` while a model loads or a worker restarts. Those failures are
short-lived, so BenchKit retries them with exponential backoff instead of
marking a task as errored.
"""

from __future__ import annotations

import os
import random
import threading
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

import httpx

T = TypeVar("T")
RetryNotice = Callable[[int, float, Exception], None]

# Gateway, overload, and rate-limit responses are worth another attempt;
# 4xx client errors such as 400/404/422 are not.
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504, 522, 524, 529})


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def is_retryable(exc: Exception) -> bool:
    """Report whether a request failure is likely transient."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS
    # Connection resets, DNS hiccups, and read timeouts against a busy server.
    return isinstance(exc, httpx.TransportError)


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff with jitter, capped at ``max_delay``."""

    attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 8.0
    jitter: float = 0.25

    @classmethod
    def from_env(cls) -> "RetryPolicy":
        return cls(
            attempts=max(1, _env_int("BENCHKIT_RETRIES", 3)),
            base_delay=max(0.0, _env_float("BENCHKIT_RETRY_BASE_DELAY", 0.5)),
            max_delay=max(0.0, _env_float("BENCHKIT_RETRY_MAX_DELAY", 8.0)),
        )

    def delay_for(self, attempt: int) -> float:
        """Backoff before retrying, where ``attempt`` is the 1-based try."""
        delay = min(self.max_delay, self.base_delay * (2 ** (attempt - 1)))
        return delay + random.uniform(0.0, self.jitter * delay)

    def sleep(self, seconds: float, cancel_event: threading.Event | None) -> bool:
        """Wait out the backoff; return False when the user stopped the run."""
        if cancel_event is not None:
            return not cancel_event.wait(seconds)
        time.sleep(seconds)
        return True


def run_with_retries(
    policy: RetryPolicy,
    operation: Callable[[], T],
    *,
    cancel_event: threading.Event | None = None,
    retryable: Callable[[Exception], bool] = is_retryable,
    on_retry: RetryNotice | None = None,
) -> T:
    """Call ``operation`` until it succeeds or its failure stops being transient."""
    for attempt in range(1, policy.attempts + 1):
        try:
            return operation()
        except Exception as exc:  # re-raised below when not retryable
            if attempt >= policy.attempts or not retryable(exc):
                raise
            if cancel_event is not None and cancel_event.is_set():
                raise
            delay = policy.delay_for(attempt)
            if on_retry is not None:
                on_retry(attempt, delay, exc)
            if not policy.sleep(delay, cancel_event):
                raise
    raise AssertionError("unreachable: retry loop exhausted without a result")
