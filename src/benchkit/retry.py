"""Retry policy for transient inference-server failures.

Inference servers behind proxies or model swappers routinely answer with
``502``/``503`` while a model loads or a worker restarts, and busy servers such
as ``llama-server`` answer with ``429`` when their slots are full. Those
failures are short-lived, so BenchKit retries them with exponential backoff
instead of marking a task as errored. When the server says how long to wait via
``Retry-After``, that hint wins over the computed backoff.
"""

from __future__ import annotations

import os
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import TypeVar

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


def retry_after_seconds(exc: Exception) -> float | None:
    """Read the server's ``Retry-After`` hint, in seconds, when it sent one.

    Rate limiters answer ``429`` with either a delta-seconds count or an
    HTTP-date. Anything unparsable, negative, or already in the past is ignored
    so the caller falls back to its own backoff.
    """
    if not isinstance(exc, httpx.HTTPStatusError):
        return None
    raw = exc.response.headers.get("retry-after")
    if not raw:
        return None
    raw = raw.strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    try:
        deadline = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if deadline is None:
        return None
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=UTC)
    return max(0.0, (deadline - datetime.now(UTC)).total_seconds())


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff with jitter, capped at ``max_delay``."""

    attempts: int = 3
    # DNS failures and refused/reset connections often outlive the short retry
    # window used for an overloaded server response. Give failures that happen
    # before a connection is established a separate, longer budget.
    connect_attempts: int = 6
    base_delay: float = 0.5
    max_delay: float = 8.0
    jitter: float = 0.25
    # A rate limiter may ask for a longer pause than our own backoff cap, so
    # ``Retry-After`` gets a separate, more generous ceiling.
    max_retry_after: float = 60.0

    @classmethod
    def from_env(cls) -> RetryPolicy:
        return cls(
            attempts=max(1, _env_int("BENCHKIT_RETRIES", 3)),
            connect_attempts=max(1, _env_int("BENCHKIT_CONNECT_RETRIES", 6)),
            base_delay=max(0.0, _env_float("BENCHKIT_RETRY_BASE_DELAY", 0.5)),
            max_delay=max(0.0, _env_float("BENCHKIT_RETRY_MAX_DELAY", 8.0)),
            max_retry_after=max(0.0, _env_float("BENCHKIT_RETRY_MAX_WAIT", 60.0)),
        )

    def delay_for(self, attempt: int, retry_after: float | None = None) -> float:
        """Backoff before retrying, where ``attempt`` is the 1-based try.

        ``retry_after`` is the server's own hint; when present it replaces the
        computed backoff (clamped to ``max_retry_after``) rather than racing it,
        since retrying sooner than asked just earns another rate-limit answer.
        """
        if retry_after is not None:
            return min(self.max_retry_after, retry_after)
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
    max_attempts = max(policy.attempts, policy.connect_attempts)
    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except Exception as exc:  # re-raised below when not retryable
            attempt_limit = (
                policy.connect_attempts
                if isinstance(exc, httpx.ConnectError)
                else policy.attempts
            )
            if attempt >= attempt_limit or not retryable(exc):
                raise
            if cancel_event is not None and cancel_event.is_set():
                raise
            delay = policy.delay_for(attempt, retry_after_seconds(exc))
            if on_retry is not None:
                on_retry(attempt, delay, exc)
            if not policy.sleep(delay, cancel_event):
                raise
    raise AssertionError("unreachable: retry loop exhausted without a result")
