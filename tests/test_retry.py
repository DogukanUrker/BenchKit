"""Retry policy behavior for rate limits and other transient failures."""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import httpx
import pytest

from benchkit.retry import (
    RetryPolicy,
    is_retryable,
    retry_after_seconds,
    run_with_retries,
)


def _status_error(
    status: int, headers: dict[str, str] | None = None
) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://localhost:8080/v1/chat/completions")
    response = httpx.Response(status, headers=headers or {}, request=request)
    return httpx.HTTPStatusError("boom", request=request, response=response)


def test_rate_limit_is_retryable():
    assert is_retryable(_status_error(429))
    assert is_retryable(_status_error(502))
    assert not is_retryable(_status_error(422))


def test_dns_connect_error_uses_longer_retry_budget():
    policy = RetryPolicy(attempts=3, connect_attempts=6, base_delay=0.0, jitter=0.0)
    calls = {"n": 0}

    def operation() -> str:
        calls["n"] += 1
        if calls["n"] < 6:
            raise httpx.ConnectError("[Errno -3] Temporary failure in name resolution")
        return "ok"

    assert run_with_retries(policy, operation) == "ok"
    assert calls["n"] == 6


def test_non_connect_transport_error_keeps_standard_retry_budget():
    policy = RetryPolicy(attempts=3, connect_attempts=6, base_delay=0.0, jitter=0.0)
    calls = {"n": 0}

    def operation() -> str:
        calls["n"] += 1
        raise httpx.ReadError("connection dropped")

    with pytest.raises(httpx.ReadError):
        run_with_retries(policy, operation)
    assert calls["n"] == 3


def test_retry_after_seconds_reads_delta():
    assert retry_after_seconds(_status_error(429, {"Retry-After": "2.5"})) == 2.5


def test_retry_after_seconds_reads_http_date():
    when = datetime.now(UTC) + timedelta(seconds=30)
    hint = retry_after_seconds(
        _status_error(429, {"Retry-After": format_datetime(when)})
    )
    assert hint is not None and 25 <= hint <= 31


def test_retry_after_seconds_ignores_junk_and_past_dates():
    assert retry_after_seconds(_status_error(429)) is None
    assert retry_after_seconds(_status_error(429, {"Retry-After": "soon"})) is None
    assert retry_after_seconds(_status_error(429, {"Retry-After": "-5"})) == 0.0
    past = format_datetime(datetime.now(UTC) - timedelta(seconds=30))
    assert retry_after_seconds(_status_error(429, {"Retry-After": past})) == 0.0
    assert retry_after_seconds(httpx.ConnectError("nope")) is None


def test_retry_after_overrides_backoff_and_is_capped():
    policy = RetryPolicy(base_delay=0.5, max_delay=8.0, max_retry_after=60.0)
    assert policy.delay_for(1, 12.0) == 12.0
    assert policy.delay_for(1, 900.0) == 60.0
    assert 0.5 <= policy.delay_for(1) <= 0.625


def test_run_with_retries_waits_the_hinted_delay():
    waits: list[float] = []

    class RecordingPolicy(RetryPolicy):
        def sleep(self, seconds, cancel_event):
            waits.append(seconds)
            return True

    policy = RecordingPolicy(attempts=3, base_delay=0.5)
    calls = {"n": 0}

    def operation() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise _status_error(429, {"Retry-After": "3"})
        return "ok"

    assert run_with_retries(policy, operation) == "ok"
    assert waits == [3.0]


def test_run_with_retries_gives_up_on_client_errors():
    policy = RetryPolicy(attempts=3, base_delay=0.0)
    calls = {"n": 0}

    def operation() -> str:
        calls["n"] += 1
        raise _status_error(422)

    with pytest.raises(httpx.HTTPStatusError):
        run_with_retries(policy, operation)
    assert calls["n"] == 1


def test_run_with_retries_stops_when_cancelled():
    policy = RetryPolicy(attempts=3, base_delay=0.0)
    cancel = threading.Event()
    cancel.set()

    with pytest.raises(httpx.HTTPStatusError):
        run_with_retries(
            policy,
            lambda: (_ for _ in ()).throw(_status_error(429)),
            cancel_event=cancel,
        )
