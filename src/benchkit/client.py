"""Inference clients for OpenAI-compatible servers and Ollama."""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import quote

import httpx

from benchkit.looping import InlineThinkingParser
from benchkit.retry import RetryPolicy, is_retryable, run_with_retries

Provider = Literal["auto", "openai", "ollama"]
ProgressCallback = Callable[["GenerationUpdate"], None]

DEFAULT_HOST = "http://localhost:11434"
MAX_DISCOVERED_CONCURRENCY = 256

_PARALLELISM_KEYS = {
    "concurrency",
    "concurrencylimit",
    "maxconcurrency",
    "maxparallelrequests",
    "nparallel",
    "parallel",
    "parallelism",
}
_PARALLELISM_CONTAINERS = {"limits", "llamaswap", "meta", "metadata"}
_CONTEXT_LENGTH_KEYS = {
    "contextlength",
    "contextsize",
    "contextwindow",
    "maxcontextlength",
    "maxpositionembeddings",
    "nctx",
    "numctx",
}


def _without_v1(url: str) -> str:
    url = url.rstrip("/")
    return url[:-3] if url.endswith("/v1") else url


def _openai_base(url: str) -> str:
    url = url.rstrip("/")
    return url if url.endswith("/v1") else f"{url}/v1"


def _positive_int(value: object) -> int | None:
    """Return a positive integer without accepting booleans or fractions."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        return int(value) if value > 0 and value.is_integer() else None
    if isinstance(value, str):
        try:
            parsed = int(value.strip())
        except ValueError:
            return None
        return parsed if parsed > 0 else None
    return None


def _parallelism_hint(value: object) -> int | None:
    """Read explicit concurrency hints from common model metadata shapes."""
    if not isinstance(value, dict):
        return None

    hints: list[int] = []
    for key, candidate in value.items():
        normalized = str(key).lower().replace("_", "").replace("-", "")
        if normalized in _PARALLELISM_KEYS:
            parsed = _positive_int(candidate)
            if parsed is not None:
                hints.append(parsed)
        elif normalized in _PARALLELISM_CONTAINERS:
            nested = _parallelism_hint(candidate)
            if nested is not None:
                hints.append(nested)
    return min(hints) if hints else None


def _context_length_hint(value: object) -> int | None:
    """Read context-window hints from model, props, slot or Ollama payloads."""
    hints: list[int] = []
    if isinstance(value, list):
        for item in value:
            hint = _context_length_hint(item)
            if hint is not None:
                hints.append(hint)
    elif isinstance(value, dict):
        for key, candidate in value.items():
            key_text = str(key).lower()
            normalized = "".join(
                character for character in key_text if character.isalnum()
            )
            leaf = key_text.rsplit(".", 1)[-1]
            leaf_normalized = "".join(
                character for character in leaf if character.isalnum()
            )
            if (
                normalized in _CONTEXT_LENGTH_KEYS
                or leaf_normalized in _CONTEXT_LENGTH_KEYS
            ):
                parsed = _positive_int(candidate)
                if parsed is not None:
                    hints.append(parsed)
            if isinstance(candidate, (dict, list)):
                nested = _context_length_hint(candidate)
                if nested is not None:
                    hints.append(nested)
    return min(hints) if hints else None


def _capacity_from_payload(payload: object) -> int | None:
    """Normalize llama.cpp slot and health responses into a worker count."""
    if isinstance(payload, list):
        return len(payload) or None
    if not isinstance(payload, dict):
        return None
    slots = payload.get("slots")
    if isinstance(slots, list) and slots:
        return len(slots)
    idle = _positive_int(payload.get("slots_idle")) or 0
    processing = _positive_int(payload.get("slots_processing")) or 0
    total = idle + processing
    return total or None


def _describe_http_error(exc: Exception) -> str:
    """Human message for an unload failure, reading error bodies when present."""
    if isinstance(exc, httpx.HTTPStatusError):
        response = exc.response
        try:
            body = response.json()
        except (ValueError, TypeError):
            return f"HTTP {response.status_code}"
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict) and isinstance(error.get("message"), str):
                return f"HTTP {response.status_code}: {error['message']}"
            if isinstance(error, str):
                return f"HTTP {response.status_code}: {error}"
        return f"HTTP {response.status_code}"
    return str(exc)


def _content_text(content: object) -> str:
    """Normalize text from OpenAI-compatible string or content-part fields."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        for key in ("text", "content"):
            if isinstance(content.get(key), str):
                return content[key]
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""


def _reasoning_text(message: dict) -> tuple[str, bool]:
    """Return reasoning text plus whether a known reasoning field was present."""
    for key in ("reasoning_content", "reasoning", "thinking"):
        if key in message:
            return _content_text(message.get(key)), True
    return "", False


def _trace_status(thinking: str, channel_seen: bool) -> str:
    if thinking:
        return "observed"
    return "available_empty" if channel_seen else "unavailable"


def _close_response(response: httpx.Response) -> None:
    """Best-effort close for deadline and user-cancellation watchers."""
    # The stream may have completed at the same instant as a watcher.
    with contextlib.suppress(Exception):
        response.close()


def _start_deadline(
    response: httpx.Response,
    seconds: float,
) -> tuple[threading.Event, threading.Timer]:
    """Close a live response when its per-task generation deadline expires."""
    expired = threading.Event()

    def expire() -> None:
        expired.set()
        _close_response(response)

    timer = threading.Timer(max(0.0, seconds), expire)
    timer.daemon = True
    timer.start()
    return expired, timer


def _start_cancellation_watch(
    response: httpx.Response,
    cancel_event: threading.Event | None,
) -> tuple[threading.Event, threading.Event]:
    """Close a live response promptly when the user stops the run."""
    cancelled = threading.Event()
    finished = threading.Event()
    if cancel_event is None:
        return cancelled, finished

    def watch() -> None:
        while not finished.is_set():
            if cancel_event.wait(0.05):
                if not finished.is_set():
                    cancelled.set()
                    _close_response(response)
                return

    thread = threading.Thread(target=watch, daemon=True)
    thread.start()
    return cancelled, finished


def _openai_metrics(data: dict, elapsed_s: float) -> dict:
    """Read token metrics from standard usage and llama.cpp timings payloads."""
    usage = data.get("usage") or {}
    timings = data.get("timings") or {}

    eval_count = int(
        usage.get("completion_tokens")
        or timings.get("predicted_n")
        or timings.get("tokens_predicted")
        or 0
    )
    input_tokens = int(
        usage.get("prompt_tokens")
        or usage.get("input_tokens")
        or timings.get("prompt_n")
        or timings.get("tokens_evaluated")
        or 0
    )
    predicted_ms = float(
        timings.get("predicted_ms") or timings.get("generation_ms") or 0.0
    )
    tok_s = float(
        timings.get("predicted_per_second")
        or timings.get("tokens_predicted_per_second")
        or 0.0
    )

    if not tok_s and predicted_ms > 0 and eval_count:
        tok_s = eval_count / (predicted_ms / 1000)
    if not tok_s and elapsed_s > 0 and eval_count:
        tok_s = eval_count / elapsed_s

    eval_duration_ns = int(predicted_ms * 1_000_000)
    if not eval_duration_ns and tok_s > 0 and eval_count:
        eval_duration_ns = int(eval_count / tok_s * 1_000_000_000)

    return {
        "tok_s": tok_s,
        "input_tokens": input_tokens,
        "eval_count": eval_count,
        "eval_duration_ns": eval_duration_ns,
        "response_time_s": elapsed_s,
    }


@dataclass
class _StreamState:
    """Tracks whether an attempt already handed output to the caller."""

    received: bool = False


@dataclass(frozen=True)
class GenerationUpdate:
    """A normalized piece of a live model generation."""

    thinking: str = ""
    response: str = ""
    elapsed_s: float = 0.0
    reasoning_channel_seen: bool = False
    done: bool = False
    attempt: int = 0


@dataclass
class InferenceClient:
    """Small synchronous client with provider auto-detection."""

    host: str
    requested_provider: Provider = "auto"
    api_key: str | None = None
    timeout: float = 300.0
    retries: RetryPolicy = field(default_factory=RetryPolicy)
    provider: Literal["openai", "ollama"] | None = field(default=None, init=False)
    _models_by_name: dict[str, dict] = field(
        default_factory=dict, init=False, repr=False
    )
    _parallelism: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _context_lengths: dict[str, int | None] = field(
        default_factory=dict, init=False, repr=False
    )

    @classmethod
    def from_env(cls) -> InferenceClient:
        provider = os.environ.get("BENCHKIT_PROVIDER", "auto").lower()
        if provider not in {"auto", "openai", "ollama"}:
            raise ValueError("BENCHKIT_PROVIDER must be one of: auto, openai, ollama")

        host = (
            os.environ.get("BENCHKIT_HOST")
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OLLAMA_HOST")
            or DEFAULT_HOST
        )
        api_key = os.environ.get("BENCHKIT_API_KEY") or os.environ.get("OPENAI_API_KEY")
        timeout = float(os.environ.get("BENCHKIT_TIMEOUT", "300"))
        return cls(host.rstrip("/"), provider, api_key, timeout, RetryPolicy.from_env())

    @property
    def label(self) -> str:
        if self.provider == "openai":
            return "OpenAI-compatible"
        if self.provider == "ollama":
            return "Ollama"
        return "Auto-detect"

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}

    def _request(
        self,
        method: str,
        url: str,
        *,
        cancel_event: threading.Event | None = None,
        **kwargs: object,
    ) -> httpx.Response:
        """Issue a non-streaming request, retrying transient server failures."""

        def attempt() -> httpx.Response:
            response = httpx.request(method, url, **kwargs)
            response.raise_for_status()
            return response

        return run_with_retries(self.retries, attempt, cancel_event=cancel_event)

    def _retry_stream(
        self,
        stream: Callable[[_StreamState], dict],
        cancel_event: threading.Event | None,
    ) -> dict:
        """Retry a generation only while nothing has been streamed to the caller.

        Once tokens have reached ``on_progress`` a replay would duplicate output,
        so a mid-stream failure is surfaced instead of retried.
        """
        state = _StreamState()

        def attempt() -> dict:
            state.received = False
            return stream(state)

        return run_with_retries(
            self.retries,
            attempt,
            cancel_event=cancel_event,
            retryable=lambda exc: not state.received and is_retryable(exc),
        )

    def list_models(self) -> list[dict]:
        """Discover models and lock in the detected provider."""
        providers: list[Literal["openai", "ollama"]]
        if self.requested_provider == "auto":
            providers = ["openai", "ollama"]
        else:
            providers = [self.requested_provider]

        errors = []
        for provider in providers:
            try:
                if provider == "openai":
                    models = self._list_openai_models()
                else:
                    models = self._list_ollama_models()
                self.provider = provider
                self._models_by_name = {model["name"]: model for model in models}
                self._parallelism.clear()
                self._context_lengths.clear()
                return models
            except (httpx.HTTPError, ValueError, TypeError) as exc:
                errors.append(f"{provider}: {exc}")

        raise RuntimeError(
            "Could not discover a compatible API (" + "; ".join(errors) + ")"
        )

    def _list_openai_models(self) -> list[dict]:
        response = self._request(
            "GET",
            f"{_openai_base(self.host)}/models",
            headers=self._headers(),
            timeout=10,
        )
        models = response.json().get("data", [])
        normalized = []
        for model in models:
            model_id = model.get("id")
            if not model_id:
                continue
            raw_status = model.get("status")
            status = (
                raw_status.get("value", "")
                if isinstance(raw_status, dict)
                else str(raw_status or "")
            )
            normalized_status = status.strip().lower()
            normalized.append(
                {
                    "name": model_id,
                    "size": model.get("size"),
                    "owned_by": model.get("owned_by", ""),
                    "status": status,
                    "loaded": (
                        True
                        if normalized_status in {"loaded", "running"}
                        else False
                        if normalized_status == "unloaded"
                        else None
                    ),
                    "meta": model.get("meta") or {},
                    "parallelism": _parallelism_hint(model),
                    "context_length": _context_length_hint(model),
                }
            )
        return sorted(normalized, key=lambda model: model["name"].lower())

    def _list_ollama_models(self) -> list[dict]:
        response = self._request(
            "GET",
            f"{_without_v1(self.host)}/api/tags",
            timeout=10,
        )
        models = response.json().get("models", [])
        running: set[str] | None = None
        running_models: list[dict] | None = None
        try:
            response = self._request(
                "GET",
                f"{_without_v1(self.host)}/api/ps",
                timeout=10,
            )
            running_models = response.json().get("models", [])
            running = {
                str(value)
                for model in running_models
                for value in (
                    model.get("name"),
                    model.get("model"),
                    model.get("digest"),
                )
                if value
            }
        except (httpx.HTTPError, ValueError, TypeError):
            # Older or non-standard Ollama-compatible servers may not expose
            # the optional running-model endpoint. Discovery should still work.
            pass

        if running is not None:
            for model in models:
                loaded = any(
                    str(value) in running
                    for value in (
                        model.get("name"),
                        model.get("model"),
                        model.get("digest"),
                    )
                    if value
                )
                model["loaded"] = loaded
                model["status"] = "loaded" if loaded else "unloaded"
                matches = [
                    entry
                    for entry in running_models or []
                    if any(
                        str(value)
                        in {
                            str(model.get("name") or ""),
                            str(model.get("model") or ""),
                            str(model.get("digest") or ""),
                        }
                        for value in (
                            entry.get("name"),
                            entry.get("model"),
                            entry.get("digest"),
                        )
                        if value
                    )
                ]
                context_length = _context_length_hint(matches)
                if context_length is not None:
                    model["context_length"] = context_length
        return sorted(models, key=lambda model: model.get("size", 0))

    def max_parallel_requests(self, model: str) -> int:
        """Return the model's automatically discovered request capacity.

        llama.cpp exposes one object per server slot from ``GET /slots``. Its
        router mode accepts the model as a query parameter, while llama-swap
        exposes the same endpoint below ``/upstream/<model>/``. Other server
        types stay at one request unless their model metadata contains an
        explicit positive concurrency value.
        """
        if model in self._parallelism:
            return self._parallelism[model]

        info = self._models_by_name.get(model, {})
        hint = _positive_int(info.get("parallelism"))
        discovered = self._discover_slot_capacity(model, info)
        candidates = [value for value in (hint, discovered) if value is not None]
        capacity = min(candidates) if candidates else 1
        capacity = min(MAX_DISCOVERED_CONCURRENCY, max(1, capacity))
        self._parallelism[model] = capacity
        return capacity

    def context_length(self, model: str) -> int | None:
        """Return the best available served-context limit for a model."""
        if model in self._context_lengths:
            return self._context_lengths[model]

        known = _context_length_hint(self._models_by_name.get(model, {}))
        if known is None:
            known = self._discover_context_length(model)
        self._context_lengths[model] = known
        return known

    def _is_llama_swap(self, model: str) -> bool:
        """Return True when the served model is routed by llama-swap."""
        info = self._models_by_name.get(model, {})
        owner = str(info.get("owned_by") or "").strip().lower()
        meta = info.get("meta") or {}
        return owner in {"llama-swap", "llamaswap"} or (
            isinstance(meta, dict) and "llamaswap" in meta
        )

    def _discover_context_length(self, model: str) -> int | None:
        if self.provider == "ollama":
            root = _without_v1(self.host)
            try:
                response = self._request("GET", f"{root}/api/ps", timeout=10)
                running = response.json().get("models", [])
                matches = [
                    entry
                    for entry in running
                    if model
                    in {
                        str(entry.get("name") or ""),
                        str(entry.get("model") or ""),
                        str(entry.get("digest") or ""),
                    }
                ]
                hint = _context_length_hint(matches)
                if hint is not None:
                    return hint
            except (httpx.HTTPError, RuntimeError, ValueError, TypeError):
                pass
            try:
                response = self._request(
                    "POST",
                    f"{root}/api/show",
                    json={"model": model},
                    timeout=10,
                )
                return _context_length_hint(response.json())
            except (httpx.HTTPError, RuntimeError, ValueError, TypeError):
                return None

        if self.provider != "openai":
            return None

        root = _without_v1(self.host)
        attempts: list[tuple[str, dict[str, str] | None]] = [
            (f"{root}/props", {"model": model}),
            (f"{root}/slots", {"model": model}),
        ]
        if self._is_llama_swap(model):
            upstream = f"{root}/upstream/{quote(model, safe='/')}"
            attempts.extend([(f"{upstream}/props", None), (f"{upstream}/slots", None)])

        for url, params in attempts:
            try:
                response = self._request(
                    "GET",
                    url,
                    headers=self._headers(),
                    params=params,
                    timeout=min(self.timeout, 30),
                )
                hint = _context_length_hint(response.json())
                if hint is not None:
                    return hint
            except (httpx.HTTPError, RuntimeError, ValueError, TypeError):
                continue
        return None

    def _discover_slot_capacity(self, model: str, info: dict) -> int | None:
        """Probe optional llama.cpp monitoring routes without breaking peers."""
        if self.provider != "openai":
            return None

        root = _without_v1(self.host)
        attempts: list[tuple[str, dict[str, str] | None]] = [
            (f"{root}/slots", {"model": model}),
            (f"{root}/health", {"model": model}),
        ]

        if self._is_llama_swap(model):
            upstream = f"{root}/upstream/{quote(model, safe='/')}"
            attempts.extend(
                [
                    (f"{upstream}/slots", None),
                    (f"{upstream}/health", None),
                ]
            )

        for url, params in attempts:
            try:
                response = self._request(
                    "GET",
                    url,
                    headers=self._headers(),
                    params=params,
                    timeout=self.timeout,
                )
                capacity = _capacity_from_payload(response.json())
                if capacity is not None:
                    return capacity
            except (httpx.HTTPError, RuntimeError, ValueError, TypeError):
                continue
        return None

    def generate(
        self,
        model: str,
        prompt: str,
        on_progress: ProgressCallback | None = None,
        cancel_event: threading.Event | None = None,
    ) -> dict:
        if self.provider is None:
            raise RuntimeError("Call list_models() before generate()")
        if self.provider == "openai":
            return self._generate_openai(model, prompt, on_progress, cancel_event)
        return self._generate_ollama(model, prompt, on_progress, cancel_event)

    def _generation_timeout(self) -> httpx.Timeout:
        """Apply the configured timeout to every transport operation."""
        return httpx.Timeout(self.timeout)

    def _generate_openai(
        self,
        model: str,
        prompt: str,
        on_progress: ProgressCallback | None,
        cancel_event: threading.Event | None,
    ) -> dict:
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "temperature": 0.0,
            "stream_options": {"include_usage": True},
        }
        try:
            return self._retry_stream(
                lambda state: self._stream_openai(
                    body, on_progress, cancel_event, state
                ),
                cancel_event,
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in {400, 404, 422}:
                raise
        # Some older compatible servers reject the optional usage request but
        # still support standard streamed chat completions.
        body.pop("stream_options")
        return self._retry_stream(
            lambda state: self._stream_openai(body, on_progress, cancel_event, state),
            cancel_event,
        )

    def _stream_openai(
        self,
        body: dict,
        on_progress: ProgressCallback | None,
        cancel_event: threading.Event | None,
        state: _StreamState | None = None,
    ) -> dict:
        state = state if state is not None else _StreamState()
        started = time.perf_counter()
        thinking_parts: list[str] = []
        response_parts: list[str] = []
        usage: dict = {}
        timings: dict = {}
        done_reason = ""
        timed_out = False
        generation_cancelled = False
        explicit_reasoning_channel = False
        inline = InlineThinkingParser()

        try:
            with httpx.stream(
                "POST",
                f"{_openai_base(self.host)}/chat/completions",
                headers=self._headers(),
                json=body,
                timeout=self._generation_timeout(),
            ) as response:
                response.raise_for_status()
                expired, watchdog = _start_deadline(
                    response,
                    self.timeout - (time.perf_counter() - started),
                )
                cancelled, cancellation_finished = _start_cancellation_watch(
                    response, cancel_event
                )
                try:
                    for line in response.iter_lines():
                        if cancelled.is_set() or (
                            cancel_event is not None and cancel_event.is_set()
                        ):
                            generation_cancelled = True
                            break
                        if (
                            expired.is_set()
                            or time.perf_counter() - started >= self.timeout
                        ):
                            timed_out = True
                            break
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if not payload or payload == "[DONE]":
                            continue
                        data = json.loads(payload)
                        if data.get("error"):
                            raise RuntimeError(f"generation failed: {data['error']}")
                        if isinstance(data.get("usage"), dict):
                            usage = data["usage"]
                        if isinstance(data.get("timings"), dict):
                            timings = data["timings"]

                        choices = data.get("choices") or []
                        choice = choices[0] if choices else {}
                        done_reason = choice.get("finish_reason") or done_reason
                        delta = choice.get("delta") or choice.get("message") or {}
                        thinking_piece, reasoning_field = _reasoning_text(delta)
                        explicit_reasoning_channel = (
                            explicit_reasoning_channel or reasoning_field
                        )
                        content_piece = _content_text(delta.get("content"))

                        if content_piece and not explicit_reasoning_channel:
                            inline_thinking, answer_piece = inline.feed(content_piece)
                            thinking_piece += inline_thinking
                        else:
                            answer_piece = content_piece

                        reasoning_channel_seen = (
                            explicit_reasoning_channel or inline.saw_marker
                        )

                        thinking_parts.append(thinking_piece)
                        response_parts.append(answer_piece)
                        if thinking_piece or answer_piece:
                            state.received = True
                        if on_progress is not None and (
                            thinking_piece or answer_piece or reasoning_field
                        ):
                            on_progress(
                                GenerationUpdate(
                                    thinking=thinking_piece,
                                    response=answer_piece,
                                    elapsed_s=time.perf_counter() - started,
                                    reasoning_channel_seen=reasoning_channel_seen,
                                )
                            )
                except (httpx.HTTPError, httpx.StreamError):
                    if cancelled.is_set() or (
                        cancel_event is not None and cancel_event.is_set()
                    ):
                        generation_cancelled = True
                    elif expired.is_set():
                        timed_out = True
                    else:
                        raise
                finally:
                    cancellation_finished.set()
                    watchdog.cancel()
                generation_cancelled = generation_cancelled or cancelled.is_set()
                timed_out = not generation_cancelled and (timed_out or expired.is_set())
        except httpx.TimeoutException:
            generation_cancelled = bool(
                cancel_event is not None and cancel_event.is_set()
            )
            timed_out = not generation_cancelled

        inline_thinking, inline_answer = inline.finish()
        if inline_thinking or inline_answer:
            thinking_parts.append(inline_thinking)
            response_parts.append(inline_answer)
            if on_progress is not None and not generation_cancelled:
                on_progress(
                    GenerationUpdate(
                        thinking=inline_thinking,
                        response=inline_answer,
                        elapsed_s=time.perf_counter() - started,
                        reasoning_channel_seen=(
                            explicit_reasoning_channel or inline.saw_marker
                        ),
                    )
                )

        elapsed_s = time.perf_counter() - started
        if on_progress is not None and not generation_cancelled:
            on_progress(
                GenerationUpdate(
                    elapsed_s=elapsed_s,
                    reasoning_channel_seen=(
                        explicit_reasoning_channel or inline.saw_marker
                    ),
                    done=True,
                )
            )

        return {
            "thinking": "".join(thinking_parts),
            "response": "".join(response_parts),
            "trace_status": _trace_status(
                "".join(thinking_parts),
                explicit_reasoning_channel or inline.saw_marker,
            ),
            "done_reason": (
                "cancelled"
                if generation_cancelled
                else "timeout"
                if timed_out
                else done_reason
            ),
            "timed_out": timed_out,
            "cancelled": generation_cancelled,
            **_openai_metrics(
                {"usage": usage, "timings": timings},
                elapsed_s,
            ),
        }

    def _generate_ollama(
        self,
        model: str,
        prompt: str,
        on_progress: ProgressCallback | None,
        cancel_event: threading.Event | None,
    ) -> dict:
        return self._retry_stream(
            lambda state: self._stream_ollama(
                model, prompt, on_progress, cancel_event, state
            ),
            cancel_event,
        )

    def _stream_ollama(
        self,
        model: str,
        prompt: str,
        on_progress: ProgressCallback | None,
        cancel_event: threading.Event | None,
        state: _StreamState | None = None,
    ) -> dict:
        state = state if state is not None else _StreamState()
        started = time.perf_counter()
        thinking_parts: list[str] = []
        response_parts: list[str] = []
        final: dict = {}
        timed_out = False
        generation_cancelled = False
        explicit_reasoning_channel = False
        inline = InlineThinkingParser()

        try:
            with httpx.stream(
                "POST",
                f"{_without_v1(self.host)}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": True,
                    "options": {"temperature": 0.0},
                },
                timeout=self._generation_timeout(),
            ) as response:
                response.raise_for_status()
                expired, watchdog = _start_deadline(
                    response,
                    self.timeout - (time.perf_counter() - started),
                )
                cancelled, cancellation_finished = _start_cancellation_watch(
                    response, cancel_event
                )
                try:
                    for line in response.iter_lines():
                        if cancelled.is_set() or (
                            cancel_event is not None and cancel_event.is_set()
                        ):
                            generation_cancelled = True
                            break
                        if (
                            expired.is_set()
                            or time.perf_counter() - started >= self.timeout
                        ):
                            timed_out = True
                            break
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        if data.get("error"):
                            raise RuntimeError(f"generation failed: {data['error']}")
                        final = data
                        reasoning_field = "thinking" in data
                        explicit_reasoning_channel = (
                            explicit_reasoning_channel or reasoning_field
                        )
                        thinking_piece = _content_text(data.get("thinking"))
                        content_piece = _content_text(data.get("response"))

                        if content_piece and not explicit_reasoning_channel:
                            inline_thinking, answer_piece = inline.feed(content_piece)
                            thinking_piece += inline_thinking
                        else:
                            answer_piece = content_piece

                        reasoning_channel_seen = (
                            explicit_reasoning_channel or inline.saw_marker
                        )

                        thinking_parts.append(thinking_piece)
                        response_parts.append(answer_piece)
                        if thinking_piece or answer_piece:
                            state.received = True
                        if on_progress is not None and (
                            thinking_piece or answer_piece or reasoning_field
                        ):
                            on_progress(
                                GenerationUpdate(
                                    thinking=thinking_piece,
                                    response=answer_piece,
                                    elapsed_s=time.perf_counter() - started,
                                    reasoning_channel_seen=reasoning_channel_seen,
                                )
                            )
                except (httpx.HTTPError, httpx.StreamError):
                    if cancelled.is_set() or (
                        cancel_event is not None and cancel_event.is_set()
                    ):
                        generation_cancelled = True
                    elif expired.is_set():
                        timed_out = True
                    else:
                        raise
                finally:
                    cancellation_finished.set()
                    watchdog.cancel()
                generation_cancelled = generation_cancelled or cancelled.is_set()
                timed_out = not generation_cancelled and (timed_out or expired.is_set())
        except httpx.TimeoutException:
            generation_cancelled = bool(
                cancel_event is not None and cancel_event.is_set()
            )
            timed_out = not generation_cancelled

        inline_thinking, inline_answer = inline.finish()
        thinking_parts.append(inline_thinking)
        response_parts.append(inline_answer)
        elapsed_s = time.perf_counter() - started
        reasoning_channel_seen = explicit_reasoning_channel or inline.saw_marker
        if on_progress is not None and not generation_cancelled:
            if inline_thinking or inline_answer:
                on_progress(
                    GenerationUpdate(
                        thinking=inline_thinking,
                        response=inline_answer,
                        elapsed_s=elapsed_s,
                        reasoning_channel_seen=reasoning_channel_seen,
                    )
                )
            on_progress(
                GenerationUpdate(
                    elapsed_s=elapsed_s,
                    reasoning_channel_seen=reasoning_channel_seen,
                    done=True,
                )
            )

        eval_count = final.get("eval_count", 0)
        eval_duration_ns = final.get("eval_duration", 0)
        total_duration_ns = final.get("total_duration", 0)
        tok_s = eval_count / (eval_duration_ns / 1e9) if eval_duration_ns > 0 else 0.0

        thinking = "".join(thinking_parts)

        return {
            "thinking": thinking,
            "response": "".join(response_parts),
            "trace_status": _trace_status(thinking, reasoning_channel_seen),
            "tok_s": tok_s,
            "input_tokens": int(final.get("prompt_eval_count") or 0),
            "eval_count": eval_count,
            "eval_duration_ns": eval_duration_ns,
            "response_time_s": (
                total_duration_ns / 1e9 if total_duration_ns else elapsed_s
            ),
            "done_reason": (
                "cancelled"
                if generation_cancelled
                else "timeout"
                if timed_out
                else final.get("done_reason", "")
            ),
            "timed_out": timed_out,
            "cancelled": generation_cancelled,
        }

    def tokenize(
        self,
        model: str,
        content: str,
        *,
        add_special: bool = True,
    ) -> list[int] | None:
        """Tokenize text through a llama.cpp-compatible native endpoint.

        The endpoint is optional. Returning ``None`` lets performance profiling
        fall back to a deterministic text prompt on generic OpenAI servers.
        """
        if self.provider != "openai":
            return None
        root = _without_v1(self.host)
        attempts = [f"{root}/tokenize"]
        if self._is_llama_swap(model):
            upstream = f"{root}/upstream/{quote(model, safe='/')}"
            attempts.append(f"{upstream}/tokenize")

        body = {
            "model": model,
            "content": content,
            "add_special": add_special,
        }
        for url in attempts:
            try:
                response = self._request(
                    "POST",
                    url,
                    headers=self._headers(),
                    json=body,
                    timeout=min(self.timeout, 60),
                )
                payload = response.json()
                tokens = payload.get("tokens") if isinstance(payload, dict) else None
                if isinstance(tokens, list) and all(
                    isinstance(token, int) for token in tokens
                ):
                    return tokens
            except (httpx.HTTPError, ValueError, TypeError):
                continue
        return None

    def profile_completion(
        self,
        model: str,
        prompt: str | list[int],
        max_tokens: int,
    ) -> dict:
        """Stream a controlled completion and return normalized perf timings."""
        if self.provider != "openai":
            raise RuntimeError(
                "Performance profiling currently targets OpenAI-compatible "
                "llama.cpp/llama-swap endpoints"
            )

        advanced = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "stream": True,
            "temperature": 0.0,
            "seed": 42,
            "ignore_eos": True,
            "cache_prompt": False,
            "timings_per_token": True,
            "stream_options": {"include_usage": True},
        }
        try:
            return run_with_retries(
                self.retries,
                lambda: self._stream_profile(advanced, advanced_options=True),
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in {400, 404, 422}:
                raise

        portable = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "stream": True,
            "temperature": 0.0,
            "seed": 42,
            "stream_options": {"include_usage": True},
        }
        return run_with_retries(
            self.retries,
            lambda: self._stream_profile(portable, advanced_options=False),
        )

    def _stream_profile(self, body: dict, *, advanced_options: bool) -> dict:
        started = time.perf_counter()
        first_token_at: float | None = None
        finished_at = started
        timings: dict = {}
        usage: dict = {}
        text_parts: list[str] = []

        with httpx.stream(
            "POST",
            f"{_openai_base(self.host)}/completions",
            headers=self._headers(),
            json=body,
            timeout=self.timeout,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if not payload or payload == "[DONE]":
                    continue
                data = json.loads(payload)
                if isinstance(data.get("timings"), dict):
                    timings = data["timings"]
                if isinstance(data.get("usage"), dict):
                    usage = data["usage"]
                choices = data.get("choices") or []
                choice = choices[0] if choices else {}
                piece = choice.get("text")
                if isinstance(piece, str) and piece:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    text_parts.append(piece)
            finished_at = time.perf_counter()

        prompt_tokens = int(timings.get("prompt_n") or usage.get("prompt_tokens") or 0)
        cached_tokens = int(timings.get("cache_n") or 0)
        output_tokens = int(
            timings.get("predicted_n") or usage.get("completion_tokens") or 0
        )
        prompt_ms = float(timings.get("prompt_ms") or 0.0)
        predicted_ms = float(timings.get("predicted_ms") or 0.0)
        ttft_s = (
            first_token_at - started
            if first_token_at is not None
            else finished_at - started
        )
        wall_time_s = finished_at - started
        wall_decode_s = (
            finished_at - first_token_at if first_token_at is not None else 0.0
        )
        server_time_s = (prompt_ms + predicted_ms) / 1000

        return {
            "response": "".join(text_parts),
            "input_tokens": prompt_tokens + cached_tokens,
            "processed_input_tokens": prompt_tokens,
            "cached_input_tokens": cached_tokens,
            "output_tokens": output_tokens,
            "pp_tps": (
                float(timings.get("prompt_per_second") or 0.0)
                or (
                    prompt_tokens / (prompt_ms / 1000)
                    if prompt_tokens and prompt_ms > 0
                    else 0.0
                )
            ),
            "tg_tps": (
                float(timings.get("predicted_per_second") or 0.0)
                or (
                    output_tokens / (predicted_ms / 1000)
                    if output_tokens and predicted_ms > 0
                    else 0.0
                )
            ),
            "ttft_s": ttft_s,
            "wall_time_s": wall_time_s,
            "wall_tg_tps": (
                (output_tokens - 1) / wall_decode_s
                if output_tokens > 1 and wall_decode_s > 0
                else 0.0
            ),
            "server_time_s": server_time_s,
            "overhead_s": (
                max(0.0, wall_time_s - server_time_s) if server_time_s > 0 else 0.0
            ),
            "advanced_options": advanced_options,
            "server_timings": bool(timings),
        }

    def unload_model(self, model: str) -> None:
        """Evict only when Ollama exposes a model-scoped unload operation.

        OpenAI-compatible APIs have no standard model unload call. In particular,
        llama-swap handles model switching itself, so BenchKit never calls its
        global ``/unload`` endpoint.
        """
        if self.provider != "ollama":
            return
        self._request(
            "POST",
            f"{_without_v1(self.host)}/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=30,
        )

    def force_unload_model(self, model: str) -> None:
        """Force a model out of VRAM before the next model loads.

        Ollama keeps the model-scoped ``keep_alive=0`` unload. Both llama.cpp's
        router mode and llama-swap unload a named model through the synchronous
        ``POST /models/unload`` endpoint, whose success response means the
        model has already been evicted. Older single-model llama-server builds
        only know the bare ``POST /unload`` endpoint, used as a fallback for
        non-swap servers. Raises ``RuntimeError`` with the server's message when
        no unload could be issued.
        """
        if self.provider == "ollama":
            self.unload_model(model)
            return
        if self.provider != "openai":
            raise RuntimeError(f"{self.label} does not support model unload")

        root = _without_v1(self.host)
        try:
            self._request(
                "POST",
                f"{root}/models/unload",
                headers=self._headers(),
                json={"model": model},
                timeout=min(self.timeout, 60),
            )
            return
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in {404, 405} and not self._is_llama_swap(
                model
            ):
                try:
                    self._request(
                        "POST",
                        f"{root}/unload",
                        headers=self._headers(),
                        timeout=min(self.timeout, 60),
                    )
                    return
                except (httpx.HTTPError, RuntimeError, ValueError, TypeError) as fb:
                    raise RuntimeError(
                        f"POST /unload: {_describe_http_error(fb)}"
                    ) from fb
            raise RuntimeError(
                f"POST /models/unload: {_describe_http_error(exc)}"
            ) from exc
        except (httpx.HTTPError, RuntimeError, ValueError, TypeError) as exc:
            raise RuntimeError(f"POST /models/unload: {exc}") from exc
