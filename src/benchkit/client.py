"""Inference clients for OpenAI-compatible servers and Ollama."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Literal

import httpx

Provider = Literal["auto", "openai", "ollama"]

DEFAULT_HOST = "http://localhost:11434"


def _without_v1(url: str) -> str:
    url = url.rstrip("/")
    return url[:-3] if url.endswith("/v1") else url


def _openai_base(url: str) -> str:
    url = url.rstrip("/")
    return url if url.endswith("/v1") else f"{url}/v1"


def _content_text(content: object) -> str:
    """Normalize OpenAI text content without exposing reasoning fields."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""


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
    predicted_ms = float(
        timings.get("predicted_ms")
        or timings.get("generation_ms")
        or 0.0
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
        "eval_count": eval_count,
        "eval_duration_ns": eval_duration_ns,
        "response_time_s": elapsed_s,
    }


@dataclass
class InferenceClient:
    """Small synchronous client with provider auto-detection."""

    host: str
    requested_provider: Provider = "auto"
    api_key: str | None = None
    timeout: float = 300.0
    provider: Literal["openai", "ollama"] | None = field(default=None, init=False)

    @classmethod
    def from_env(cls) -> "InferenceClient":
        provider = os.environ.get("BENCHKIT_PROVIDER", "auto").lower()
        if provider not in {"auto", "openai", "ollama"}:
            raise ValueError(
                "BENCHKIT_PROVIDER must be one of: auto, openai, ollama"
            )

        host = (
            os.environ.get("BENCHKIT_HOST")
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OLLAMA_HOST")
            or DEFAULT_HOST
        )
        api_key = os.environ.get("BENCHKIT_API_KEY") or os.environ.get(
            "OPENAI_API_KEY"
        )
        timeout = float(os.environ.get("BENCHKIT_TIMEOUT", "300"))
        return cls(host.rstrip("/"), provider, api_key, timeout)

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
                return models
            except (httpx.HTTPError, ValueError, TypeError) as exc:
                errors.append(f"{provider}: {exc}")

        raise RuntimeError(
            "Could not discover a compatible API (" + "; ".join(errors) + ")"
        )

    def _list_openai_models(self) -> list[dict]:
        response = httpx.get(
            f"{_openai_base(self.host)}/models",
            headers=self._headers(),
            timeout=10,
        )
        response.raise_for_status()
        models = response.json().get("data", [])
        normalized = []
        for model in models:
            model_id = model.get("id")
            if not model_id:
                continue
            normalized.append(
                {
                    "name": model_id,
                    "size": model.get("size"),
                    "owned_by": model.get("owned_by", ""),
                    "status": (model.get("status") or {}).get("value", ""),
                }
            )
        return sorted(normalized, key=lambda model: model["name"].lower())

    def _list_ollama_models(self) -> list[dict]:
        response = httpx.get(
            f"{_without_v1(self.host)}/api/tags",
            timeout=10,
        )
        response.raise_for_status()
        models = response.json().get("models", [])
        return sorted(models, key=lambda model: model.get("size", 0))

    def generate(self, model: str, prompt: str) -> dict:
        if self.provider is None:
            raise RuntimeError("Call list_models() before generate()")
        if self.provider == "openai":
            return self._generate_openai(model, prompt)
        return self._generate_ollama(model, prompt)

    def _generate_openai(self, model: str, prompt: str) -> dict:
        started = time.perf_counter()
        response = httpx.post(
            f"{_openai_base(self.host)}/chat/completions",
            headers=self._headers(),
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "temperature": 0.0,
            },
            timeout=self.timeout,
        )
        elapsed_s = time.perf_counter() - started
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        choice = choices[0] if choices else {}
        message = choice.get("message") or {}

        return {
            "response": _content_text(message.get("content")),
            "done_reason": choice.get("finish_reason", ""),
            **_openai_metrics(data, elapsed_s),
        }

    def _generate_ollama(self, model: str, prompt: str) -> dict:
        response = httpx.post(
            f"{_without_v1(self.host)}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()

        eval_count = data.get("eval_count", 0)
        eval_duration_ns = data.get("eval_duration", 0)
        total_duration_ns = data.get("total_duration", 0)
        tok_s = (
            eval_count / (eval_duration_ns / 1e9) if eval_duration_ns > 0 else 0.0
        )

        return {
            "response": data.get("response", ""),
            "tok_s": tok_s,
            "eval_count": eval_count,
            "eval_duration_ns": eval_duration_ns,
            "response_time_s": total_duration_ns / 1e9,
            "done_reason": data.get("done_reason", ""),
        }

    def unload_model(self, model: str) -> None:
        """Evict only when Ollama exposes a model-scoped unload operation.

        OpenAI-compatible APIs have no standard model unload call. In particular,
        llama-swap handles model switching itself, so BenchKit never calls its
        global ``/unload`` endpoint.
        """
        if self.provider != "ollama":
            return
        httpx.post(
            f"{_without_v1(self.host)}/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=30,
        )
