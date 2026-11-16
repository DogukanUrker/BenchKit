"""Tests for who decides the sampling temperature: the server, unless pinned."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from benchkit import config
from benchkit.client import InferenceClient, temperature_from_env
from benchkit.engine import Engine, JobSpec


def _openai_body(client: InferenceClient) -> dict:
    client.provider = "openai"
    with patch.object(client, "_retry_stream", return_value={}) as stream:
        client.generate("m", "hi")
        builder = stream.call_args.args[0]
    with patch.object(client, "_stream_openai", return_value={}) as inner:
        builder(None)
    return inner.call_args.args[0]


class _OllamaStream:
    def __enter__(self) -> _OllamaStream:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def close(self) -> None:
        return None

    def iter_lines(self):
        yield json.dumps({"response": "4", "done": True, "done_reason": "stop"})


def _ollama_body(client: InferenceClient) -> dict:
    client.provider = "ollama"
    with patch("benchkit.client.httpx.stream", return_value=_OllamaStream()) as stream:
        client.generate("m", "hi")
    return stream.call_args.kwargs["json"]


def test_openai_requests_leave_sampling_to_the_server_by_default() -> None:
    body = _openai_body(InferenceClient("http://local/v1", "openai"))

    assert "temperature" not in body
    assert not {"top_p", "top_k", "seed"} & body.keys()


def test_openai_requests_send_a_pinned_temperature() -> None:
    body = _openai_body(InferenceClient("http://local/v1", "openai", temperature=0.7))

    assert body["temperature"] == 0.7


def test_a_pinned_zero_is_sent_rather_than_dropped_as_falsy() -> None:
    body = _openai_body(InferenceClient("http://local/v1", "openai", temperature=0.0))

    assert body["temperature"] == 0.0


def test_ollama_requests_leave_sampling_to_the_modelfile_by_default() -> None:
    body = _ollama_body(InferenceClient("http://local", "ollama"))

    assert "options" not in body


def test_ollama_requests_send_a_pinned_temperature() -> None:
    body = _ollama_body(InferenceClient("http://local", "ollama", temperature=0.0))

    assert body["options"] == {"temperature": 0.0}


@pytest.mark.parametrize("raw", ["", "   "])
def test_an_unset_temperature_means_the_server_decides(monkeypatch, raw) -> None:
    monkeypatch.setenv("BENCHKIT_TEMPERATURE", raw)

    assert temperature_from_env() is None
    assert InferenceClient.from_env().temperature is None


def test_the_environment_pins_the_temperature(monkeypatch) -> None:
    monkeypatch.setenv("BENCHKIT_TEMPERATURE", "0")
    assert InferenceClient.from_env().temperature == 0.0

    monkeypatch.setenv("BENCHKIT_TEMPERATURE", "0.65")
    assert InferenceClient.from_env().temperature == 0.65


@pytest.mark.parametrize("raw", ["warm", "-0.1", "nan", "inf"])
def test_a_bad_temperature_is_refused(monkeypatch, raw) -> None:
    monkeypatch.setenv("BENCHKIT_TEMPERATURE", raw)

    with pytest.raises(ValueError, match="BENCHKIT_TEMPERATURE"):
        temperature_from_env()


def test_the_setting_is_listed_by_benchkit_config() -> None:
    assert ("BENCHKIT_TEMPERATURE", "") in config.SETTINGS


class _Client:
    timeout = 10.0
    label = "test"
    host = "test://local"
    provider = "openai"

    def __init__(self, temperature: float | None) -> None:
        self.temperature = temperature

    def max_parallel_requests(self, _model: str) -> int:
        return 1

    def generate(self, _model: str, _prompt: str, **_kwargs: object) -> dict:
        return {
            "thinking": "",
            "response": "4",
            "trace_status": "unavailable",
            "tok_s": 10.0,
            "eval_count": 1,
            "eval_duration_ns": 100_000_000,
            "response_time_s": 0.1,
            "done_reason": "stop",
            "timed_out": False,
            "cancelled": False,
            "input_tokens": 5,
        }


@pytest.mark.parametrize("temperature", [None, 0.0, 0.7])
def test_each_job_records_the_temperature_it_was_sent(temperature) -> None:
    results = Engine(
        client=_Client(temperature), jobs=[JobSpec("m", "sanity", "1")]
    ).run()

    assert results[0]["request_temperature"] == temperature
