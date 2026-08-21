"""Tests for force-unloading models between jobs in a multi-model run."""

from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import Mock, patch

import httpx

from benchkit.client import InferenceClient
from benchkit.engine import Engine, JobSpec, ModelUnloaded, RunControls, benchmark


def _http_error(
    status: int, url: str, body: dict | None = None
) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", url)
    response = httpx.Response(status, request=request, json=body)
    return httpx.HTTPStatusError(f"status {status}", request=request, response=response)


def _generation() -> dict:
    return {
        "thinking": "",
        "response": "pass",
        "trace_status": "unavailable",
        "tok_s": 10.0,
        "eval_count": 1,
        "eval_duration_ns": 100_000_000,
        "response_time_s": 0.02,
        "done_reason": "stop",
        "timed_out": False,
        "cancelled": False,
    }


class _OrderingClient:
    """Records the interleaving of generation and unload calls."""

    timeout = 1.0
    label = "test"
    host = "test://local"

    def __init__(self, capacity: int = 4) -> None:
        self.capacity = capacity
        self.events: list[tuple[str, str]] = []
        self.lock = threading.Lock()

    def max_parallel_requests(self, model: str) -> int:
        return self.capacity

    def generate(self, model: str, prompt: str, **_kwargs: object) -> dict:
        with self.lock:
            self.events.append(("generate", model))
        time.sleep(0.01)
        return _generation()

    def unload_model(self, model: str) -> None:
        with self.lock:
            self.events.append(("unload", model))

    def force_unload_model(self, model: str) -> None:
        with self.lock:
            self.events.append(("force_unload", model))


class RunControlsForceUnloadTests(unittest.TestCase):
    def test_force_unload_defaults_off_and_toggles(self) -> None:
        controls = RunControls()
        self.assertFalse(controls.force_unload)

        self.assertTrue(controls.toggle_force_unload())
        self.assertTrue(controls.force_unload)

        self.assertFalse(controls.toggle_force_unload())
        self.assertFalse(controls.force_unload)

    def test_force_unload_accepts_an_initial_value(self) -> None:
        self.assertTrue(RunControls(force_unload=True).force_unload)


class ForceUnloadClientTests(unittest.TestCase):
    def test_ollama_force_unload_keeps_the_model_scoped_request(self) -> None:
        client = InferenceClient("http://local", "ollama")
        client.provider = "ollama"

        with patch.object(client, "_request", return_value=Mock()) as request:
            client.force_unload_model("model")

        request.assert_called_once()
        self.assertEqual(request.call_args.args[1], "http://local/api/generate")
        self.assertEqual(
            request.call_args.kwargs["json"],
            {"model": "model", "keep_alive": 0},
        )

    def test_llamacpp_force_unload_calls_models_unload(self) -> None:
        client = InferenceClient("http://local/v1", "openai")
        client.provider = "openai"
        client._models_by_name = {"model-x": {"owned_by": "llamacpp"}}

        with patch.object(client, "_request", return_value=Mock()) as request:
            client.force_unload_model("model-x")

        self.assertEqual(request.call_count, 1)
        self.assertEqual(request.call_args.args[1], "http://local/models/unload")
        self.assertEqual(request.call_args.kwargs["json"], {"model": "model-x"})

    def test_force_unload_falls_back_to_single_model_unload(self) -> None:
        client = InferenceClient("http://local/v1", "openai")
        client.provider = "openai"
        client._models_by_name = {"model-x": {"owned_by": "llamacpp"}}

        with patch.object(
            client,
            "_request",
            side_effect=[
                _http_error(404, "http://local/models/unload"),
                Mock(),
            ],
        ) as request:
            client.force_unload_model("model-x")

        self.assertEqual(request.call_count, 2)
        self.assertEqual(request.call_args.args[1], "http://local/unload")

    def test_force_unload_llama_swap_uses_named_models_unload(self) -> None:
        client = InferenceClient("http://swap/v1", "openai")
        client.provider = "openai"
        client._models_by_name = {
            "org/model": {"owned_by": "llama-swap", "meta": {}},
        }

        with patch.object(client, "_request", return_value=Mock()) as request:
            client.force_unload_model("org/model")

        self.assertEqual(request.call_count, 1)
        self.assertEqual(request.call_args.args[1], "http://swap/models/unload")
        self.assertEqual(request.call_args.kwargs["json"], {"model": "org/model"})

    def test_force_unload_llama_swap_never_hits_global_unload(self) -> None:
        client = InferenceClient("http://swap/v1", "openai")
        client.provider = "openai"
        client._models_by_name = {
            "org/model": {"owned_by": "llama-swap", "meta": {}},
        }

        with (
            patch.object(
                client,
                "_request",
                side_effect=[_http_error(404, "http://swap/models/unload")],
            ) as request,
            self.assertRaises(RuntimeError),
        ):
            client.force_unload_model("org/model")

        self.assertEqual(request.call_count, 1)
        self.assertEqual(request.call_args.args[1], "http://swap/models/unload")

    def test_force_unload_surfaces_the_server_error_message(self) -> None:
        client = InferenceClient("http://local/v1", "openai")
        client.provider = "openai"
        client._models_by_name = {"model-x": {"owned_by": "llamacpp"}}

        with (
            patch.object(
                client,
                "_request",
                side_effect=[
                    _http_error(
                        400,
                        "http://local/models/unload",
                        {"error": {"message": "model is not found"}},
                    ),
                ],
            ),
            self.assertRaises(RuntimeError) as raised,
        ):
            client.force_unload_model("model-x")

        self.assertIn("model is not found", str(raised.exception))


class ForceUnloadSchedulingTests(unittest.TestCase):
    def test_unload_waits_for_drain_then_precedes_the_next_model(self) -> None:
        client = _OrderingClient()
        controls = RunControls(force_unload=True)
        jobs = [
            JobSpec("model-a", "sanity", "6"),
            JobSpec("model-b", "sanity", "6"),
        ]

        with patch.object(benchmark("sanity"), "evaluate", return_value=True):
            Engine(client=client, jobs=jobs, controls=controls).run()

        events = client.events
        last_a = max(
            index
            for index, (kind, model) in enumerate(events)
            if kind == "generate" and model == "model-a"
        )
        unload_a = next(
            index
            for index, (kind, model) in enumerate(events)
            if kind == "force_unload" and model == "model-a"
        )
        first_b = next(
            index
            for index, (kind, model) in enumerate(events)
            if kind == "generate" and model == "model-b"
        )

        self.assertEqual(
            sum(
                1 for kind, model in events if kind == "generate" and model == "model-a"
            ),
            6,
        )
        self.assertEqual(
            sum(
                1 for kind, model in events if kind == "generate" and model == "model-b"
            ),
            6,
        )
        self.assertLess(last_a, unload_a)
        self.assertLess(unload_a, first_b)

    def test_unload_stays_off_unless_forced(self) -> None:
        client = _OrderingClient()
        jobs = [
            JobSpec("model-a", "sanity", "3"),
            JobSpec("model-b", "sanity", "3"),
        ]

        with patch.object(benchmark("sanity"), "evaluate", return_value=True):
            Engine(client=client, jobs=jobs).run()

        self.assertNotIn(
            "force_unload",
            {kind for kind, _model in client.events},
        )
        # Soft unload still fires at every model boundary by default.
        self.assertIn(
            "unload",
            {kind for kind, _model in client.events},
        )

    def test_force_unload_emits_an_event_per_model(self) -> None:
        client = _OrderingClient()
        controls = RunControls(force_unload=True)
        events: list[object] = []
        jobs = [
            JobSpec("model-a", "sanity", "2"),
            JobSpec("model-b", "sanity", "2"),
        ]

        with patch.object(benchmark("sanity"), "evaluate", return_value=True):
            Engine(
                client=client, jobs=jobs, controls=controls, sink=events.append
            ).run()

        unloaded = [event for event in events if isinstance(event, ModelUnloaded)]
        self.assertEqual([event.model for event in unloaded], ["model-a", "model-b"])


if __name__ == "__main__":
    unittest.main()
