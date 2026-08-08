"""Tests for automatic server-capacity discovery and concurrent scheduling."""

from __future__ import annotations

import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import httpx

from benchkit.client import GenerationUpdate
from benchkit.engine import Engine, JobSpec, JobStarted, RunControls, benchmark
from benchkit.metrics import (
    aggregate_tok_s,
    effective_concurrency,
    stream_tok_s,
    throughput_metrics,
)
from benchkit.tui.formatting import throughput_stat


def _http_error(status: int, url: str) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", url)
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError(f"status {status}", request=request, response=response)


def _generation(*, cancelled: bool = False) -> dict:
    return {
        "thinking": "",
        "response": "pass",
        "trace_status": "unavailable",
        "tok_s": 10.0,
        "eval_count": 1,
        "eval_duration_ns": 100_000_000,
        "response_time_s": 0.02,
        "done_reason": "cancelled" if cancelled else "stop",
        "timed_out": False,
        "cancelled": cancelled,
    }


class ParallelClient:
    timeout = 1.0
    label = "test"
    host = "test://local"

    def __init__(self, capacities: dict[str, int]) -> None:
        self.capacities = capacities
        self.active: dict[str, int] = {}
        self.max_active: dict[str, int] = {}
        self.calls: dict[str, int] = {}
        self.lock = threading.Lock()

    def max_parallel_requests(self, model: str) -> int:
        return self.capacities[model]

    def generate(self, model: str, prompt: str, **_kwargs: object) -> dict:
        with self.lock:
            self.active[model] = self.active.get(model, 0) + 1
            self.calls[model] = self.calls.get(model, 0) + 1
            self.max_active[model] = max(
                self.max_active.get(model, 0), self.active[model]
            )
        try:
            time.sleep(0.025)
            return _generation()
        finally:
            with self.lock:
                self.active[model] -= 1

    def unload_model(self, _model: str) -> None:
        return None


class BlockingParallelClient(ParallelClient):
    def generate(
        self,
        model: str,
        prompt: str,
        *,
        cancel_event: threading.Event | None = None,
        **_kwargs: object,
    ) -> dict:
        with self.lock:
            self.active[model] = self.active.get(model, 0) + 1
            self.calls[model] = self.calls.get(model, 0) + 1
            self.max_active[model] = max(
                self.max_active.get(model, 0), self.active[model]
            )
        try:
            while cancel_event is None or not cancel_event.wait(0.002):
                pass
            return _generation(cancelled=True)
        finally:
            with self.lock:
                self.active[model] -= 1


class ProgressParallelClient(ParallelClient):
    def generate(
        self,
        model: str,
        prompt: str,
        *,
        on_progress=None,
        **kwargs: object,
    ) -> dict:
        if on_progress is not None:
            on_progress(
                GenerationUpdate(
                    response="pass",
                    elapsed_s=0.01,
                )
            )
        return super().generate(model, prompt, **kwargs)


class CapacityDiscoveryTests(unittest.TestCase):
    def test_model_metadata_is_used_as_a_capacity_cap(self) -> None:
        from benchkit.client import InferenceClient

        models = Mock()
        models.json.return_value = {
            "data": [
                {
                    "id": "model",
                    "owned_by": "llamacpp",
                    "meta": {"n_parallel": "5"},
                }
            ]
        }
        slots = Mock()
        slots.json.return_value = [{"id": index} for index in range(8)]
        client = InferenceClient("http://local", "openai")

        with patch.object(client, "_request", side_effect=[models, slots]):
            discovered = client.list_models()
            capacity = client.max_parallel_requests("model")

        self.assertEqual(discovered[0]["parallelism"], 5)
        self.assertEqual(capacity, 5)

    def test_direct_llama_cpp_counts_model_slots(self) -> None:
        from benchkit.client import InferenceClient

        client = InferenceClient("http://local/v1", "openai")
        client.provider = "openai"
        client._models_by_name = {"model-x": {"owned_by": "llamacpp"}}
        response = Mock()
        response.json.return_value = [{"id": index} for index in range(4)]

        with patch.object(client, "_request", return_value=response) as request:
            self.assertEqual(client.max_parallel_requests("model-x"), 4)
            self.assertEqual(client.max_parallel_requests("model-x"), 4)

        request.assert_called_once()
        self.assertEqual(request.call_args.kwargs["params"], {"model": "model-x"})

    def test_llama_swap_probes_the_selected_models_upstream(self) -> None:
        from benchkit.client import InferenceClient

        client = InferenceClient("http://swap/v1", "openai")
        client.provider = "openai"
        client._models_by_name = {
            "org/model y": {
                "owned_by": "llama-swap",
                "meta": {"llamaswap": {"type": "model"}},
            }
        }
        slots = Mock()
        slots.json.return_value = [{"id": index} for index in range(6)]

        with patch.object(
            client,
            "_request",
            side_effect=[
                _http_error(404, "http://swap/slots"),
                _http_error(404, "http://swap/health"),
                slots,
            ],
        ) as request:
            self.assertEqual(client.max_parallel_requests("org/model y"), 6)

        self.assertEqual(request.call_count, 3)
        self.assertEqual(
            request.call_args.args[1],
            "http://swap/upstream/org/model%20y/slots",
        )

    def test_unknown_openai_server_safely_stays_serial(self) -> None:
        from benchkit.client import InferenceClient

        client = InferenceClient("http://generic", "openai")
        client.provider = "openai"
        client._models_by_name = {"model": {"owned_by": "provider"}}

        with patch.object(
            client,
            "_request",
            side_effect=[
                _http_error(404, "http://generic/slots"),
                _http_error(404, "http://generic/health"),
            ],
        ):
            self.assertEqual(client.max_parallel_requests("model"), 1)

    def test_explicit_metadata_caps_the_discovered_slots(self) -> None:
        from benchkit.client import InferenceClient

        client = InferenceClient("http://local", "openai")
        client.provider = "openai"
        client._models_by_name = {"model": {"parallelism": 3, "owned_by": "llamacpp"}}
        response = Mock()
        response.json.return_value = [{"id": index} for index in range(8)]

        with patch.object(client, "_request", return_value=response):
            self.assertEqual(client.max_parallel_requests("model"), 3)


class ThroughputMetricTests(unittest.TestCase):
    def test_parallel_metrics_separate_stream_speed_from_total_throughput(
        self,
    ) -> None:
        metrics = throughput_metrics(
            output_tokens=25_938,
            generation_time_s=594.0,
            request_time_s=594.0,
            wall_time_s=99.0,
        )

        self.assertAlmostEqual(metrics["tok_s_per_stream"], 43.67, places=2)
        self.assertAlmostEqual(metrics["tok_s_aggregate"], 262.0, places=2)
        self.assertAlmostEqual(metrics["concurrency_eff"], 6.0, places=2)

    def test_legacy_result_metrics_can_be_derived(self) -> None:
        legacy = {
            "tok_s": 43.7,
            "avg_response_time": 29.7,
            "total": 20,
            "total_time": 99.0,
        }

        self.assertEqual(stream_tok_s(legacy), 43.7)
        self.assertAlmostEqual(effective_concurrency(legacy), 6.0)
        self.assertAlmostEqual(aggregate_tok_s(legacy), 262.2)

    def test_serial_throughput_renders_stream_speed_only(self) -> None:
        self.assertEqual(
            throughput_stat(
                concurrency=1,
                aggregate=64.0,
                stream=96.2,
                effective=0.67,
                precision=0,
            ),
            ("96 tok/s", "single stream"),
        )

    def test_parallel_throughput_renders_aggregate_breakdown(self) -> None:
        self.assertEqual(
            throughput_stat(
                concurrency=4,
                aggregate=180.0,
                stream=52.0,
                effective=3.46,
                precision=0,
            ),
            ("180 tok/s", "aggregate · 52 stream · 3.46x effective"),
        )


class ConcurrentEngineTests(unittest.TestCase):
    def test_pi_uses_normal_discovered_capacity(self) -> None:
        client = ParallelClient({"model": 4})
        engine = Engine(client=client, jobs=[])

        concurrency = engine._max_parallel_requests(
            JobSpec("model", "quickbench", "5", harness="pi"),
            5,
        )

        self.assertEqual(concurrency, 4)

    def test_each_model_uses_its_own_discovered_capacity(self) -> None:
        client = ParallelClient({"model-x": 4, "model-y": 6})
        events: list[object] = []
        jobs = [
            JobSpec("model-x", "quickbench", "8"),
            JobSpec("model-y", "quickbench", "12"),
        ]

        with patch.object(benchmark("quickbench"), "evaluate", return_value=True):
            results = Engine(client=client, jobs=jobs, sink=events.append).run()

        self.assertEqual(client.calls, {"model-x": 8, "model-y": 12})
        self.assertEqual(client.max_active, {"model-x": 4, "model-y": 6})
        self.assertEqual([result["concurrency"] for result in results], [4, 6])
        self.assertEqual([result["tok_s_per_stream"] for result in results], [10, 10])
        self.assertEqual(
            [result["tok_s"] for result in results],
            [result["tok_s_per_stream"] for result in results],
        )
        self.assertEqual([result["total_output_tokens"] for result in results], [8, 12])
        self.assertEqual(
            [result["sum_generation_time"] for result in results],
            [0.8, 1.2],
        )
        self.assertEqual(
            [result["sum_request_time"] for result in results],
            [0.16, 0.24],
        )
        self.assertTrue(all(result["tok_s_aggregate"] > 0 for result in results))
        self.assertTrue(all(result["concurrency_eff"] > 0 for result in results))
        starts = [event for event in events if isinstance(event, JobStarted)]
        self.assertEqual([event.concurrency for event in starts], [4, 6])
        self.assertEqual(
            [task["task_id"] for task in results[1]["tasks"]],
            [f"QuickBench/{index}" for index in range(12)],
        )

    def test_task_count_caps_a_larger_server_capacity(self) -> None:
        client = ParallelClient({"model": 16})

        with patch.object(benchmark("quickbench"), "evaluate", return_value=True):
            result = Engine(
                client=client,
                jobs=[JobSpec("model", "quickbench", "3")],
            ).run()[0]

        self.assertEqual(result["concurrency"], 3)
        self.assertEqual(client.max_active["model"], 3)

    def test_stop_cancels_every_active_request(self) -> None:
        controls = RunControls()
        client = BlockingParallelClient({"model": 4})
        timer = threading.Timer(0.03, controls.stop)

        started = time.perf_counter()
        timer.start()
        try:
            results = Engine(
                client=client,
                jobs=[JobSpec("model", "quickbench", "10")],
                controls=controls,
            ).run()
        finally:
            timer.cancel()

        self.assertLess(time.perf_counter() - started, 0.5)
        self.assertEqual(client.calls["model"], 4)
        self.assertEqual(client.max_active["model"], 4)
        self.assertEqual(results, [])


class ConcurrentTUITests(unittest.IsolatedAsyncioTestCase):
    async def test_run_screen_accepts_interleaved_task_events(self) -> None:
        from textual.widgets import Static

        import benchkit.tui.app as tui_app_module
        from benchkit.tui.app import BenchKitApp
        from benchkit.tui.screens.run import RunScreen
        from benchkit.tui.widgets import StatCard

        class Harness(BenchKitApp):
            CSS_PATH = Path(tui_app_module.__file__).with_name("app.tcss")

            def __init__(self) -> None:
                super().__init__(demo=True)
                self.client = ProgressParallelClient({"model": 3})
                self.results: list[dict] | None = None

            def on_mount(self) -> None:
                self.push_screen(RunScreen([JobSpec("model", "quickbench", "6")]))

            def show_results(self, results: list[dict], output: Path | None) -> None:
                self.results = results

            def show_setup(self, _client, _models: list[dict]) -> None:
                return None

        app = Harness()
        with (
            patch.object(benchmark("quickbench"), "evaluate", return_value=True),
            patch(
                "benchkit.tui.screens.run.save",
                return_value=Path("/tmp/benchkit-test-results"),
            ),
        ):
            async with app.run_test(size=(120, 40)) as pilot:
                for _ in range(100):
                    if app.results is not None:
                        break
                    await pilot.pause(0.02)

                self.assertIsNotNone(app.results)
                assert app.results is not None
                self.assertEqual(app.results[0]["concurrency"], 3)
                screen = next(
                    mounted
                    for mounted in app.screen_stack
                    if isinstance(mounted, RunScreen)
                )
                self.assertEqual(len(screen.records[0]), 6)
                self.assertEqual(screen.completed_generated_chars, 24)
                job_summary = str(
                    screen.query_one("#job-progress-label", Static).render()
                )
                live_card = screen.query_one("#stat-live", StatCard)
                self.assertIn("24 chars generated", job_summary)
                self.assertNotIn("QuickBench/", job_summary)
                self.assertEqual(live_card.hint, "24 chars generated this job")


if __name__ == "__main__":
    unittest.main()
