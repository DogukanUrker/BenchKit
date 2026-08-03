"""Tests for passive reasoning capture and loop detection."""

from __future__ import annotations

import json
import os
import threading
import time
import unittest
from unittest.mock import patch

import httpx

from benchkit.client import GenerationUpdate, InferenceClient
from benchkit.engine import Engine, GenerationProgress, JobSpec, RunControls
from benchkit.looping import InlineThinkingParser, LoopAnalyzer


class FakeStreamResponse:
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines
        self.status_code = 200
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        self.close()
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self):
        yield from self.lines

    def close(self) -> None:
        self.closed = True


class DelayedStreamResponse(FakeStreamResponse):
    def iter_lines(self):
        yield self.lines[0]
        time.sleep(0.05)
        yield from self.lines[1:]


class BlockingStreamResponse(FakeStreamResponse):
    def iter_lines(self):
        yield self.lines[0]
        while not self.closed:
            time.sleep(0.001)
        raise httpx.StreamClosed


class LoopAnalyzerTests(unittest.TestCase):
    def test_inline_tags_can_cross_chunks(self) -> None:
        parser = InlineThinkingParser()
        pieces = [
            parser.feed("<thi"),
            parser.feed("nk>check this</thi"),
            parser.feed("nk>final"),
            parser.finish(),
        ]

        self.assertTrue(parser.saw_marker)
        self.assertEqual("".join(piece[0] for piece in pieces), "check this")
        self.assertEqual("".join(piece[1] for piece in pieces), "final")

    def test_repeated_reasoning_is_flagged(self) -> None:
        analyzer = LoopAnalyzer()
        analyzer.add(
            thinking=(
                "I should reconsider the same evidence before deciding again "
                "because this path may still be wrong "
            )
            * 24
        )

        snapshot = analyzer.snapshot(final=True)

        self.assertEqual(snapshot.state, "looping")
        self.assertGreater(snapshot.repeated_ngram_coverage, 0.8)
        self.assertEqual(snapshot.source, "thinking")

    def test_novel_reasoning_is_clear(self) -> None:
        analyzer = LoopAnalyzer()
        analyzer.add(thinking=" ".join(f"distinct{i}" for i in range(256)))

        snapshot = analyzer.snapshot(final=True)

        self.assertEqual(snapshot.state, "clear")
        self.assertEqual(snapshot.repeated_ngram_coverage, 0.0)

    def test_short_final_answer_is_clear(self) -> None:
        analyzer = LoopAnalyzer()
        analyzer.add(answer="The answer is B.")

        self.assertEqual(analyzer.snapshot(final=True).state, "clear")

    def test_answer_loop_is_detected_even_with_short_thinking(self) -> None:
        analyzer = LoopAnalyzer()
        analyzer.add(
            thinking="A brief useful check.",
            answer=("the visible answer repeats the same broken output " * 30),
        )

        snapshot = analyzer.snapshot(final=True)

        self.assertEqual(snapshot.state, "looping")
        self.assertEqual(snapshot.source, "answer")

    def test_live_analysis_retains_a_bounded_tail(self) -> None:
        analyzer = LoopAnalyzer()
        analyzer.add(thinking="x" * 200_000)

        self.assertEqual(analyzer.thinking_chars, 200_000)
        self.assertLess(len(analyzer.thinking), 200_000)


class ClientStreamingTests(unittest.TestCase):
    def test_generation_uses_configured_transport_timeout(self) -> None:
        client = InferenceClient("http://local", "openai", timeout=12)

        timeout = client._generation_timeout()

        self.assertEqual(timeout.read, 12)
        self.assertEqual(timeout.connect, 12)

    def test_task_deadline_keeps_partial_trace_and_returns_timeout(self) -> None:
        stream = DelayedStreamResponse(
            [
                _sse({"choices": [{"delta": {"reasoning_content": "partial"}}]}),
                _sse({"choices": [{"delta": {"content": "too late"}}]}),
            ]
        )
        client = InferenceClient("http://local", "openai", timeout=0.02)
        client.provider = "openai"

        with patch("benchkit.client.httpx.stream", return_value=stream):
            result = client.generate("model", "prompt")

        self.assertTrue(result["timed_out"])
        self.assertEqual(result["done_reason"], "timeout")
        self.assertEqual(result["thinking"], "partial")
        self.assertEqual(result["response"], "")
        self.assertTrue(stream.closed)

    def test_user_cancel_closes_an_inflight_stream(self) -> None:
        stream = BlockingStreamResponse(
            [_sse({"choices": [{"delta": {"reasoning_content": "partial"}}]})]
        )
        cancel_event = threading.Event()
        cancel_timer = threading.Timer(0.01, cancel_event.set)
        client = InferenceClient("http://local", "openai", timeout=1)
        client.provider = "openai"

        started = time.perf_counter()
        cancel_timer.start()
        try:
            with patch("benchkit.client.httpx.stream", return_value=stream):
                result = client.generate("model", "prompt", cancel_event=cancel_event)
        finally:
            cancel_timer.cancel()

        self.assertLess(time.perf_counter() - started, 0.25)
        self.assertTrue(stream.closed)
        self.assertTrue(result["cancelled"])
        self.assertFalse(result["timed_out"])
        self.assertEqual(result["done_reason"], "cancelled")
        self.assertEqual(result["thinking"], "partial")

    def test_progress_abort_unwinds_and_closes_the_stream(self) -> None:
        stream = FakeStreamResponse(
            [_sse({"choices": [{"delta": {"reasoning_content": "loop"}}]})]
        )
        client = InferenceClient("http://local", "openai", timeout=1)
        client.provider = "openai"

        def abort(_update: GenerationUpdate) -> None:
            raise RuntimeError("kill generation")

        with patch("benchkit.client.httpx.stream", return_value=stream):
            with self.assertRaisesRegex(RuntimeError, "kill generation"):
                client.generate("model", "prompt", on_progress=abort)

        self.assertTrue(stream.closed)

    def test_openai_reasoning_and_answer_streams_are_separated(self) -> None:
        lines = [
            _sse({"choices": [{"delta": {"reasoning_content": "check "}}]}),
            _sse({"choices": [{"delta": {"reasoning_content": "twice"}}]}),
            _sse({"choices": [{"delta": {"content": "B"}}]}),
            _sse(
                {
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                    "usage": {"completion_tokens": 3},
                }
            ),
            "data: [DONE]",
        ]
        updates: list[GenerationUpdate] = []
        client = InferenceClient("http://local", "openai")
        client.provider = "openai"

        with patch(
            "benchkit.client.httpx.stream",
            return_value=FakeStreamResponse(lines),
        ):
            result = client.generate("model", "prompt", updates.append)

        self.assertEqual(result["thinking"], "check twice")
        self.assertEqual(result["response"], "B")
        self.assertEqual(result["trace_status"], "observed")
        self.assertEqual(result["eval_count"], 3)
        self.assertEqual(result["done_reason"], "stop")
        self.assertTrue(updates[-1].done)

    def test_ollama_inline_thinking_fallback_is_streamed(self) -> None:
        lines = [
            json.dumps({"response": "<thi", "done": False}),
            json.dumps({"response": "nk>check</thi", "done": False}),
            json.dumps({"response": "nk>A", "done": False}),
            json.dumps(
                {
                    "response": "",
                    "done": True,
                    "done_reason": "stop",
                    "eval_count": 4,
                    "eval_duration": 1_000_000_000,
                    "total_duration": 1_200_000_000,
                }
            ),
        ]
        client = InferenceClient("http://local", "ollama")
        client.provider = "ollama"

        with patch(
            "benchkit.client.httpx.stream",
            return_value=FakeStreamResponse(lines),
        ):
            result = client.generate("model", "prompt")

        self.assertEqual(result["thinking"], "check")
        self.assertEqual(result["response"], "A")
        self.assertEqual(result["trace_status"], "observed")
        self.assertEqual(result["done_reason"], "stop")


class EngineIntegrationTests(unittest.TestCase):
    def test_live_loop_state_reaches_task_and_job_results(self) -> None:
        events: list[object] = []
        client = LoopingClient()

        results = Engine(
            client=client,
            jobs=[JobSpec("looping-model", "quickbench", "1")],
            sink=events.append,
        ).run()

        live = [event for event in events if isinstance(event, GenerationProgress)]
        self.assertTrue(any(event.loop_state == "looping" for event in live))
        self.assertEqual(results[0]["loops"], 1)
        self.assertEqual(results[0]["loop_rate"], 100.0)
        self.assertEqual(results[0]["trace_coverage"], 100.0)
        self.assertEqual(results[0]["tasks"][0]["loop_state"], "looping")

    def test_timeout_skips_evaluation_and_continues_to_next_task(self) -> None:
        client = TimeoutClient()

        results = Engine(
            client=client,
            jobs=[JobSpec("slow-model", "quickbench", "2")],
        ).run()

        result = results[0]
        self.assertEqual(client.calls, 2)
        self.assertEqual(result["total"], 2)
        self.assertEqual(result["passed"], 0)
        self.assertEqual(result["errors"], 2)
        self.assertEqual(result["timeouts"], 2)
        self.assertTrue(all(task["timed_out"] for task in result["tasks"]))
        self.assertTrue(
            all("timed out after 9s" in task["error"] for task in result["tasks"])
        )

    def test_stop_abandons_the_inflight_task_without_waiting(self) -> None:
        controls = RunControls()
        client = TokenLoopClient()
        stop_timer = threading.Timer(0.01, controls.stop)

        started = time.perf_counter()
        stop_timer.start()
        try:
            results = Engine(
                client=client,
                jobs=[JobSpec("looping-model", "quickbench", "2")],
                controls=controls,
            ).run()
        finally:
            stop_timer.cancel()

        self.assertLess(time.perf_counter() - started, 0.25)
        self.assertEqual(client.calls, 1)
        self.assertGreater(client.tokens, 0)
        self.assertEqual(results, [])
        self.assertTrue(controls.stopped)

    def test_skip_kills_request_and_runs_the_next_job(self) -> None:
        controls = RunControls()
        client = SkipThenCompleteClient()
        skip_timer = threading.Timer(0.01, controls.skip_job)

        skip_timer.start()
        try:
            results = Engine(
                client=client,
                jobs=[
                    JobSpec("looping-model", "quickbench", "1"),
                    JobSpec("looping-model", "quickbench", "1"),
                ],
                controls=controls,
            ).run()
        finally:
            skip_timer.cancel()

        self.assertEqual(client.calls, 2)
        self.assertTrue(client.cancel_was_clear_for_next_job)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["passed"], 1)

    def test_sustained_doom_loop_score_kills_the_task(self) -> None:
        with patch("benchkit.engine.time.monotonic", side_effect=[0.0, 2.0]):
            results = Engine(
                client=SustainedLoopClient(),
                jobs=[JobSpec("doom-model", "quickbench", "1")],
                loop_kill_percent=80,
                loop_kill_seconds=1,
            ).run()

        result = results[0]
        task = result["tasks"][0]
        self.assertEqual(result["loop_kills"], 1)
        self.assertEqual(result["errors"], 1)
        self.assertEqual(result["passed"], 0)
        self.assertTrue(task["loop_killed"])
        self.assertEqual(task["done_reason"], "loop_killed")
        self.assertGreaterEqual(task["loop_kill_score"], 0.8)
        self.assertIn("at or above 80% for 1s", task["error"])

    def test_loop_kill_timer_resets_below_threshold(self) -> None:
        with patch("benchkit.engine.time.monotonic", side_effect=[0.0, 1.0, 2.0]):
            results = Engine(
                client=ResettingLoopClient(),
                jobs=[JobSpec("recovering-model", "quickbench", "1")],
                loop_kill_percent=80,
                loop_kill_seconds=1.5,
            ).run()

        self.assertEqual(results[0]["loop_kills"], 0)
        self.assertEqual(results[0]["passed"], 1)

    def test_loop_kill_settings_are_read_from_environment(self) -> None:
        with patch.dict(
            os.environ,
            {
                "BENCHKIT_LOOP_KILL_PERCENT": "87.5",
                "BENCHKIT_LOOP_KILL_SECONDS": "6",
            },
        ):
            engine = Engine(client=LoopingClient(), jobs=[])

        self.assertEqual(engine.loop_kill_percent, 87.5)
        self.assertEqual(engine.loop_kill_seconds, 6)


class LoopingClient:
    def generate(
        self, model: str, prompt: str, on_progress=None, cancel_event=None
    ) -> dict:
        thinking = (
            "I should inspect the same evidence and reconsider the same path "
            "before making the same decision again "
        ) * 24
        response = "return len(set(string.lower()))"
        if on_progress is not None:
            on_progress(
                GenerationUpdate(
                    thinking=thinking,
                    elapsed_s=2.0,
                    reasoning_channel_seen=True,
                )
            )
            on_progress(
                GenerationUpdate(
                    response=response,
                    elapsed_s=2.1,
                    reasoning_channel_seen=True,
                )
            )
            on_progress(
                GenerationUpdate(
                    elapsed_s=2.1,
                    reasoning_channel_seen=True,
                    done=True,
                )
            )
        return {
            "thinking": thinking,
            "response": response,
            "trace_status": "observed",
            "tok_s": 100.0,
            "eval_count": 300,
            "eval_duration_ns": 3_000_000_000,
            "response_time_s": 3.0,
            "done_reason": "stop",
        }


class TimeoutClient:
    timeout = 9.0

    def __init__(self) -> None:
        self.calls = 0

    def generate(
        self, model: str, prompt: str, on_progress=None, cancel_event=None
    ) -> dict:
        self.calls += 1
        return {
            "thinking": "partial reasoning",
            # This is deliberately a valid first QuickBench solution. A zero
            # score proves the evaluator was skipped after the timeout.
            "response": "return len(set(string.lower()))",
            "trace_status": "observed",
            "tok_s": 0.0,
            "eval_count": 8,
            "eval_duration_ns": 0,
            "response_time_s": self.timeout,
            "done_reason": "timeout",
            "timed_out": True,
        }


class TokenLoopClient:
    timeout = 30.0

    def __init__(self) -> None:
        self.calls = 0
        self.tokens = 0

    def generate(
        self, model: str, prompt: str, on_progress=None, cancel_event=None
    ) -> dict:
        self.calls += 1
        while True:
            time.sleep(0.001)
            self.tokens += 1
            on_progress(
                GenerationUpdate(
                    thinking="loop token ",
                    elapsed_s=self.tokens / 1000,
                    reasoning_channel_seen=True,
                )
            )


class SkipThenCompleteClient:
    timeout = 30.0

    def __init__(self) -> None:
        self.calls = 0
        self.cancel_was_clear_for_next_job = False

    def generate(
        self, model: str, prompt: str, on_progress=None, cancel_event=None
    ) -> dict:
        self.calls += 1
        assert cancel_event is not None
        if self.calls == 1:
            cancel_event.wait(1)
            return {
                "thinking": "partial loop",
                "response": "",
                "trace_status": "observed",
                "tok_s": 0.0,
                "eval_count": 0,
                "eval_duration_ns": 0,
                "response_time_s": 0.01,
                "done_reason": "cancelled",
                "timed_out": False,
                "cancelled": True,
            }

        self.cancel_was_clear_for_next_job = not cancel_event.is_set()
        return {
            "thinking": "brief check",
            "response": "return len(set(string.lower()))",
            "trace_status": "observed",
            "tok_s": 50.0,
            "eval_count": 10,
            "eval_duration_ns": 200_000_000,
            "response_time_s": 0.2,
            "done_reason": "stop",
            "timed_out": False,
            "cancelled": False,
        }


class SustainedLoopClient:
    timeout = 30.0

    def generate(
        self, model: str, prompt: str, on_progress=None, cancel_event=None
    ) -> dict:
        repeated = (
            "I must reconsider the same evidence and follow the same path again " * 500
        )
        on_progress(
            GenerationUpdate(
                thinking=repeated,
                elapsed_s=1.0,
                reasoning_channel_seen=True,
            )
        )
        on_progress(
            GenerationUpdate(
                thinking="same evidence same path again " * 50,
                elapsed_s=3.0,
                reasoning_channel_seen=True,
            )
        )
        raise AssertionError("sustained loop should have killed the request")


class ResettingLoopClient:
    timeout = 30.0

    def generate(
        self, model: str, prompt: str, on_progress=None, cancel_event=None
    ) -> dict:
        on_progress(
            GenerationUpdate(
                thinking="same evidence same path again " * 1000,
                elapsed_s=1.0,
                reasoning_channel_seen=True,
            )
        )
        on_progress(
            GenerationUpdate(
                thinking=" ".join(f"novel{index}" for index in range(5000)),
                elapsed_s=2.0,
                reasoning_channel_seen=True,
            )
        )
        on_progress(
            GenerationUpdate(
                thinking="same evidence same path again " * 1200,
                elapsed_s=3.0,
                reasoning_channel_seen=True,
            )
        )
        return {
            "thinking": "brief recovered reasoning",
            "response": "return len(set(string.lower()))",
            "trace_status": "observed",
            "tok_s": 50.0,
            "eval_count": 10,
            "eval_duration_ns": 200_000_000,
            "response_time_s": 0.2,
            "done_reason": "stop",
            "timed_out": False,
            "cancelled": False,
        }


def _sse(data: dict) -> str:
    return "data: " + json.dumps(data)


if __name__ == "__main__":
    unittest.main()
