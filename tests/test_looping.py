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

    def test_loop_drifting_by_one_token_still_reaches_looping(self) -> None:
        analyzer = LoopAnalyzer()
        analyzer.add(
            answer=" ".join(
                f"Step {index}: I need to reconsider the problem and check my work."
                for index in range(1, 101)
            )
        )

        snapshot = analyzer.snapshot(final=True)

        # The literal cycle changes at the counter, so normalized suffix
        # evidence and high literal coverage must agree.
        self.assertEqual(snapshot.state, "looping")
        self.assertGreater(snapshot.repeated_ngram_coverage, 0.8)
        self.assertEqual(snapshot.max_repeated_block, 1)
        self.assertGreaterEqual(snapshot.score, 0.8)
        self.assertEqual(snapshot.evidence, "numeric_cycle")

    def test_worst_channel_wins_when_both_share_a_state(self) -> None:
        analyzer = LoopAnalyzer()
        analyzer.add(
            thinking=(
                " ".join(f"distinct{index}" for index in range(200))
                + " "
                + " ".join(f"distinct{index}" for index in range(20))
            ),
            answer=("the answer is the answer is the answer is " * 20),
        )

        snapshot = analyzer.snapshot()

        # Both channels land in "suspected"; the reported score drives the kill
        # decision, so the looping answer must not hide behind calmer thinking.
        self.assertEqual(snapshot.source, "answer")
        self.assertGreaterEqual(snapshot.score, 0.8)

    def test_structured_answer_without_repetition_stays_clear(self) -> None:
        analyzer = LoopAnalyzer()
        analyzer.add(
            answer="\n".join(
                f"| model{index} | {index * 3} | {index * 7} | pass |"
                for index in range(60)
            )
        )

        self.assertEqual(analyzer.snapshot(final=True).state, "clear")

    def test_two_finite_copies_are_not_called_a_loop(self) -> None:
        paragraph = " ".join(f"unique{index}" for index in range(100))
        analyzer = LoopAnalyzer()
        analyzer.add(answer=f"{paragraph} {paragraph}")

        snapshot = analyzer.snapshot(final=True)

        self.assertEqual(snapshot.state, "suspected")
        self.assertFalse(snapshot.active_cycle)
        self.assertLess(snapshot.score, 0.8)

    def test_punctuation_only_loop_is_detected(self) -> None:
        analyzer = LoopAnalyzer()
        analyzer.add(answer="#" * 100)

        snapshot = analyzer.snapshot(final=True)

        self.assertEqual(snapshot.state, "looping")
        self.assertEqual(snapshot.evidence, "exact_cycle")
        self.assertEqual(snapshot.cycle_period_tokens, 1)
        self.assertEqual(snapshot.analyzed_words, 0)

    def test_live_cycle_needs_more_generated_evidence_to_confirm(self) -> None:
        cycle = "same evidence leads down the same broken path again "
        analyzer = LoopAnalyzer()
        analyzer.add(thinking=cycle * 12)

        first = analyzer.snapshot()
        analyzer.add(thinking=cycle * 12)
        second = analyzer.snapshot()

        self.assertEqual(first.state, "suspected")
        self.assertTrue(first.active_cycle)
        self.assertFalse(first.confirmed_cycle)
        self.assertEqual(second.state, "looping")
        self.assertTrue(second.confirmed_cycle)
        self.assertGreaterEqual(second.evidence_growth_tokens, 32)

    def test_live_cycle_confirmation_survives_chunk_rotation(self) -> None:
        cycle = "same evidence leads down the same broken path again "
        analyzer = LoopAnalyzer()
        analyzer.add(thinking=cycle * 12 + "same evidence ")

        first = analyzer.snapshot()
        analyzer.add(
            thinking="leads down the same broken path again " + cycle * 12,
        )
        second = analyzer.snapshot()

        self.assertEqual(first.cycle_period_tokens, second.cycle_period_tokens)
        self.assertTrue(second.confirmed_cycle)
        self.assertEqual(second.state, "looping")

    def test_answer_phase_cannot_be_killed_by_historical_thinking(self) -> None:
        cycle = "same evidence leads down the same broken path again "
        analyzer = LoopAnalyzer()
        analyzer.add(thinking=cycle * 12)
        analyzer.snapshot()
        analyzer.add(thinking=cycle * 12)
        self.assertTrue(analyzer.snapshot().confirmed_cycle)

        analyzer.add(answer="The final answer is B.")
        recovered = analyzer.snapshot()

        self.assertEqual(recovered.source, "answer")
        self.assertFalse(recovered.active_cycle)
        self.assertNotEqual(recovered.state, "looping")

        final = analyzer.snapshot(final=True)
        self.assertEqual(final.source, "thinking")
        self.assertEqual(final.state, "suspected")
        self.assertTrue(final.recovered_cycle)
        self.assertFalse(final.active_cycle)

    def test_visible_answer_loop_wins_over_a_thinking_loop(self) -> None:
        analyzer = LoopAnalyzer()
        analyzer.add(
            thinking="thinking repeats the same longer argument forever " * 30,
            answer="answer repeats the same conclusion forever " * 20,
        )

        snapshot = analyzer.snapshot(final=True)

        self.assertEqual(snapshot.source, "answer")
        self.assertEqual(snapshot.state, "looping")
        self.assertTrue(snapshot.active_cycle)
        self.assertFalse(snapshot.recovered_cycle)

    def test_confirmed_cycle_that_breaks_is_reported_as_recovered(self) -> None:
        cycle = "same evidence leads down the same broken path again "
        analyzer = LoopAnalyzer()
        analyzer.add(thinking=cycle * 12)
        analyzer.snapshot()
        analyzer.add(thinking=cycle * 12)
        self.assertTrue(analyzer.snapshot().confirmed_cycle)

        analyzer.add(thinking=" ".join(f"novel{index}" for index in range(300)))
        recovered_live = analyzer.snapshot()
        final = analyzer.snapshot(final=True)

        self.assertFalse(recovered_live.active_cycle)
        self.assertNotEqual(recovered_live.state, "looping")
        self.assertEqual(final.state, "suspected")
        self.assertTrue(final.recovered_cycle)
        self.assertFalse(final.active_cycle)

    def test_final_payload_reconciliation_preserves_live_cycle_history(self) -> None:
        cycle = "same evidence leads down the same broken path again "
        streamed = cycle * 24
        analyzer = LoopAnalyzer()
        analyzer.add(thinking=streamed[: len(streamed) // 2])
        analyzer.snapshot()
        analyzer.add(thinking=streamed[len(streamed) // 2 :])
        self.assertTrue(analyzer.snapshot().confirmed_cycle)

        final_thinking = streamed + " A new argument resolves the issue."
        self.assertTrue(
            analyzer.reconcile(
                thinking=final_thinking,
                answer="The final answer is B.",
            )
        )
        final = analyzer.snapshot(final=True)

        self.assertEqual(analyzer.thinking_chars, len(final_thinking))
        self.assertTrue(final.recovered_cycle)
        self.assertEqual(final.state, "suspected")

    def test_mismatched_final_payload_is_not_reconciled(self) -> None:
        analyzer = LoopAnalyzer()
        analyzer.add(thinking="streamed reasoning")

        self.assertFalse(
            analyzer.reconcile(
                thinking="different final reasoning",
                answer="answer",
            )
        )
        self.assertEqual(analyzer.thinking, "streamed reasoning")
        self.assertEqual(analyzer.answer, "")

    def test_long_exact_cycle_is_detected_beyond_the_old_period_limit(self) -> None:
        cycle = " ".join(f"unique{index}" for index in range(180)) + " "
        analyzer = LoopAnalyzer()
        analyzer.add(answer=cycle * 4)

        snapshot = analyzer.snapshot(final=True)

        self.assertEqual(snapshot.state, "looping")
        self.assertTrue(snapshot.active_cycle)
        self.assertEqual(snapshot.cycle_period_tokens, 180)
        self.assertEqual(snapshot.cycle_repetitions, 4)

    def test_finite_repeated_code_structure_is_advisory_only(self) -> None:
        block = """
for i in range(n):
    for j in range(n):
        ni, nj = i + {di}, j + {dj}
        while 0 <= ni < n and 0 <= nj < n:
            count += grid[ni][nj]
            ni += {di}
            nj += {dj}
"""
        answer = "\n".join(
            block.format(di=di, dj=dj) for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1))
        )
        analyzer = LoopAnalyzer()
        analyzer.add(answer=answer)

        snapshot = analyzer.snapshot(final=True)

        self.assertNotEqual(snapshot.state, "looping")
        self.assertFalse(snapshot.active_cycle)
        self.assertLess(snapshot.score, 0.8)

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

        with (
            patch("benchkit.client.httpx.stream", return_value=stream),
            self.assertRaisesRegex(RuntimeError, "kill generation"),
        ):
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

        with patch(
            "benchkit.engine.time.monotonic",
            side_effect=[0.0, 1.0, 2.0, 3.0],
        ):
            results = Engine(
                client=client,
                jobs=[JobSpec("looping-model", "quickbench", "1")],
                sink=events.append,
            ).run()

        live = [event for event in events if isinstance(event, GenerationProgress)]
        self.assertTrue(any(event.loop_state == "looping" for event in live))
        self.assertEqual(results[0]["loops"], 0)
        self.assertEqual(results[0]["suspected_loops"], 0)
        self.assertEqual(results[0]["recovered_loops"], 1)
        self.assertEqual(results[0]["loop_rate"], 0.0)
        self.assertEqual(results[0]["trace_coverage"], 100.0)
        task = results[0]["tasks"][0]
        self.assertEqual(task["loop_state"], "suspected")
        self.assertTrue(task["recovered_cycle"])
        self.assertFalse(task["active_cycle"])

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
        self.assertTrue(all(task["tokens_recovered"] for task in result["tasks"]))
        self.assertTrue(all(task["output_tokens"] > 0 for task in result["tasks"]))
        self.assertEqual(result["throughput_coverage"], 0.0)

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
        events: list[object] = []
        with patch("benchkit.engine.time.monotonic", side_effect=[0.0, 1.0, 3.0]):
            results = Engine(
                client=SustainedLoopClient(),
                jobs=[JobSpec("doom-model", "quickbench", "1")],
                sink=events.append,
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
        self.assertIn("confirmed active cycle", task["error"])
        self.assertIn("at or above 80% for 1s", task["error"])
        self.assertTrue(task["tokens_recovered"])
        self.assertGreater(task["output_tokens"], 0)
        confirmed = [
            event
            for event in events
            if isinstance(event, GenerationProgress) and event.loop_state == "looping"
        ]
        self.assertEqual(confirmed[-1].loop_kill_remaining_s, 1.0)

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

    def test_completed_stream_is_not_killed_on_its_done_update(self) -> None:
        with patch("benchkit.engine.time.monotonic", side_effect=[0.0, 2.0]):
            results = Engine(
                client=CompletedLoopClient(),
                jobs=[JobSpec("recovering-model", "quickbench", "1")],
                loop_kill_percent=80,
                loop_kill_seconds=1,
            ).run()

        self.assertEqual(results[0]["loop_kills"], 0)
        self.assertEqual(results[0]["passed"], 1)

    def test_loop_kill_settings_are_read_from_environment(self) -> None:
        with patch.dict(
            os.environ,
            {
                "BENCHKIT_LOOP_KILL": "false",
                "BENCHKIT_LOOP_KILL_PERCENT": "87.5",
                "BENCHKIT_LOOP_KILL_SECONDS": "6",
            },
        ):
            engine = Engine(client=LoopingClient(), jobs=[])

        self.assertFalse(engine.loop_kill_enabled)
        self.assertEqual(engine.loop_kill_percent, 87.5)
        self.assertEqual(engine.loop_kill_seconds, 6)

    def test_loop_kill_is_enabled_by_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            engine = Engine(client=LoopingClient(), jobs=[])

        self.assertTrue(engine.loop_kill_enabled)

    def test_disabled_loop_kill_lets_the_generation_finish(self) -> None:
        results = Engine(
            client=SurvivingLoopClient(),
            jobs=[JobSpec("doom-model", "quickbench", "1")],
            loop_kill_enabled=False,
            loop_kill_percent=80,
            loop_kill_seconds=1,
        ).run()

        result = results[0]
        task = result["tasks"][0]
        self.assertEqual(result["loop_kills"], 0)
        self.assertEqual(result["passed"], 1)
        self.assertFalse(task["loop_killed"])
        self.assertFalse(result["loop_kill_enabled"])


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
                    thinking=thinking[: len(thinking) // 2],
                    elapsed_s=1.0,
                    reasoning_channel_seen=True,
                )
            )
            on_progress(
                GenerationUpdate(
                    thinking=thinking[len(thinking) // 2 :],
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
        if on_progress is not None:
            on_progress(
                GenerationUpdate(
                    thinking="partial reasoning",
                    response="return len(set(string.lower()))",
                    elapsed_s=self.timeout,
                    reasoning_channel_seen=True,
                )
            )
        return {
            "thinking": "partial reasoning",
            # This is deliberately a valid first QuickBench solution. A zero
            # score proves the evaluator was skipped after the timeout.
            "response": "return len(set(string.lower()))",
            "trace_status": "observed",
            "tok_s": 0.0,
            "eval_count": 0,
            "eval_duration_ns": 0,
            "response_time_s": self.timeout,
            "done_reason": "timeout",
            "timed_out": True,
        }

    def tokenize(
        self, model: str, content: str, *, add_special: bool = True
    ) -> list[int]:
        return list(range(len(content.split()) + int(add_special)))


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
        cycle = "I must reconsider the same evidence and follow the same path again "
        repeated = cycle * 500
        on_progress(
            GenerationUpdate(
                thinking=repeated,
                elapsed_s=1.0,
                reasoning_channel_seen=True,
            )
        )
        on_progress(
            GenerationUpdate(
                thinking=cycle * 50,
                elapsed_s=3.0,
                reasoning_channel_seen=True,
            )
        )
        on_progress(
            GenerationUpdate(
                thinking=cycle * 50,
                elapsed_s=5.0,
                reasoning_channel_seen=True,
            )
        )
        raise AssertionError("sustained loop should have killed the request")

    def tokenize(
        self, model: str, content: str, *, add_special: bool = True
    ) -> list[int]:
        return list(range(len(content.split()) + int(add_special)))


class SurvivingLoopClient:
    """Loops hard, then finishes — used when loop killing is turned off."""

    timeout = 30.0

    def generate(
        self, model: str, prompt: str, on_progress=None, cancel_event=None
    ) -> dict:
        repeated = (
            "I must reconsider the same evidence and follow the same path again " * 500
        )
        for elapsed in (1.0, 3.0, 5.0):
            on_progress(
                GenerationUpdate(
                    thinking=repeated,
                    elapsed_s=elapsed,
                    reasoning_channel_seen=True,
                )
            )
        return {
            "thinking": repeated,
            "response": "return len(set(string.lower()))",
            "trace_status": "observed",
            "tok_s": 50.0,
            "eval_count": 10,
            "eval_duration_ns": 200_000_000,
            "response_time_s": 5.0,
            "done_reason": "stop",
            "timed_out": False,
            "cancelled": False,
        }


class CompletedLoopClient:
    """Ends on the update that confirms a loop, so there is nothing to kill."""

    timeout = 30.0

    def generate(
        self, model: str, prompt: str, on_progress=None, cancel_event=None
    ) -> dict:
        cycle = "I keep reconsidering the same evidence along the same path "
        on_progress(
            GenerationUpdate(
                thinking=cycle * 20,
                elapsed_s=1.0,
                reasoning_channel_seen=True,
            )
        )
        on_progress(
            GenerationUpdate(
                thinking=cycle * 20,
                elapsed_s=3.0,
                reasoning_channel_seen=True,
                done=True,
            )
        )
        return {
            "thinking": cycle * 40,
            "response": "return len(set(string.lower()))",
            "trace_status": "observed",
            "tok_s": 50.0,
            "eval_count": 10,
            "eval_duration_ns": 200_000_000,
            "response_time_s": 3.0,
            "done_reason": "stop",
            "timed_out": False,
            "cancelled": False,
        }


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
