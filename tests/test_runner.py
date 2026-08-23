"""Tests for the headless live counters and dashboard rendering."""

from __future__ import annotations

import io
import signal
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rich.console import Console

from benchkit.cli import _outcome_counts
from benchkit.engine import (
    GenerationProgress,
    JobCompleted,
    JobSpec,
    JobStarted,
    RunStarted,
    TaskCompleted,
    TaskPhase,
    TaskRecord,
)
from benchkit.runner import (
    _bar,
    _counters,
    _Glyphs,
    _LiveStats,
    _Reporter,
    _spread,
    run,
)


def _record(
    *,
    passed: bool = False,
    error: str = "",
    loop: bool = False,
    response: str = "response",
    score: float = 0.0,
) -> TaskRecord:
    return TaskRecord(
        index=0,
        task_id="task",
        passed=passed,
        tok_s=0.0,
        response_time_s=0.0,
        prompt="prompt",
        response=response,
        score=score,
        error=error,
        loop_state="looping" if loop else "clear",
    )


def _console(**kwargs: object) -> Console:
    kwargs.setdefault("force_terminal", False)
    kwargs.setdefault("width", 100)
    return Console(**kwargs)


def _ascii_console() -> Console:
    """A console whose output stream cannot carry the block-drawing glyphs."""
    stream = io.TextIOWrapper(io.BytesIO(), encoding="ascii")
    return Console(width=100, force_terminal=False, file=stream)


class LiveHeadlessStatsTests(unittest.TestCase):
    def test_outcomes_are_counted_separately_from_loops(self) -> None:
        stats = _LiveStats()
        stats.add(_record(passed=True))
        stats.add(_record())
        stats.add(_record(error="timed out", loop=True))

        self.assertEqual(
            stats.fields(),
            {"passed": 1, "failed": 1, "errors": 1, "loops": 1},
        )
        self.assertEqual(stats.fields(live_looping=True)["loops"], 2)
        self.assertEqual(stats.fields()["loops"], 1)
        self.assertEqual(stats.completed, 3)
        self.assertAlmostEqual(stats.score or 0.0, 100 / 3)

    def test_score_is_unknown_before_the_first_task(self) -> None:
        self.assertIsNone(_LiveStats().score)

    def test_live_score_includes_partial_credit(self) -> None:
        stats = _LiveStats()
        stats.add(_record(score=0.6))
        stats.add(_record(passed=True))

        self.assertEqual(stats.score, 80.0)

    def test_counters_render_all_live_values(self) -> None:
        stats = _LiveStats(passed=1, failed=1, errors=1, loops=1)

        self.assertEqual(str(_counters(stats)), "PASS 1  FAIL 1  ERR 1  LOOP 1")
        self.assertEqual(
            str(_counters(stats, live_looping=True)),
            "PASS 1  FAIL 1  ERR 1  LOOP 2",
        )

    def test_result_table_uses_the_same_outcome_split(self) -> None:
        result = {
            "passed": 1,
            "tasks": [
                {"passed": True, "error": ""},
                {"passed": False, "error": ""},
                {"passed": False, "error": "generation timed out"},
            ],
        }

        self.assertEqual(_outcome_counts(result), (1, 1, 1))


class DrawingTests(unittest.TestCase):
    def test_bar_fills_proportionally(self) -> None:
        glyphs = _Glyphs.for_console(_ascii_console())

        self.assertEqual(str(_bar(0.0, 10, glyphs)), "." * 10)
        self.assertEqual(str(_bar(1.0, 10, glyphs)), "#" * 10)
        self.assertEqual(str(_bar(0.5, 10, glyphs)), "#" * 5 + "." * 5)
        self.assertEqual(len(str(_bar(0.37, 20, glyphs))), 20)

    def test_unicode_bar_keeps_its_width_with_partial_blocks(self) -> None:
        glyphs = _Glyphs.for_console(_console())

        for fraction in (0.0, 0.13, 0.5, 0.99, 1.0):
            self.assertEqual(len(str(_bar(fraction, 24, glyphs))), 24)

    def test_glyphs_fall_back_to_ascii_without_utf8(self) -> None:
        self.assertEqual(_Glyphs.for_console(_ascii_console()).passed, "+")

    def test_spread_justifies_and_truncates(self) -> None:
        from rich.text import Text

        line = _spread(Text("left"), Text("right"), 20)
        self.assertEqual(str(line), "left" + " " * 11 + "right")

        crowded = _spread(Text("a very long left side"), Text("right"), 20)
        self.assertEqual(len(str(crowded)), 20)
        self.assertTrue(str(crowded).endswith("right"))


class PlainReporterTests(unittest.TestCase):
    """A non-terminal console must log lines instead of drawing a dashboard."""

    def test_task_lines_are_logged_with_outcome_words(self) -> None:
        console = _console(record=True)
        reporter = _Reporter(console, verbose=False)
        self.assertTrue(reporter.plain)

        job = JobSpec("model", "sanity", None)
        with reporter:
            reporter(TaskCompleted(0, job, _record(passed=True), 1, 1))
            reporter(TaskCompleted(0, job, _record(error="boom"), 1, 2))

        output = console.export_text()
        self.assertIn("PASS", output)
        self.assertIn("ERROR", output)
        self.assertIn("boom", output)
        self.assertEqual(reporter.state.overall_done, 2)

    def test_concurrent_activity_aggregates_streams_without_task_name_flicker(
        self,
    ) -> None:
        console = _console(record=True)
        reporter = _Reporter(console, verbose=False)
        job = JobSpec("model", "sanity", None)
        reporter(RunStarted([job], 2))
        reporter(JobStarted(0, job, 2, 2, concurrency=2))

        for position, task_id in ((1, "question-a"), (2, "question-b")):
            reporter(
                TaskPhase(
                    0,
                    job,
                    position,
                    2,
                    task_id,
                    "",
                    "generating",
                    "waiting for model",
                )
            )
        for position, task_id, chars in (
            (1, "question-a", 100),
            (2, "question-b", 20),
        ):
            reporter(
                GenerationProgress(
                    0,
                    job,
                    position,
                    2,
                    task_id,
                    "",
                    "answering",
                    1.0,
                    0,
                    chars,
                    "unavailable",
                    "clear",
                    0.0,
                    "none",
                    None,
                    "prompt",
                    "",
                    "x" * chars,
                )
            )

        view = reporter.state.current
        assert view is not None
        activity = str(view.activity)
        self.assertIn("2 active requests", activity)
        self.assertIn("120 chars generated", activity)
        self.assertNotIn("question-a", activity)
        self.assertNotIn("question-b", activity)

        reporter(TaskCompleted(0, job, _record(response="x" * 100), 1, 1))
        self.assertEqual(view.generated_chars, 120)
        self.assertIn("1 active request", str(view.activity))
        self.assertIn("120 chars generated", str(view.activity))

    def test_serial_job_summary_only_shows_stream_speed_as_tok_s(self) -> None:
        console = _console(record=True, width=200)
        reporter = _Reporter(console, verbose=False)
        job = JobSpec("model", "sanity", None)
        result = {
            "model": "model",
            "benchmark": "sanity",
            "score": 50.0,
            "passed": 1,
            "total": 2,
            "tok_s_per_stream": 62.6,
            "tok_s_aggregate": 58.4,
            "concurrency_eff": 1.0,
            "concurrency": 1,
            "loop_kills": 0,
            "total_time": 2.0,
        }

        reporter(JobCompleted(0, job, result, skipped=False))

        output = console.export_text()
        self.assertIn("62.6 tok/s", output)
        self.assertNotIn("aggregate tok/s", output)
        self.assertNotIn("stream tok/s", output)
        self.assertNotIn("effective", output)

    def test_parallel_job_summary_keeps_parallel_throughput_metrics(self) -> None:
        console = _console(record=True, width=200)
        reporter = _Reporter(console, verbose=False)
        job = JobSpec("model", "sanity", None)
        result = {
            "model": "model",
            "benchmark": "sanity",
            "score": 50.0,
            "passed": 1,
            "total": 2,
            "tok_s_per_stream": 62.6,
            "tok_s_aggregate": 116.8,
            "concurrency_eff": 1.87,
            "concurrency": 2,
            "loop_kills": 0,
            "total_time": 1.0,
        }

        reporter(JobCompleted(0, job, result, skipped=False))

        output = console.export_text()
        self.assertIn("116.8 aggregate tok/s", output)
        self.assertIn("62.6 stream tok/s", output)
        self.assertIn("1.87x effective", output)


class HeadlessSignalTests(unittest.TestCase):
    def test_sigterm_unwinds_engine_cleanup_and_restores_handlers(self) -> None:
        console = _console(record=True)
        installed: dict[int, object] = {}
        original = object()
        engine = Mock()

        def install(signum: int, handler: object) -> object:
            installed[signum] = handler
            return original

        def interrupt_run() -> list[dict]:
            handler = installed[signal.SIGTERM]
            assert callable(handler)
            handler(signal.SIGTERM, None)
            return []

        engine.run.side_effect = interrupt_run
        with (
            patch("benchkit.runner.Engine", return_value=engine),
            patch("benchkit.runner.signal.signal", side_effect=install) as set_signal,
            self.assertRaisesRegex(SystemExit, str(128 + signal.SIGTERM)),
        ):
            run(
                SimpleNamespace(),
                [JobSpec("model", "sanity", "1")],
                console,
            )

        self.assertIn(
            (signal.SIGTERM, signal.SIG_IGN),
            [call.args for call in set_signal.call_args_list],
        )
        restored = {
            call.args[0]: call.args[1]
            for call in set_signal.call_args_list
            if call.args[1] is original
        }
        self.assertEqual(restored[signal.SIGINT], original)
        self.assertEqual(restored[signal.SIGTERM], original)


if __name__ == "__main__":
    unittest.main()
