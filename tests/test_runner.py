"""Tests for the headless live counters and dashboard rendering."""

from __future__ import annotations

import io
import unittest

from rich.console import Console

from benchkit.cli import _outcome_counts
from benchkit.engine import JobSpec, TaskCompleted, TaskRecord
from benchkit.runner import _bar, _counters, _Glyphs, _LiveStats, _Reporter, _spread


def _record(*, passed: bool = False, error: str = "", loop: bool = False) -> TaskRecord:
    return TaskRecord(
        index=0,
        task_id="task",
        passed=passed,
        tok_s=0.0,
        response_time_s=0.0,
        prompt="prompt",
        response="response",
        error=error,
        loop_state="looping" if loop else "clear",
    )


def _console(**kwargs: object) -> Console:
    kwargs.setdefault("force_terminal", False)
    return Console(width=100, **kwargs)


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

        job = JobSpec("model", "quickbench", None)
        with reporter:
            reporter(TaskCompleted(0, job, _record(passed=True), 1, 1))
            reporter(TaskCompleted(0, job, _record(error="boom"), 1, 2))

        output = console.export_text()
        self.assertIn("PASS", output)
        self.assertIn("ERROR", output)
        self.assertIn("boom", output)
        self.assertEqual(reporter.state.overall_done, 2)


if __name__ == "__main__":
    unittest.main()
