"""Tests for the live headless score and outcome counters."""

from __future__ import annotations

import time
import unittest

from rich.progress import Task

from benchkit.cli import _outcome_counts
from benchkit.engine import TaskRecord
from benchkit.runner import _LiveScoreColumn, _LiveStats


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

    def test_score_column_renders_all_live_values(self) -> None:
        task = Task(
            id=0,
            description="quickbench",
            total=10,
            completed=3,
            _get_time=time.monotonic,
            fields={"passed": 1, "failed": 1, "errors": 1, "loops": 2},
        )

        rendered = _LiveScoreColumn().render(task)

        self.assertEqual(str(rendered), " 33.3% PASS 1 FAIL 1 ERR 1 LOOP 2")

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


if __name__ == "__main__":
    unittest.main()
