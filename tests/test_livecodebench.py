"""Tests for the LiveCodeBench window, dataset preparation and scoring."""

from __future__ import annotations

import base64
import json
import os
import pickle
import tempfile
import unittest
import zlib
from datetime import date
from pathlib import Path

from benchkit.benchmarks.livecodebench import (
    DatasetUnavailable,
    LiveCodeBench,
    _decode_tests,
    extract_code,
    prepare,
    window,
)
from benchkit.engine import TaskRecord, group_breakdown

FUNCTIONAL = {
    "question_id": "func-1",
    "question_title": "Add two numbers",
    "platform": "leetcode",
    "contest_date": "2025-01-15T00:00:00",
    "difficulty": "easy",
    "question_content": "Return the sum of a and b.",
    "starter_code": "class Solution:\n    def add(self, a: int, b: int) -> int:\n        ",
    "metadata": json.dumps({"func_name": "add"}),
    "public_test_cases": json.dumps(
        [{"input": "1\n2", "output": "3", "testtype": "functional"}]
    ),
}

STDIN = {
    "question_id": "stdin-1",
    "question_title": "Echo doubled",
    "platform": "codeforces",
    "contest_date": "2024-03-02T00:00:00",
    "difficulty": "hard",
    "question_content": "Read one integer and print twice its value.",
    "starter_code": "",
    "metadata": "{}",
    "public_test_cases": json.dumps(
        [{"input": "4\n", "output": "8\n", "testtype": "stdin"}]
    ),
}


def _pickled(cases: list[dict]) -> str:
    """Encode test cases the way the released dataset does."""
    return base64.b64encode(zlib.compress(pickle.dumps(json.dumps(cases)))).decode()


class WindowTests(unittest.TestCase):
    def setUp(self) -> None:
        for name in ("BENCHKIT_LCB_AFTER", "BENCHKIT_LCB_BEFORE"):
            os.environ.pop(name, None)

    tearDown = setUp

    def test_default_window_has_a_lower_bound(self) -> None:
        active = window()
        self.assertIsNotNone(active.after)
        self.assertIsNone(active.before)

    def test_month_bounds_are_inclusive_at_both_ends(self) -> None:
        os.environ["BENCHKIT_LCB_AFTER"] = "2025-02"
        os.environ["BENCHKIT_LCB_BEFORE"] = "2025-03"
        active = window()

        self.assertFalse(active.contains(date(2025, 1, 31)))
        self.assertTrue(active.contains(date(2025, 2, 1)))
        self.assertTrue(active.contains(date(2025, 3, 31)))
        self.assertFalse(active.contains(date(2025, 4, 1)))
        self.assertEqual(active.label, "2025-02-01 to 2025-03-31")

    def test_all_removes_the_lower_bound(self) -> None:
        os.environ["BENCHKIT_LCB_AFTER"] = "all"
        self.assertIsNone(window().after)

    def test_inverted_window_is_rejected(self) -> None:
        os.environ["BENCHKIT_LCB_AFTER"] = "2025-06"
        os.environ["BENCHKIT_LCB_BEFORE"] = "2025-01"
        with self.assertRaises(ValueError):
            window()

    def test_unparseable_date_is_rejected(self) -> None:
        os.environ["BENCHKIT_LCB_AFTER"] = "June 2025"
        with self.assertRaises(ValueError):
            window()


class TestCaseDecodingTests(unittest.TestCase):
    def test_plain_json_cases_are_read_directly(self) -> None:
        cases = [{"input": "1", "output": "2", "testtype": "stdin"}]
        self.assertEqual(_decode_tests(json.dumps(cases)), cases)

    def test_compressed_pickle_payload_is_decoded(self) -> None:
        cases = [{"input": "1", "output": "2", "testtype": "stdin"}]
        self.assertEqual(_decode_tests(_pickled(cases)), cases)

    def test_pickled_globals_are_refused(self) -> None:
        payload = base64.b64encode(zlib.compress(pickle.dumps(range(3)))).decode()
        with self.assertRaises(pickle.UnpicklingError):
            _decode_tests(payload)


class ExtractionTests(unittest.TestCase):
    def test_last_fenced_block_wins(self) -> None:
        response = (
            "First attempt:\n```python\nprint(1)\n```\n"
            "Actually:\n```python\nprint(2)\n```"
        )
        self.assertEqual(extract_code(response), "print(2)")

    def test_reasoning_is_stripped_before_extraction(self) -> None:
        response = "<think>```python\nwrong()\n```</think>```python\nright()\n```"
        self.assertEqual(extract_code(response), "right()")

    def test_unterminated_fence_still_yields_code(self) -> None:
        self.assertEqual(extract_code("```python\nprint(3)\n"), "print(3)")

    def test_bare_code_is_returned_unchanged(self) -> None:
        self.assertEqual(extract_code("print(4)"), "print(4)")


class DatasetTests(unittest.TestCase):
    """Prepare a miniature release in a temporary cache and score against it."""

    def setUp(self) -> None:
        self._environ = dict(os.environ)
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)

        source = root / "source"
        source.mkdir()
        functional = dict(FUNCTIONAL)
        functional["private_test_cases"] = _pickled(
            [{"input": "10\n20", "output": "30", "testtype": "functional"}]
        )
        stdin = dict(STDIN)
        stdin["private_test_cases"] = _pickled(
            [{"input": "7\n", "output": "14\n", "testtype": "stdin"}]
        )
        with open(source / "test.jsonl", "w", encoding="utf-8") as handle:
            for row in (functional, stdin):
                handle.write(json.dumps(row) + "\n")

        os.environ["BENCHKIT_LCB_CACHE"] = str(root / "cache")
        os.environ["BENCHKIT_LCB_DATASET"] = str(source)
        os.environ["BENCHKIT_LCB_AFTER"] = "all"
        os.environ.pop("BENCHKIT_LCB_BEFORE", None)
        self.bench = LiveCodeBench()

    def tearDown(self) -> None:
        self._tmp.cleanup()
        os.environ.clear()
        os.environ.update(self._environ)

    def test_prepare_reports_the_cached_release(self) -> None:
        manifest = prepare()
        self.assertEqual(manifest["problems"], 2)
        self.assertEqual(manifest["first_contest_date"], "2024-03-02")
        self.assertEqual(manifest["last_contest_date"], "2025-01-15")

    def test_tasks_carry_difficulty_and_run_oldest_first(self) -> None:
        tasks = self.bench.load_tasks()
        self.assertEqual([task.id for task in tasks], ["stdin-1", "func-1"])
        self.assertEqual([task.metadata["group"] for task in tasks], ["hard", "easy"])
        self.assertEqual(self.bench.task_count, 2)

    def test_window_filters_by_contest_date(self) -> None:
        os.environ["BENCHKIT_LCB_AFTER"] = "2024-06"
        tasks = LiveCodeBench().load_tasks()
        self.assertEqual([task.id for task in tasks], ["func-1"])

    def test_empty_window_is_an_explicit_failure(self) -> None:
        os.environ["BENCHKIT_LCB_AFTER"] = "2026-01"
        with self.assertRaises(DatasetUnavailable):
            LiveCodeBench().load_tasks()

    def test_report_note_documents_the_window(self) -> None:
        os.environ["BENCHKIT_LCB_AFTER"] = "2025-01"
        note = LiveCodeBench().report_note
        self.assertIn("2025-01-01 onwards", note)

    def _task(self, task_id: str):
        return next(task for task in self.bench.load_tasks() if task.id == task_id)

    def test_correct_functional_solution_passes_hidden_tests(self) -> None:
        response = (
            "```python\nclass Solution:\n"
            "    def add(self, a, b):\n        return a + b\n```"
        )
        self.assertTrue(self.bench.evaluate(self._task("func-1"), response))

    def test_functional_solution_failing_a_hidden_test_fails(self) -> None:
        # Correct on the public case (1 + 2), wrong on the private one.
        response = (
            "```python\nclass Solution:\n"
            "    def add(self, a, b):\n        return 3\n```"
        )
        self.assertFalse(self.bench.evaluate(self._task("func-1"), response))

    def test_correct_stdin_solution_passes(self) -> None:
        response = "```python\nprint(int(input()) * 2)\n```"
        self.assertTrue(self.bench.evaluate(self._task("stdin-1"), response))

    def test_wrong_stdin_output_fails(self) -> None:
        response = "```python\nprint(int(input()))\n```"
        self.assertFalse(self.bench.evaluate(self._task("stdin-1"), response))

    def test_code_that_never_terminates_fails(self) -> None:
        response = "```python\nwhile True:\n    pass\n```"
        self.assertFalse(self.bench.evaluate(self._task("stdin-1"), response))

    def test_empty_response_fails(self) -> None:
        self.assertFalse(self.bench.evaluate(self._task("stdin-1"), ""))


class GroupBreakdownTests(unittest.TestCase):
    def _record(self, group: str, passed: bool) -> TaskRecord:
        return TaskRecord(
            index=0,
            task_id=group,
            passed=passed,
            tok_s=0.0,
            response_time_s=0.0,
            prompt="",
            response="",
            group=group,
        )

    def test_groups_follow_the_declared_order(self) -> None:
        records = [
            self._record("hard", False),
            self._record("easy", True),
            self._record("easy", False),
            self._record("medium", False),
        ]
        breakdown = group_breakdown(records, ("easy", "medium", "hard"))
        self.assertEqual(
            breakdown,
            [
                {"name": "easy", "passed": 1, "total": 2, "score": 50.0},
                {"name": "medium", "passed": 0, "total": 1, "score": 0.0},
                {"name": "hard", "passed": 0, "total": 1, "score": 0.0},
            ],
        )

    def test_ungrouped_benchmarks_report_nothing(self) -> None:
        self.assertEqual(group_breakdown([self._record("", True)]), [])


if __name__ == "__main__":
    unittest.main()
