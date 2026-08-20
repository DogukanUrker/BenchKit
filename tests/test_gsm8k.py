"""Tests for GSM8K answer extraction and evaluation."""

from __future__ import annotations

import unittest

from benchkit.benchmarks.base import Task
from benchkit.benchmarks.gsm8k import GSM8K


def _task(answer: str = "42") -> Task:
    return Task(
        id="GSM8K/test",
        prompt="What is the answer?",
        metadata={"answer": answer},
    )


class GSM8KTest(unittest.TestCase):
    def test_prompt_requires_the_final_answer_delimiter(self) -> None:
        prompt = GSM8K().build_prompt(_task())

        self.assertIn('after "####"', prompt)
        self.assertTrue(prompt.endswith("What is the answer?"))

    def test_evaluate_accepts_common_number_formats(self) -> None:
        benchmark = GSM8K()
        cases = (
            ("1,234", "Work shown.\n#### 1,234."),
            ("-12.5", "Final answer: **#### -12.5**"),
            ("0.25", "#### +0.25"),
        )

        for answer, response in cases:
            with self.subTest(answer=answer, response=response):
                self.assertTrue(benchmark.evaluate(_task(answer), response))

    def test_last_delimiter_is_the_final_commitment(self) -> None:
        response = "First attempt: #### 41\nCorrection: #### 42"

        self.assertTrue(GSM8K().evaluate(_task(), response))

    def test_reasoning_number_does_not_override_wrong_final_answer(self) -> None:
        response = "The intermediate total is 42, but my answer is #### 41"

        self.assertFalse(GSM8K().evaluate(_task(), response))

    def test_think_block_numbers_are_ignored(self) -> None:
        response = "<think>I considered 41 and 999.</think>\n#### 42"

        self.assertTrue(GSM8K().evaluate(_task(), response))

    def test_unclosed_think_block_is_not_scored_as_an_answer(self) -> None:
        response = "<think>The partial calculation reached 42"

        self.assertFalse(GSM8K().evaluate(_task(), response))

    def test_fallback_uses_the_last_number_without_a_delimiter(self) -> None:
        response = "I tried 41 first. The final result is 42."

        self.assertTrue(GSM8K().evaluate(_task(), response))

    def test_empty_and_unparseable_responses_fail(self) -> None:
        benchmark = GSM8K()

        for response in ("", "No numeric answer is available.", "####"):
            with self.subTest(response=response):
                self.assertFalse(benchmark.evaluate(_task(), response))


if __name__ == "__main__":
    unittest.main()
