"""Tests for multiple-choice answer extraction and the MMLU-Pro benchmark."""

from __future__ import annotations

import unittest

from benchkit.benchmarks.base import Task
from benchkit.benchmarks.mcq import extract_choice
from benchkit.benchmarks.mmlu_pro import LETTERS, MMLUPro, _stratified


class ExtractChoiceTest(unittest.TestCase):
    def test_bare_letter(self) -> None:
        self.assertEqual(extract_choice("C"), "C")
        self.assertEqual(extract_choice(" b "), "B")

    def test_marked_answer(self) -> None:
        self.assertEqual(extract_choice("The answer is (C)."), "C")
        self.assertEqual(extract_choice("Answer: D"), "D")
        self.assertEqual(extract_choice("C) Paris"), "C")

    def test_last_commitment_wins(self) -> None:
        response = "Option A looks plausible, but the answer is B."
        self.assertEqual(extract_choice(response), "B")

    def test_falls_back_to_choice_text(self) -> None:
        choices = ["Paris", "Rome", "Madrid", "Lisbon"]
        self.assertEqual(extract_choice("Rome, obviously", choices), "B")

    def test_unparseable(self) -> None:
        self.assertIsNone(extract_choice(""))
        self.assertIsNone(extract_choice("I have no idea whatsoever."))


class ExtractChoiceTenOptionTest(unittest.TestCase):
    def test_letters_past_d(self) -> None:
        for text, expected in (
            ("The answer is (J).", "J"),
            ("Answer: H", "H"),
            ("F.", "F"),
            ("**G**", "G"),
            ("I", "I"),
        ):
            with self.subTest(text=text):
                self.assertEqual(extract_choice(text, letters=LETTERS), expected)

    def test_four_option_parser_ignores_wider_letters(self) -> None:
        self.assertIsNone(extract_choice("The answer is (J)."))

    def test_prose_pronoun_is_not_an_answer(self) -> None:
        response = "I think a careful reading rules that out. The answer is E."
        self.assertEqual(extract_choice(response, letters=LETTERS), "E")

    def test_bare_article_mid_sentence_is_ignored(self) -> None:
        response = "This is a case where the correct option must be G somehow"
        self.assertEqual(extract_choice(response, letters=LETTERS), "G")

    def test_choice_text_uses_wide_alphabet(self) -> None:
        choices = [f"option {index}" for index in range(10)]
        self.assertEqual(
            extract_choice("clearly option 9 fits", choices, letters=LETTERS), "J"
        )


def _task(index: int, category: str, choices: int = 10) -> Task:
    return Task(
        id=f"MMLU-Pro/{category}/{index}",
        prompt="question?",
        metadata={
            "choices": [f"choice {i}" for i in range(choices)],
            "answer": "D",
            "category": category,
            "source": "test",
        },
    )


class MMLUProTest(unittest.TestCase):
    def test_prompt_lists_every_option(self) -> None:
        prompt = MMLUPro().build_prompt(_task(0, "math"))
        self.assertIn("A through J", prompt)
        for letter in LETTERS:
            self.assertIn(f"{letter}) choice", prompt)

    def test_prompt_shortens_letter_range_for_pruned_options(self) -> None:
        prompt = MMLUPro().build_prompt(_task(0, "law", choices=6))
        self.assertIn("A through F", prompt)
        self.assertNotIn("G) ", prompt)

    def test_evaluate_accepts_reasoned_answers(self) -> None:
        task = _task(0, "physics")
        benchmark = MMLUPro()
        self.assertTrue(benchmark.evaluate(task, "Working through it, the answer is D"))
        self.assertFalse(benchmark.evaluate(task, "The answer is (J)."))

    def test_stratified_is_balanced_deterministic_and_ordered(self) -> None:
        tasks = [
            _task(index, category)
            for category in ("math", "law", "health")
            for index in range(30)
        ]
        picked = _stratified(tasks, 10)
        self.assertEqual(len(picked), 30)
        for category in ("math", "law", "health"):
            found = [t for t in picked if t.metadata["category"] == category]
            self.assertEqual(len(found), 10)

        self.assertEqual(
            [task.id for task in picked], [task.id for task in _stratified(tasks, 10)]
        )
        order = [tasks.index(task) for task in picked]
        self.assertEqual(order, sorted(order))

    def test_stratified_keeps_small_categories_whole(self) -> None:
        tasks = [_task(index, "other") for index in range(4)]
        self.assertEqual(len(_stratified(tasks, 10)), 4)


if __name__ == "__main__":
    unittest.main()
