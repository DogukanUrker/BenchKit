"""MATH-500: answer extraction, normalization and SymPy equivalence."""

from __future__ import annotations

import io

import pytest

from benchkit import report_markdown
from benchkit.benchmarks import REGISTRY
from benchkit.benchmarks.base import Task
from benchkit.benchmarks.math500 import (
    MATH500,
    SympyChecker,
    answers_match,
    extract_answer,
    normalize,
)


def _task(answer: str, level: int = 3, subject: str = "Algebra") -> Task:
    return Task(
        id="MATH500/x",
        prompt="?",
        metadata={"answer": answer, "level": level, "subject": subject},
    )


def test_registered_with_all_problems() -> None:
    tasks = REGISTRY["math-500"]().load_tasks()
    assert len(tasks) == 500
    assert {task.metadata["level"] for task in tasks} == {1, 2, 3, 4, 5}
    assert len({task.metadata["subject"] for task in tasks}) == 7
    assert len({task.id for task in tasks}) == 500


def test_prompt_asks_for_a_boxed_answer() -> None:
    bench = MATH500()
    task = bench.load_tasks()[0]
    prompt = bench.build_prompt(task)
    assert prompt.startswith(task.prompt)
    assert prompt.endswith("put your final answer within \\boxed{}.")
    assert task.metadata["answer"] not in prompt.split("\n\n")[-1]


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("so $x = \\boxed{\\frac{1}{2}}$.", "\\frac{1}{2}"),
        (
            "\\boxed{1} then \\boxed{\\left( 3, \\frac{\\pi}{2} \\right)}",
            "\\left( 3, \\frac{\\pi}{2} \\right)",
        ),
        ("\\fbox{7}", "7"),
        ("\\boxed 5$", "5"),
        ("<think>\\boxed{9}</think>The final answer is $12$.", "12"),
        ("Therefore, the answer is: 2\\sqrt{3}.", "2\\sqrt{3}"),
        ("We get $a=1$ and then $b = 4$", "b = 4"),
        ("no math at all", None),
        ("\\boxed{}", None),
    ],
)
def test_extract_answer(response: str, expected: str | None) -> None:
    assert extract_answer(response) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("\\dfrac{17}{50}", "\\frac{17}{50}"),
        ("\\frac43", "\\frac{4}{3}"),
        ("\\frac 59", "\\frac{5}{9}"),
        ("\\frac9{19}", "\\frac{9}{19}"),
        ("11\\sqrt2", "11\\sqrt{2}"),
        ("\\left( 3, \\frac{\\pi}{2} \\right)", "(3,\\frac{\\pi}{2})"),
        ("90^\\circ", "90"),
        ("\\$32,\\!348", "32348"),
        ("11,\\! 111,\\! 111,\\! 100", "11111111100"),
        ("5.4 \\text{ cents}", "5.4"),
        ("864 \\mbox{ inches}^2", "864"),
        ("\\frac{270}7\\text{ degrees}", "\\frac{270}{7}"),
        ("\\text{(C)}", "C"),
        ("\\text{Evelyn}", "evelyn"),
        ("x \\in [-2,7]", "[-2,7]"),
        ("y = 2x + 3", "2x+3"),
        ("4210_{5}", "4210_5"),
        ("3/4", "\\frac{3}{4}"),
        (".5", "0.5"),
    ],
)
def test_normalize(raw: str, expected: str) -> None:
    assert normalize(raw) == expected


@pytest.mark.parametrize(
    ("predicted", "reference"),
    [
        ("0.5", "\\frac{1}{2}"),
        ("1/2", "\\dfrac12"),
        ("\\frac{\\sqrt{3}}{3}", "\\frac{1}{\\sqrt{3}}"),
        ("\\sqrt{12}", "2\\sqrt{3}"),
        ("\\pi/4", "\\frac{\\pi}{4}"),
        ("-5i+6", "6 - 5i"),
        ("(3, \\pi/2)", "\\left( 3, \\frac{\\pi}{2} \\right)"),
        ("-2, 1", "1,-2"),
        ("1+\\sqrt{19}, 1-\\sqrt{19}", "1 \\pm \\sqrt{19}"),
        ("\\{-2, 1+\\sqrt5, 1-\\sqrt5\\}", "\\{1\\pm\\sqrt{5},-2\\}"),
        ("(-\\infty,0]", "(-\\infty, 0]"),
        ("(-\\infty,2)\\cup(3,\\infty)", "(-\\infty, 2) \\cup (3, \\infty)"),
        ("x^3 + 3x - 6", "x^3+3x-6"),
        ("(x-1)(x+1)", "x^2-1"),
        ("\\frac{\\cos x}{\\sin x}", "\\cot x"),
        ("-x + 5x^2 + 3", "5x^2 - x + 3"),
        ("-5x+7y-11z-4=0", "5x - 7y + 11z + 4 = 0"),
        (
            "\\begin{pmatrix} -1/3 \\\\ 2/3 \\\\ 5/3 \\end{pmatrix}",
            "\\begin{pmatrix} -\\frac13 \\\\ \\frac{2}{3} \\\\ \\frac53 \\end{pmatrix}",
        ),
        ("1\\frac{4}{5}", "\\frac{9}{5}"),
        ("137.5", "137 \\frac{1}{2}"),
        ("1100_2", "12"),
        ("5!", "120"),
        ("\\binom{5}{2}", "10"),
    ],
)
def test_equivalent_answers_match(predicted: str, reference: str) -> None:
    assert answers_match(predicted, reference) is not None


@pytest.mark.parametrize(
    ("predicted", "reference"),
    [
        ("0.333", "\\frac{1}{3}"),
        ("2", "\\frac{1}{2}"),
        ("(3,4)", "(3,4]"),
        ("(4,3)", "(3,4)"),
        ("1", "1,-2"),
        ("x^2+1", "x^2-1"),
        ("\\text{west}", "\\text{east}"),
        ("B", "\\text{(C)}"),
        ("\\infty", "1"),
        ("__import__('os').system('true')", "0"),
        ("", "1"),
    ],
)
def test_different_answers_do_not_match(predicted: str, reference: str) -> None:
    assert answers_match(predicted, reference) is None


def test_hanging_comparison_is_killed_and_the_worker_restarts() -> None:
    checker = SympyChecker(timeout=1.0)
    try:
        assert checker.equivalent("\\frac{1}{2}", "0.5") is True
        assert checker.equivalent("9^{9^{9^{9}}}", "1") is False
        assert checker.equivalent("\\sqrt{4}", "2") is True
    finally:
        checker.close()


def test_evaluate_reports_how_it_matched_without_leaking_the_answer() -> None:
    bench = MATH500()
    task = _task("\\frac{14}{3}")

    exact = bench.evaluate_with_feedback(task, "\\boxed{\\dfrac{14}{3}}")
    assert exact.passed
    assert exact.details == {"extracted": "\\dfrac{14}{3}", "match": "string"}

    symbolic = bench.evaluate_with_feedback(task, "\\boxed{28/6}")
    assert symbolic.passed
    assert symbolic.details["match"] == "sympy"

    wrong = bench.evaluate_with_feedback(task, "\\boxed{5}")
    assert not wrong.passed
    assert "14" not in wrong.feedback

    missing = bench.evaluate_with_feedback(task, "I am not sure.")
    assert not missing.passed
    assert "\\boxed{}" in missing.feedback


def test_demo_answer_passes() -> None:
    bench = MATH500()
    for task in bench.load_tasks()[:20]:
        assert bench.evaluate(task, task.metadata["canonical_solution"])


def test_summary_fields_and_markdown_section() -> None:
    bench = MATH500()
    tasks = bench.load_tasks()

    class Record:
        def __init__(self, task_id: str, score: float) -> None:
            self.task_id = task_id
            self.score = score

    records = [Record(task.id, float(task.metadata["level"] <= 2)) for task in tasks]
    fields = bench.summary_fields(records)
    assert fields["math_level_scores"]["Level 1"] == 100.0
    assert fields["math_level_scores"]["Level 5"] == 0.0
    assert set(fields["math_subject_scores"]) == {
        task.metadata["subject"] for task in tasks
    }
    assert bench.summary_fields([]) == {}

    out = io.StringIO()
    report_markdown._math_levels(
        out,
        [{"model": "m", "benchmark": "math-500", "score": 26.6, **fields}],
    )
    markdown = out.getvalue()
    assert "## MATH-500 by level and subject" in markdown
    assert "| Level 1 | Level 2 | Level 3 | Level 4 | Level 5 |" in markdown
    assert "100.0%" in markdown

    empty = io.StringIO()
    report_markdown._math_levels(empty, [{"model": "m", "score": 1.0}])
    assert empty.getvalue() == ""
