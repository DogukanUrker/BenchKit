"""Regression tests for EvalPlus response assembly."""

from benchkit.benchmarks.evalplus import _extract_code, _human_solution


def test_humaneval_body_recovers_first_line_only_indent_loss() -> None:
    problem = {
        "entry_point": "has_close_elements",
        "prompt": "def has_close_elements(numbers, threshold):\n",
    }
    response = """sorted_numbers = sorted(numbers)
    for i in range(len(sorted_numbers) - 1):
        if sorted_numbers[i+1] - sorted_numbers[i] < threshold:
            return True
    return False"""

    solution = _human_solution(problem, response)

    compile(solution, "<test-humaneval-body>", "exec")


def test_orphan_think_close_grades_the_final_fence() -> None:
    response = (
        "Let me draft it.\n```python\nreturn 'draft'\n```\n"
        "Done.</think>\n```python\nreturn 'final'\n```"
    )

    assert _extract_code(response) == "return 'final'"
