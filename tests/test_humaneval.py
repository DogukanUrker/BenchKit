"""Regression tests for HumanEval response assembly."""

from benchkit.benchmarks.humaneval import HumanEval, _assemble_solution, _extract_code

CLOSE_ELEMENTS_FLUSH = """\
sorted_numbers = sorted(numbers)
for i in range(len(sorted_numbers) - 1):
    if sorted_numbers[i + 1] - sorted_numbers[i] < threshold:
        return True
return False"""

CLOSE_ELEMENTS_INDENTED = """\
    sorted_numbers = sorted(numbers)
    for i in range(len(sorted_numbers) - 1):
        if sorted_numbers[i + 1] - sorted_numbers[i] < threshold:
            return True
    return False"""

CLOSE_ELEMENTS_FIRST_LINE_FLUSH = """\
sorted_numbers = sorted(numbers)
    for i in range(len(sorted_numbers) - 1):
        if sorted_numbers[i + 1] - sorted_numbers[i] < threshold:
            return True
    return False"""


def _task(task_id: str = "HumanEval/0"):
    for task in HumanEval().load_tasks():
        if task.id == task_id:
            return task
    raise AssertionError(f"missing {task_id}")


def test_body_recovers_first_line_only_indent_loss() -> None:
    task = _task()
    solution = _assemble_solution(
        task.prompt, task.metadata["entry_point"], CLOSE_ELEMENTS_FIRST_LINE_FLUSH
    )
    compile(solution, "<test-humaneval-body>", "exec")
    assert HumanEval().evaluate(task, CLOSE_ELEMENTS_FIRST_LINE_FLUSH)


def test_flush_and_indented_bodies_still_pass() -> None:
    task = _task()
    bench = HumanEval()
    assert bench.evaluate(task, CLOSE_ELEMENTS_FLUSH)
    assert bench.evaluate(task, CLOSE_ELEMENTS_INDENTED)
    assert bench.evaluate(task, f"```python\n{CLOSE_ELEMENTS_FIRST_LINE_FLUSH}\n```")


def test_full_function_completion_still_passes() -> None:
    task = _task()
    response = (
        "def has_close_elements(numbers: list[float], threshold: float) -> bool:\n"
        + CLOSE_ELEMENTS_INDENTED
    )
    assert HumanEval().evaluate(task, response)


def test_wrong_body_still_fails() -> None:
    task = _task()
    assert not HumanEval().evaluate(task, "    return False")


def test_extract_code_strips_fences_without_reindenting() -> None:
    extracted = _extract_code(f"```python\n{CLOSE_ELEMENTS_FIRST_LINE_FLUSH}\n```")
    assert extracted == CLOSE_ELEMENTS_FIRST_LINE_FLUSH
