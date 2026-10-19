"""Tests for the CRUXEval input and output prediction suites."""

import pytest

from benchkit.benchmarks import DESCRIPTIONS, REGISTRY
from benchkit.benchmarks.base import Task
from benchkit.benchmarks.cruxeval import (
    CRUXEvalInput,
    CRUXEvalInputCoT,
    CRUXEvalOutput,
    CRUXEvalOutputCoT,
    outputs_match,
    predicted_call,
    predicted_output,
)

KEYS = ("cruxeval-o", "cruxeval-i", "cruxeval-o-cot", "cruxeval-i-cot")

CONCAT = Task(
    id="CRUXEval/test",
    prompt="def f(s1, s2):\n    return s1 + s2",
    metadata={"input": "'ba', 'nana'", "output": "'banana'"},
)


def test_every_key_loads_the_same_800_functions() -> None:
    for key in KEYS:
        bench = REGISTRY[key]()
        tasks = bench.load_tasks()
        assert bench.name == key
        assert len(tasks) == bench.task_count == 800
        assert len({task.id for task in tasks}) == 800
        assert key in DESCRIPTIONS


def test_prompts_hide_the_answer_they_ask_for() -> None:
    task = CRUXEvalOutput().load_tasks()[0]
    output_prompt = CRUXEvalOutput().build_prompt(task)
    input_prompt = CRUXEvalInput().build_prompt(task)

    assert f"assert f({task.metadata['input']}) == ??" in output_prompt
    assert task.metadata["output"] not in output_prompt
    assert f"assert f(??) == {task.metadata['output']}" in input_prompt
    assert output_prompt.endswith("[ANSWER]\n")
    assert CRUXEvalOutputCoT().build_prompt(task).endswith("[THOUGHT]\n")
    assert CRUXEvalInputCoT().build_prompt(task).endswith("[THOUGHT]\n")


def test_mode_is_recorded_with_the_score() -> None:
    assert CRUXEvalInputCoT().summary_fields([]) == {
        "cruxeval_task": "input",
        "cruxeval_prompt": "cot",
    }


@pytest.mark.parametrize(
    "response",
    [
        'assert f("ba", "nana") == "banana"\n[/ANSWER]',
        "[ANSWER]\nassert f('x', 'y') == 'banana'\n[/ANSWER]",
        "Step 1...\n[/THOUGHT]\n[ANSWER]\nassert f(1) == 'banana' # done\n[/ANSWER]",
        "```python\nassert f(1) == 'banana'\n```",
        "'banana'",
        '"banana"',
        "banana",
        "assert f(1) == banana",
        "<think>it is 'bananas'</think>[ANSWER]assert f(1) == 'banana'[/ANSWER]",
    ],
)
def test_output_predictions_compare_as_literals(response: str) -> None:
    assert CRUXEvalOutput().evaluate(CONCAT, response)


@pytest.mark.parametrize(
    "response",
    [
        "assert f(1) == 'banan'",
        "assert f(1) == 'ba' + 'nana'",  # unsimplified expression
        "assert f(1) == f('ba', 'nana')",
        "[ANSWER]\n[/ANSWER]",
        "",
    ],
)
def test_wrong_or_non_literal_outputs_fail(response: str) -> None:
    result = CRUXEvalOutput().evaluate_with_feedback(CONCAT, response)
    assert not result.passed
    assert "banana" not in result.feedback


def test_literal_comparison_ignores_repr_spelling() -> None:
    assert outputs_match("[(4,1),( 2, 3 )]", "[(4, 1), (2, 3)]")
    assert outputs_match("{'b': 2, 'a': 1}", "{'a': 1, 'b': 2}")
    assert outputs_match("0.30000000000000004", "0.30000000000000004")
    assert not outputs_match("[(4, 1)]", "[[4, 1]]")
    # The bare-text fallback is for strings only.
    assert not outputs_match("four", "4")


def test_output_side_is_taken_from_either_side_of_the_assertion() -> None:
    assert predicted_output("assert 'banana' == f(1)") == "'banana'"
    assert predicted_call("assert 'banana' == f('ba', 'nana')") == "f('ba', 'nana')"
    assert predicted_call("'ba', 'nana'") == "f('ba', 'nana')"
    assert predicted_call("assert 'banana' == 'banana'") is None


@pytest.mark.parametrize(
    "response",
    [
        '[ANSWER]\nassert f("ba", "nana") == "banana"\n[/ANSWER]',
        'assert f("b", "anana") == "banana"\n[/ANSWER]',  # a different valid input
        "assert f('banana', '') == 'banana'",
        "'bana', 'na'",
    ],
)
def test_any_input_that_produces_the_output_passes(response: str) -> None:
    assert CRUXEvalInput().evaluate(CONCAT, response)


def test_inputs_that_miss_or_cheat_fail() -> None:
    bench = CRUXEvalInput()
    wrong = bench.evaluate_with_feedback(CONCAT, "assert f('ba', 'nan') == 'banana'")
    assert not wrong.passed
    assert "different value" in wrong.feedback

    exits = bench.evaluate_with_feedback(
        CONCAT, "assert f(__import__('os')._exit(0), 'x') == 'banana'"
    )
    assert not exits.passed

    missing = bench.evaluate_with_feedback(CONCAT, "I am not sure.")
    assert not missing.passed


def test_canonical_answers_pass_a_slice_of_the_real_data() -> None:
    output, inputs = CRUXEvalOutput(), CRUXEvalInput()
    for task in output.load_tasks()[:10]:
        answer = task.metadata["canonical_solution"]
        assert output.evaluate(task, answer), task.id
        assert inputs.evaluate(task, answer), task.id
