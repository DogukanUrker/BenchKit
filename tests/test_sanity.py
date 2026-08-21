"""Tests for the curated Sanity benchmark."""

from collections import Counter

from benchkit.benchmarks.sanity import Sanity


def test_sanity_has_five_checks_per_category() -> None:
    benchmark = Sanity()
    tasks = benchmark.load_tasks()

    assert len(tasks) == benchmark.task_count == 25
    assert [task.id for task in tasks] == [f"Sanity/{index}" for index in range(25)]
    assert Counter(task.metadata["sanity_category"] for task in tasks) == {
        "code": 5,
        "math": 5,
        "instruction": 5,
        "science": 5,
        "commonsense": 5,
    }


def test_sanity_dispatches_to_native_prompt_and_evaluator() -> None:
    benchmark = Sanity()
    tasks = benchmark.load_tasks()
    code, math, science, commonsense = tasks[0], tasks[5], tasks[15], tasks[20]

    assert "Complete the following Python function" in benchmark.build_prompt(code)
    assert "####" in benchmark.build_prompt(math)
    assert benchmark.evaluate(math, f"#### {math.metadata['answer']}")
    assert "A)" in benchmark.build_prompt(science)
    assert benchmark.evaluate(science, science.metadata["answer"])
    assert "Goal:" in benchmark.build_prompt(commonsense)
    assert benchmark.evaluate(commonsense, commonsense.metadata["answer"])
