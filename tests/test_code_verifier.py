"""Tests for the code-verifier benchmark."""

import json
from collections import Counter
from types import SimpleNamespace

from benchkit import decision
from benchkit.benchmarks import DESCRIPTIONS
from benchkit.benchmarks.code_verifier import CodeVerifier


def test_loads_a_balanced_interleaved_suite() -> None:
    benchmark = CodeVerifier()
    tasks = benchmark.load_tasks()

    assert len(tasks) == benchmark.task_count == 500
    assert Counter(task.metadata["answer"] for task in tasks) == {
        "yes": 250,
        "no": 250,
    }
    assert [task.metadata["answer"] for task in tasks[:4]] == ["no", "yes"] * 2
    assert len({task.id for task in tasks}) == 500
    assert "code-verifier" in DESCRIPTIONS


def test_state_shows_problem_and_candidate_but_not_the_label() -> None:
    task = CodeVerifier().load_tasks()[0]

    assert task.prompt.startswith("Problem:\n```python\n")
    assert "Solution:\n```python\n" in task.prompt
    assert task.metadata["model"] not in task.prompt
    assert '"passes"' not in task.prompt


def test_chat_answers_are_graded_on_the_last_yes_or_no() -> None:
    benchmark = CodeVerifier()
    tasks = benchmark.load_tasks()
    failing = next(t for t in tasks if t.metadata["answer"] == "no")
    passing = next(t for t in tasks if t.metadata["answer"] == "yes")

    assert benchmark.evaluate(failing, "NO")
    assert benchmark.evaluate(passing, "<think>maybe no</think> Yes.")
    assert not benchmark.evaluate(failing, "Yes, it passes.")
    assert not benchmark.evaluate(passing, "I cannot tell.")


def test_decision_models_get_a_yes_no_question() -> None:
    benchmark = CodeVerifier()
    task = benchmark.load_tasks()[0]
    body = json.loads(decision.render_request(benchmark, task))

    assert body["state"] == task.prompt
    assert body["questions"]["answer"]["type"] == "noul"


def test_summary_splits_accuracy_by_true_label() -> None:
    benchmark = CodeVerifier()
    tasks = benchmark.load_tasks()
    # A verifier that always says yes: right on every passing solution only.
    records = [
        SimpleNamespace(task_id=t.id, passed=t.metadata["answer"] == "yes")
        for t in tasks
    ]

    assert benchmark.summary_fields(records) == {
        "verifier_pass_total": 250,
        "verifier_fail_total": 250,
        "verifier_pass_accuracy": 100.0,
        "verifier_fail_accuracy": 0.0,
    }
