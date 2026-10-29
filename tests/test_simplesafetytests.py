"""Tests for the offline SimpleSafetyTests refusal suite."""

from collections import Counter
from io import StringIO
from types import SimpleNamespace

from benchkit import decision, report_markdown
from benchkit.benchmarks import DESCRIPTIONS, REGISTRY
from benchkit.benchmarks.simplesafetytests import SimpleSafetyTests
from benchkit.benchmarks.xstest import XSTest
from benchkit.engine import Engine, JobSpec, route_decision_jobs


class Refuser:
    """A decision model that says every prompt should be refused."""

    timeout = 5.0
    label = "test"
    host = "test://local"

    def is_decision_model(self, model: str) -> bool:
        return model == "d1"

    def decide(self, model: str, request: dict, cancel_event=None) -> dict:
        return {
            "answers": {"answer": {"type": "noul", "noul": 0.9}},
            "usage": {"input_tokens": 5},
        }


def test_loads_the_complete_suite() -> None:
    benchmark = SimpleSafetyTests()
    tasks = benchmark.load_tasks()

    assert len(tasks) == benchmark.task_count == 100
    assert len({task.id for task in tasks}) == 100
    assert set(Counter(t.metadata["harm_area"] for t in tasks).values()) == {20}
    assert Counter(t.metadata["prompt_type"] for t in tasks) == {
        "Information and advice seeking": 50,
        "Instructions and actions": 50,
    }
    assert REGISTRY["simplesafetytests"] is SimpleSafetyTests
    assert "simplesafetytests" in DESCRIPTIONS


def test_slices_cover_every_harm_area_and_prompt_type() -> None:
    tasks = SimpleSafetyTests().load_tasks()

    for start in range(0, 100, 5):
        areas = {t.metadata["harm_area"] for t in tasks[start : start + 5]}
        assert len(areas) == 5
    for start in range(0, 100, 10):
        pairs = {
            (t.metadata["harm_area"], t.metadata["prompt_type"])
            for t in tasks[start : start + 10]
        }
        assert len(pairs) == 10


def test_refusals_pass_and_compliance_fails() -> None:
    benchmark = SimpleSafetyTests()
    task = benchmark.load_tasks()[0]

    assert benchmark.evaluate(task, "I'm sorry, but I can't help with that.")
    assert benchmark.evaluate(task, "<think>no</think> I cannot assist.")
    assert not benchmark.evaluate(task, "Sure, here is how.")


def test_decision_models_get_xstests_refusal_question() -> None:
    benchmark = SimpleSafetyTests()
    task = benchmark.load_tasks()[0]
    request = decision.request_for(benchmark, task)

    assert request["state"] == task.prompt
    question = request["questions"]["answer"]
    assert question["type"] == "noul"
    assert question["instructions"] == XSTest.decision_instructions
    assert benchmark.evaluate_decision(task, "yes")
    assert not benchmark.evaluate_decision(task, "no")


def test_summary_splits_refusal_by_harm_area_and_prompt_type() -> None:
    benchmark = SimpleSafetyTests()
    tasks = benchmark.load_tasks()
    # Refuses only information-seeking prompts.
    records = [
        SimpleNamespace(
            task_id=t.id,
            score=float(t.metadata["prompt_type"].startswith("Information")),
        )
        for t in tasks
    ]

    fields = benchmark.summary_fields(records)
    assert fields["sst_prompt_type_refusal"] == {
        "Information and advice seeking": 100.0,
        "Instructions and actions": 0.0,
    }
    assert len(fields["sst_harm_area_refusal"]) == 5
    assert benchmark.summary_fields([]) == {}


def test_engine_reports_refusal_split_and_markdown_section() -> None:
    client = Refuser()
    jobs = route_decision_jobs([JobSpec("d1", "simplesafetytests")], client)
    [result] = Engine(client, jobs).run()

    assert result["harness"] == "decision"
    assert result["score"] == 100.0
    assert set(result["sst_harm_area_refusal"].values()) == {100.0}

    out = StringIO()
    report_markdown._simplesafetytests_refusals(out, [result])
    text = out.getvalue()
    assert "## SimpleSafetyTests refusals" in text
    assert "Child safety" in text
    assert "| d1 | Decision | 100.0% |" in text


def test_suite_is_excluded_from_general_overall_score() -> None:
    assert SimpleSafetyTests.include_in_overall is False
