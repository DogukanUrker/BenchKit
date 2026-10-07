"""BANKING77 intent routing: label extraction, macro-F1, and decision keys."""

from __future__ import annotations

import json
import unittest
from collections import Counter

from benchkit import decision
from benchkit.benchmarks.banking77 import extract_intent, intents, macro_f1
from benchkit.benchmarks.base import Task
from benchkit.engine import Engine, JobSpec, benchmark, route_decision_jobs, tasks_for
from benchkit.report import save


class FakeRouter:
    """Routes each message to a fixed intent with probability 0.7."""

    timeout = 5.0
    label = "test"
    host = "test://local"

    def __init__(self, routes: dict[str, str]) -> None:
        self.routes = routes
        self.requests: list[dict] = []

    def is_decision_model(self, model: str) -> bool:
        return model == "d1"

    def decide(self, model: str, request: dict, cancel_event=None) -> dict:
        self.requests.append(request)
        keys = list(request["questions"]["answer"]["criteria"])
        probs = dict.fromkeys(keys, 0.3 / (len(keys) - 1))
        probs[self.routes[request["state"]]] = 0.7
        answer = {"type": "choice", "probabilities": probs}
        return {"answers": {"answer": answer}, "usage": {"input_tokens": 12}}


class DatasetTests(unittest.TestCase):
    def test_full_test_split_with_every_intent(self) -> None:
        tasks = tasks_for("banking77")
        self.assertEqual(len(tasks), 3080)
        self.assertEqual(len(intents()), 77)
        counts = Counter(task.metadata["answer"] for task in tasks)
        self.assertEqual(set(counts), set(intents()))
        self.assertEqual(len({task.id for task in tasks}), 3080)

    def test_any_prefix_slice_is_spread_across_intents(self) -> None:
        first = [task.metadata["answer"] for task in tasks_for("banking77")[:77]]
        self.assertEqual(sorted(first), sorted(intents()))

    def test_chat_prompt_lists_every_intent(self) -> None:
        bench = benchmark("banking77")
        task = tasks_for("banking77")[0]
        prompt = bench.build_prompt(task)
        self.assertIn(task.prompt, prompt)
        for label in intents():
            self.assertIn(f"- {label}\n", prompt)


class ExtractionTests(unittest.TestCase):
    def test_exact_label_up_to_case_and_wrapping(self) -> None:
        for response in (
            "card_arrival",
            "Card_Arrival",
            "`card_arrival`",
            "**card_arrival**.",
            "#### card_arrival",
            '"card_arrival"',
            "Intent: card_arrival",
            "<think>maybe card_linking</think>card_arrival",
            "The customer is waiting on a new card.\ncard_arrival",
        ):
            with self.subTest(response=response):
                self.assertEqual(extract_intent(response), "card_arrival")

    def test_labels_keep_their_upstream_spelling(self) -> None:
        self.assertEqual(
            extract_intent("refund_not_showing_up"), "Refund_not_showing_up"
        )

    def test_anything_but_a_label_is_no_answer(self) -> None:
        for response in (
            "",
            "card arrival",
            "It is card_arrival or card_linking",
            "not_an_intent",
        ):
            with self.subTest(response=response):
                self.assertIsNone(extract_intent(response))

    def test_feedback_never_names_the_expected_intent(self) -> None:
        bench = benchmark("banking77")
        task = Task("t", "Where is my card?", {"answer": "card_arrival"})
        result = bench.evaluate_with_feedback(task, "card_linking")
        self.assertEqual(result.score, 0.0)
        self.assertNotIn("card_arrival", result.feedback)
        self.assertEqual(result.details["predicted_intent"], "card_linking")
        self.assertEqual(bench.evaluate_with_feedback(task, "card_arrival").score, 1.0)


class MacroF1Tests(unittest.TestCase):
    def test_perfect_and_empty(self) -> None:
        self.assertEqual(macro_f1([("a", "a"), ("b", "b")]), 1.0)
        self.assertEqual(macro_f1([]), 0.0)

    def test_collapsing_onto_one_intent_is_punished(self) -> None:
        # 3 of 4 accurate, but intent b is never predicted.
        pairs = [("a", "a"), ("a", "a"), ("a", "a"), ("b", "a")]
        # F1(a) = 2*3/(3+4) = 6/7, F1(b) = 0.
        self.assertAlmostEqual(macro_f1(pairs), 3 / 7)

    def test_missing_prediction_counts_against_recall_only(self) -> None:
        # F1(a) = 2*1/(2+1), F1(b) = 1.
        self.assertAlmostEqual(
            macro_f1([("a", "a"), ("a", None), ("b", "b")]), (2 / 3 + 1) / 2
        )


class DecisionTests(unittest.TestCase):
    def test_request_offers_all_intents_keyed_by_label(self) -> None:
        bench = benchmark("banking77")
        task = tasks_for("banking77")[0]
        request = decision.request_for(bench, task)
        self.assertEqual(request["state"], task.prompt)
        question = request["questions"][decision.QUESTION]
        self.assertEqual(question["type"], "choice")
        self.assertEqual(list(question["criteria"]), list(intents()))
        self.assertEqual(question["criteria"]["card_arrival"], "card arrival")
        self.assertIn("banking77", decision.supported_benchmarks())

    def test_engine_scores_the_chosen_label(self) -> None:
        tasks = tasks_for("banking77")[:4]
        routes = {task.prompt: task.metadata["answer"] for task in tasks}
        # The last message is routed to the wrong intent.
        routes[tasks[3].prompt] = next(
            label for label in intents() if label != tasks[3].metadata["answer"]
        )
        client = FakeRouter(routes)

        jobs = route_decision_jobs([JobSpec("d1", "banking77", "4")], client)
        [result] = Engine(client, jobs).run()
        self.assertEqual(result["harness"], "decision")
        self.assertEqual(result["score"], 75.0)
        task = result["tasks"][0]
        self.assertEqual(task["response"], tasks[0].metadata["answer"])
        self.assertTrue(task["decision"]["correct"])
        self.assertAlmostEqual(task["decision"]["p_correct"], 0.7)
        self.assertEqual(len(task["decision"]["probabilities"]), 77)
        self.assertEqual(json.loads(task["prompt"]), client.requests[0])
        expected = macro_f1([(t.metadata["answer"], routes[t.prompt]) for t in tasks])
        self.assertEqual(result["macro_f1"], round(expected * 100, 1))
        self.assertIn("decision_ece", result)


def test_markdown_report_has_routing_section(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    tasks = tasks_for("banking77")[:2]
    client = FakeRouter({task.prompt: task.metadata["answer"] for task in tasks})
    jobs = route_decision_jobs([JobSpec("d1", "banking77", "2")], client)
    out = save(Engine(client, jobs).run())
    text = (out / "results.md").read_text()
    assert "## Intent routing" in text
    assert "| 100.0% | 100.0% |" in text
    assert "dec/s" in text


if __name__ == "__main__":
    unittest.main()
