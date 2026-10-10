"""Native decision models: routing, /v1/systemone requests, and calibration."""

from __future__ import annotations

import json
import unittest

from benchkit import decision
from benchkit.benchmarks.base import Task
from benchkit.client import InferenceClient, _is_decision_model
from benchkit.engine import Engine, JobSpec, benchmark, route_decision_jobs, tasks_for


class FakeDecisionClient:
    """Answers every choice with the right option at 0.7, yes/no with P=0.8."""

    timeout = 5.0
    label = "test"
    host = "test://local"

    def __init__(self, decision_models: set[str]) -> None:
        self.decision_models = decision_models
        self.requests: list[dict] = []

    def is_decision_model(self, model: str) -> bool:
        return model in self.decision_models

    def decide(self, model: str, request: dict, cancel_event=None) -> dict:
        self.requests.append(request)
        question = request["questions"]["answer"]
        if question["type"] == "noul":
            answer = {"type": "noul", "noul": 0.8}
        else:
            keys = list(question["criteria"])
            rest = 0.3 / (len(keys) - 1)
            probs = dict.fromkeys(keys, rest)
            probs[self.answer_for(request)] = 0.7
            answer = {"type": "choice", "probabilities": probs}
        return {"answers": {"answer": answer}, "usage": {"input_tokens": 12}}

    def answer_for(self, request: dict) -> str:
        return self.expected[request["state"]]


class RequestTests(unittest.TestCase):
    def test_choice_question_is_keyed_by_option_letter(self) -> None:
        bench = benchmark("arc")
        task = Task("t", "Why?", {"choices": ["x", "y", "z"], "answer": "B"})
        request = decision.request_for(bench, task)
        self.assertEqual(request["state"], "Why?")
        question = request["questions"][decision.QUESTION]
        self.assertEqual(question["type"], "choice")
        self.assertEqual(question["criteria"], {"A": "x", "B": "y", "C": "z"})
        self.assertEqual(question["instructions"], bench.decision_instructions)

    def test_truthfulqa_only_sends_visible_choices(self) -> None:
        bench = benchmark("truthfulqa")
        task = Task("t", "Q", {"choices": list("abcdef"), "answer": "A"})
        criteria = decision.request_for(bench, task)["questions"]["answer"]["criteria"]
        self.assertEqual(list(criteria), ["A", "B", "C", "D"])

    def test_boolq_is_a_yes_no_question_with_the_passage(self) -> None:
        bench = benchmark("boolq")
        task = Task("t", "is it?", {"passage": "Some text.", "answer": "yes"})
        request = decision.request_for(bench, task)
        self.assertEqual(request["questions"]["answer"]["type"], "noul")
        self.assertIn("Some text.", request["state"])
        self.assertIn("is it?", request["state"])

    def test_supported_benchmarks_are_exactly_the_choice_and_yes_no_suites(
        self,
    ) -> None:
        self.assertEqual(
            set(decision.supported_benchmarks()),
            {
                "arc",
                "banking77",
                "boolq",
                "code-verifier",
                "gpqa",
                "hellaswag",
                "mmlu",
                "mmlu-pro",
                "mmvp",
                "openbookqa",
                "piqa",
                "truthfulqa",
                "vstar",
                "winogrande",
                "xstest",
            },
        )


class CalibrationTests(unittest.TestCase):
    def test_brier_and_correctness(self) -> None:
        row = decision.calibration({"probabilities": {"A": 0.7, "B": 0.3}}, "A")
        self.assertTrue(row["correct"])
        self.assertAlmostEqual(row["p_correct"], 0.7)
        self.assertAlmostEqual(row["brier"], 0.18)

    def test_confidently_wrong_is_two(self) -> None:
        row = decision.calibration({"probabilities": {"A": 1.0, "B": 0.0}}, "B")
        self.assertFalse(row["correct"])
        self.assertAlmostEqual(row["brier"], 2.0)

    def test_summary_ece(self) -> None:
        rows = [
            decision.calibration({"probabilities": {"A": 0.9, "B": 0.1}}, "A"),
            decision.calibration({"probabilities": {"A": 0.9, "B": 0.1}}, "B"),
        ]
        stats = decision.summary(rows)
        # One bin at confidence 0.9 with 50% accuracy.
        self.assertAlmostEqual(stats["decision_ece"], 0.4)
        self.assertAlmostEqual(stats["decision_confidence"], 90.0)
        self.assertEqual(decision.summary([{}, {}]), {})

    def test_summary_speed(self) -> None:
        rows = [
            decision.calibration(
                {"probabilities": {"A": 1.0, "B": 0.0}, "latency_ms": ms}, "A"
            )
            for ms in (10, 20, 30, 40, 100)
        ]
        stats = decision.summary(rows, wall_time_s=0.5)
        self.assertEqual(stats["decisions_per_s"], 10.0)
        self.assertEqual(stats["decision_latency_p50_ms"], 30.0)
        self.assertEqual(stats["decision_latency_p95_ms"], 100.0)
        self.assertEqual(decision.summary(rows)["decisions_per_s"], 0.0)

    def test_display_helpers(self) -> None:
        from benchkit.metrics import decision_speed, latency_text

        result = {
            "harness": "decision",
            "decisions_per_s": 20.44,
            "decision_latency_p50_ms": 48.2,
            "decision_latency_p95_ms": 61.0,
            "avg_response_time": 0.0,
        }
        self.assertEqual(
            decision_speed(result), ("20.4 dec/s", "p50 48 ms · p95 61 ms")
        )
        self.assertEqual(latency_text(result), "48ms")
        self.assertEqual(latency_text({"avg_response_time": 1.2}), "1.2s")


class RoutingTests(unittest.TestCase):
    def test_decision_models_are_routed_and_chat_models_are_untouched(self) -> None:
        client = FakeDecisionClient({"d1"})
        jobs = [
            JobSpec("d1", "mmlu", harness="direct", repair_attempts=1),
            JobSpec("d1", "mmlu", harness="pi", repair_attempts=1),
            JobSpec("qwen", "mmlu", harness="direct", repair_attempts=1),
        ]
        routed = route_decision_jobs(jobs, client)
        self.assertEqual(len(routed), 2)
        self.assertEqual(routed[0].harness, "decision")
        self.assertEqual(routed[0].repair_attempts, 0)
        self.assertEqual(routed[0].harness_label, "Decision")
        self.assertEqual(routed[1], jobs[2])

    def test_generative_benchmark_on_decision_model_is_an_error(self) -> None:
        client = FakeDecisionClient({"d1"})
        jobs = [JobSpec("d1", "humaneval"), JobSpec("qwen", "humaneval")]
        with self.assertRaisesRegex(ValueError, "d1 × humaneval"):
            route_decision_jobs(jobs, client)

    def test_forced_models_route_without_server_detection(self) -> None:
        client = FakeDecisionClient(set())
        jobs = [JobSpec("d1", "arc"), JobSpec("qwen", "arc")]
        routed = route_decision_jobs(jobs, client, forced={"d1"})
        self.assertEqual([job.harness for job in routed], ["decision", "direct"])
        # A client with no detection at all still honors the forced list.
        routed = route_decision_jobs(jobs, object(), forced={"d1"})
        self.assertEqual(routed[0].harness, "decision")
        with self.assertRaisesRegex(ValueError, "d1 × humaneval"):
            route_decision_jobs([JobSpec("d1", "humaneval")], object(), {"d1"})

    def test_clients_without_detection_are_unchanged(self) -> None:
        jobs = [JobSpec("m", "humaneval")]
        self.assertIs(route_decision_jobs(jobs, object()), jobs)

    def test_models_endpoint_marker(self) -> None:
        self.assertTrue(
            _is_decision_model({"architecture": {"output_modalities": ["decisions"]}})
        )
        self.assertTrue(
            _is_decision_model(
                {"architecture": {"output_modalities": ["text", "decisions"]}}
            )
        )
        self.assertFalse(
            _is_decision_model({"architecture": {"output_modalities": ["text"]}})
        )
        self.assertFalse(_is_decision_model({}))

    def test_llama_swap_metadata_marker(self) -> None:
        def entry(metadata: dict) -> dict:
            return {"meta": {"llamaswap": {"type": "model", **metadata}}}

        self.assertTrue(_is_decision_model(entry({"decision": True})))
        self.assertFalse(_is_decision_model(entry({"decision": "yes"})))
        self.assertFalse(_is_decision_model(entry({})))

    def test_client_reads_marker_from_discovery(self) -> None:
        client = InferenceClient("http://x")
        client._models_by_name = {"d1": {"decision": True}, "qwen": {}}
        self.assertTrue(client.is_decision_model("d1"))
        self.assertFalse(client.is_decision_model("qwen"))
        self.assertFalse(client.is_decision_model("missing"))


class EngineTests(unittest.TestCase):
    def _client(self, key: str, count: int) -> FakeDecisionClient:
        client = FakeDecisionClient({"d1"})
        bench = benchmark(key)
        client.expected = {
            decision.request_for(bench, task)["state"]: task.metadata["answer"]
            for task in tasks_for(key)[:count]
        }
        return client

    def test_choice_benchmark_scores_through_existing_evaluate(self) -> None:
        client = self._client("arc", 3)
        jobs = route_decision_jobs([JobSpec("d1", "arc", "3")], client)
        [result] = Engine(client, jobs).run()
        self.assertEqual(result["harness"], "decision")
        self.assertEqual(result["score"], 100.0)
        self.assertEqual(result["total_output_tokens"], 0)
        self.assertEqual(result["total_input_tokens"], 36)
        self.assertAlmostEqual(result["decision_p_correct"], 70.0)
        self.assertGreater(result["decisions_per_s"], 0)
        self.assertGreaterEqual(
            result["decision_latency_p95_ms"], result["decision_latency_p50_ms"]
        )
        task = result["tasks"][0]
        self.assertEqual(len(task["response"]), 1)
        self.assertEqual(json.loads(task["prompt"]), client.requests[0])
        self.assertTrue(task["decision"]["correct"])

    def test_choice_order_perturbation_still_pairs(self) -> None:
        client = FakeDecisionClient({"d1"})
        bench = benchmark("arc")
        tasks = tasks_for("arc")[:2]
        from benchkit.perturbations import perturb_task

        client.expected = {}
        for task in tasks:
            for name in (None, "choice-order"):
                case = perturb_task("arc", task, name, 42)
                state = decision.request_for(bench, case.prompt_task)["state"]
                client.expected.setdefault(state, {})[name] = case
        # Same state for clean and perturbed; answer by matching criteria.
        client.answer_for = lambda request: next(
            case.evaluation_task.metadata["answer"]
            for case in client.expected[request["state"]].values()
            if decision.request_for(bench, case.prompt_task)["questions"]
            == request["questions"]
        )
        jobs = route_decision_jobs(
            [
                JobSpec("d1", "arc", "2"),
                JobSpec("d1", "arc", "2", perturbation="choice-order"),
            ],
            client,
        )
        clean, perturbed = Engine(client, jobs).run()
        self.assertEqual(clean["score"], 100.0)
        self.assertEqual(perturbed["perturbed_score"], 100.0)

    def test_yes_no_benchmark(self) -> None:
        client = FakeDecisionClient({"d1"})
        jobs = route_decision_jobs([JobSpec("d1", "boolq", "4")], client)
        [result] = Engine(client, jobs).run()
        expected = sum(
            task.metadata["answer"] == "yes" for task in tasks_for("boolq")[:4]
        )
        self.assertEqual(result["passed"], expected)
        self.assertEqual(result["tasks"][0]["response"], "yes")


class XSTestDecisionTests(unittest.TestCase):
    def test_request_is_a_yes_no_moderation_question(self) -> None:
        bench = benchmark("xstest")
        task = tasks_for("xstest")[0]
        request = decision.request_for(bench, task)
        self.assertEqual(request["state"], task.prompt)
        question = request["questions"]["answer"]
        self.assertEqual(question["type"], "noul")
        self.assertIn("refuse", question["instructions"])

    def test_yes_means_refuse(self) -> None:
        bench = benchmark("xstest")
        tasks = tasks_for("xstest")
        safe = next(t for t in tasks if t.metadata["label"] == "safe")
        unsafe = next(t for t in tasks if t.metadata["label"] == "unsafe")
        self.assertTrue(bench.evaluate_decision(unsafe, "yes"))
        self.assertFalse(bench.evaluate_decision(unsafe, "no"))
        self.assertTrue(bench.evaluate_decision(safe, "no"))
        self.assertFalse(bench.evaluate_decision(safe, "yes"))

    def test_engine_uses_the_decision_evaluator_and_splits_failures(self) -> None:
        tasks = tasks_for("xstest")
        labels = {t.prompt: t.metadata["label"] for t in tasks}

        class Refuser(FakeDecisionClient):
            """Refuses everything: every safe prompt becomes a false refusal."""

            def decide(self, model, request, cancel_event=None):
                assert labels[request["state"]] in {"safe", "unsafe"}
                return {
                    "answers": {"answer": {"type": "noul", "noul": 0.9}},
                    "usage": {"input_tokens": 5},
                }

        client = Refuser({"d1"})
        jobs = route_decision_jobs([JobSpec("d1", "xstest")], client)
        [result] = Engine(client, jobs).run()
        self.assertEqual(result["harness"], "decision")
        self.assertEqual(result["xstest_safe_total"], 250)
        self.assertEqual(result["xstest_unsafe_total"], 200)
        self.assertEqual(result["xstest_false_refusal_rate"], 100.0)
        self.assertEqual(result["xstest_missed_refusal_rate"], 0.0)
        self.assertAlmostEqual(result["score"], 200 / 450 * 100, places=1)


if __name__ == "__main__":
    unittest.main()
