"""Tests for bounded, answer-safe verifier repair."""

from __future__ import annotations

import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from benchkit.cli import _headless_jobs, _parse_args
from benchkit.engine import MAX_REPAIR_ATTEMPTS, Engine, JobSpec, benchmark
from benchkit.evaluation import EvaluationResult, evaluate_response
from benchkit.executor import execute_with_feedback
from benchkit.pi_agent import PiAgentRunner


def _generation(response: str, tokens: int = 3) -> dict:
    return {
        "thinking": "",
        "response": response,
        "trace_status": "unavailable",
        "tok_s": 10.0,
        "eval_count": tokens,
        "eval_duration_ns": tokens * 100_000_000,
        "response_time_s": tokens / 10,
        "done_reason": "stop",
        "timed_out": False,
        "cancelled": False,
        "input_tokens": 5,
    }


class RepairClient:
    timeout = 10.0
    label = "test"
    host = "test://local"
    provider = "openai"

    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.prompts: list[str] = []

    def max_parallel_requests(self, _model: str) -> int:
        return 1

    def generate(self, _model: str, prompt: str, **_kwargs: object) -> dict:
        self.prompts.append(prompt)
        return _generation(next(self.responses))


class EvaluationFeedbackTests(unittest.TestCase):
    def test_generic_feedback_never_includes_the_expected_answer(self) -> None:
        task = SimpleNamespace(metadata={"answer": "SECRET-42"})
        bench = SimpleNamespace(evaluate=lambda _task, _response: False)

        result = evaluate_response(bench, task, "wrong")

        self.assertFalse(result.passed)
        self.assertIn("incorrect", result.feedback)
        self.assertNotIn("SECRET-42", result.feedback)

    def test_python_assertion_feedback_hides_the_test_source(self) -> None:
        result = execute_with_feedback(
            "hidden_value = 41\nassert hidden_value == 42  # SECRET-ASSERTION"
        )

        self.assertFalse(result.passed)
        self.assertEqual(
            result.feedback,
            "The submitted Python ran, but a verifier assertion failed.",
        )
        self.assertNotIn("SECRET-ASSERTION", result.feedback)


class DirectRepairTests(unittest.TestCase):
    def test_failed_direct_answer_gets_one_follow_up_raw_call(self) -> None:
        client = RepairClient(["wrong", "right"])
        job = JobSpec(
            "model",
            "quickbench",
            "1",
            harness="direct",
            repair_attempts=1,
        )
        engine = Engine(client=client, jobs=[job])
        quickbench = benchmark("quickbench")

        def verify(_task, response: str) -> EvaluationResult:
            if response == "right":
                return EvaluationResult(1.0)
            return EvaluationResult(0.0, "Sanitized verifier failure.")

        with patch.object(quickbench, "evaluate_with_feedback", side_effect=verify):
            result = engine.run()[0]

        self.assertEqual(len(client.prompts), 2)
        self.assertIn("Previous assistant answer:\nwrong", client.prompts[1])
        self.assertIn("Sanitized verifier failure.", client.prompts[1])
        self.assertNotIn("only verifier repair", client.prompts[0])
        self.assertEqual(result["first_attempt_score"], 0.0)
        self.assertEqual(result["score"], 100.0)
        self.assertEqual(result["repair_delta_pp"], 100.0)
        self.assertEqual(result["repair_attempted"], 1)
        self.assertEqual(result["repair_successes"], 1)
        task = result["tasks"][0]
        self.assertTrue(task["repaired"])
        self.assertEqual(task["repair_attempts_used"], 1)
        self.assertEqual(
            [item["response"] for item in task["attempts"]], ["wrong", "right"]
        )
        self.assertEqual(task["input_tokens"], 10)
        self.assertEqual(task["output_tokens"], 6)

    def test_direct_repair_stops_after_exactly_one_retry(self) -> None:
        client = RepairClient(["wrong-one", "wrong-two", "unused"])
        job = JobSpec("model", "quickbench", "1", repair_attempts=1)
        engine = Engine(client=client, jobs=[job])
        quickbench = benchmark("quickbench")

        with patch.object(
            quickbench,
            "evaluate_with_feedback",
            return_value=EvaluationResult(0.0, "Still incorrect."),
        ):
            result = engine.run()[0]

        self.assertEqual(len(client.prompts), 2)
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["repair_attempted"], 1)
        self.assertEqual(result["repair_successes"], 0)
        self.assertEqual(result["tasks"][0]["repair_attempts_used"], 1)

    def test_direct_repair_supports_multiple_retries(self) -> None:
        client = RepairClient(["wrong-one", "wrong-two", "right"])
        job = JobSpec("model", "quickbench", "1", repair_attempts=3)
        engine = Engine(client=client, jobs=[job])
        quickbench = benchmark("quickbench")

        def verify(_task, response: str) -> EvaluationResult:
            if response == "right":
                return EvaluationResult(1.0)
            return EvaluationResult(0.0, "Sanitized verifier failure.")

        with patch.object(quickbench, "evaluate_with_feedback", side_effect=verify):
            result = engine.run()[0]

        self.assertEqual(len(client.prompts), 3)
        self.assertIn("2 repair attempt(s) remaining", client.prompts[1])
        self.assertIn("1 repair attempt(s) remaining", client.prompts[2])
        self.assertEqual(result["repair_attempted"], 1)
        self.assertEqual(result["repair_turns"], 2)
        self.assertEqual(result["repair_successes"], 1)
        self.assertEqual(
            result["repair_score_curve"],
            [
                {
                    "round": 0,
                    "score": 0.0,
                    "delta_pp": 0.0,
                    "cumulative_delta_pp": 0.0,
                    "attempted": 1,
                    "new_passes": 0,
                },
                {
                    "round": 1,
                    "score": 0.0,
                    "delta_pp": 0.0,
                    "cumulative_delta_pp": 0.0,
                    "attempted": 1,
                    "new_passes": 0,
                },
                {
                    "round": 2,
                    "score": 100.0,
                    "delta_pp": 100.0,
                    "cumulative_delta_pp": 100.0,
                    "attempted": 1,
                    "new_passes": 1,
                },
                {
                    "round": 3,
                    "score": 100.0,
                    "delta_pp": 0.0,
                    "cumulative_delta_pp": 100.0,
                    "attempted": 0,
                    "new_passes": 0,
                },
            ],
        )
        self.assertEqual(result["repair_round_2_delta_pp"], 100.0)
        self.assertEqual(result["repair_round_2_new_passes"], 1)
        self.assertEqual(result["repair_round_3_attempted"], 0)
        self.assertEqual(result["tasks"][0]["repair_attempts_used"], 2)

    def test_repair_limit_is_bounded(self) -> None:
        JobSpec("model", "quickbench", repair_attempts=MAX_REPAIR_ATTEMPTS)
        with self.assertRaisesRegex(ValueError, "between 0 and"):
            JobSpec(
                "model",
                "quickbench",
                repair_attempts=MAX_REPAIR_ATTEMPTS + 1,
            )

    def test_cli_repair_flag_applies_to_direct_and_pi_jobs(self) -> None:
        args = _parse_args(
            [
                "--headless",
                "--models",
                "model",
                "--benchmarks",
                "quickbench:1",
                "--harness",
                "both",
                "--repair-attempts",
                "3",
            ]
        )

        jobs = _headless_jobs(args, ["model"])

        self.assertEqual([job.harness for job in jobs], ["direct", "pi"])
        self.assertEqual([job.repair_attempts for job in jobs], [3, 3])
        self.assertEqual(
            [job.harness_label for job in jobs],
            ["Direct + repair", "Pi agent + repair"],
        )


class _FakeProcess:
    def __init__(self, events: list[dict]) -> None:
        self.stdin = io.StringIO()
        self.stdout = io.StringIO("".join(json.dumps(event) + "\n" for event in events))
        self.stderr = io.StringIO("")

    def poll(self) -> int:
        return 0


class PiRepairTests(unittest.TestCase):
    def test_pi_repair_reuses_one_rpc_process_and_workspace(self) -> None:
        events = [
            {"type": "response", "id": "benchkit-prompt-0", "success": True},
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "wrong"}],
                    "usage": {"input": 10, "output": 2},
                    "stopReason": "stop",
                },
            },
            {"type": "agent_settled"},
            {
                "type": "response",
                "id": "benchkit-stats-0",
                "data": {
                    "tokens": {"input": 10, "output": 2},
                    "assistantMessages": 1,
                },
            },
            {
                "type": "response",
                "id": "benchkit-final-0",
                "data": {"text": "wrong"},
            },
            {"type": "response", "id": "benchkit-prompt-1", "success": True},
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "right"}],
                    "usage": {"input": 15, "output": 3},
                    "stopReason": "stop",
                },
            },
            {"type": "agent_settled"},
            {
                "type": "response",
                "id": "benchkit-stats-1",
                "data": {
                    "tokens": {"input": 25, "output": 5},
                    "assistantMessages": 2,
                },
            },
            {
                "type": "response",
                "id": "benchkit-final-1",
                "data": {"text": "right"},
            },
        ]
        process = _FakeProcess(events)
        environment = Mock()
        environment.__enter__ = Mock(return_value=environment)
        environment.__exit__ = Mock(return_value=None)
        environment.start_pi.return_value = process
        image = Mock(version="test")
        image.prepare.return_value = "test"
        runner = PiAgentRunner(SimpleNamespace(), image=image)
        verifier = Mock(
            side_effect=[
                EvaluationResult(0.0, "Sanitized failure."),
                EvaluationResult(1.0),
            ]
        )

        with patch(
            "benchkit.pi_agent.DockerTaskEnvironment",
            return_value=environment,
        ) as task_environment:
            result = runner.generate(
                "model",
                "original prompt",
                verifier=verifier,
                repair_attempts=1,
            )

        task_environment.assert_called_once()
        environment.start_pi.assert_called_once_with()
        sent = [json.loads(line) for line in process.stdin.getvalue().splitlines()]
        prompts = [item for item in sent if item.get("type") == "prompt"]
        self.assertEqual(len(prompts), 2)
        self.assertEqual(prompts[0]["message"], "original prompt")
        self.assertIn("Sanitized failure.", prompts[1]["message"])
        self.assertEqual(result["response"], "right")
        self.assertTrue(result["repaired"])
        self.assertEqual(result["repair_attempts_used"], 1)
        self.assertEqual(result["input_tokens"], 25)
        self.assertEqual(result["eval_count"], 5)
        self.assertEqual(result["model_turns"], 2)


if __name__ == "__main__":
    unittest.main()
