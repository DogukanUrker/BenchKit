"""Tests for the synthetic RULER effective-context benchmark."""

from __future__ import annotations

import unittest
from collections import Counter
from unittest.mock import Mock, patch

from benchkit.benchmarks.ruler import RULER
from benchkit.benchmarks.ruler_generator import (
    CONTEXT_BUCKETS,
    OUTPUT_RESERVE_TOKENS,
    TASKS_PER_BUCKET,
    context_label,
    fit_prompt,
    generate_tasks,
)
from benchkit.client import InferenceClient, _context_length_hint
from benchkit.engine import (
    Engine,
    JobSpec,
    benchmark,
    expand_jobs,
    slice_task_count,
    task_count,
    tasks_for_job,
)


class ContextClient:
    timeout = 1.0
    label = "test"
    host = "test://local"

    def __init__(self, context: int | None) -> None:
        self.context = context

    def context_length(self, _model: str) -> int | None:
        return self.context

    def max_parallel_requests(self, _model: str) -> int:
        return 1

    def generate(self, _model: str, _prompt: str, **_kwargs: object) -> dict:
        return {
            "thinking": "",
            "response": "partial",
            "trace_status": "unavailable",
            "tok_s": 10.0,
            "eval_count": 1,
            "eval_duration_ns": 100_000_000,
            "response_time_s": 0.1,
            "done_reason": "stop",
            "timed_out": False,
            "cancelled": False,
        }

    def unload_model(self, _model: str) -> None:
        return None


class RulerGenerationTests(unittest.TestCase):
    def test_generates_twenty_lightweight_specs_per_context(self) -> None:
        tasks = generate_tasks()
        counts = Counter(task.metadata["context_tokens"] for task in tasks)

        self.assertEqual(len(tasks), TASKS_PER_BUCKET * len(CONTEXT_BUCKETS))
        self.assertEqual(task_count("ruler"), len(tasks))
        self.assertEqual(slice_task_count("ruler"), TASKS_PER_BUCKET)
        self.assertEqual(
            counts,
            Counter({bucket: TASKS_PER_BUCKET for bucket in CONTEXT_BUCKETS}),
        )
        self.assertTrue(all(not task.prompt for task in tasks))
        self.assertEqual(
            {task.metadata["task_type"] for task in tasks},
            {"niah_multikey", "variable_tracking"},
        )

    def test_prompt_fitting_is_deterministic_and_token_aware(self) -> None:
        task = generate_tasks()[0]

        def counter(text: str) -> int:
            return len(text.split())

        first, units, measured = fit_prompt(task, counter)
        second, second_units, second_measured = fit_prompt(task, counter)
        target = int(task.metadata["context_tokens"]) - OUTPUT_RESERVE_TOKENS

        self.assertEqual(first, second)
        self.assertEqual(units, second_units)
        self.assertEqual(measured, second_measured)
        assert measured is not None
        self.assertLessEqual(measured, target)
        self.assertGreaterEqual(measured, target * 0.99)
        self.assertIn(str(task.metadata["query"]), first)
        self.assertIn(str(task.metadata["answers"][0]), first)

    def test_variable_tracking_uses_fractional_ruler_scoring(self) -> None:
        task = next(
            task
            for task in generate_tasks()
            if task.metadata["task_type"] == "variable_tracking"
        )
        answers = task.metadata["answers"]
        response = f"The variables are {answers[0]}, {answers[2]}, {answers[4]}."

        self.assertEqual(RULER().evaluate(task, response), 3 / 5)
        self.assertEqual(RULER().evaluate(task, "none found"), 0.0)


class RulerJobTests(unittest.TestCase):
    def test_job_expansion_stops_at_discovered_context(self) -> None:
        jobs = expand_jobs(
            [JobSpec("model", "ruler", "3")],
            ContextClient(32768),
        )

        self.assertEqual(
            [job.variant for job in jobs],
            ["4k", "8k", "16k", "32k"],
        )
        self.assertTrue(all(job.planned_total() == 3 for job in jobs))
        self.assertTrue(
            all(len(tasks_for_job(job)) == TASKS_PER_BUCKET for job in jobs)
        )

    def test_unknown_context_keeps_the_128k_bucket(self) -> None:
        jobs = expand_jobs(
            [JobSpec("model", "ruler", "1")],
            ContextClient(None),
        )

        self.assertEqual(
            [job.variant for job in jobs],
            [context_label(bucket) for bucket in CONTEXT_BUCKETS],
        )

    def test_decimal_128k_served_limit_keeps_the_128k_bucket(self) -> None:
        jobs = expand_jobs(
            [JobSpec("model", "ruler", "1")],
            ContextClient(128000),
        )

        self.assertEqual(CONTEXT_BUCKETS[-1], 128000)
        self.assertEqual(context_label(CONTEXT_BUCKETS[-1]), "128k")
        self.assertEqual(len(jobs), 6)
        self.assertEqual(jobs[-1].variant, "128k")

    def test_engine_emits_one_fractional_score_per_bucket(self) -> None:
        client = ContextClient(8192)
        ruler = benchmark("ruler")
        with patch.object(ruler, "evaluate", return_value=0.5):
            results = Engine(
                client=client,
                jobs=[JobSpec("model", "ruler", "1")],
            ).run()

        self.assertEqual([result["context_label"] for result in results], ["4k", "8k"])
        self.assertEqual([result["score"] for result in results], [50.0, 50.0])
        self.assertEqual([result["passed"] for result in results], [0, 0])
        self.assertTrue(
            all(result["include_in_overall"] is False for result in results)
        )
        self.assertTrue(all(result["tasks"][0]["score"] == 50.0 for result in results))


class ContextDiscoveryTests(unittest.TestCase):
    def test_nested_context_hints_use_the_smallest_served_limit(self) -> None:
        payload = {
            "model": {"max_position_embeddings": 131072},
            "default_generation_settings": {"n_ctx": "32768"},
        }

        self.assertEqual(_context_length_hint(payload), 32768)

    def test_openai_props_context_is_cached(self) -> None:
        client = InferenceClient("http://local", "openai")
        client.provider = "openai"
        client._models_by_name = {"model": {"owned_by": "llamacpp"}}
        response = Mock()
        response.json.return_value = {"default_generation_settings": {"n_ctx": 65536}}

        with patch.object(client, "_request", return_value=response) as request:
            self.assertEqual(client.context_length("model"), 65536)
            self.assertEqual(client.context_length("model"), 65536)

        request.assert_called_once()


if __name__ == "__main__":
    unittest.main()
