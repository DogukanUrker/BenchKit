"""Tests for deterministic, paired choice-order robustness runs."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from benchkit.benchmarks.base import Task
from benchkit.cli import _headless_jobs, _parse_args
from benchkit.engine import Engine, JobSpec, benchmark, prompt_for, tasks_for
from benchkit.perturbations import (
    CHOICE_ORDER,
    CHOICE_ORDER_BENCHMARKS,
    annotate_robustness,
    perturb_task,
    perturbations_for,
)


def _generation(response: str) -> dict:
    return {
        "thinking": "",
        "response": response,
        "trace_status": "unavailable",
        "tok_s": 10.0,
        "eval_count": 1,
        "eval_duration_ns": 100_000_000,
        "response_time_s": 0.1,
        "done_reason": "stop",
        "timed_out": False,
        "cancelled": False,
    }


class PromptClient:
    timeout = 1.0
    label = "test"
    host = "test://local"

    def __init__(self, answers: dict[str, str]) -> None:
        self.answers = answers

    def max_parallel_requests(self, _model: str) -> int:
        return 1

    def generate(self, _model: str, prompt: str, **_kwargs: object) -> dict:
        return _generation(self.answers[prompt])

    def unload_model(self, _model: str) -> None:
        return None


class ChoiceOrderTests(unittest.TestCase):
    def test_supported_benchmarks_are_explicit(self) -> None:
        expected = {
            "arc",
            "gpqa",
            "hellaswag",
            "mmlu",
            "mmlu-pro",
            "openbookqa",
            "piqa",
            "truthfulqa",
            "winogrande",
        }

        self.assertEqual(set(CHOICE_ORDER_BENCHMARKS), expected)
        for key in expected:
            self.assertEqual(perturbations_for(key), (CHOICE_ORDER,))
        self.assertEqual(perturbations_for("boolq"), ())

    def test_permutation_is_deterministic_and_does_not_mutate_source(self) -> None:
        task = Task(
            id="example/1",
            prompt="Question?",
            metadata={"choices": ["one", "two", "three", "four"], "answer": "B"},
        )

        first = perturb_task("arc", task, CHOICE_ORDER, 42)
        second = perturb_task("arc", task, CHOICE_ORDER, 42)

        self.assertEqual(first.prompt_task, second.prompt_task)
        self.assertEqual(first.details, second.details)
        self.assertIs(first.prompt_task, first.evaluation_task)
        self.assertIsNot(first.prompt_task, task)
        self.assertEqual(task.metadata["choices"], ["one", "two", "three", "four"])
        self.assertEqual(task.metadata["answer"], "B")
        self.assertCountEqual(
            first.prompt_task.metadata["choices"],
            task.metadata["choices"],
        )
        self.assertNotEqual(first.prompt_task.metadata["answer"], "B")
        self.assertNotEqual(first.details["choice_order"], ["A", "B", "C", "D"])

    def test_binary_choice_always_swaps_answer_position(self) -> None:
        for answer in ("A", "B"):
            with self.subTest(answer=answer):
                task = Task(
                    id=f"binary/{answer}",
                    prompt="Question?",
                    metadata={"choices": ["left", "right"], "answer": answer},
                )

                case = perturb_task("piqa", task, CHOICE_ORDER, 7)

                self.assertEqual(case.details["choice_order"], ["B", "A"])
                self.assertNotEqual(case.evaluation_task.metadata["answer"], answer)

    def test_truthfulqa_only_reorders_four_visible_choices(self) -> None:
        choices = [f"choice-{index}" for index in range(7)]
        task = Task(
            id="TruthfulQA/0",
            prompt="Question?",
            metadata={"choices": choices, "answer": "A"},
        )

        case = perturb_task("truthfulqa", task, CHOICE_ORDER, 42)
        changed = case.prompt_task.metadata["choices"]

        self.assertCountEqual(changed[:4], choices[:4])
        self.assertEqual(changed[4:], choices[4:])
        self.assertEqual(len(case.details["choice_order"]), 4)
        self.assertNotEqual(case.prompt_task.metadata["answer"], "A")

    def test_unsupported_benchmark_is_rejected(self) -> None:
        task = Task(
            id="BoolQ/0",
            prompt="Question?",
            metadata={"choices": ["yes", "no"], "answer": "A"},
        )

        with self.assertRaisesRegex(ValueError, "does not support"):
            perturb_task("boolq", task, CHOICE_ORDER, 42)


class RobustnessMetricsTests(unittest.TestCase):
    def test_pairs_tasks_and_reports_regressions_and_recoveries(self) -> None:
        clean = {
            "model": "model",
            "benchmark": "arc",
            "benchmark_label": "arc",
            "variant": None,
            "slice": "3",
            "tasks": [
                {"task_id": "a", "passed": True, "score": 100.0},
                {"task_id": "b", "passed": False, "score": 0.0},
                {"task_id": "c", "passed": True, "score": 100.0},
            ],
        }
        perturbed = {
            "model": "model",
            "benchmark": "arc",
            "benchmark_label": "arc · choice-order",
            "variant": None,
            "slice": "3",
            "perturbation": CHOICE_ORDER,
            "perturbation_seed": 42,
            "tasks": [
                {"task_id": "a", "passed": False, "score": 0.0},
                {"task_id": "b", "passed": True, "score": 100.0},
                {"task_id": "c", "passed": True, "score": 100.0},
            ],
        }

        annotate_robustness([clean, perturbed])

        self.assertEqual(perturbed["paired_total"], 3)
        self.assertEqual(perturbed["baseline_score"], 66.7)
        self.assertEqual(perturbed["perturbed_score"], 66.7)
        self.assertEqual(perturbed["score_delta_pp"], 0.0)
        self.assertEqual(perturbed["regressions"], 1)
        self.assertEqual(perturbed["regression_rate"], 33.3)
        self.assertEqual(perturbed["recoveries"], 1)


class PerturbationJobTests(unittest.TestCase):
    def test_cli_adds_clean_and_perturbed_jobs(self) -> None:
        args = _parse_args(
            [
                "--headless",
                "--models",
                "model",
                "--benchmarks",
                "arc:2",
                "--perturbation",
                CHOICE_ORDER,
                "--perturbation-seed",
                "9",
            ]
        )

        jobs = _headless_jobs(args, ["model"])

        self.assertEqual(len(jobs), 2)
        self.assertIsNone(jobs[0].perturbation)
        self.assertEqual(jobs[1].perturbation, CHOICE_ORDER)
        self.assertEqual(jobs[1].perturbation_seed, 9)
        self.assertEqual(jobs[1].benchmark_label, "arc · choice-order")
        self.assertNotEqual(jobs[0].key, jobs[1].key)

    def test_cli_rejects_unsupported_selection_before_inference(self) -> None:
        args = _parse_args(
            [
                "--headless",
                "--models",
                "model",
                "--benchmarks",
                "boolq:1",
                "--perturbation",
                CHOICE_ORDER,
            ]
        )

        with self.assertRaises(SystemExit):
            _headless_jobs(args, ["model"])

    def test_tui_overall_score_excludes_perturbed_jobs(self) -> None:
        from benchkit.tui.screens.run import _include_in_overall

        self.assertTrue(_include_in_overall(JobSpec("model", "arc", "1")))
        self.assertFalse(
            _include_in_overall(JobSpec("model", "arc", "1", perturbation=CHOICE_ORDER))
        )

    def test_engine_scores_remapped_answer_and_adds_paired_metrics(self) -> None:
        bench = benchmark("arc")
        task = tasks_for("arc")[0]
        case = perturb_task("arc", task, CHOICE_ORDER, 42)
        clean_prompt = prompt_for(bench, task, object(), "model")
        perturbed_prompt = prompt_for(bench, case.prompt_task, object(), "model")
        client = PromptClient(
            {
                clean_prompt: task.metadata["answer"],
                perturbed_prompt: case.evaluation_task.metadata["answer"],
            }
        )

        results = Engine(
            client=client,
            jobs=[
                JobSpec("model", "arc", "1"),
                JobSpec(
                    "model",
                    "arc",
                    "1",
                    perturbation=CHOICE_ORDER,
                    perturbation_seed=42,
                ),
            ],
        ).run()

        self.assertEqual([result["score"] for result in results], [100.0, 100.0])
        self.assertEqual(results[1]["score_delta_pp"], 0.0)
        self.assertEqual(results[1]["regressions"], 0)
        self.assertFalse(results[1]["include_in_overall"])
        self.assertEqual(
            results[1]["tasks"][0]["perturbation"]["perturbed_answer"],
            case.evaluation_task.metadata["answer"],
        )
        self.assertNotEqual(
            results[0]["tasks"][0]["prompt"],
            results[1]["tasks"][0]["prompt"],
        )


class SetupPerturbationTUITests(unittest.IsolatedAsyncioTestCase):
    async def test_setup_builds_paired_jobs_and_validates_support(self) -> None:
        from textual.widgets import Checkbox, Input, Static

        import benchkit.tui.app as tui_app_module
        from benchkit.demo import DemoClient
        from benchkit.tui.app import BenchKitApp
        from benchkit.tui.screens.setup import SetupScreen

        class TestSetupScreen(SetupScreen):
            def _count_tasks(self) -> None:
                return None

        class Harness(BenchKitApp):
            CSS_PATH = Path(tui_app_module.__file__).with_name("app.tcss")

            def __init__(self) -> None:
                super().__init__(demo=True)
                self.client = DemoClient(speed=1000)
                self.models = self.client.list_models()
                self.started_jobs: list[JobSpec] | None = None

            def on_mount(self) -> None:
                self.push_screen(TestSetupScreen())

            def start_run(
                self, jobs: list[JobSpec], *, force_unload: bool = False
            ) -> None:
                self.started_jobs = jobs

        app = Harness()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SetupScreen)
            screen.selected_models = {"demo-mini:3b"}
            screen.selected_benchmarks = {"arc"}
            screen.counts["arc"] = 1
            screen.query_one("#global-limit", Input).value = "2"

            seed = screen.query_one("#perturbation-seed", Input)
            self.assertTrue(seed.disabled)
            await pilot.press("p")
            await pilot.pause()
            self.assertTrue(screen.query_one("#choice-order", Checkbox).value)
            self.assertFalse(seed.disabled)
            seed.value = "7"
            screen._refresh_summary()

            summary = str(screen.query_one("#plan-summary", Static).render())
            self.assertIn("2 runs", summary)
            self.assertIn("4 tasks total", summary)
            self.assertIn("clean + choice-order", summary)

            with patch.object(app.client, "prime") as prime:
                screen.action_start()

            assert app.started_jobs is not None
            self.assertEqual(len(app.started_jobs), 2)
            self.assertIsNone(app.started_jobs[0].perturbation)
            self.assertEqual(app.started_jobs[1].perturbation, CHOICE_ORDER)
            self.assertEqual(app.started_jobs[1].perturbation_seed, 7)
            prime.assert_called_once_with(["arc"], perturbations=[(CHOICE_ORDER, 7)])

            screen.selected_benchmarks = {"boolq"}
            screen.counts["boolq"] = 1
            screen._refresh_summary()
            summary = str(screen.query_one("#plan-summary", Static).render())
            self.assertIn("Choice-order unsupported for: boolq", summary)
            self.assertIsNone(screen._build_jobs())


if __name__ == "__main__":
    unittest.main()
