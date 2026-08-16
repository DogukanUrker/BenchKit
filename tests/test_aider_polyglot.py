"""Tests for Aider Polyglot task discovery and workspace verification."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchkit.benchmarks.aider_polyglot import (
    _EXPECTED_COUNTS,
    _LANGUAGES,
    _TEST_COMMANDS,
    AiderPolyglot,
)
from benchkit.benchmarks.base import Task
from benchkit.cli import _headless_jobs, _parse_args
from benchkit.sandbox import AIDER_PI_DOCKERFILE, AIDER_POLYGLOT_COMMIT


class FakeEnvironment:
    def __init__(self, results: list[subprocess.CompletedProcess[str]] | None = None):
        self.commands: list[list[str]] = []
        self.calls: list[tuple[list[str], dict]] = []
        self.workdir = "/workspace"
        self.results = list(results or [])

    def exec(self, command, **kwargs):
        self.commands.append(command)
        self.calls.append((command, kwargs))
        if self.results:
            return self.results.pop(0)
        return subprocess.CompletedProcess(command, 0, "", "")


class AiderPolyglotTests(unittest.TestCase):
    def test_cli_selects_language_alias_and_optional_slice(self) -> None:
        args = _parse_args(
            [
                "--headless",
                "--models",
                "model",
                "--benchmarks",
                "aider-polyglot:js:5",
                "--harness",
                "pi",
            ]
        )

        jobs = _headless_jobs(args, ["model"])

        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].variant, "javascript")
        self.assertEqual(jobs[0].slice_spec, "5")

    def test_cli_accepts_every_short_language_selector(self) -> None:
        args = _parse_args(
            [
                "--headless",
                "--models",
                "model",
                "--benchmarks",
                "aider-polyglot:cpp,aider-polyglot:go,aider-polyglot:java,"
                "aider-polyglot:js,aider-polyglot:py,aider-polyglot:rs",
                "--harness",
                "pi",
            ]
        )

        jobs = _headless_jobs(args, ["model"])

        self.assertEqual(
            [job.variant for job in jobs],
            ["cpp", "go", "java", "javascript", "python", "rust"],
        )

    def test_loads_all_225_pinned_tasks_and_language_variants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for language, count in _EXPECTED_COUNTS.items():
                exercises = root / language / "exercises" / "practice"
                for number in range(count):
                    (exercises / f"exercise-{number:03d}").mkdir(parents=True)

            with patch(
                "benchkit.benchmarks.aider_polyglot._dataset_root",
                return_value=root,
            ):
                benchmark = AiderPolyglot()
                tasks = benchmark.load_tasks()

        self.assertEqual(len(tasks), 225)
        self.assertEqual(
            benchmark.variants(None, "model"),
            ("cpp", "go", "java", "javascript", "python", "rust"),
        )
        python = benchmark.tasks_for_variant(tasks, "python")
        self.assertEqual(len(python), 34)
        self.assertTrue(all(task.id.startswith("python/") for task in python))

    def test_workspace_is_copied_and_committed_before_agent_starts(self) -> None:
        environment = FakeEnvironment()
        task = Task(
            "go/book-store",
            "",
            {"language": "go", "exercise": "book-store"},
        )

        AiderPolyglot().prepare_workspace(task, environment)

        self.assertEqual(environment.workdir, "/workspace/book-store")
        self.assertEqual(environment.commands[0], ["mkdir", "-p", environment.workdir])
        self.assertEqual(
            environment.commands[1],
            [
                "cp",
                "-a",
                "/opt/aider-polyglot/go/exercises/practice/book-store/.",
                "/workspace/book-store/",
            ],
        )
        self.assertIn(
            ["git", "-C", "/workspace/book-store", "add", "."],
            environment.commands,
        )

    def test_cpp_workspace_preserves_exercise_name_for_cmake(self) -> None:
        environment = FakeEnvironment()
        task = Task("cpp/allergies", "", {"language": "cpp", "exercise": "allergies"})

        benchmark = AiderPolyglot()
        benchmark.prepare_workspace(task, environment)
        environment.results = [
            subprocess.CompletedProcess(["git", "diff"], 0, "", ""),
            subprocess.CompletedProcess(["git", "diff"], 0, "", ""),
            subprocess.CompletedProcess(["cmake"], 0, "tests passed", ""),
        ]
        result = benchmark.verify_workspace(task, environment)

        self.assertTrue(result.passed)
        self.assertEqual(environment.workdir, "/workspace/allergies")
        test_command, test_options = environment.calls[-1]
        self.assertEqual(test_command, _TEST_COMMANDS["cpp"])
        self.assertEqual(test_options["workdir"], "/workspace/allergies")

    def test_cpp_verifier_runs_all_tests_with_an_in_container_deadline(self) -> None:
        command = _TEST_COMMANDS["cpp"]

        self.assertEqual(command[:2], ["bash", "-lc"])
        self.assertIn("timeout --signal=KILL 120s", command[2])
        self.assertIn("-DEXERCISM_RUN_ALL_TESTS=ON", command[2])

    def test_workspace_verifier_returns_tests_and_patch(self) -> None:
        environment = FakeEnvironment(
            [
                subprocess.CompletedProcess(
                    ["git", "diff"], 0, "diff --git a/x b/x\n", ""
                ),
                subprocess.CompletedProcess(["git", "diff"], 0, "", ""),
                subprocess.CompletedProcess(["go", "test"], 0, "ok\n", ""),
            ]
        )
        task = Task("go/pov", "", {"language": "go", "exercise": "pov"})

        result = AiderPolyglot().verify_workspace(task, environment)

        self.assertTrue(result.passed)
        self.assertEqual(result.details["test_exit_code"], 0)
        self.assertIn("diff --git", result.details["patch"])

    def test_modified_tests_are_restored_before_verification(self) -> None:
        environment = FakeEnvironment(
            [
                subprocess.CompletedProcess(["git", "diff"], 0, "patch", ""),
                subprocess.CompletedProcess(
                    ["git", "diff"], 0, "pov.go\npov_test.go\n", ""
                ),
                subprocess.CompletedProcess(["git", "checkout"], 0, "", ""),
                subprocess.CompletedProcess(["go", "test"], 1, "failed", ""),
            ]
        )
        task = Task("go/pov", "", {"language": "go", "exercise": "pov"})

        result = AiderPolyglot().verify_workspace(task, environment)

        self.assertFalse(result.passed)
        self.assertEqual(result.details["test_files_restored"], ["pov_test.go"])
        self.assertIn(
            [
                "git",
                "-C",
                "/workspace/pov",
                "checkout",
                "HEAD",
                "--",
                "pov_test.go",
            ],
            environment.commands,
        )

    def test_full_image_pins_dataset_and_contains_every_toolchain(self) -> None:
        self.assertIn(AIDER_POLYGLOT_COMMIT, AIDER_PI_DOCKERFILE)
        for executable in (
            "cmake",
            "go1.21.5",
            "openjdk-17-jdk",
            "libboost-date-time-dev",
            "rustup.rs",
            "jest",
        ):
            self.assertIn(executable, AIDER_PI_DOCKERFILE)
        self.assertEqual(tuple(_EXPECTED_COUNTS), _LANGUAGES)
        self.assertIn(
            "mkdir -p /workspace /home/node/.pi/agent /opt/go",
            AIDER_PI_DOCKERFILE,
        )
        self.assertIn(
            "gradle testClasses --no-daemon",
            AIDER_PI_DOCKERFILE,
        )

    def test_aider_sandbox_allows_the_bank_account_thread_suite(self) -> None:
        self.assertEqual(AiderPolyglot().pi_image().pids_limit, 2048)


if __name__ == "__main__":
    unittest.main()
