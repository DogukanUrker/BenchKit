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
    AiderPolyglot,
)
from benchkit.benchmarks.base import Task
from benchkit.sandbox import AIDER_PI_DOCKERFILE, AIDER_POLYGLOT_COMMIT


class FakeEnvironment:
    def __init__(self, results: list[subprocess.CompletedProcess[str]] | None = None):
        self.commands: list[list[str]] = []
        self.results = list(results or [])

    def exec(self, command, **_kwargs):
        self.commands.append(command)
        if self.results:
            return self.results.pop(0)
        return subprocess.CompletedProcess(command, 0, "", "")


class AiderPolyglotTests(unittest.TestCase):
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

        self.assertEqual(
            environment.commands[0],
            [
                "cp",
                "-a",
                "/opt/aider-polyglot/go/exercises/practice/book-store/.",
                "/workspace/",
            ],
        )
        self.assertIn(["git", "-C", "/workspace", "add", "."], environment.commands)

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
                "/workspace",
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
            "rustup.rs",
            "jest",
        ):
            self.assertIn(executable, AIDER_PI_DOCKERFILE)
        self.assertEqual(tuple(_EXPECTED_COUNTS), _LANGUAGES)
        self.assertIn(
            "mkdir -p /workspace /home/node/.pi/agent /opt/go",
            AIDER_PI_DOCKERFILE,
        )


if __name__ == "__main__":
    unittest.main()
