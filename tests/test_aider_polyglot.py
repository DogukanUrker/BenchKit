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
        sanitize = environment.commands[2]
        self.assertEqual(sanitize[:3], ["find", environment.workdir, "-depth"])
        for forbidden in (
            ".meta",
            ".approaches",
            "*example*",
            "*reference*",
            "*proof*",
        ):
            self.assertIn(forbidden, sanitize)
        self.assertIn(
            ["rm", "-rf", f"{environment.workdir}/.git"], environment.commands
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
        test_command, test_options = next(
            call for call in environment.calls if call[0] == _TEST_COMMANDS["cpp"]
        )
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

    def test_disabled_java_tests_are_enabled_after_pristine_restore(self) -> None:
        environment = FakeEnvironment(
            [
                subprocess.CompletedProcess(["git", "diff"], 0, "patch", ""),
                subprocess.CompletedProcess(
                    ["git", "diff"],
                    0,
                    "src/main/java/WordProblemSolver.java\n"
                    "src/test/java/WordProblemSolverTest.java\n",
                    "",
                ),
                subprocess.CompletedProcess(["git", "checkout"], 0, "", ""),
                subprocess.CompletedProcess(["sed"], 0, "", ""),
                subprocess.CompletedProcess(["gradle", "test"], 0, "tests passed", ""),
            ]
        )
        task = Task(
            "java/wordy",
            "",
            {"language": "java", "exercise": "wordy"},
        )

        result = AiderPolyglot().verify_workspace(task, environment)

        self.assertTrue(result.passed)
        self.assertTrue(
            any(
                "@Disabled" in command[2]
                for command in environment.commands
                if command[:2] == ["bash", "-lc"]
            )
        )
        self.assertIn(_TEST_COMMANDS["java"], environment.commands)

    def test_javascript_and_rust_verifiers_enable_skipped_tests(self) -> None:
        for language, exercise, marker in (
            ("javascript", "wordy", "xtest"),
            ("rust", "wordy", "ignore"),
        ):
            with self.subTest(language=language):
                environment = FakeEnvironment(
                    [
                        subprocess.CompletedProcess(["git", "diff"], 0, "patch", ""),
                        subprocess.CompletedProcess(["git", "diff"], 0, "", ""),
                        subprocess.CompletedProcess(["sed"], 0, "", ""),
                        subprocess.CompletedProcess(["test"], 0, "tests passed", ""),
                    ]
                )
                task = Task(
                    f"{language}/{exercise}",
                    "",
                    {"language": language, "exercise": exercise},
                )

                result = AiderPolyglot().verify_workspace(task, environment)

                self.assertTrue(result.passed)
                self.assertTrue(
                    any(
                        marker in command[2]
                        for command in environment.commands
                        if command[:2] == ["bash", "-lc"]
                    )
                )
                self.assertIn(_TEST_COMMANDS[language], environment.commands)

    def test_counter_keeps_authored_tests_and_checks_every_implementation(self) -> None:
        environment = FakeEnvironment(
            [
                subprocess.CompletedProcess(["git", "diff"], 0, "test patch", ""),
                subprocess.CompletedProcess(
                    ["git", "diff"], 0, "counter_test.go\n", ""
                ),
                subprocess.CompletedProcess(
                    ["bash"], 0, "implementation checks passed", ""
                ),
            ]
        )
        task = Task(
            "go/counter",
            "",
            {"language": "go", "exercise": "counter"},
        )

        result = AiderPolyglot().verify_workspace(task, environment)

        self.assertTrue(result.passed)
        self.assertEqual(result.details["test_files_restored"], [])
        self.assertFalse(any("checkout" in command for command in environment.commands))
        self.assertTrue(
            any(
                "COUNTER_IMPL=4" in command[2]
                for command in environment.commands
                if command[:2] == ["bash", "-lc"]
            )
        )

    def test_guard_hit_marks_passing_dirty_workspace_contaminated(self) -> None:
        hit = (
            '{"tool":"read","arguments":{"path":".meta/example.go"},'
            '"matches":[".meta/example.go"]}\n'
        )
        environment = FakeEnvironment(
            [
                subprocess.CompletedProcess(["git", "diff"], 0, "patch", ""),
                subprocess.CompletedProcess(["git", "diff"], 0, "", ""),
                subprocess.CompletedProcess(["go", "test"], 0, "ok", ""),
                subprocess.CompletedProcess(["cat"], 0, hit, ""),
            ]
        )
        task = Task("go/pov", "", {"language": "go", "exercise": "pov"})

        result = AiderPolyglot().verify_workspace(task, environment)

        self.assertFalse(result.passed)
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.details["contamination_verdict"], "CONTAMINATED")
        self.assertEqual(
            result.details["answer_key_guard_hits"][0]["matches"],
            [".meta/example.go"],
        )

    def test_counter_prompt_describes_test_authoring_protocol(self) -> None:
        task = Task(
            "go/counter",
            "",
            {"language": "go", "exercise": "counter"},
        )

        prompt = AiderPolyglot().build_prompt(task)

        self.assertIn("test-authoring exercise", prompt)
        self.assertIn("counter_test.go", prompt)

    def test_full_image_pins_dataset_and_contains_every_toolchain(self) -> None:
        self.assertIn(AIDER_POLYGLOT_COMMIT, AIDER_PI_DOCKERFILE)
        for executable in (
            "cmake",
            "go1.21.5",
            "openjdk-17-jdk",
            "libboost-date-time-dev",
            "rustup.rs",
            "jest@29.7.0",
        ):
            self.assertIn(executable, AIDER_PI_DOCKERFILE)
        self.assertEqual(tuple(_EXPECTED_COUNTS), _LANGUAGES)
        self.assertIn(
            "mkdir -p /workspace /home/node/.pi/agent /opt/go",
            AIDER_PI_DOCKERFILE,
        )
        self.assertIn(
            "gradle test --test-dry-run --no-daemon",
            AIDER_PI_DOCKERFILE,
        )
        self.assertIn("-name Cargo-example.toml", AIDER_PI_DOCKERFILE)
        self.assertIn(
            'cargo fetch --manifest-path "$cache_dir/Cargo.toml"',
            AIDER_PI_DOCKERFILE,
        )
        self.assertIn("answer_key_guard.ts", AIDER_PI_DOCKERFILE)
        self.assertIn("rm -rf /opt/aider-polyglot/.git", AIDER_PI_DOCKERFILE)

    def test_aider_sandbox_allows_the_bank_account_thread_suite(self) -> None:
        image = AiderPolyglot().pi_image()
        self.assertEqual(image.pids_limit, 2048)
        self.assertTrue(image.transient)
        self.assertTrue(image.answer_key_guard)

    def test_java_verifier_uses_utf8_for_unicode_exercises(self) -> None:
        self.assertEqual(
            _TEST_COMMANDS["java"],
            [
                "gradle",
                "-Dfile.encoding=UTF-8",
                "test",
                "--offline",
                "--no-daemon",
            ],
        )


if __name__ == "__main__":
    unittest.main()
