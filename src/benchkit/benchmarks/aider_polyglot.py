"""Aider Polyglot exercises executed by the stock Pi coding agent."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
from pathlib import Path

from benchkit.benchmarks.base import Task
from benchkit.evaluation import EvaluationResult
from benchkit.sandbox import (
    AIDER_POLYGLOT_COMMIT,
    DockerTaskEnvironment,
    aider_pi_image,
)

_REPOSITORY = "https://github.com/Aider-AI/polyglot-benchmark.git"
_LANGUAGES = ("cpp", "go", "java", "javascript", "python", "rust")
_LANGUAGE_ALIASES = {
    "cpp": "cpp",
    "c++": "cpp",
    "go": "go",
    "java": "java",
    "js": "javascript",
    "javascript": "javascript",
    "py": "python",
    "python": "python",
    "rs": "rust",
    "rust": "rust",
}
_EXPECTED_COUNTS = {
    "cpp": 26,
    "go": 39,
    "java": 47,
    "javascript": 49,
    "python": 34,
    "rust": 30,
}
_TEST_COMMANDS = {
    "cpp": [
        "bash",
        "-lc",
        "timeout --signal=KILL 120s bash -lc "
        "'cmake -S . -B build -DEXERCISM_RUN_ALL_TESTS=ON "
        "&& cmake --build build --parallel 2'",
    ],
    "go": ["go", "test", "./..."],
    "java": ["gradle", "test", "--offline", "--no-daemon"],
    "javascript": ["npm", "test", "--", "--runInBand"],
    "python": ["python3", "-m", "unittest", "discover", "-v", "-p", "*_test.py"],
    "rust": ["cargo", "test", "--offline"],
}
_SAFE_NAME = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_CACHE_LOCK = threading.Lock()


def _cache_root() -> Path:
    configured = os.environ.get("BENCHKIT_AIDER_DATASET")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "benchkit" / "aider-polyglot"


def _run_git(args: list[str], *, timeout: float = 300) -> None:
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("Aider Polyglot requires git to download its task corpus")
    completed = subprocess.run(
        [git, *args],
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()[-2000:]
        raise RuntimeError(f"could not prepare Aider Polyglot dataset: {detail}")


def _dataset_root() -> Path:
    """Return a pinned upstream checkout, downloading it once when necessary."""
    root = _cache_root()
    marker = root / ".git"
    with _CACHE_LOCK:
        if not marker.exists():
            root.parent.mkdir(parents=True, exist_ok=True)
            _run_git(["clone", _REPOSITORY, str(root)])
        completed = subprocess.run(
            [shutil.which("git") or "git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            timeout=30,
        )
        if completed.returncode or completed.stdout.strip() != AIDER_POLYGLOT_COMMIT:
            _run_git(["-C", str(root), "fetch", "origin", AIDER_POLYGLOT_COMMIT])
            _run_git(["-C", str(root), "checkout", "--detach", AIDER_POLYGLOT_COMMIT])
    return root


class AiderPolyglot:
    """Run the complete Aider exercise corpus through Pi's native tools."""

    name = "aider-polyglot"
    task_count = sum(_EXPECTED_COUNTS.values())
    variant_count = len(_LANGUAGES)
    workspace_task = True
    evaluation_activity = "running exercise tests in the Pi sandbox"

    def variants(self, _client: object, _model: str) -> tuple[str, ...]:
        return _LANGUAGES

    def resolve_variant(self, selector: str) -> str | None:
        """Resolve a CLI language selector to its canonical variant name."""
        return _LANGUAGE_ALIASES.get(selector.strip().lower())

    def variant_task_count(self, variant: str) -> int:
        return _EXPECTED_COUNTS[variant]

    def load_tasks(self) -> list[Task]:
        root = _dataset_root()
        tasks: list[Task] = []
        for language in _LANGUAGES:
            exercises = root / language / "exercises" / "practice"
            names = sorted(path.name for path in exercises.iterdir() if path.is_dir())
            expected = _EXPECTED_COUNTS[language]
            if len(names) != expected:
                raise RuntimeError(
                    f"Aider Polyglot {language} checkout has {len(names)} tasks; "
                    f"expected {expected} at {AIDER_POLYGLOT_COMMIT[:12]}"
                )
            tasks.extend(
                Task(
                    id=f"{language}/{name}",
                    prompt="",
                    metadata={"language": language, "exercise": name},
                )
                for name in names
            )
        return tasks

    def tasks_for_variant(self, tasks: list[Task], variant: str) -> list[Task]:
        return [task for task in tasks if task.metadata["language"] == variant]

    def build_prompt(self, task: Task) -> str:
        language = task.metadata["language"]
        exercise = task.metadata["exercise"]
        return (
            f"Solve the Aider Polyglot {language} exercise `{exercise}` in the "
            "current workspace. Read `.docs/instructions.md` and any appendices, "
            "inspect the starter code and tests, then edit the implementation so "
            "all tests pass. Use the terminal to run the tests yourself, but bound "
            "every test command with `timeout --signal=KILL 120s` so a broken "
            "implementation cannot deadlock the agent. Do not change the tests or "
            "exercise instructions."
        )

    def evaluate(self, _task: Task, _response: str) -> bool:
        raise RuntimeError("Aider Polyglot must be evaluated inside its workspace")

    def pi_image(self):
        return aider_pi_image()

    def prepare_workspace(
        self,
        task: Task,
        environment: DockerTaskEnvironment,
    ) -> None:
        language, exercise = self._coordinates(task)
        workspace = f"/workspace/{exercise}"
        environment.workdir = workspace
        source = f"/opt/aider-polyglot/{language}/exercises/practice/{exercise}/."
        environment.exec(["mkdir", "-p", workspace])
        environment.exec(["cp", "-a", source, f"{workspace}/"])
        environment.exec(["git", "init", "-q", workspace])
        environment.exec(["git", "-C", workspace, "config", "user.name", "BenchKit"])
        environment.exec(
            [
                "git",
                "-C",
                workspace,
                "config",
                "user.email",
                "benchkit@localhost",
            ]
        )
        environment.exec(["git", "-C", workspace, "add", "."])
        environment.exec(["git", "-C", workspace, "commit", "-q", "-m", "baseline"])

    def verify_workspace(
        self,
        task: Task,
        environment: DockerTaskEnvironment,
    ) -> EvaluationResult:
        language, exercise = self._coordinates(task)
        workspace = f"/workspace/{exercise}"
        command = _TEST_COMMANDS[language]
        try:
            patch = environment.exec(
                ["git", "-C", workspace, "diff", "--no-ext-diff"],
                timeout=30,
                check=False,
            ).stdout
            changed = environment.exec(
                ["git", "-C", workspace, "diff", "--name-only"],
                timeout=30,
                check=False,
            ).stdout.splitlines()
            changed_tests = [
                path for path in changed if self._is_test_file(language, path)
            ]
            if changed_tests:
                environment.exec(
                    [
                        "git",
                        "-C",
                        workspace,
                        "checkout",
                        "HEAD",
                        "--",
                        *changed_tests,
                    ],
                    timeout=30,
                )
            completed = environment.exec(
                command,
                workdir=workspace,
                timeout=180,
                check=False,
            )
        except Exception as exc:
            return EvaluationResult(
                0.0,
                error=f"workspace verification failed: {type(exc).__name__}: {exc}",
            )
        output = (completed.stdout + completed.stderr).strip()
        passed = completed.returncode == 0
        feedback = ""
        if not passed:
            feedback = (
                "The exercise tests still fail. Review this test output, fix the "
                f"workspace, and run the tests again:\n\n{output[-6000:]}"
            )
        return EvaluationResult(
            score=1.0 if passed else 0.0,
            feedback=feedback,
            details={
                "language": language,
                "test_command": command,
                "test_exit_code": completed.returncode,
                "test_output": output[-8000:],
                "test_output_truncated": len(output) > 8000,
                "test_files_restored": changed_tests,
                "patch": patch[-16000:],
                "patch_truncated": len(patch) > 16000,
            },
        )

    @staticmethod
    def _coordinates(task: Task) -> tuple[str, str]:
        language = str(task.metadata.get("language") or "")
        exercise = str(task.metadata.get("exercise") or "")
        if language not in _LANGUAGES or not _SAFE_NAME.fullmatch(exercise):
            raise ValueError(f"invalid Aider Polyglot task coordinates: {task.id}")
        return language, exercise

    @staticmethod
    def _is_test_file(language: str, path: str) -> bool:
        name = Path(path).name
        if language == "cpp":
            return name.endswith("_test.cpp") or path.startswith("test/")
        if language == "go":
            return name.endswith("_test.go")
        if language == "java":
            return path.startswith("src/test/")
        if language == "javascript":
            return name.endswith(".spec.js")
        if language == "python":
            return name.endswith("_test.py")
        return language == "rust" and path.startswith("tests/")

    def result_metadata(self, variant: str | None) -> dict:
        return {
            "dataset_commit": AIDER_POLYGLOT_COMMIT,
            "language": variant,
            "protocol": "pi-agent",
            "official_aider_comparable": False,
        }
