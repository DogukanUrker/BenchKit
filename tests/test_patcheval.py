"""Tests for PatchEval dataset validation and hidden grading boundaries."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from benchkit.benchmarks.patcheval import (
    _GENERIC_REPAIR,
    PatchEval,
    _trusted_patch,
)
from benchkit.engine import Engine
from benchkit.sandbox import (
    PI_VERSION,
    LatestPiImage,
    PatchEvalRuntimeRecipe,
    SandboxError,
    patcheval_pi_image,
)

_RUNTIME_RECIPE = {
    "schema_version": 1,
    "base_image": "ghcr.io/astral-sh/uv:0.12.1-python3.14-trixie-slim",
    "sync_command": ["uv", "sync", "--frozen", "--no-install-project"],
    "bootstrap_command": ["uv", "pip", "install", "setuptools==80.9.0"],
    "environment": ["UV_OFFLINE=1", "PYTHONHASHSEED=0"],
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dataset(root: Path, *, validated: bool = True) -> dict:
    source_root = root / "source"
    source_root.mkdir()
    (source_root / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    tests = source_root / "tests"
    tests.mkdir()
    (tests / "test_existing.py").write_text("def test_old(): pass\n", encoding="utf-8")
    archive = root / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for path in sorted(source_root.rglob("*")):
            handle.add(path, arcname=path.relative_to(source_root), recursive=False)
    hidden = root / "hidden.patch"
    hidden.write_text("hidden grader patch\n", encoding="utf-8")
    record = {
        "id": "owner-repo-123",
        "repository": "owner/repo",
        "issue_title": "Handle empty input",
        "issue_body": "Calling parse with an empty value should return None.",
        "runtime_recipe": _RUNTIME_RECIPE,
        "source_archive": archive.name,
        "hidden_test_patch": hidden.name,
        "source_sha256": _digest(archive),
        "hidden_test_sha256": _digest(hidden),
        "fail_to_pass_command": ["python3", "-m", "pytest", "grader_test.py"],
        "regression_command": ["python3", "-m", "pytest", "tests"],
        "protected_globs": ["tests/**", "grader_test.py"],
        "ignored_globs": [".venv/**", "*.egg-info/**"],
        "timeout_s": 120,
        "validated": validated,
    }
    (root / "dataset.json").write_text(
        json.dumps({"schema_version": 1, "release": "pilot-20", "task_count": 1}),
        encoding="utf-8",
    )
    (root / "tasks.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    return record


class DownloadEnvironment:
    def __init__(self, candidate: Path) -> None:
        self.candidate = candidate
        self.workdir = "/workspace/repo"

    def download(self, _source: str, destination: Path) -> None:
        shutil.copytree(self.candidate, destination, dirs_exist_ok=True, symlinks=True)


def _docker_result(stdout: str = "", returncode: int = 0) -> SimpleNamespace:
    """Stand in for the CompletedProcess that sandbox._run returns."""
    return SimpleNamespace(stdout=stdout, stderr="", returncode=returncode)


def _absent() -> SimpleNamespace:
    """Docker's answer when an inspected resource does not exist."""
    return _docker_result(returncode=1)


class PatchEvalTests(unittest.TestCase):
    def test_loads_only_validated_checksum_pinned_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _dataset(root)
            benchmark = PatchEval(root)
            tasks = benchmark.load_tasks()

        self.assertEqual([task.id for task in tasks], ["owner-repo-123"])
        self.assertEqual(benchmark.task_count, 1)
        self.assertEqual(
            benchmark.result_metadata(None), {"dataset_release": "pilot-20"}
        )
        prompt = benchmark.build_prompt(tasks[0])
        self.assertEqual(
            prompt,
            "Fix the following issue in the current repository. Inspect the code, "
            "make the necessary changes, and verify your solution.\n\n"
            "# Handle empty input\n\n"
            "Calling parse with an empty value should return None.",
        )
        self.assertNotIn("benchmark", prompt.lower())
        self.assertNotIn("hidden", prompt.lower())

    def test_rejects_unvalidated_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _dataset(root, validated=False)
            with self.assertRaisesRegex(RuntimeError, "not miner-validated"):
                PatchEval(root).load_tasks()

    def test_rejects_invalid_runtime_recipes(self) -> None:
        invalid_recipes = (
            {**_RUNTIME_RECIPE, "schema_version": True},
            {**_RUNTIME_RECIPE, "schema_version": 2},
            {**_RUNTIME_RECIPE, "base_image": "python"},
            {**_RUNTIME_RECIPE, "base_image": "python:latest"},
            {
                **_RUNTIME_RECIPE,
                "base_image": "python@sha256:" + "a" * 64,
            },
            {**_RUNTIME_RECIPE, "sync_command": []},
            {**_RUNTIME_RECIPE, "environment": ["NOT-VALID"]},
            {**_RUNTIME_RECIPE, "unexpected": True},
        )
        for runtime_recipe in invalid_recipes:
            with (
                self.subTest(runtime_recipe=runtime_recipe),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                record = _dataset(root)
                record["runtime_recipe"] = runtime_recipe
                (root / "tasks.jsonl").write_text(
                    json.dumps(record) + "\n", encoding="utf-8"
                )
                with self.assertRaisesRegex(RuntimeError, "runtime_recipe"):
                    PatchEval(root).load_tasks()

    def test_rejects_modified_dataset_assets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _dataset(root)
            (root / "hidden.patch").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                PatchEval(root).load_tasks()

    def test_trusted_patch_ignores_agent_git_and_test_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _dataset(root)
            task = PatchEval(root).load_tasks()[0]
            spec = task.metadata["spec"]
            candidate = root / "candidate"
            candidate.mkdir()
            (candidate / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
            tests = candidate / "tests"
            tests.mkdir()
            (tests / "test_existing.py").write_text(
                "def test_old(): assert False\n", encoding="utf-8"
            )
            (tests / "test_agent.py").write_text(
                "def test_new(): pass\n", encoding="utf-8"
            )
            (candidate / ".git").mkdir()
            (candidate / ".git" / "config").write_text(
                "malicious metadata", encoding="utf-8"
            )

            submission, changed, excluded = _trusted_patch(
                spec, DownloadEnvironment(candidate)
            )

        self.assertIn("module.py", submission)
        self.assertNotIn("test_agent.py", submission)
        self.assertEqual(
            changed,
            ["module.py", "tests/test_agent.py", "tests/test_existing.py"],
        )
        self.assertEqual(excluded, ["tests/test_agent.py", "tests/test_existing.py"])

    def test_standard_root_python_tests_are_always_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _dataset(root)
            task = PatchEval(root).load_tasks()[0]
            spec = task.metadata["spec"]
            candidate = root / "candidate"
            candidate.mkdir()
            (candidate / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
            shutil.copytree(root / "source" / "tests", candidate / "tests")
            (candidate / "test_agent.py").write_text(
                "def test_agent(): pass\n", encoding="utf-8"
            )

            submission, changed, excluded = _trusted_patch(
                spec, DownloadEnvironment(candidate)
            )

        self.assertIn("module.py", submission)
        self.assertNotIn("test_agent.py", submission)
        self.assertIn("test_agent.py", changed)
        self.assertIn("test_agent.py", excluded)

    def test_verifier_runs_hidden_and_regression_graders_separately(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _dataset(root)
            benchmark = PatchEval(root)
            task = benchmark.load_tasks()[0]
            environment = SimpleNamespace(
                image=SimpleNamespace(), owner_id="owner", workdir="/workspace/repo"
            )
            hidden_failure = {
                "exit_code": 1,
                "output": "SECRET_EXPECTATION",
                "output_truncated": False,
            }
            regression_pass = {
                "exit_code": 0,
                "output": "all old tests passed",
                "output_truncated": False,
            }
            with (
                patch(
                    "benchkit.benchmarks.patcheval._trusted_patch",
                    return_value=("diff", ["module.py"], []),
                ),
                patch(
                    "benchkit.benchmarks.patcheval._grade_once",
                    side_effect=[hidden_failure, regression_pass],
                ) as grade,
            ):
                result = benchmark.verify_workspace(task, environment)

        self.assertFalse(result.passed)
        self.assertEqual(result.feedback, _GENERIC_REPAIR)
        self.assertNotIn("SECRET_EXPECTATION", result.feedback)
        self.assertEqual(grade.call_count, 2)
        self.assertTrue(grade.call_args_list[0].kwargs["hidden_tests"])
        self.assertFalse(grade.call_args_list[1].kwargs["hidden_tests"])
        self.assertEqual(result.details["f2p_exit_code"], 1)
        self.assertEqual(result.details["regression_exit_code"], 0)

    def test_workspace_setup_exposes_source_but_not_hidden_patch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _dataset(root)
            benchmark = PatchEval(root)
            task = benchmark.load_tasks()[0]
            environment = Mock(workdir="/workspace", exec=Mock())

            benchmark.prepare_workspace(task, environment)

        environment.upload.assert_called_once()
        uploaded = environment.upload.call_args.args
        self.assertEqual(uploaded[0].name, "source.tar.gz")
        self.assertNotIn("hidden", str(uploaded[0]))
        commands = [call.args[0] for call in environment.exec.call_args_list]
        self.assertIn(["git", "init", "-q", "/workspace/repo"], commands)
        self.assertTrue(
            all("hidden.patch" not in " ".join(command) for command in commands)
        )

    def test_repair_prompt_is_generic_and_does_not_mention_grading(self) -> None:
        prompt = PatchEval().build_repair_prompt("secret output", 1, 3)

        self.assertEqual(prompt, _GENERIC_REPAIR)
        self.assertNotIn("verifier", prompt.lower())
        self.assertNotIn("test output", prompt.lower())

    def test_engine_caches_one_runner_per_locally_built_task_image(self) -> None:
        image = LatestPiImage(
            docker="docker", image="benchkit-pi-patcheval:runtime", transient=False
        )
        benchmark = SimpleNamespace(
            name="patcheval", pi_image_for_task=Mock(return_value=image)
        )
        task = SimpleNamespace()
        engine = Engine(client=SimpleNamespace(), jobs=[])

        first = engine._pi(benchmark, task)
        second = engine._pi(benchmark, task)

        self.assertIs(first, second)
        self.assertEqual(benchmark.pi_image_for_task.call_count, 2)

    def test_runtime_image_is_local_content_addressed_and_context_is_minimal(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _dataset(root)
            benchmark = PatchEval(root)
            task = benchmark.load_tasks()[0]
            spec = task.metadata["spec"]

            image = benchmark.pi_image_for_task(task)
            same = benchmark.pi_image_for_task(task)
            changed_recipe = PatchEvalRuntimeRecipe(
                **{
                    **spec.runtime_recipe.__dict__,
                    "environment": (*spec.runtime_recipe.environment, "TZ=UTC"),
                }
            )
            changed = patcheval_pi_image(
                spec.source_archive, spec.source_sha256, changed_recipe
            )

        self.assertEqual(image.image, same.image)
        self.assertNotEqual(image.image, changed.image)
        self.assertTrue(image.image.startswith("benchkit-pi-patcheval:"))
        self.assertEqual(
            image.build_files,
            ((spec.source_archive, "parent-source.tar"),),
        )
        self.assertTrue(image.no_cache)
        self.assertTrue(image.transient)
        self.assertTrue(image.ephemeral_buildx)
        self.assertTrue(image.always_cleanup_image)
        self.assertIn("FROM node:24-bookworm-slim AS benchkit-node", image.dockerfile)
        self.assertIn(f"FROM {_RUNTIME_RECIPE['base_image']}", image.dockerfile)
        self.assertIn("COPY parent-source.tar", image.dockerfile)
        self.assertIn("rm -rf /opt/project", image.dockerfile)
        self.assertIn("HOME=/home/node USER=node LOGNAME=node", image.dockerfile)
        self.assertGreater(
            image.dockerfile.index('ENV UV_OFFLINE="1"'),
            image.dockerfile.index('RUN ["uv","sync"'),
        )
        self.assertNotIn("hidden.patch", image.dockerfile)
        self.assertNotIn("gold", image.dockerfile.lower())

    def test_ephemeral_builder_and_image_are_removed_after_prepare(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _dataset(root)
            benchmark = PatchEval(root)
            task = benchmark.load_tasks()[0]
            image = benchmark.pi_image_for_task(task)
            observed_context: set[str] = set()

            def run(args, **_kwargs):
                if args[2:3] == ["inspect"]:
                    return _absent()
                if args[1:3] == ["buildx", "build"]:
                    context = Path(args[-1])
                    observed_context.update(
                        path.relative_to(context).as_posix()
                        for path in context.rglob("*")
                        if path.is_file()
                    )
                return _docker_result(f"{PI_VERSION}\n")

            with patch("benchkit.sandbox._run", side_effect=run) as docker_run:
                self.assertEqual(image.prepare(), PI_VERSION)
                image.cleanup()

        commands = [call.args[0] for call in docker_run.call_args_list]
        create = next(
            command for command in commands if command[1:3] == ["buildx", "create"]
        )
        builder = create[create.index("--name") + 1]
        docker = commands[0][0]
        self.assertTrue(builder.startswith("benchkit-patcheval-build-"))
        self.assertIn([docker, "buildx", "rm", "--force", builder], commands)
        self.assertIn(
            [docker, "rm", "--force", f"buildx_buildkit_{builder}0"], commands
        )
        self.assertIn(
            [
                docker,
                "volume",
                "rm",
                "--force",
                f"buildx_buildkit_{builder}0_state",
            ],
            commands,
        )
        self.assertIn(
            [docker, "image", "rm", "--force", "moby/buildkit:buildx-stable-1"],
            commands,
        )
        self.assertEqual(
            commands[-2:],
            [
                [docker, "image", "rm", "--force", image.image],
                [docker, "image", "inspect", image.image],
            ],
        )
        self.assertIn("parent-source.tar", observed_context)
        self.assertIn("Dockerfile", observed_context)
        self.assertFalse(any("hidden" in path for path in observed_context))
        self.assertFalse(any("gold" in path for path in observed_context))

    def test_failed_build_removes_builder_cache_and_partial_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _dataset(root)
            benchmark = PatchEval(root)
            image = benchmark.pi_image_for_task(benchmark.load_tasks()[0])

            def run(args, **_kwargs):
                if args[1:3] == ["buildx", "build"]:
                    raise SandboxError("build failed")
                if args[2:3] == ["inspect"]:
                    return _absent()
                return _docker_result(f"{PI_VERSION}\n")

            with patch("benchkit.sandbox._run", side_effect=run) as docker_run:
                with self.assertRaisesRegex(SandboxError, "build failed"):
                    image.prepare()
                image.cleanup()

        commands = [call.args[0] for call in docker_run.call_args_list]
        create = next(
            command for command in commands if command[1:3] == ["buildx", "create"]
        )
        builder = create[create.index("--name") + 1]
        docker = commands[0][0]
        self.assertIn([docker, "buildx", "rm", "--force", builder], commands)
        self.assertIn(
            [
                docker,
                "volume",
                "rm",
                "--force",
                f"buildx_buildkit_{builder}0_state",
            ],
            commands,
        )
        image_removals = [
            command
            for command in commands
            if command == [docker, "image", "rm", "--force", image.image]
        ]
        self.assertGreaterEqual(len(image_removals), 2)


if __name__ == "__main__":
    unittest.main()
