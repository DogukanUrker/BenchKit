"""Hermetic bug-fix tasks scored by hidden tests outside the agent sandbox."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from benchkit.benchmarks.base import Task
from benchkit.evaluation import EvaluationResult
from benchkit.sandbox import (
    DockerTaskEnvironment,
    LatestPiImage,
    PatchEvalRuntimeRecipe,
    SandboxError,
    _run,
    patcheval_pi_image,
    resource_labels,
)

_SCHEMA_VERSION = 1
_TASKS_FILE = "tasks.jsonl"
_RELEASE_FILE = "dataset.json"
_GENERIC_REPAIR = (
    "The issue is not fully resolved yet. Re-examine your changes and continue "
    "working on the fix. Use the repository and any tests you can run to validate "
    "your next attempt."
)
_MAX_PATCH_BYTES = 5 * 1024 * 1024
_MAX_CANDIDATE_FILES = 50_000
_MAX_CANDIDATE_BYTES = 512 * 1024 * 1024
_BASE_IMAGE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/:+-]*")
_ENVIRONMENT_KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PYTHON_TEST_GLOBS = (
    "tests/**",
    "test/**",
    "**/tests/**",
    "**/test/**",
    "test_*.py",
    "*_test.py",
    "**/test_*.py",
    "**/*_test.py",
    "conftest.py",
    "**/conftest.py",
)


@dataclass(frozen=True)
class PatchEvalSpec:
    """One frozen PatchEval task and its host-only grading assets."""

    id: str
    repository: str
    issue_title: str
    issue_body: str
    runtime_recipe: PatchEvalRuntimeRecipe
    source_archive: Path
    hidden_test_patch: Path
    source_sha256: str
    hidden_test_sha256: str
    fail_to_pass_command: tuple[str, ...]
    regression_command: tuple[str, ...]
    setup_command: tuple[str, ...]
    protected_globs: tuple[str, ...]
    ignored_globs: tuple[str, ...]
    timeout_s: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(root: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must stay inside the dataset root")
    resolved_root = root.resolve(strict=True)
    candidate = root
    for part in path.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError(f"{field} must not traverse symlinks")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise ValueError(f"{field} must stay inside the dataset root") from exc
    return resolved


def _strings(record: dict, field: str, *, required: bool = True) -> tuple[str, ...]:
    value = record.get(field)
    if value is None and not required:
        return ()
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise ValueError(f"{field} must be a non-empty list of strings")
    return tuple(value)


def _runtime_recipe(record: dict) -> PatchEvalRuntimeRecipe:
    value = record.get("runtime_recipe")
    if not isinstance(value, dict):
        raise ValueError("runtime_recipe must be an object")
    allowed = {
        "schema_version",
        "base_image",
        "sync_command",
        "bootstrap_command",
        "environment",
    }
    unknown = set(value).difference(allowed)
    if unknown:
        raise ValueError(
            "runtime_recipe contains unsupported fields: " + ", ".join(sorted(unknown))
        )
    if (
        isinstance(value.get("schema_version"), bool)
        or value.get("schema_version") != 1
    ):
        raise ValueError("runtime_recipe schema_version must be 1")
    base_image = value.get("base_image")
    if not isinstance(base_image, str) or not base_image:
        raise ValueError("runtime_recipe base_image must be a versioned image tag")
    if (
        "@" in base_image
        or any(character.isspace() for character in base_image)
        or _BASE_IMAGE_RE.fullmatch(base_image) is None
        or ":" not in base_image.rsplit("/", 1)[-1]
    ):
        raise ValueError("runtime_recipe base_image must be a versioned image tag")
    tag = base_image.rsplit(":", 1)[-1]
    if tag == "latest" or not any(character.isdigit() for character in tag):
        raise ValueError("runtime_recipe base_image must be a versioned image tag")
    try:
        sync_command = _strings(value, "sync_command")
        bootstrap_command = _strings(value, "bootstrap_command", required=False)
        environment = _strings(value, "environment", required=False)
    except ValueError as exc:
        raise ValueError(f"runtime_recipe {exc}") from exc
    keys: list[str] = []
    for item in environment:
        key, separator, _environment_value = item.partition("=")
        if (
            not separator
            or _ENVIRONMENT_KEY_RE.fullmatch(key) is None
            or "\n" in item
            or "\r" in item
            or "\0" in item
        ):
            raise ValueError("runtime_recipe environment entries must use KEY=value")
        keys.append(key)
    if len(keys) != len(set(keys)):
        raise ValueError("runtime_recipe environment keys must be unique")
    return PatchEvalRuntimeRecipe(
        schema_version=1,
        base_image=base_image,
        sync_command=sync_command,
        bootstrap_command=bootstrap_command,
        environment=environment,
    )


def _load_spec(root: Path, record: object) -> PatchEvalSpec:
    if not isinstance(record, dict):
        raise ValueError("each PatchEval task must be a JSON object")
    if "runtime_image" in record:
        raise ValueError("runtime_image is unsupported; use runtime_recipe")
    required_strings = (
        "id",
        "repository",
        "issue_title",
        "issue_body",
        "source_sha256",
        "hidden_test_sha256",
    )
    for field in required_strings:
        if not isinstance(record.get(field), str) or not record[field].strip():
            raise ValueError(f"{field} must be a non-empty string")
    if record.get("validated") is not True:
        raise ValueError(f"PatchEval task {record['id']!r} is not miner-validated")

    source = _safe_relative(root, record.get("source_archive"), "source_archive")
    hidden = _safe_relative(root, record.get("hidden_test_patch"), "hidden_test_patch")
    for path, field, expected in (
        (source, "source_archive", record["source_sha256"]),
        (hidden, "hidden_test_patch", record["hidden_test_sha256"]),
    ):
        if not path.is_file():
            raise ValueError(f"{field} does not exist for task {record['id']!r}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(
                f"{field} checksum mismatch for task {record['id']!r}: "
                f"expected {expected}, got {actual}"
            )
    _validate_source_archive(source)

    timeout_s = record.get("timeout_s", 300)
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, int):
        raise ValueError("timeout_s must be an integer")
    if not 10 <= timeout_s <= 1800:
        raise ValueError("timeout_s must be between 10 and 1800 seconds")

    return PatchEvalSpec(
        id=record["id"],
        repository=record["repository"],
        issue_title=record["issue_title"].strip(),
        issue_body=record["issue_body"].strip(),
        runtime_recipe=_runtime_recipe(record),
        source_archive=source,
        hidden_test_patch=hidden,
        source_sha256=record["source_sha256"],
        hidden_test_sha256=record["hidden_test_sha256"],
        fail_to_pass_command=_strings(record, "fail_to_pass_command"),
        regression_command=_strings(record, "regression_command"),
        setup_command=_strings(record, "setup_command", required=False),
        protected_globs=tuple(
            dict.fromkeys(_PYTHON_TEST_GLOBS + _strings(record, "protected_globs"))
        ),
        ignored_globs=_strings(record, "ignored_globs", required=False),
        timeout_s=timeout_s,
    )


def _source_archive_members(handle: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = handle.getmembers()
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe path in source archive: {member.name!r}")
        if member.issym() or member.islnk():
            raise ValueError(f"source archives must dereference links: {member.name!r}")
        if not member.isdir() and not member.isreg():
            raise ValueError(f"unsupported archive entry: {member.name!r}")
    return members


def _validate_source_archive(archive: Path) -> None:
    with tarfile.open(archive, "r:*") as handle:
        _source_archive_members(handle)


def _safe_extract(archive: Path, destination: Path) -> None:
    """Extract a miner-produced source archive without following archive links."""
    with tarfile.open(archive, "r:*") as handle:
        members = _source_archive_members(handle)
        handle.extractall(destination, members=members, filter="fully_trusted")


def _git(
    worktree: Path, args: list[str], *, check: bool = True
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=os.devnull,
    )
    completed = subprocess.run(
        ["git", "-C", str(worktree), *args],
        text=True,
        capture_output=True,
        env=environment,
        timeout=60,
    )
    if check and completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()[-2000:]
        raise RuntimeError(f"trusted patch extraction failed: {detail}")
    return completed


def _remove_entry(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _copy_candidate(source: Path, destination: Path) -> None:
    """Copy only ordinary files, directories, and symlinks from the sandbox."""
    file_count = 0
    byte_count = 0

    def copy_entry(current: Path, target: Path) -> None:
        nonlocal file_count, byte_count
        if current.name == ".git":
            return
        info = current.lstat()
        file_count += 1
        if file_count > _MAX_CANDIDATE_FILES:
            raise ValueError("candidate workspace contains too many files")
        if stat.S_ISLNK(info.st_mode):
            target.symlink_to(os.readlink(current))
        elif stat.S_ISDIR(info.st_mode):
            target.mkdir(mode=info.st_mode & 0o777, exist_ok=True)
            for child in current.iterdir():
                copy_entry(child, target / child.name)
        elif stat.S_ISREG(info.st_mode):
            byte_count += info.st_size
            if byte_count > _MAX_CANDIDATE_BYTES:
                raise ValueError("candidate workspace is too large")
            shutil.copyfile(current, target, follow_symlinks=False)
            target.chmod(info.st_mode & 0o777)
        else:
            raise ValueError(f"candidate contains unsupported file: {current.name!r}")

    for child in source.iterdir():
        copy_entry(child, destination / child.name)


def _matches(path: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def _trusted_patch(
    spec: PatchEvalSpec, environment: DockerTaskEnvironment
) -> tuple[str, list[str], list[str]]:
    """Diff a downloaded workspace against the host's pristine source archive."""
    with tempfile.TemporaryDirectory(prefix="benchkit-patcheval-diff-") as directory:
        root = Path(directory)
        worktree = root / "trusted"
        candidate = root / "candidate"
        worktree.mkdir()
        candidate.mkdir()
        _safe_extract(spec.source_archive, worktree)
        environment.download(f"{environment.workdir}/.", candidate)

        _git(worktree, ["init", "-q"])
        _git(worktree, ["add", "-A"])
        _git(
            worktree,
            [
                "-c",
                "user.name=BenchKit",
                "-c",
                "user.email=benchkit@localhost",
                "commit",
                "-q",
                "--no-gpg-sign",
                "-m",
                "pristine",
            ],
        )
        for child in worktree.iterdir():
            if child.name != ".git":
                _remove_entry(child)
        _copy_candidate(candidate, worktree)
        _git(worktree, ["add", "--intent-to-add", "--", "."])
        # -z keeps paths verbatim. Git otherwise quotes non-ASCII, quote,
        # backslash, and control characters, and a quoted path matches neither
        # its protected glob nor the pathspec of the second diff, so the
        # agent's real edits are silently dropped from the submission.
        changed = [
            path
            for path in _git(
                worktree,
                ["diff", "-z", "--name-only", "--no-renames", "HEAD", "--"],
            ).stdout.split("\0")
            if path
        ]
        excluded = [
            path
            for path in changed
            if _matches(path, spec.protected_globs + spec.ignored_globs)
        ]
        included = [path for path in changed if path not in excluded]
        if not included:
            return "", changed, excluded
        patch = _git(
            worktree,
            ["diff", "--binary", "--no-renames", "HEAD", "--", *included],
        ).stdout
        if len(patch.encode()) > _MAX_PATCH_BYTES:
            raise ValueError("candidate patch exceeds the 5 MiB limit")
        return patch, changed, excluded


def _docker_exec(
    docker: str,
    container: str,
    command: list[str],
    *,
    workdir: str | None = None,
    timeout: float = 60,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return _run(
        [
            docker,
            "exec",
            *(["--workdir", workdir] if workdir else []),
            container,
            *command,
        ],
        timeout=timeout,
        check=check,
    )


def _grade_once(
    spec: PatchEvalSpec,
    image: LatestPiImage,
    patch: str,
    command: tuple[str, ...],
    *,
    hidden_tests: bool,
    owner_id: str,
) -> dict:
    docker = image.docker
    label = "f2p" if hidden_tests else "regression"
    container = f"benchkit-patcheval-{label}-{uuid.uuid4().hex[:12]}"
    labels = resource_labels(owner_id)
    with tempfile.TemporaryDirectory(prefix="benchkit-patcheval-grade-") as directory:
        patch_path = Path(directory) / "submission.patch"
        patch_path.write_text(patch, encoding="utf-8")
        try:
            _run(
                [
                    docker,
                    "run",
                    "--detach",
                    "--name",
                    container,
                    *labels,
                    "--network",
                    "none",
                    "--cap-drop",
                    "ALL",
                    "--security-opt",
                    "no-new-privileges",
                    "--pids-limit",
                    os.environ.get("BENCHKIT_SANDBOX_PIDS", "256"),
                    "--memory",
                    os.environ.get("BENCHKIT_SANDBOX_MEMORY", "2g"),
                    "--cpus",
                    os.environ.get("BENCHKIT_SANDBOX_CPUS", "2"),
                    image.image,
                    "sleep",
                    "infinity",
                ],
                timeout=60,
            )
            _run([docker, "cp", str(spec.source_archive), f"{container}:/tmp/source"])
            _run([docker, "cp", str(patch_path), f"{container}:/tmp/submission.patch"])
            if hidden_tests:
                _run(
                    [
                        docker,
                        "cp",
                        str(spec.hidden_test_patch),
                        f"{container}:/tmp/hidden-tests.patch",
                    ]
                )
            _docker_exec(docker, container, ["mkdir", "-p", "/workspace/repo"])
            _docker_exec(
                docker,
                container,
                ["tar", "-xf", "/tmp/source", "-C", "/workspace/repo"],
            )
            if patch:
                _docker_exec(
                    docker,
                    container,
                    ["git", "apply", "--binary", "/tmp/submission.patch"],
                    workdir="/workspace/repo",
                )
            if hidden_tests:
                _docker_exec(
                    docker,
                    container,
                    ["git", "apply", "--binary", "/tmp/hidden-tests.patch"],
                    workdir="/workspace/repo",
                )
            if spec.setup_command:
                _docker_exec(
                    docker,
                    container,
                    list(spec.setup_command),
                    workdir="/workspace/repo",
                    timeout=spec.timeout_s + 30,
                )
            completed = _docker_exec(
                docker,
                container,
                ["timeout", "--signal=KILL", f"{spec.timeout_s}s", *command],
                workdir="/workspace/repo",
                timeout=spec.timeout_s + 30,
                check=False,
            )
            output = (completed.stdout + completed.stderr).strip()
            return {
                "exit_code": completed.returncode,
                "output": output[-8000:],
                "output_truncated": len(output) > 8000,
            }
        finally:
            _run([docker, "rm", "--force", container], timeout=30, check=False)


class PatchEval:
    """Fix real Python repository issues with hidden, deterministic grading."""

    name = "patcheval"
    task_count = 20
    workspace_task = True
    evaluation_activity = "running hidden PatchEval checks"
    list_note = "pilot-20 · requires Pi and Docker"

    def __init__(self, dataset_root: Path | None = None) -> None:
        configured = dataset_root or (
            Path(os.environ["BENCHKIT_PATCHEVAL_DATASET"]).expanduser()
            if os.environ.get("BENCHKIT_PATCHEVAL_DATASET")
            else None
        )
        self.dataset_root = configured
        self.release = "unloaded"

    def _root(self) -> Path:
        if self.dataset_root is None:
            raise RuntimeError(
                "PatchEval's frozen dataset is not configured; download the "
                "pilot-20 release from DogukanUrker/PatchEval and set "
                "BENCHKIT_PATCHEVAL_DATASET to its local root"
            )
        return self.dataset_root.resolve()

    @staticmethod
    def _spec(task: Task) -> PatchEvalSpec:
        spec = task.metadata.get("spec")
        if not isinstance(spec, PatchEvalSpec):
            raise TypeError("PatchEval task metadata is missing its validated spec")
        return spec

    def load_tasks(self) -> list[Task]:
        root = self._root()
        release_path = root / _RELEASE_FILE
        tasks_path = root / _TASKS_FILE
        try:
            release = json.loads(release_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"could not read PatchEval dataset metadata: {exc}"
            ) from exc
        if (
            not isinstance(release, dict)
            or release.get("schema_version") != _SCHEMA_VERSION
        ):
            raise RuntimeError(
                f"PatchEval dataset must use schema_version {_SCHEMA_VERSION}"
            )
        self.release = str(release.get("release") or "unknown")
        try:
            records = [
                json.loads(line)
                for line in tasks_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            specs = [_load_spec(root, record) for record in records]
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError(f"invalid PatchEval dataset: {exc}") from exc
        expected = release.get("task_count")
        if expected != len(specs):
            raise RuntimeError(
                f"PatchEval release declares {expected} tasks but contains {len(specs)}"
            )
        ids = [spec.id for spec in specs]
        if len(ids) != len(set(ids)):
            raise RuntimeError("PatchEval task ids must be unique")
        self.task_count = len(specs)
        return [Task(spec.id, "", {"spec": spec}) for spec in specs]

    def result_metadata(self, _variant: str | None) -> dict:
        return {"dataset_release": self.release}

    def build_prompt(self, task: Task) -> str:
        spec = self._spec(task)
        return (
            "Fix the following issue in the current repository. Inspect the code, "
            "make the necessary changes, and verify your solution.\n\n"
            f"# {spec.issue_title}\n\n{spec.issue_body}"
        )

    def build_repair_prompt(self, _feedback: str, _attempt: int, _total: int) -> str:
        return _GENERIC_REPAIR

    def evaluate(self, _task: Task, _response: str) -> bool:
        raise RuntimeError("PatchEval must be evaluated by its external grader")

    def pi_image_for_task(self, task: Task) -> LatestPiImage:
        spec = self._spec(task)
        return patcheval_pi_image(
            spec.source_archive,
            spec.source_sha256,
            spec.runtime_recipe,
        )

    def prepare_workspace(self, task: Task, environment: DockerTaskEnvironment) -> None:
        spec = self._spec(task)
        environment.workdir = "/workspace/repo"
        environment.upload(spec.source_archive, "/tmp/patch-eval-source")
        environment.exec(["mkdir", "-p", environment.workdir])
        environment.exec(
            ["tar", "-xf", "/tmp/patch-eval-source", "-C", environment.workdir]
        )
        environment.exec(
            [
                "find",
                environment.workdir,
                "-exec",
                "touch",
                "-t",
                "198001010000",
                "{}",
                "+",
            ]
        )
        environment.exec(["rm", "-rf", f"{environment.workdir}/.git"])
        if spec.setup_command:
            environment.exec(
                list(spec.setup_command),
                workdir=environment.workdir,
                timeout=spec.timeout_s + 30,
            )
        environment.exec(["git", "init", "-q", environment.workdir])
        environment.exec(
            ["git", "-C", environment.workdir, "config", "user.name", "BenchKit"]
        )
        environment.exec(
            [
                "git",
                "-C",
                environment.workdir,
                "config",
                "user.email",
                "benchkit@localhost",
            ]
        )
        environment.exec(["git", "-C", environment.workdir, "add", "."])
        environment.exec(
            [
                "git",
                "-C",
                environment.workdir,
                "commit",
                "-q",
                "-m",
                "starting state",
            ]
        )

    def verify_workspace(
        self,
        task: Task,
        environment: DockerTaskEnvironment,
        _tool_trace: list[dict] | None = None,
    ) -> EvaluationResult:
        spec = self._spec(task)
        try:
            patch, changed, excluded = _trusted_patch(spec, environment)
            # The two graders share nothing but the read-only task image and
            # the submitted patch, so they run at the same time.
            with ThreadPoolExecutor(max_workers=2) as pool:
                f2p_grade = pool.submit(
                    _grade_once,
                    spec,
                    environment.image,
                    patch,
                    spec.fail_to_pass_command,
                    hidden_tests=True,
                    owner_id=environment.owner_id,
                )
                regression_grade = pool.submit(
                    _grade_once,
                    spec,
                    environment.image,
                    patch,
                    spec.regression_command,
                    hidden_tests=False,
                    owner_id=environment.owner_id,
                )
                f2p = f2p_grade.result()
                regression = regression_grade.result()
        except (OSError, RuntimeError, ValueError, SandboxError) as exc:
            return EvaluationResult(
                0.0,
                error=f"PatchEval grader infrastructure failed: {type(exc).__name__}: {exc}",
            )
        passed = f2p["exit_code"] == 0 and regression["exit_code"] == 0
        return EvaluationResult(
            1.0 if passed else 0.0,
            feedback="" if passed else _GENERIC_REPAIR,
            details={
                "repository": spec.repository,
                "f2p_exit_code": f2p["exit_code"],
                "regression_exit_code": regression["exit_code"],
                "f2p_output": f2p["output"],
                "f2p_output_truncated": f2p["output_truncated"],
                "regression_output": regression["output"],
                "regression_output_truncated": regression["output_truncated"],
                "changed_paths": changed,
                "excluded_agent_paths": excluded,
                "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
                "patch": patch,
            },
        )
