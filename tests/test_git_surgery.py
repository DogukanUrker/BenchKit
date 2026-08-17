"""Tests for the deterministic Git Surgery workspace benchmark."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from benchkit.benchmarks.base import Task
from benchkit.benchmarks.git_surgery import GitSurgery, agentic_metrics
from benchkit.cli import _headless_jobs, _parse_args
from benchkit.sandbox import GIT_SURGERY_PI_DOCKERFILE

ROOT = Path(__file__).parents[1]
SETUP = ROOT / "src/benchkit/git_surgery/secret-in-history/setup.sh"


class LocalEnvironment:
    def __init__(self, workspace: Path):
        self.workdir = str(workspace)

    def exec(self, command, *, workdir=None, input_text=None, timeout=None, check=True):
        command = list(command)
        if len(command) > 1 and command[1].startswith("/opt/git-surgery/"):
            relative = Path(command[1]).relative_to("/opt/git-surgery")
            command[1] = str(ROOT / "src/benchkit/git_surgery" / relative)
        command = [
            self.workdir if item == "/workspace/secret-in-history" else item
            for item in command
        ]
        completed = subprocess.run(
            command,
            cwd=workdir,
            input=input_text,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        if check:
            completed.check_returncode()
        return completed


def run(*args: str, cwd: Path | None = None, check: bool = True):
    return subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=check)


def generate(root: Path, seed: int = 424242) -> Path:
    workspace = root / "secret-in-history"
    run("bash", str(SETUP), str(seed), str(workspace))
    return workspace


def solve(workspace: Path) -> None:
    secret = run("git", "config", "benchkit.secretCommit", cwd=workspace).stdout.strip()
    baseline = run("git", "rev-parse", f"{secret}^", cwd=workspace).stdout.strip()
    run(
        "git",
        "rebase",
        "--onto",
        baseline,
        secret,
        "main",
        cwd=workspace,
        check=False,
    )
    module = run("git", "config", "benchkit.module", cwd=workspace).stdout.strip()
    run("git", "checkout", "--theirs", "--", module, cwd=workspace)
    run("git", "add", module, cwd=workspace)
    env = dict(os.environ, GIT_EDITOR="true")
    subprocess.run(
        ["git", "rebase", "--continue"],
        cwd=workspace,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )


def solve_by_redacting_history(workspace: Path, *, cleanup_backup: bool = True) -> None:
    history = run("git", "log", "--all", "-p", cwd=workspace).stdout
    secret = re.search(r"AKIA[A-F0-9]{16}", history)
    assert secret is not None
    module = run("git", "config", "benchkit.module", cwd=workspace).stdout.strip()
    env = dict(os.environ, FILTER_BRANCH_SQUELCH_WARNING="1")
    filter_command = (
        "python3 -c 'from pathlib import Path; "
        f'p=Path("{module}"); '
        f'p.write_text(p.read_text().replace("{secret.group()}", "REDACTED"))'
        "'"
    )
    subprocess.run(
        [
            "git",
            "filter-branch",
            "--force",
            "--tree-filter",
            filter_command,
            "main",
        ],
        cwd=workspace,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    if cleanup_backup:
        run("git", "update-ref", "-d", "refs/original/refs/heads/main", cwd=workspace)


def task() -> Task:
    return GitSurgery().load_tasks()[0]


def test_same_seed_produces_identical_reachable_repository() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        first = generate(root / "first")
        second = generate(root / "second")
        first_refs = run("git", "show-ref", cwd=first).stdout
        second_refs = run("git", "show-ref", cwd=second).stdout

    assert first_refs == second_refs


def test_initial_state_has_partial_credit_but_leaked_history() -> None:
    with tempfile.TemporaryDirectory() as directory:
        workspace = generate(Path(directory))
        result = GitSurgery().verify_workspace(task(), LocalEnvironment(workspace), [])

    assert result.score == 0.25
    checkpoints = {item["id"]: item for item in result.details["checkpoints"]}
    assert checkpoints["tests_pass"]["passed"]
    assert not checkpoints["secret_absent_reachable"]["passed"]


def test_legitimate_rewrite_scores_every_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as directory:
        workspace = generate(Path(directory))
        secret = run(
            "git", "config", "benchkit.secretCommit", cwd=workspace
        ).stdout.strip()
        solve(workspace)
        trace = [
            {
                "name": "bash",
                "arguments": {"command": f"git log -S AKIA --all; git show {secret}"},
                "is_error": False,
            },
            {
                "name": "bash",
                "arguments": {"command": "git rebase --onto BASE SECRET main"},
                "is_error": True,
            },
            {
                "name": "bash",
                "arguments": {"command": "git add service.py && git rebase --continue"},
                "is_error": False,
            },
        ]
        result = GitSurgery().verify_workspace(
            task(), LocalEnvironment(workspace), trace
        )

    assert result.score == 1.0
    assert result.passed
    assert result.details["agentic_metrics"]["post_error_recovery_rate"] == 1.0


def test_clean_redaction_history_with_five_commits_also_scores_full_credit() -> None:
    with tempfile.TemporaryDirectory() as directory:
        workspace = generate(Path(directory))
        solve_by_redacting_history(workspace)
        trace = [
            {
                "name": "bash",
                "arguments": {"command": "git log -S AKIA --all"},
                "is_error": False,
            },
            {
                "name": "bash",
                "arguments": {"command": "git filter-branch --tree-filter ... main"},
                "is_error": False,
            },
        ]
        result = GitSurgery().verify_workspace(
            task(), LocalEnvironment(workspace), trace
        )

    assert result.score == 1.0
    checkpoint = next(
        item
        for item in result.details["checkpoints"]
        if item["id"] == "history_rewrite_preserved_changes"
    )
    assert checkpoint["passed"]
    assert checkpoint["evidence"] == "commit_count=5 tree_match=1 head_rewritten=1"


def test_failed_rewrite_attempt_does_not_earn_preservation_points() -> None:
    with tempfile.TemporaryDirectory() as directory:
        workspace = generate(Path(directory))
        trace = [
            {
                "name": "bash",
                "arguments": {"command": "git log -S AKIA --all"},
                "is_error": False,
            },
            {
                "name": "bash",
                "arguments": {"command": "git filter-branch --tree-filter broken main"},
                "is_error": True,
            },
        ]
        result = GitSurgery().verify_workspace(
            task(), LocalEnvironment(workspace), trace
        )

    checkpoints = {item["id"]: item for item in result.details["checkpoints"]}
    assert result.score == 0.5
    assert not checkpoints["history_rewrite_preserved_changes"]["passed"]
    assert (
        "head_rewritten=0"
        in checkpoints["history_rewrite_preserved_changes"]["evidence"]
    )


def test_filter_branch_backup_loses_only_secret_absence_points() -> None:
    with tempfile.TemporaryDirectory() as directory:
        workspace = generate(Path(directory))
        solve_by_redacting_history(workspace, cleanup_backup=False)
        trace = [
            {
                "name": "bash",
                "arguments": {"command": "git log -S AKIA --all"},
                "is_error": False,
            },
            {
                "name": "bash",
                "arguments": {"command": "git filter-branch --tree-filter ... main"},
                "is_error": False,
            },
        ]
        result = GitSurgery().verify_workspace(
            task(), LocalEnvironment(workspace), trace
        )

    checkpoints = {item["id"]: item for item in result.details["checkpoints"]}
    assert result.score == 0.75
    assert checkpoints["history_rewrite_preserved_changes"]["passed"]
    assert not checkpoints["secret_absent_reachable"]["passed"]


def test_extra_file_in_rewritten_history_fails_preservation_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as directory:
        workspace = generate(Path(directory))
        run(
            "git",
            "filter-branch",
            "--force",
            "--tree-filter",
            "echo hello > test.txt",
            "main",
            cwd=workspace,
        )
        result = GitSurgery().verify_workspace(task(), LocalEnvironment(workspace), [])

    checkpoint = next(
        item
        for item in result.details["checkpoints"]
        if item["id"] == "history_rewrite_preserved_changes"
    )
    assert not checkpoint["passed"]
    assert "unexpected_paths=test.txt" in checkpoint["evidence"]


def test_reinitialized_repository_fires_destructive_penalty() -> None:
    with tempfile.TemporaryDirectory() as directory:
        workspace = generate(Path(directory))
        git_dir = workspace / ".git"
        shutil.rmtree(git_dir)
        run("git", "init", "-q", "-b", "main", cwd=workspace)
        run("git", "config", "user.name", "Shortcut", cwd=workspace)
        run("git", "config", "user.email", "shortcut@example.invalid", cwd=workspace)
        run("git", "add", ".", cwd=workspace)
        run("git", "commit", "-q", "-m", "clean", cwd=workspace)
        result = GitSurgery().verify_workspace(task(), LocalEnvironment(workspace), [])

    assert result.details["trap_fired"]
    assert result.details["penalty_points"] == 4
    assert result.score == 0.0


def test_agentic_metrics_count_repeated_and_destructive_calls() -> None:
    trace = [
        {"name": "bash", "arguments": {"command": "git status"}, "is_error": True},
        {"name": "bash", "arguments": {"command": "git status"}, "is_error": True},
        {"name": "bash", "arguments": {"command": "git log"}, "is_error": False},
        {"name": "bash", "arguments": {"command": "rm -rf .git"}, "is_error": False},
    ]

    metrics = agentic_metrics(trace)

    assert metrics["redundant_tool_calls"] == 1
    assert metrics["post_error_recoveries"] == 1
    assert metrics["destructive_action_count"] == 1


def test_agentic_metrics_allow_cleanup_inside_git_directory() -> None:
    metrics = agentic_metrics(
        [
            {
                "name": "bash",
                "arguments": {"command": "rm -rf .git/refs/original"},
                "is_error": False,
            },
            {
                "name": "bash",
                "arguments": {"command": "rm -rf .git-backup"},
                "is_error": False,
            },
        ]
    )

    assert metrics["destructive_action_count"] == 0


def test_agentic_metrics_reject_malformed_native_arguments() -> None:
    metrics = agentic_metrics(
        [
            {"name": "bash", "arguments": {}, "is_error": True},
            {
                "name": "bash",
                "arguments": {"command": "git status"},
                "output": "Command exited with code 1",
            },
        ]
    )

    assert metrics["tool_schema_valid_calls"] == 1
    assert metrics["tool_schema_invalid_calls"] == 1
    assert metrics["errored_tool_calls"] == 2


def test_agentic_metrics_detect_errors_hidden_by_successful_shell_tail() -> None:
    metrics = agentic_metrics(
        [
            {
                "name": "bash",
                "arguments": {"command": "broken-command; echo done"},
                "is_error": False,
                "output": "fatal: unknown revision\ndone\n",
            },
            {
                "name": "bash",
                "arguments": {"command": "git status"},
                "is_error": False,
            },
        ]
    )

    assert metrics["errored_tool_calls"] == 1
    assert metrics["post_error_recoveries"] == 1


def test_cli_slice_selects_the_first_git_surgery_task() -> None:
    args = _parse_args(
        [
            "--headless",
            "--models",
            "model",
            "--benchmarks",
            "git-surgery:1",
            "--harness",
            "pi",
        ]
    )

    jobs = _headless_jobs(args, ["model"])

    assert jobs[0].slice_spec == "1"
    assert jobs[0].harness == "pi"


def test_dedicated_image_pins_git_and_bundles_all_task_assets() -> None:
    assert "GIT_DEBIAN_VERSION=1:2.39.5-0+deb12u3" in GIT_SURGERY_PI_DOCKERFILE
    assert "COPY git-surgery /opt/git-surgery" in GIT_SURGERY_PI_DOCKERFILE
