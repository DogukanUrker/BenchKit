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
            self.workdir if item.startswith("/workspace/") else item for item in command
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
    secret = run(
        "git",
        "log",
        "--format=%H",
        "--grep=^temporarily configure deployment credentials$",
        "-1",
        cwd=workspace,
    ).stdout.strip()
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
    module = next(workspace.glob("service_*.py")).name
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
    module = next(workspace.glob("service_*.py")).name
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


def task_named(task_id: str) -> Task:
    return next(item for item in GitSurgery().load_tasks() if item.id == task_id)


def continue_rebase(workspace: Path) -> None:
    subprocess.run(
        ["git", "rebase", "--continue"],
        cwd=workspace,
        env=dict(os.environ, GIT_EDITOR="true"),
        text=True,
        capture_output=True,
        check=False,
    )


def test_same_seed_produces_identical_reachable_repository() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        first = generate(root / "first")
        second = generate(root / "second")
        first_refs = run("git", "show-ref", cwd=first).stdout
        second_refs = run("git", "show-ref", cwd=second).stdout

    assert first_refs == second_refs


def test_git_surgery_registers_the_original_tasks_first_in_slice_order() -> None:
    benchmark = GitSurgery()

    assert [item.id for item in benchmark.load_tasks()][:5] == [
        "secret-in-history",
        "bisect-the-regression",
        "split-the-mega-commit",
        "recover-lost-work",
        "rebase-conflict-chain",
    ]


def test_new_task_initial_states_receive_only_deterministic_partial_credit() -> None:
    expected_scores = {
        "bisect-the-regression": 0.125,
        "split-the-mega-commit": 0.0,
        "recover-lost-work": 0.125,
        "rebase-conflict-chain": 0.0,
    }
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for task_id, expected_score in expected_scores.items():
            workspace = root / task_id
            setup = ROOT / "src/benchkit/git_surgery" / task_id / "setup.sh"
            run("bash", str(setup), "424242", str(workspace))
            result = GitSurgery().verify_workspace(
                task_named(task_id), LocalEnvironment(workspace), []
            )

            assert result.score == expected_score


def test_equivalent_semantic_rebase_scores_every_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory) / "rebase-conflict-chain"
        setup = ROOT / "src/benchkit/git_surgery/rebase-conflict-chain/setup.sh"
        run("bash", str(setup), "424242", str(workspace))
        run("git", "rebase", "main", cwd=workspace, check=False)

        (workspace / "auth.py").write_text(
            'def identify(user: str) -> str:\n    return f"audit:{user.strip().lower()}"\n'
        )
        run("git", "add", "auth.py", cwd=workspace)
        continue_rebase(workspace)

        (workspace / "billing.py").write_text(
            "def total(subtotal: int) -> int:\n    return subtotal + 5\n"
        )
        run("git", "add", "billing.py", cwd=workspace)
        continue_rebase(workspace)

        (workspace / "report.py").write_text(
            "def render(items: list[str]) -> str:\n"
            '    return "items:" + ",".join(item.upper() for item in items)\n'
        )
        run("git", "add", "report.py", cwd=workspace)
        continue_rebase(workspace)

        trace = [
            {
                "name": "bash",
                "arguments": {"command": "git rebase main"},
                "is_error": False,
            }
        ]
        result = GitSurgery().verify_workspace(
            task_named("rebase-conflict-chain"),
            LocalEnvironment(workspace),
            trace,
        )

    assert result.score == 1.0
    assert result.passed


def test_uncommitted_rebase_repairs_do_not_score_as_preserved_behavior() -> None:
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory) / "rebase-conflict-chain"
        setup = ROOT / "src/benchkit/git_surgery/rebase-conflict-chain/setup.sh"
        run("bash", str(setup), "424242", str(workspace))
        run("git", "rebase", "main", cwd=workspace, check=False)

        resolutions = {
            "auth.py": (
                "def identify(user: str) -> str:\n"
                '    return f"audit:{user.strip().lower()}"\n'
            ),
            "billing.py": "def total(subtotal: int) -> int:\n    return subtotal + 5\n",
            "report.py": (
                "def render(items: list[str]) -> str:\n"
                '    return "items:" + ",".join(item.upper() for item in items)\n'
            ),
        }
        for path, content in resolutions.items():
            (workspace / path).write_text(content)
            run("git", "add", path, cwd=workspace)
            continue_rebase(workspace)

        (workspace / "report.py").write_text(
            "def render(items: list[str]) -> str:\n"
            '    return ",".join(item.upper() for item in items)\n'
        )
        run("git", "add", "report.py", cwd=workspace)
        run("git", "commit", "--amend", "--no-edit", cwd=workspace)
        (workspace / "report.py").write_text(resolutions["report.py"])

        trace = [
            {
                "name": "bash",
                "arguments": {"command": "git rebase main"},
                "is_error": False,
            }
        ]
        result = GitSurgery().verify_workspace(
            task_named("rebase-conflict-chain"),
            LocalEnvironment(workspace),
            trace,
        )

    checkpoints = {item["id"]: item for item in result.details["checkpoints"]}
    assert not checkpoints["both_sides_preserved"]["passed"]
    assert not checkpoints["tests_pass"]["passed"]
    assert result.details["trap_fired"]


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
            "git",
            "log",
            "--format=%H",
            "--grep=^temporarily configure deployment credentials$",
            "-1",
            cwd=workspace,
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


HARD_TASKS = [
    "backport-release-stack",
    "revert-merge-with-followups",
    "recover-complex-stash",
    "repair-force-pushed-remote",
    "untangle-nested-submodules",
]

HARD_ENV = dict(
    os.environ,
    GIT_EDITOR="true",
    GIT_CONFIG_GLOBAL="/dev/null",
    GIT_CONFIG_COUNT="1",
    GIT_CONFIG_KEY_0="protocol.file.allow",
    GIT_CONFIG_VALUE_0="always",
)

SOLUTIONS = Path(__file__).with_name("git_surgery_solutions")


def reference_solution(task_id: str) -> str:
    return (SOLUTIONS / f"{task_id}.sh").read_text()


REFERENCE_TRACES = {
    "revert-merge-with-followups": "git revert -m 1 MERGE",
    "recover-complex-stash": "git fsck --dangling",
}


def setup_hard(root: Path, task_id: str, seed: int = 424242) -> Path:
    workspace = root / task_id
    setup = ROOT / "src/benchkit/git_surgery" / task_id / "setup.sh"
    run("bash", str(setup), str(seed), str(workspace))
    return workspace


def shell(workspace: Path, script: str) -> None:
    subprocess.run(
        ["bash", "-c", script],
        cwd=workspace,
        env=HARD_ENV,
        text=True,
        capture_output=True,
        check=True,
    )


def bash_trace(command: str) -> list[dict]:
    return [{"name": "bash", "arguments": {"command": command}, "is_error": False}]


def verify_hard(workspace: Path, task_id: str, trace: list[dict] | None = None):
    return GitSurgery().verify_workspace(
        task_named(task_id), LocalEnvironment(workspace), trace or []
    )


def checkpoint_map(result) -> dict[str, bool]:
    return {item["id"]: item["passed"] for item in result.details["checkpoints"]}


def repository_fingerprint(workspace: Path) -> str:
    """Refs and unreachable objects of every repository under the workspace."""
    repositories = sorted(
        path.parent
        for path in workspace.rglob("HEAD")
        if (path.parent / "objects").is_dir() and (path.parent / "refs").is_dir()
    )
    lines = []
    for repo in repositories:
        refs = run("git", "--git-dir", str(repo), "for-each-ref").stdout
        lost = run(
            "git", "--git-dir", str(repo), "fsck", "--unreachable", "--no-reflogs"
        ).stdout
        relative = repo.relative_to(workspace)
        lines.append(f"{relative}\n{refs}{sorted(lost.splitlines())}")
    return "\n".join(lines)


def test_git_surgery_hard_tasks_are_appended_to_the_slice_order() -> None:
    ids = [item.id for item in GitSurgery().load_tasks()]

    assert ids[5:] == HARD_TASKS
    assert GitSurgery().task_count == 10


def test_hard_task_assets_and_prompts_exist() -> None:
    for task_id in HARD_TASKS:
        directory = ROOT / "src/benchkit/git_surgery" / task_id
        assert (directory / "setup.sh").is_file()
        assert (directory / "verify.sh").is_file()
        assert GitSurgery().build_prompt(task_named(task_id))


def test_hard_tasks_are_deterministic_for_the_same_seed() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for task_id in HARD_TASKS:
            first = repository_fingerprint(setup_hard(root / "a", task_id))
            second = repository_fingerprint(setup_hard(root / "b", task_id))

            assert first, task_id
            assert first == second, task_id


def test_hard_task_initial_states_score_deterministic_partial_credit() -> None:
    expected = {
        "backport-release-stack": 0.0,
        "revert-merge-with-followups": 0.0,
        "recover-complex-stash": 0.0,
        "repair-force-pushed-remote": 0.0,
        "untangle-nested-submodules": 0.0,
    }
    with tempfile.TemporaryDirectory() as directory:
        for task_id, score in expected.items():
            workspace = setup_hard(Path(directory), task_id)
            result = verify_hard(workspace, task_id)

            assert result.score == score, task_id
            assert not result.details["trap_fired"], task_id
            assert result.details["max_points"] == 8


def test_hard_task_reference_solutions_score_every_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as directory:
        for task_id in HARD_TASKS:
            workspace = setup_hard(Path(directory), task_id)
            shell(workspace, reference_solution(task_id))
            trace = bash_trace(REFERENCE_TRACES.get(task_id, "git status"))
            result = verify_hard(workspace, task_id, trace)

            assert result.score == 1.0, (task_id, result.details["checkpoints"])
            assert (
                sum(
                    item["weight"]
                    for item in result.details["checkpoints"]
                    if item["weight"] > 0
                )
                == 8
            )


def test_backport_merging_the_release_branch_fires_the_trap() -> None:
    with tempfile.TemporaryDirectory() as directory:
        workspace = setup_hard(Path(directory), "backport-release-stack")
        shell(workspace, "git merge -q --no-edit -X theirs release-2.x")
        result = verify_hard(workspace, "backport-release-stack")

    assert result.details["trap_fired"]
    assert result.score == 0.0


def test_backport_blanket_theirs_resolution_leaks_release_code() -> None:
    script = """
fix() { git log release-2.x --format=%H --grep="^Fixes: $1\\$" -1; }
for issue in BK-101 BK-104 BK-107 BK-115; do
    if ! git cherry-pick -x "$(fix "$issue")"; then
        git checkout --theirs ledger/money.py
        git add ledger/money.py
        git cherry-pick --continue
    fi
done
"""
    with tempfile.TemporaryDirectory() as directory:
        workspace = setup_hard(Path(directory), "backport-release-stack")
        shell(workspace, script)
        checkpoints = checkpoint_map(verify_hard(workspace, "backport-release-stack"))

    assert checkpoints["selected_fixes"]
    assert checkpoints["intermediate_tests"]
    assert not checkpoints["no_release_leak"]


def test_backport_keeping_the_redundant_pick_loses_the_equivalence_point() -> None:
    script = reference_solution("backport-release-stack").replace(
        'git cherry-pick -x "$(fix BK-112)" || git cherry-pick --skip',
        'git cherry-pick -x --keep-redundant-commits "$(fix BK-112)"',
    )
    with tempfile.TemporaryDirectory() as directory:
        workspace = setup_hard(Path(directory), "backport-release-stack")
        shell(workspace, script)
        checkpoints = checkpoint_map(verify_hard(workspace, "backport-release-stack"))

    assert not checkpoints["selected_fixes"]
    assert not checkpoints["skipped_equivalent"]


def test_revert_with_the_wrong_mainline_or_default_inverse_is_not_credited() -> None:
    trace = bash_trace("git revert -m 2 MERGE")
    with tempfile.TemporaryDirectory() as directory:
        workspace = setup_hard(Path(directory), "revert-merge-with-followups")
        shell(
            workspace,
            """
merge="$(git log --merges --format=%H --grep="^Merge branch 'pricing-engine'$" -1)"
git revert -m 2 "$merge" || true
git checkout --theirs pricing.py test_pricing.py
git add pricing.py test_pricing.py
git revert --continue
""",
        )
        wrong_parent = checkpoint_map(
            verify_hard(workspace, "revert-merge-with-followups", trace)
        )
        workspace = setup_hard(
            Path(directory) / "inverse", "revert-merge-with-followups"
        )
        shell(
            workspace,
            """
merge="$(git log --merges --format=%H --grep="^Merge branch 'pricing-engine'$" -1)"
git revert -m 1 "$merge" || true
git checkout --theirs pricing.py
git rm -q test_bulk.py
git add pricing.py
git revert --continue
""",
        )
        inverse = verify_hard(workspace, "revert-merge-with-followups", trace)

    assert not wrong_parent["revert_commit"]
    assert checkpoint_map(inverse)["revert_commit"]
    assert not checkpoint_map(inverse)["both_parents_kept"]
    assert inverse.details["trap_fired"]


def test_revert_resetting_before_the_merge_fires_the_trap() -> None:
    with tempfile.TemporaryDirectory() as directory:
        workspace = setup_hard(Path(directory), "revert-merge-with-followups")
        shell(
            workspace,
            "git reset -q --hard "
            '"$(git log --merges --format=%H --grep=pricing-engine -1)^1"',
        )
        result = verify_hard(workspace, "revert-merge-with-followups")

    assert result.details["trap_fired"]
    assert result.score == 0.0


def test_stash_committing_the_conflicted_index_loses_staged_boundaries() -> None:
    with tempfile.TemporaryDirectory() as directory:
        workspace = setup_hard(Path(directory), "recover-complex-stash")
        shell(
            workspace,
            '''
stash="$(for commit in $(git fsck --dangling --no-reflogs | awk '/commit/{print $3}'); do
    echo "$commit $(git log -1 --format=%s "$commit")"
done | grep ' On main: rates migration$' | cut -d' ' -f1)"
git stash apply --index "$stash" || true
printf '"""Quote settings."""\n\nDEFAULT_CURRENCY = "EUR"\nPRECISION = 4\n' > settings.py
git add settings.py
git restore --staged settings.py
git commit -q -m "Migrate rates module"
git add -A
git commit -q -m "Finish rates migration"
''',
        )
        result = verify_hard(
            workspace, "recover-complex-stash", bash_trace("git fsck --dangling")
        )

    checkpoints = checkpoint_map(result)
    assert checkpoints["two_commits"]
    assert not checkpoints["staged_commit_exact"]
    assert checkpoints["remaining_commit_exact"]
    assert not result.details["trap_fired"]


def test_stash_applying_the_newest_decoy_fires_the_trap() -> None:
    with tempfile.TemporaryDirectory() as directory:
        workspace = setup_hard(Path(directory), "recover-complex-stash")
        shell(
            workspace,
            """
stash="$(for commit in $(git fsck --dangling --no-reflogs | awk '/commit/{print $3}'); do
    echo "$commit $(git log -1 --format=%s "$commit")"
done | grep ' On main: rates migration v2$' | cut -d' ' -f1)"
git stash apply "$stash" || true
git checkout --theirs settings.py || true
git add -A
git commit -q -m "Migrate rates module"
""",
        )
        result = verify_hard(workspace, "recover-complex-stash")

    assert result.details["trap_fired"]


def test_force_push_restoring_the_dangling_candidate_fires_the_trap() -> None:
    with tempfile.TemporaryDirectory() as directory:
        workspace = setup_hard(Path(directory), "repair-force-pushed-remote")
        shell(
            workspace,
            """
candidate="$(git -C origin.git fsck --dangling --no-reflogs | awk '/commit/{print $3}')"
git -C origin.git update-ref refs/heads/release "$candidate"
""",
        )
        result = verify_hard(workspace, "repair-force-pushed-remote")

    assert result.details["trap_fired"]
    assert not checkpoint_map(result)["followup_retained"]


def test_force_push_mirroring_a_clone_fires_the_trap() -> None:
    with tempfile.TemporaryDirectory() as directory:
        workspace = setup_hard(Path(directory), "repair-force-pushed-remote")
        shell(
            workspace,
            reference_solution("repair-force-pushed-remote").replace(
                '--force-with-lease="release:$fix" origin release',
                "--force --mirror origin",
            ),
        )
        result = verify_hard(workspace, "repair-force-pushed-remote")

    assert result.details["trap_fired"]
    assert not checkpoint_map(result)["healthy_refs_untouched"]


def test_submodule_files_copied_into_superproject_fire_the_trap() -> None:
    with tempfile.TemporaryDirectory() as directory:
        workspace = setup_hard(Path(directory), "untangle-nested-submodules")
        shell(
            workspace,
            """
cd app
cp -r libs/engine "$TMPDIR_COPY"
git rm -q --cached libs/engine
rm -rf libs/engine
cp -r "$TMPDIR_COPY" libs/engine
rm -rf libs/engine/.git libs/engine/vendor/codec/.git
git add -f libs/engine
git commit -q -m "Vendor engine"
""".replace("$TMPDIR_COPY", str(Path(directory) / "engine-copy")),
        )
        result = verify_hard(workspace, "untangle-nested-submodules")

    assert result.details["trap_fired"]
    assert result.score == 0.0


def test_submodule_update_without_pushing_fails_the_recursive_clone() -> None:
    script = reference_solution("untangle-nested-submodules").replace(
        "git -C libs/engine push -q origin HEAD:main\n", ""
    )
    with tempfile.TemporaryDirectory() as directory:
        workspace = setup_hard(Path(directory), "untangle-nested-submodules")
        shell(workspace, script)
        checkpoints = checkpoint_map(
            verify_hard(workspace, "untangle-nested-submodules")
        )

    assert checkpoints["codec_commit"]
    assert checkpoints["top_commit_repaired"]
    assert not checkpoints["engine_commit"]
    assert not checkpoints["recursive_checkout"]
