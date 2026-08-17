"""Deterministic Git repository repair tasks executed by stock Pi."""

from __future__ import annotations

import json
import re
from pathlib import Path

from benchkit.benchmarks.base import Task
from benchkit.evaluation import EvaluationResult
from benchkit.sandbox import DockerTaskEnvironment, git_surgery_pi_image

_TASK_ROOT = Path(__file__).parents[1] / "git_surgery"
_SEED = 424242
_SEARCH_RE = re.compile(r"\bgit\s+.*(?:log|show|rev-list|grep|bisect)\b")
_REWRITE_RE = re.compile(
    r"\bgit\s+(?:rebase|filter-branch|filter-repo|replace|commit-tree)\b"
)


def _command(call: dict) -> str:
    arguments = call.get("arguments")
    if not isinstance(arguments, dict):
        return ""
    return str(arguments.get("command") or "")


def _is_error(call: dict) -> bool:
    output = str(call.get("output") or "")
    return bool(call.get("is_error")) or bool(
        re.search(
            r"(?:"
            r"(?:exited with (?:code|status)|exit code)\s+[1-9]\d*"
            r"|(?:^|\n)\s*(?:fatal|error):"
            r"|index filter failed"
            r"|unknown (?:option|switch)"
            r"|usage:\s+git\b"
            r")",
            output,
            re.I,
        )
    )


def _valid_tool_arguments(call: dict) -> bool:
    arguments = call.get("arguments")
    if not isinstance(arguments, dict):
        return False
    required: dict[str, tuple[str, ...]] = {
        "bash": ("command",),
        "read": ("path",),
        "write": ("path", "content"),
        "edit": ("path", "oldText", "newText"),
    }
    fields = required.get(str(call.get("name") or ""))
    if fields is None:
        return bool(arguments)
    return all(isinstance(arguments.get(field), str) for field in fields)


def agentic_metrics(tool_trace: list[dict]) -> dict:
    """Derive deterministic agent behavior metrics from Pi's native trace."""
    calls = [call for call in tool_trace if isinstance(call, dict)]
    valid = sum(_valid_tool_arguments(call) for call in calls)
    invalid = len(calls) - valid
    errored = sum(_is_error(call) for call in calls)
    redundant = 0
    recoveries = 0
    previous_key = None
    for index, call in enumerate(calls):
        key = (
            str(call.get("name") or ""),
            json.dumps(call.get("arguments"), sort_keys=True, default=str),
        )
        if key == previous_key:
            redundant += 1
        previous_key = key
        if _is_error(call) and index + 1 < len(calls):
            next_call = calls[index + 1]
            next_key = (
                str(next_call.get("name") or ""),
                json.dumps(next_call.get("arguments"), sort_keys=True, default=str),
            )
            recoveries += int(next_key != key and not _is_error(next_call))
    destructive = sum(
        bool(
            re.search(
                r"(?:rm\s+-[^\n]*r[^\n]*\s+\.git(?=$|[\s;&|])|git\s+init\b)",
                _command(call),
            )
        )
        for call in calls
    )
    total = len(calls)
    return {
        "tool_schema_valid_calls": valid,
        "tool_schema_invalid_calls": invalid,
        "tool_schema_validity_rate": round(valid / total, 4) if total else 1.0,
        "errored_tool_calls": errored,
        "post_error_recoveries": recoveries,
        "post_error_recovery_rate": (
            round(recoveries / errored, 4) if errored else 1.0
        ),
        "redundant_tool_calls": redundant,
        "redundant_action_rate": round(redundant / total, 4) if total else 0.0,
        "destructive_action_count": destructive,
        "destructive_action_rate": round(destructive / total, 4) if total else 0.0,
    }


class GitSurgery:
    """Small agentic benchmark over real, stateful Git repositories."""

    name = "git-surgery"
    task_count = 1
    workspace_task = True
    evaluation_activity = "checking Git history with plumbing commands"
    list_note = "1 of 5 tasks implemented · requires Pi"

    def load_tasks(self) -> list[Task]:
        return [
            Task(
                id="secret-in-history",
                prompt="",
                metadata={"seed": _SEED},
            )
        ]

    def build_prompt(self, _task: Task) -> str:
        return (
            "Repair the Git repository in the current workspace. An AWS-style "
            "credential was committed earlier in the reachable history. Remove "
            "that credential from the entire reachable history while preserving "
            "all legitimate later changes and a reviewable commit history. The "
            "working tree includes tests; run them and verify your work. Do not "
            "delete or reinitialize the repository, and do not merely add a revert "
            "or remove the credential in a new commit. Work directly in the "
            "repository using Pi's stock tools."
        )

    def evaluate(self, _task: Task, _response: str) -> bool:
        raise RuntimeError("Git Surgery must be evaluated inside its workspace")

    def pi_image(self):
        return git_surgery_pi_image()

    def prepare_workspace(self, task: Task, environment: DockerTaskEnvironment) -> None:
        workspace = "/workspace/secret-in-history"
        environment.workdir = workspace
        environment.exec(["mkdir", "-p", workspace])
        environment.exec(
            [
                "bash",
                "/opt/git-surgery/secret-in-history/setup.sh",
                str(task.metadata["seed"]),
                workspace,
            ],
            timeout=30,
        )

    def verify_workspace(
        self,
        task: Task,
        environment: DockerTaskEnvironment,
        tool_trace: list[dict] | None = None,
    ) -> EvaluationResult:
        workspace = "/workspace/secret-in-history"
        trace = list(tool_trace or [])
        completed = environment.exec(
            [
                "bash",
                "/opt/git-surgery/secret-in-history/verify.sh",
                str(task.metadata["seed"]),
                workspace,
            ],
            timeout=60,
            check=False,
        )
        state: dict[str, tuple[bool, str]] = {}
        for line in completed.stdout.splitlines():
            parts = line.split("\t", 2)
            if len(parts) == 3:
                state[parts[0]] = (parts[1] == "1", parts[2])

        commands = [_command(call) for call in trace]
        searched = any(_SEARCH_RE.search(command) for command in commands)
        rewrote = any(_REWRITE_RE.search(command) for command in commands)
        history_preserved = state.get("history_preserved", (False, "missing"))
        secret_absent = state.get("secret_absent", (False, "missing"))
        checkpoint_specs = [
            (
                "located_offending_commit",
                1,
                searched,
                "history-search command observed",
            ),
            ("history_rewrite_path", 1, rewrote, "history-rewrite command observed"),
            (
                "history_rewrite_preserved_changes",
                2,
                rewrote and history_preserved[0],
                history_preserved[1],
            ),
            (
                "secret_absent_reachable",
                2,
                secret_absent[0],
                secret_absent[1],
            ),
            (
                "tests_pass",
                2,
                state.get("tests_pass", (False, "missing"))[0],
                state.get("tests_pass", (False, "missing"))[1],
            ),
        ]
        destructive = state.get("destructive_reinit", (True, "missing"))[0]
        checkpoints = [
            {
                "id": checkpoint_id,
                "weight": weight,
                "passed": passed,
                "awarded": weight if passed else 0,
                "evidence": evidence,
            }
            for checkpoint_id, weight, passed, evidence in checkpoint_specs
        ]
        checkpoints.append(
            {
                "id": "destructive_reinitialization",
                "weight": -4,
                "passed": destructive,
                "awarded": -4 if destructive else 0,
                "evidence": state.get("destructive_reinit", (True, "missing"))[1],
            }
        )
        positive = sum(item["awarded"] for item in checkpoints if item["weight"] > 0)
        penalty = -sum(item["awarded"] for item in checkpoints if item["weight"] < 0)
        score = max(0, positive - penalty) / 8
        metrics = agentic_metrics(trace)
        details = {
            "checkpoints": checkpoints,
            "positive_points": positive,
            "penalty_points": penalty,
            "max_points": 8,
            "trap_fired": destructive,
            "verifier_exit_code": completed.returncode,
            "verifier_stderr": completed.stderr[-4000:],
            "git_version": environment.exec(["git", "--version"]).stdout.strip(),
            "agentic_metrics": metrics,
        }
        return EvaluationResult(score=score, details=details)
