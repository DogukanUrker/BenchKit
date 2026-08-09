"""EvalPlus benchmarks with stronger HumanEval+ and MBPP+ test suites."""

from __future__ import annotations

import contextlib
import multiprocessing
import os
import platform
import textwrap
from functools import cache, lru_cache
from typing import Any

from benchkit.benchmarks.base import Task
from benchkit.benchmarks.utils import strip_think_tags

HUMANEVAL_SYSTEM = (
    "Complete the following Python function. "
    "Output ONLY the function body. "
    "No markdown fences, no explanation, no examples."
)
MBPP_SYSTEM = (
    "Write a Python function to solve the following task. "
    "Output ONLY the function. "
    "No markdown fences, no explanation."
)


@contextlib.contextmanager
def _fork_multiprocessing():
    """Temporarily swap ``multiprocessing`` primitives to use the *fork* context.

    EvalPlus calls ``multiprocessing.Process`` (and friends) using the
    module-level default, which on Python 3.13 / macOS is ``spawn``.  The
    ``spawn`` path validates every FD it intends to keep open across
    ``fork_exec``, raising ``ValueError: bad value(s) in fds_to_keep`` when
    the Textual TUI has high-numbered or already-closed descriptors.

    Switching to ``fork`` side-steps the issue entirely: the child inherits a
    copy of the parent's address space and no FDs need to be explicitly
    forwarded.  The swap is scoped so the rest of the process is unaffected.

    EvalPlus's ``evalplus.eval`` does ``from multiprocessing import Array, Value``
    so those names are bound on that module and must be patched there as well.
    """
    import evalplus.eval as _ev

    fork_ctx = multiprocessing.get_context("fork")
    mp_originals = (
        multiprocessing.Process,
        multiprocessing.Value,
        multiprocessing.Array,
    )
    ev_originals = (_ev.Value, _ev.Array)
    try:
        multiprocessing.Process = fork_ctx.Process
        multiprocessing.Value = fork_ctx.Value
        multiprocessing.Array = fork_ctx.Array
        _ev.Value = fork_ctx.Value
        _ev.Array = fork_ctx.Array
        yield
    finally:
        multiprocessing.Process, multiprocessing.Value, multiprocessing.Array = (
            mp_originals
        )
        _ev.Value, _ev.Array = ev_originals


def _evalplus_api():
    """Import EvalPlus lazily so normal startup stays lightweight."""
    # RLIMIT_AS is not reliably supported on macOS. EvalPlus otherwise tries
    # to impose a 4 GiB address-space limit that can make every check fail
    # before generated code starts.
    if platform.system() == "Darwin":
        os.environ.setdefault("EVALPLUS_MAX_MEMORY_BYTES", "-1")

    try:
        from evalplus.data import get_human_eval_plus, get_mbpp_plus
        from evalplus.eval import PASS
        from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
        from evalplus.evaluate import check_correctness
        from evalplus.gen.util import trusted_exec
    except ImportError as exc:
        raise RuntimeError(
            "EvalPlus is missing from the environment. Run: uv sync"
        ) from exc

    return (
        get_human_eval_plus,
        get_mbpp_plus,
        PASS,
        MBPP_OUTPUT_NOT_NONE_TASKS,
        check_correctness,
        trusted_exec,
    )


@lru_cache(maxsize=2)
def _problems(dataset: str) -> dict[str, dict[str, Any]]:
    get_human_eval_plus, get_mbpp_plus, *_ = _evalplus_api()
    if dataset == "humaneval":
        return get_human_eval_plus()
    if dataset == "mbpp":
        return get_mbpp_plus()
    raise ValueError(f"Unsupported EvalPlus dataset: {dataset}")


@cache
def _expected_output(dataset: str, task_id: str) -> dict[str, list[Any]]:
    """Compute the official oracle outputs once per task and process."""
    (
        _,
        _,
        _,
        output_not_none_tasks,
        _,
        trusted_exec,
    ) = _evalplus_api()
    problem = _problems(dataset)[task_id]
    oracle_code = problem["prompt"] + problem["canonical_solution"]
    output_not_none = (
        dataset == "mbpp" and problem["entry_point"] in output_not_none_tasks
    )

    base, base_time = trusted_exec(
        oracle_code,
        problem["base_input"],
        problem["entry_point"],
        record_time=True,
        output_not_none=output_not_none,
    )
    plus, plus_time = trusted_exec(
        oracle_code,
        problem["plus_input"],
        problem["entry_point"],
        record_time=True,
        output_not_none=output_not_none,
    )
    return {
        "base": base,
        "base_time": base_time,
        "plus": plus,
        "plus_time": plus_time,
    }


def _extract_code(response: str) -> str:
    """Strip reasoning and a single Markdown code fence."""
    text = strip_think_tags(response).rstrip()
    if "```python" in text:
        text = text.split("```python", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]
    return text.strip("\n")


def _human_solution(problem: dict[str, Any], response: str) -> str:
    code = _extract_code(response)
    entry_point = problem["entry_point"]
    if f"def {entry_point}" not in code:
        normalized = textwrap.dedent(code)
        solution = problem["prompt"] + textwrap.indent(normalized, "    ")
        try:
            compile(solution, "<benchkit-humaneval>", "exec")
        except (IndentationError, TabError):
            lines = code.splitlines()
            first_index = next(
                (index for index, line in enumerate(lines) if line.strip()), None
            )
            if first_index is not None and not lines[first_index].startswith(
                (" ", "\t")
            ):
                lines[first_index] = "    " + lines[first_index]
                normalized = textwrap.dedent("\n".join(lines))
                repaired = problem["prompt"] + textwrap.indent(normalized, "    ")
                try:
                    compile(repaired, "<benchkit-humaneval>", "exec")
                except (IndentationError, TabError):
                    pass
                else:
                    solution = repaired
        return solution

    imports = [
        line
        for line in problem["prompt"].splitlines()
        if line.startswith(("import ", "from "))
    ]
    import_block = "\n".join(imports)
    return f"{import_block}\n\n{code}" if imports else code


class _EvalPlusBenchmark:
    dataset: str
    name: str
    task_count: int
    system: str
    evaluation_activity = "executing code and running EvalPlus tests"

    def load_tasks(self) -> list[Task]:
        return [
            Task(
                id=task_id,
                prompt=problem["prompt"],
                metadata={
                    "entry_point": problem["entry_point"],
                    "canonical_solution": problem["canonical_solution"],
                },
            )
            for task_id, problem in _problems(self.dataset).items()
        ]

    def build_prompt(self, task: Task) -> str:
        return f"{self.system}\n\n{task.prompt}"

    def evaluate(self, task: Task, response: str) -> bool:
        *_, pass_status, _, check_correctness, _ = _evalplus_api()
        problem = _problems(self.dataset)[task.id]
        if self.dataset == "humaneval":
            solution = _human_solution(problem, response)
        else:
            solution = _extract_code(response)

        with _fork_multiprocessing():
            result = check_correctness(
                self.dataset,
                completion_id=0,
                problem=problem,
                solution=solution,
                expected_output=_expected_output(self.dataset, task.id),
                fast_check=True,
            )
        return result["base"][0] == pass_status and result["plus"][0] == pass_status


class HumanEvalPlus(_EvalPlusBenchmark):
    name = "humaneval-plus"
    dataset = "humaneval"
    task_count = 164
    system = HUMANEVAL_SYSTEM


class MBPPPlus(_EvalPlusBenchmark):
    name = "mbpp-plus"
    dataset = "mbpp"
    task_count = 378
    system = MBPP_SYSTEM
