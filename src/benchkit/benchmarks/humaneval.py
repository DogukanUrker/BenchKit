"""HumanEval benchmark - 164 code generation tasks, scored by pass@1."""

import json
import textwrap
from pathlib import Path

from benchkit.benchmarks.base import Task
from benchkit.benchmarks.utils import strip_think_tags
from benchkit.evaluation import EvaluationResult
from benchkit.executor import execute_with_feedback

DATASET = Path(__file__).parent.parent / "datasets" / "humaneval.jsonl"

SYSTEM = (
    "Complete the following Python function. "
    "Output ONLY the function body. "
    "No markdown fences, no explanation, no examples."
)


def _extract_code(response: str) -> str:
    """Strip reasoning and a single Markdown code fence."""
    text = strip_think_tags(response).rstrip()

    if "```python" in text:
        text = text.split("```python", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]

    return text.strip("\n")


def _assemble_solution(prompt: str, entry: str, code: str) -> str:
    """Join a completion onto the HumanEval prompt.

    Bare bodies are indented as a function body. If the first line is flush
    and the rest is already indented, indenting every line over-indents the
    body, so recover by indenting only that first line.
    """
    if f"def {entry}" in code:
        imports = [
            line for line in prompt.split("\n") if line.startswith(("import ", "from "))
        ]
        return "\n".join(imports) + "\n\n" + code if imports else code

    normalized = textwrap.dedent(code)
    solution = prompt + textwrap.indent(normalized, "    ")
    try:
        compile(solution, "<benchkit-humaneval>", "exec")
        return solution
    except (IndentationError, TabError):
        pass

    lines = code.splitlines()
    first_index = next(
        (index for index, line in enumerate(lines) if line.strip()), None
    )
    if first_index is not None and not lines[first_index].startswith((" ", "\t")):
        lines[first_index] = "    " + lines[first_index]
        normalized = textwrap.dedent("\n".join(lines))
        repaired = prompt + textwrap.indent(normalized, "    ")
        try:
            compile(repaired, "<benchkit-humaneval>", "exec")
        except (IndentationError, TabError):
            return solution
        return repaired
    return solution


class HumanEval:
    name = "humaneval"
    evaluation_activity = "executing code and running tests"

    def load_tasks(self) -> list[Task]:
        tasks = []
        with open(DATASET) as f:
            for line in f:
                d = json.loads(line)
                tasks.append(
                    Task(
                        id=d["task_id"],
                        prompt=d["prompt"],
                        metadata={
                            "test": d["test"],
                            "entry_point": d["entry_point"],
                        },
                    )
                )
        return tasks

    def build_prompt(self, task: Task) -> str:
        return f"{SYSTEM}\n\n{task.prompt}"

    def evaluate(self, task: Task, response: str) -> bool:
        return self.evaluate_with_feedback(task, response).passed

    def evaluate_with_feedback(self, task: Task, response: str) -> EvaluationResult:
        code = _extract_code(response)
        entry = task.metadata["entry_point"]
        fn_code = _assemble_solution(task.prompt, entry, code)
        full = fn_code + "\n\n" + task.metadata["test"] + f"\ncheck({entry})\n"
        result = execute_with_feedback(full)
        return EvaluationResult(float(result.passed), result.feedback)
