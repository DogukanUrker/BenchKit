"""HumanEval benchmark - 164 code generation tasks, scored by pass@1."""

import json
from pathlib import Path

from benchkit.benchmarks.base import Task
from benchkit.executor import execute

DATASET = Path(__file__).parent.parent / "datasets" / "humaneval.jsonl"

SYSTEM = (
    "Complete the following Python function. "
    "Output ONLY the function body. "
    "No markdown fences, no explanation, no examples."
)


def _extract_code(response: str) -> str:
    """Strip markdown fences if the model wrapped its output."""
    text = response.rstrip()

    # Strip <think>...</think> blocks (reasoning models like DeepSeek-R1)
    if "<think>" in text:
        start = text.index("<think>")
        if "</think>" in text:
            end = text.index("</think>") + len("</think>")
            text = text[:start] + text[end:]
        else:
            text = text[:start]
        text = text.rstrip()

    if "```python" in text:
        text = text.split("```python", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]

    # Ensure body is indented (models sometimes return unindented code)
    # But skip if it's a full function definition
    first = text.lstrip("\n").split("\n")[0] if text.strip() else ""
    if first and not first.startswith((" ", "\t", "def ", "class ")):
        lines = text.split("\n")
        lines = ["    " + line if line.strip() else line for line in lines]
        text = "\n".join(lines)

    return text


class HumanEval:
    name = "humaneval"

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
        code = _extract_code(response)
        entry = task.metadata["entry_point"]

        if f"def {entry}" in code:
            # Model gave full function - prepend any imports from the prompt
            imports = [
                line
                for line in task.prompt.split("\n")
                if line.startswith(("import ", "from "))
            ]
            fn_code = "\n".join(imports) + "\n\n" + code if imports else code
        else:
            fn_code = task.prompt + code

        full = fn_code + "\n\n" + task.metadata["test"] + f"\ncheck({entry})\n"
        return execute(full)
