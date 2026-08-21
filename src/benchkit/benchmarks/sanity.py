"""Fast 25-task sanity check across core model and harness behaviors."""

import json
from pathlib import Path

from benchkit.benchmarks.arc import ARC
from benchkit.benchmarks.base import Task
from benchkit.benchmarks.gsm8k import GSM8K
from benchkit.benchmarks.ifeval import IFEval
from benchkit.benchmarks.piqa import PIQA
from benchkit.benchmarks.utils import strip_think_tags
from benchkit.evaluation import EvaluationResult
from benchkit.executor import execute_with_feedback

CODE_DATASET = Path(__file__).parent.parent / "datasets" / "sanity_code.jsonl"
CODE_SYSTEM = (
    "Complete the following Python function. "
    "Output ONLY the function body. "
    "No markdown fences, no explanation, no examples."
)

# Hand-reviewed for clarity and complementary coverage. These are intentionally
# easy-to-moderate diagnostics: a strong model failing several usually indicates
# a prompt template, output parsing, quantization, or generation problem.
SELECTION = (
    ("code", "sanity-code", "SanityCode/0"),  # case-insensitive sets
    ("code", "sanity-code", "SanityCode/2"),  # overlapping substrings
    ("code", "sanity-code", "SanityCode/4"),  # numeric edge cases
    ("code", "sanity-code", "SanityCode/7"),  # parsing and validation
    ("code", "sanity-code", "SanityCode/8"),  # branching string logic
    ("math", "gsm8k", "GSM8K/0"),  # chained arithmetic
    ("math", "gsm8k", "GSM8K/5"),  # discount pattern
    ("math", "gsm8k", "GSM8K/7"),  # percentage and rates
    ("math", "gsm8k", "GSM8K/13"),  # reverse word problem
    ("math", "gsm8k", "GSM8K/19"),  # average-speed constraint
    ("instruction", "ifeval", "IFEval/1001"),  # forbidden punctuation
    ("instruction", "ifeval", "IFEval/1005"),  # required placeholders
    ("instruction", "ifeval", "IFEval/1019"),  # lowercase output
    ("instruction", "ifeval", "IFEval/1075"),  # valid JSON only
    ("instruction", "ifeval", "IFEval/1107"),  # two simultaneous constraints
    ("science", "arc", "ARC/4"),  # gravity and mass
    ("science", "arc", "ARC/9"),  # atomic units
    ("science", "arc", "ARC/13"),  # experimental method
    ("science", "arc", "ARC/17"),  # heat transfer
    ("science", "arc", "ARC/19"),  # evidence and hypothesis
    ("commonsense", "piqa", "PIQA/6"),  # safe container reuse
    ("commonsense", "piqa", "PIQA/14"),  # tool construction
    ("commonsense", "piqa", "PIQA/18"),  # kitchen tool choice
    ("commonsense", "piqa", "PIQA/23"),  # appliance operation
    ("commonsense", "piqa", "PIQA/24"),  # material suitability
)


def _extract_code(response: str) -> str:
    text = strip_think_tags(response).rstrip()
    if "```python" in text:
        text = text.split("```python", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]
    text = text.strip("\n")
    lines = text.split("\n")
    if any(
        line.startswith(("def ", "class ", "import ", "from ", "@")) for line in lines
    ):
        return text
    first = next((line for line in lines if line.strip()), "")
    if first and not first.startswith((" ", "\t")):
        return "\n".join(("    " + line) if line.strip() else line for line in lines)
    return text


class _SanityCode:
    name = "sanity-code"

    def load_tasks(self) -> list[Task]:
        tasks = []
        with open(CODE_DATASET, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                tasks.append(
                    Task(
                        id=row["task_id"],
                        prompt=row["prompt"],
                        metadata={
                            "test": row["test"],
                            "entry_point": row["entry_point"],
                            "canonical_solution": row["canonical_solution"],
                        },
                    )
                )
        return tasks

    def build_prompt(self, task: Task) -> str:
        return f"{CODE_SYSTEM}\n\n{task.prompt}"

    def evaluate(self, task: Task, response: str) -> bool:
        return self.evaluate_with_feedback(task, response).passed

    def evaluate_with_feedback(self, task: Task, response: str) -> EvaluationResult:
        code = _extract_code(response)
        entry = task.metadata["entry_point"]
        if f"def {entry}" in code:
            imports = [
                line
                for line in task.prompt.split("\n")
                if line.startswith(("import ", "from "))
            ]
            function_code = "\n".join(imports) + "\n\n" + code if imports else code
        else:
            function_code = task.prompt + code
        full = function_code + "\n\n" + task.metadata["test"] + f"\ncheck({entry})\n"
        result = execute_with_feedback(full)
        return EvaluationResult(float(result.passed), result.feedback)


class Sanity:
    """Dispatch the curated checks to their native deterministic evaluators."""

    name = "sanity"
    task_count = len(SELECTION)

    def __init__(self) -> None:
        self._benchmarks = {
            "sanity-code": _SanityCode(),
            "gsm8k": GSM8K(),
            "ifeval": IFEval(),
            "arc": ARC(),
            "piqa": PIQA(),
        }

    def load_tasks(self) -> list[Task]:
        tasks_by_source = {
            name: {task.id: task for task in benchmark.load_tasks()}
            for name, benchmark in self._benchmarks.items()
        }
        tasks = []
        for index, (category, source, source_id) in enumerate(SELECTION):
            source_task = tasks_by_source[source][source_id]
            metadata = dict(source_task.metadata)
            metadata.update(
                sanity_category=category,
                source_benchmark=source,
                source_task_id=source_id,
            )
            tasks.append(
                Task(id=f"Sanity/{index}", prompt=source_task.prompt, metadata=metadata)
            )
        return tasks

    @staticmethod
    def _source_task(task: Task) -> Task:
        metadata = dict(task.metadata)
        source_id = metadata.pop("source_task_id")
        metadata.pop("source_benchmark")
        metadata.pop("sanity_category")
        return Task(id=source_id, prompt=task.prompt, metadata=metadata)

    def build_prompt(self, task: Task) -> str:
        benchmark = self._benchmarks[task.metadata["source_benchmark"]]
        return benchmark.build_prompt(self._source_task(task))

    def evaluate(self, task: Task, response: str) -> bool | float:
        benchmark = self._benchmarks[task.metadata["source_benchmark"]]
        return benchmark.evaluate(self._source_task(task), response)

    def evaluate_with_feedback(self, task: Task, response: str) -> EvaluationResult:
        benchmark = self._benchmarks[task.metadata["source_benchmark"]]
        source_task = self._source_task(task)
        evaluator = getattr(benchmark, "evaluate_with_feedback", None)
        if evaluator is not None:
            return evaluator(source_task, response)
        return EvaluationResult(float(benchmark.evaluate(source_task, response)))
