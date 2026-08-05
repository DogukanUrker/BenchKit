"""IFEval benchmark - verifiable instruction following.

541 prompts, each carrying one or more instructions that can be checked with
code ("write at least 300 words", "no commas", "wrap the title in <<>>").
A task counts as passed only when every instruction attached to it is
followed, which is the strict prompt-level accuracy reported in the paper.
"""

import json
from pathlib import Path

from benchkit.benchmarks.base import Task
from benchkit.benchmarks.ifeval_instructions import follows_instruction
from benchkit.benchmarks.utils import strip_think_tags

DATASET = Path(__file__).parent.parent / "datasets" / "ifeval.jsonl"


class IFEval:
    name = "ifeval"

    def load_tasks(self) -> list[Task]:
        tasks = []
        with open(DATASET, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                tasks.append(
                    Task(
                        id=d["task_id"],
                        prompt=d["prompt"],
                        metadata={
                            "instruction_id_list": d["instruction_id_list"],
                            "kwargs": d["kwargs"],
                        },
                    )
                )
        return tasks

    def build_prompt(self, task: Task) -> str:
        # IFEval prompts are self-contained: every constraint is spelled out in
        # the prompt itself, so no extra system framing is added.
        return task.prompt

    def evaluate(self, task: Task, response: str) -> bool:
        text = strip_think_tags(response)
        instructions = task.metadata["instruction_id_list"]
        kwargs_list = task.metadata["kwargs"]

        for instruction_id, kwargs in zip(instructions, kwargs_list, strict=False):
            args = dict(kwargs or {})
            if instruction_id == "combination:repeat_prompt":
                args.setdefault("prompt_to_repeat", task.prompt)
            if not follows_instruction(instruction_id, text, args):
                return False

        return True
