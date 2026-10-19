"""CRUXEval - predict what a short Python function returns, or what it was given.

800 functions, each with one input and the output running it produces. The
output task (``cruxeval-o``) shows the input and asks for the output; the input
task (``cruxeval-i``) shows the output and asks for any input that produces it.
Both come in the upstream direct and chain-of-thought prompts, as separate
registry keys so the gap between them shows up as two jobs side by side.

Scoring needs no judge. A predicted output is parsed with ``ast.literal_eval``
and compared with ``==`` against the recorded output, so ``'abc'`` and
``"abc"``, or ``(1, 2)`` written with odd spacing, all match. The one fallback:
when the recorded output is a string and the prediction is not a literal at
all, the bare text is compared with that string, so ``== abc`` counts for
``'abc'``. Anything else that is not a literal - an unsimplified expression, a
call - is wrong, which is stricter than upstream (it evaluates expressions)
and keeps model-written code out of output scoring entirely.

A predicted input is never string-matched, because many inputs can produce
the same output: the call is run against the function in the sandbox and
passes when it returns the recorded output.
"""

import ast
import json
import re
import secrets
from pathlib import Path

from benchkit.benchmarks.base import Task
from benchkit.benchmarks.utils import strip_think_tags
from benchkit.evaluation import EvaluationResult
from benchkit.executor import execute_with_feedback

DATASET = Path(__file__).parent.parent / "datasets" / "cruxeval.jsonl"

# Prompts are the upstream ones from facebookresearch/cruxeval (MIT), verbatim.
DIRECT_OUTPUT = """You are given a Python function and an assertion containing an input to the function. Complete the assertion with a literal (no unsimplified expressions, no function calls) containing the output when executing the provided code on the given input, even if the function is incorrect or incomplete. Do NOT output any extra information. Provide the full assertion with the correct output in [ANSWER] and [/ANSWER] tags, following the examples.

[PYTHON]
def f(n):
    return n
assert f(17) == ??
[/PYTHON]
[ANSWER]
assert f(17) == 17
[/ANSWER]

[PYTHON]
def f(s):
    return s + "a"
assert f("x9j") == ??
[/PYTHON]
[ANSWER]
assert f("x9j") == "x9ja"
[/ANSWER]

[PYTHON]
{code}
assert f({input}) == ??
[/PYTHON]
[ANSWER]
"""

COT_OUTPUT = """You are given a Python function and an assertion containing an input to the function. Complete the assertion with a literal (no unsimplified expressions, no function calls) containing the output when executing the provided code on the given input, even if the function is incorrect or incomplete. Do NOT output any extra information. Execute the program step by step before arriving at an answer, and provide the full assertion with the correct output in [ANSWER] and [/ANSWER] tags, following the examples.

[PYTHON]
def f(s):
    s = s + s
    return "b" + s + "a"
assert f("hi") == ??
[/PYTHON]
[THOUGHT]
Let's execute the code step by step:

1. The function f is defined, which takes a single argument s.
2. The function is called with the argument "hi", so within the function, s is initially "hi".
3. Inside the function, s is concatenated with itself, so s becomes "hihi".
4. The function then returns a new string that starts with "b", followed by the value of s (which is now "hihi"), and ends with "a".
5. The return value of the function is therefore "bhihia".
[/THOUGHT]
[ANSWER]
assert f("hi") == "bhihia"
[/ANSWER]

[PYTHON]
{code}
assert f({input}) == ??
[/PYTHON]
[THOUGHT]
"""

DIRECT_INPUT = """You will be given a function f and an output in the form f(??) == output. Find any input such that executing f on the input leads to the given output. There may be multiple answers, but you should only output one. In [ANSWER] and [/ANSWER] tags, complete the assertion with one such input that will produce the output when executing the function.

[PYTHON]
def f(my_list):
    count = 0
    for i in my_list:
        if len(i) % 2 == 0:
            count += 1
    return count
assert f(??) == 3
[/PYTHON]
[ANSWER]
assert f(["mq", "px", "zy"]) == 3
[/ANSWER]

[PYTHON]
def f(s1, s2):
    return s1 + s2
assert f(??) == "banana"
[/PYTHON]
[ANSWER]
assert f("ba", "nana") == "banana"
[/ANSWER]

[PYTHON]
{code}
assert f(??) == {output}
[/PYTHON]
[ANSWER]
"""

COT_INPUT = """You will be given a function f and an output in the form f(??) == output. Your task is to find any input such that executing f on the input leads to the given output. There may be multiple answers, but only output one. First, think step by step. You MUST surround the answer with [ANSWER] and [/ANSWER] tags. Express your answer as a passing assertion containing the input and the given output.

[PYTHON]
def f(x):
    return x + 1
assert f(??) == 17
[/PYTHON]
[THOUGHT]
To find an input such that executing f on the input leads to the given output, we can work backwards from the given assertion. We know that f(??) == 17.

Since the function f(x) returns x + 1, for f(??) to be equal to 17, the value of ?? should be 16.
[/THOUGHT]
[ANSWER]
assert f(16) == 17
[/ANSWER]

[PYTHON]
{code}
assert f(??) == {output}
[/PYTHON]
[THOUGHT]
"""

TEMPLATES = {
    ("output", "direct"): DIRECT_OUTPUT,
    ("output", "cot"): COT_OUTPUT,
    ("input", "direct"): DIRECT_INPUT,
    ("input", "cot"): COT_INPUT,
}

NO_ASSERTION = (
    "No assertion of the form assert f(...) == ... was found in the answer. "
    "Reply with one such assertion inside [ANSWER] and [/ANSWER] tags."
)


def _answer_block(response: str) -> str:
    """The text the model meant as its answer.

    The prompt itself opens ``[ANSWER]`` (direct) or ``[THOUGHT]`` (CoT), so a
    response may carry both tags, only the closing one, or neither. The last
    complete block wins; a Markdown fence inside it is unwrapped.
    """
    text = strip_think_tags(response)
    if "[ANSWER]" in text:
        text = text.rsplit("[ANSWER]", 1)[1].split("[/ANSWER]", 1)[0]
    elif "[/ANSWER]" in text:
        text = text.rsplit("[/ANSWER]", 1)[0]
        text = text.rsplit("[/THOUGHT]", 1)[-1]
    fences = re.findall(r"```[\w-]*\n?(.*?)```", text, re.DOTALL)
    if fences:
        text = fences[-1]
    return text.strip()


def _assertion(block: str) -> str | None:
    """The last ``assert`` statement in the block, if any."""
    lines = block.splitlines()
    for index in range(len(lines) - 1, -1, -1):
        if lines[index].lstrip().startswith("assert"):
            tail = "\n".join(line.strip() for line in lines[index:])
            return tail if _parses(tail) else lines[index].strip()
    return None


def _parses(source: str) -> bool:
    try:
        ast.parse(source)
    except SyntaxError:
        return False
    return True


def _is_f_call(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "f"
    )


def _equality(source: str) -> tuple[str, ast.expr, ast.expr] | None:
    """Split ``[assert] left == right`` into its parsed sides."""
    text = source.strip()
    if text.startswith("assert"):
        text = text[len("assert") :].strip()
    try:
        node = ast.parse(text, mode="eval").body
    except SyntaxError:
        return None
    if (
        isinstance(node, ast.Compare)
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.Eq)
    ):
        return text, node.left, node.comparators[0]
    return None


def predicted_output(response: str) -> str | None:
    """The output the model predicted, as source text."""
    block = _answer_block(response)
    candidate = _assertion(block) or block
    split = _equality(candidate)
    if split is not None:
        text, left, right = split
        side = left if _is_f_call(right) and not _is_f_call(left) else right
        return ast.get_source_segment(text, side)
    # Not parseable: a bare word after ``==`` is still a usable prediction.
    if "==" in candidate:
        candidate = candidate.rsplit("==", 1)[1]
    candidate = candidate.split("#", 1)[0].strip()
    return candidate or None


def predicted_call(response: str) -> str | None:
    """The ``f(...)`` call the model proposed, as source text."""
    block = _answer_block(response)
    candidate = _assertion(block) or block
    split = _equality(candidate)
    if split is not None:
        text, left, right = split
        for side in (left, right):
            if _is_f_call(side):
                return ast.get_source_segment(text, side)
        return None
    try:
        node = ast.parse(candidate, mode="eval").body
    except SyntaxError:
        return None
    if _is_f_call(node):
        return ast.get_source_segment(candidate, node)
    # A bare input, such as ``"ba", "nana"``, is the argument list itself.
    return f"f({candidate})"


def outputs_match(predicted: str | None, expected: str) -> bool:
    """Literal comparison of a predicted output with the recorded one."""
    if predicted is None:
        return False
    target = ast.literal_eval(expected)
    try:
        value = ast.literal_eval(predicted)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return isinstance(target, str) and predicted.strip() == target
    try:
        return bool(value == target)
    except Exception:
        return False


class _CRUXEval:
    name: str
    task: str  # "output" or "input"
    style: str  # "direct" or "cot"
    task_count = 800

    def load_tasks(self) -> list[Task]:
        tasks = []
        with open(DATASET, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                tasks.append(
                    Task(
                        id=d["task_id"],
                        prompt=d["function"],
                        metadata={
                            "input": d["input"],
                            "output": d["output"],
                            # The demo client answers with this.
                            "canonical_solution": (
                                f"[ANSWER]\nassert f({d['input']}) == "
                                f"{d['output']}\n[/ANSWER]"
                            ),
                        },
                    )
                )
        return tasks

    def build_prompt(self, task: Task) -> str:
        template = TEMPLATES[(self.task, self.style)]
        return template.format(
            code=task.prompt,
            input=task.metadata["input"],
            output=task.metadata["output"],
        )

    def evaluate(self, task: Task, response: str) -> bool:
        return self.evaluate_with_feedback(task, response).passed

    def summary_fields(self, records: list[object]) -> dict:
        """Record which half and which prompt style a score came from."""
        return {"cruxeval_task": self.task, "cruxeval_prompt": self.style}


class _OutputPrediction(_CRUXEval):
    task = "output"

    def evaluate_with_feedback(self, task: Task, response: str) -> EvaluationResult:
        predicted = predicted_output(response)
        if predicted is None:
            return EvaluationResult(0.0, NO_ASSERTION)
        if outputs_match(predicted, task.metadata["output"]):
            return EvaluationResult(1.0)
        return EvaluationResult(
            0.0,
            "The predicted output does not equal the value the function returns. "
            "Trace the code again; the answer must be a Python literal.",
        )


class _InputPrediction(_CRUXEval):
    task = "input"
    evaluation_activity = "running the predicted input"

    def evaluate_with_feedback(self, task: Task, response: str) -> EvaluationResult:
        call = predicted_call(response)
        if call is None:
            return EvaluationResult(0.0, NO_ASSERTION)
        # Same check as upstream, ``assert output == f(input)``. The input is
        # model-written code, so a pass also needs a fresh sentinel printed
        # after the assertion: an input that exits early cannot fake one.
        sentinel = secrets.token_hex(16)
        program = (
            f"{task.prompt}\n\n"
            f"assert {task.metadata['output']} == {call}\n"
            f"print({sentinel!r})\n"
        )
        result = execute_with_feedback(program, timeout=5, sentinel=sentinel)
        if result.passed:
            return EvaluationResult(1.0)
        if "assertion failed" in result.feedback:
            return EvaluationResult(
                0.0,
                "The proposed input ran, but the function returned a different "
                "value than the given output.",
            )
        return EvaluationResult(0.0, result.feedback)


class CRUXEvalOutput(_OutputPrediction):
    name = "cruxeval-o"
    style = "direct"


class CRUXEvalOutputCoT(_OutputPrediction):
    name = "cruxeval-o-cot"
    style = "cot"


class CRUXEvalInput(_InputPrediction):
    name = "cruxeval-i"
    style = "direct"


class CRUXEvalInputCoT(_InputPrediction):
    name = "cruxeval-i-cot"
    style = "cot"
