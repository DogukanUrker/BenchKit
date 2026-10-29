"""MATH-500 - competition mathematics with LaTeX answers.

The 500-problem MATH subset from "Let's Verify Step by Step" (Lightman et al.,
2023), drawn from Hendrycks et al.'s MATH test split: seven subjects at
difficulty levels 1-5. The prompt is the common ``\\boxed{}`` instruction, and
results report accuracy per level and per subject, because in small models
levels 4-5 sit near zero and flatten the aggregate.

Answers are LaTeX expressions, so string equality is wrong (``1/2``,
``\\frac{1}{2}`` and ``\\dfrac12`` are the same answer). Grading has three
steps and no judge:

1. Extraction: the last ``\\boxed{...}`` (or ``\\fbox``) in the response, with
   balanced braces. Without one, the text after the last "final answer"
   or "answer is", else the last ``$...$`` math span. Nothing found is wrong.
2. Normalization, after the official MATH ``strip_string``: drop ``\\left`` /
   ``\\right``, spacing commands, ``\\$``, ``%``, degree marks, text units and
   thousands separators; unify ``\\dfrac`` / ``\\tfrac`` and brace-less
   ``\\frac12`` / ``\\sqrt3``; drop a short ``x =`` prefix; unwrap
   ``\\text{...}``; remove spaces. Equal strings pass.
3. SymPy, only when the strings differ: both sides are parsed by
   ``math_equiv`` (never ``eval``) and compared symbolically, then numerically.
   It runs in a worker process killed after ``EQUIVALENCE_TIMEOUT`` seconds,
   and a timeout is a mismatch.

Decimals are exact, so a rounded ``0.333`` is not ``\\frac13``; tuples and
intervals are ordered and their brackets must match; comma lists, ``\\{...\\}``
sets and ``\\pm`` are unordered. Harnesses that differ on any of these rules
(string-only graders, or ``math_verify`` with its float tolerance) will score
the same responses differently, so compare MATH-500 numbers across tools with
care.
"""

from __future__ import annotations

import atexit
import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from benchkit.benchmarks.base import Task
from benchkit.benchmarks.utils import strip_think_tags
from benchkit.benchmarks.vision import category_scores
from benchkit.evaluation import EvaluationResult

DATASET = Path(__file__).parent.parent / "datasets" / "math500.jsonl"
WORKER = Path(__file__).with_name("math_equiv.py")
EQUIVALENCE_TIMEOUT = 5.0

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."

NO_ANSWER = "No final answer was found. Finish with the final answer inside \\boxed{}."
WRONG_ANSWER = (
    "The final answer is not equivalent to the correct answer. Re-check the "
    "work and give a corrected final answer inside \\boxed{}."
)


# Extraction ----------------------------------------------------------------


def _braced(text: str, start: int) -> str | None:
    """The contents of the brace group opening at ``start``."""
    depth = 0
    for pos in range(start, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : pos]
    return None


def last_boxed(text: str) -> str | None:
    """The contents of the last ``\\boxed{}`` or ``\\fbox{}``."""
    pos = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if pos < 0:
        return None
    rest = text[pos:]
    brace = re.match(r"\\(?:boxed|fbox)\s*\{", rest)
    if brace:
        return _braced(text, pos + brace.end() - 1)
    bare = re.match(r"\\boxed\s+([^\s$]+)", rest)
    return bare.group(1) if bare else None


def extract_answer(response: str) -> str | None:
    text = strip_think_tags(response)
    boxed = last_boxed(text)
    if boxed is not None:
        return boxed.strip() or None
    stated = list(
        re.finditer(
            r"(?:final answer|answer is)\s*(?:is)?\s*:?\s*(.+)", text, re.IGNORECASE
        )
    )
    if stated:
        answer = stated[-1].group(1).strip()
        span = re.search(r"\$+(.+?)\$+|\\\((.+?)\\\)", answer)
        if span:
            answer = span.group(1) or span.group(2)
        return answer.strip().rstrip(".").strip() or None
    spans = re.findall(r"\$\$(.+?)\$\$|\\\[(.+?)\\\]|\$(.+?)\$|\\\((.+?)\\\)", text)
    if spans:
        last = next(part for part in spans[-1] if part)
        return last.strip() or None
    return None


# Normalization ---------------------------------------------------------------


def _fix_fracs(text: str) -> str:
    """``\\frac12`` -> ``\\frac{1}{2}``; braced arguments are left alone."""
    parts = text.split("\\frac")
    out = parts[0]
    for part in parts[1:]:
        out += "\\frac"
        if part.startswith("{"):
            close = _braced(part, 0)
            if close is None:
                return text
            numerator, rest = close, part[len(close) + 2 :]
        elif part:
            numerator, rest = part[0], part[1:]
        else:
            return text
        if rest.startswith("{"):
            out += "{" + numerator + "}" + rest
        elif rest:
            out += "{" + numerator + "}{" + rest[0] + "}" + rest[1:]
        else:
            return text
    return out


def _fix_sqrt(text: str) -> str:
    return re.sub(r"\\sqrt(\w)", r"\\sqrt{\1}", text)


def _fix_slash(text: str) -> str:
    match = re.fullmatch(r"(-?\d+)/(\d+)", text)
    if match:
        return f"\\frac{{{match.group(1)}}}{{{match.group(2)}}}"
    return text


_UNIT = r"\\(?:text|mbox|mathrm)\{\s*[a-zA-Z][a-zA-Z .]*\}(?:\^\{?\d\}?)?"


def normalize(answer: str) -> str:
    """Canonical string form of an answer, after MATH's ``strip_string``."""
    s = answer.strip().replace("\n", "")
    s = s.replace("\\!", "")
    # Spacing commands, but not the \\ row separator of a matrix.
    s = re.sub(
        r"\\\\|\\[,;:> ]", lambda m: m.group() if m.group() == "\\\\" else " ", s
    )
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\\displaystyle", "", s)
    s = s.replace("^{\\circ}", "").replace("^\\circ", "").replace("°", "")
    s = s.replace("\\$", "").replace("$", "")
    s = s.replace("\\%", "").replace("%", "")
    # Units after a value: "5.4 \text{ cents}", "864 \mbox{ inches}^2".
    s = re.sub(rf"(?<=\S)\s*{_UNIT}$", "", s)
    s = re.sub(r"(?<=\d)\s+(?:degrees?|units?|cents?|dollars?)$", "", s)
    s = re.sub(r"\\(?:text|textbf|mathrm|mbox|operatorname)\{([^{}]*)\}", r"\1", s)
    s = s.strip().rstrip(".").strip()
    s = re.sub(r"^[a-zA-Z]\s*\\in\s*", "", s)
    if s.count("=") == 1:
        lhs, rhs = s.split("=")
        if len(lhs.strip()) <= 2:
            s = rhs
    s = s.replace(" ", "")
    s = _fix_sqrt(s)
    s = _fix_fracs(s)
    s = _fix_slash(s)
    s = re.sub(r"(?<![\d.])\.(\d)", r"0.\1", s)
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", s):
        s = s.replace(",", "")
    # Base subscripts: 4210_{5} and 4210_5 are the same.
    s = re.sub(r"_\{(\w+)\}", r"_\1", s)
    choice = re.fullmatch(r"\(?([A-Ea-e])\)", s)
    if choice:
        s = choice.group(1).upper()
    if re.fullmatch(r"[A-Za-z]+", s) and len(s) > 1:
        s = s.lower()
    return s


# Symbolic equivalence ----------------------------------------------------------


class SympyChecker:
    """One long-lived ``math_equiv`` worker, killed and restarted on a deadline.

    Comparisons are serialized; each costs milliseconds once SymPy is loaded,
    which is why the worker is reused rather than started per answer.
    """

    def __init__(self, timeout: float = EQUIVALENCE_TIMEOUT) -> None:
        self.timeout = timeout
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        atexit.register(self.close)

    def _start(self) -> subprocess.Popen[str]:
        home = tempfile.gettempdir()
        env = {
            "PATH": os.environ.get("PATH") or os.defpath,
            "HOME": home,
            "TMPDIR": home,
            "LANG": os.environ.get("LANG") or "C.UTF-8",
        }
        process = subprocess.Popen(
            [sys.executable, "-I", str(WORKER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            env=env,
        )
        lines: queue.Queue[str | None] = queue.Queue()

        def pump() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                lines.put(line)
            lines.put(None)

        threading.Thread(target=pump, daemon=True, name="math-equiv").start()
        self._lines = lines
        return process

    def close(self) -> None:
        process, self._process = self._process, None
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()

    def equivalent(self, a: str, b: str) -> bool | None:
        """True or False, None if a side did not parse; a timeout is False."""
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                # The first call also pays for importing SymPy.
                self._process = self._start()
                deadline = self.timeout + 30.0
            else:
                deadline = self.timeout
            assert self._process.stdin is not None
            try:
                self._process.stdin.write(json.dumps({"a": a, "b": b}) + "\n")
                self._process.stdin.flush()
                line = self._lines.get(timeout=deadline)
            except (OSError, queue.Empty):
                line = None
            if line is None:
                self.close()
                return False
            return json.loads(line)["equal"]


_CHECKER: SympyChecker | None = None
_CHECKER_LOCK = threading.Lock()


def _checker() -> SympyChecker:
    global _CHECKER
    with _CHECKER_LOCK:
        if _CHECKER is None:
            _CHECKER = SympyChecker()
        return _CHECKER


def answers_match(predicted: str, reference: str) -> str | None:
    """How the answers matched (``"string"`` or ``"sympy"``), or None."""
    left, right = normalize(predicted), normalize(reference)
    if not left:
        return None
    if left == right:
        return "string"
    if _checker().equivalent(left, right):
        return "sympy"
    return None


# Benchmark -----------------------------------------------------------------------


class MATH500:
    """Competition math with LaTeX answers, graded without a judge."""

    name = "math-500"
    task_count = 500

    def load_tasks(self) -> list[Task]:
        tasks = []
        with open(DATASET, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                tasks.append(
                    Task(
                        id=row["task_id"],
                        prompt=row["problem"],
                        metadata={
                            "answer": row["answer"],
                            "subject": row["subject"],
                            "level": row["level"],
                            "unique_id": row["unique_id"],
                            # What the demo client answers with.
                            "canonical_solution": (
                                f"The final answer is $\\boxed{{{row['answer']}}}$."
                            ),
                        },
                    )
                )
        return tasks

    def build_prompt(self, task: Task) -> str:
        return f"{task.prompt}\n\n{INSTRUCTION}"

    def evaluate_with_feedback(self, task: Task, response: str) -> EvaluationResult:
        predicted = extract_answer(response)
        if predicted is None:
            return EvaluationResult(0.0, NO_ANSWER, details={"extracted": None})
        method = answers_match(predicted, task.metadata["answer"])
        details = {"extracted": predicted, "match": method}
        if method is None:
            return EvaluationResult(0.0, WRONG_ANSWER, details=details)
        return EvaluationResult(1.0, details=details)

    def evaluate(self, task: Task, response: str) -> bool:
        return self.evaluate_with_feedback(task, response).passed

    def summary_fields(self, records: list[object]) -> dict:
        """Accuracy per difficulty level and per subject."""
        tasks = self.load_tasks()
        levels = {task.id: f"Level {task.metadata['level']}" for task in tasks}
        subjects = {task.id: task.metadata["subject"] for task in tasks}
        by_level = category_scores(records, levels)
        if not by_level:
            return {}
        return {
            "math_level_scores": by_level,
            "math_subject_scores": category_scores(records, subjects),
        }
