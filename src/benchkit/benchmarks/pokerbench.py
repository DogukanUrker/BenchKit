"""PokerBench - pick the solver's play in a No Limit Texas Hold'em spot.

Each task is one decision from the PokerBench test splits (1k pre-flop and 10k
post-flop spots, Zhuang et al., AAAI 2025): the authors' natural-language game
scenario, scored against the action a GTO solver chose. The prompt is upstream's
verbatim and does not list the legal moves, so a bet or raise only counts with
the solver's exact size - upstream's exact-match metric. Getting the action
right with the wrong size is reported separately as action accuracy.

Data is fetched from the authors' Hugging Face release at a pinned revision
on first use (or with ``benchkit datasets install pokerbench``) into
``~/.cache/benchkit/datasets/pokerbench/``. The legal moves come from the
structured CSV that sits beside each prompt file.
They give decision models their options and the repair policy its option
count; every row's label is checked against them when the data loads, so a
misaligned file fails loudly instead of scoring against the wrong spot.
"""

from __future__ import annotations

import ast
import csv
import json
import re
from functools import cache
from pathlib import Path

from benchkit.benchmarks.base import Task
from benchkit.benchmarks.downloads import Dataset, Progress, cache_root
from benchkit.benchmarks.utils import strip_think_tags
from benchkit.benchmarks.vision import (
    _download_file,
    _http_client,
    _url,
    category_scores,
    spread,
)
from benchkit.evaluation import EvaluationResult

# The authors' Hub release, pinned; a new revision is a deliberate suite update.
DATASET = "RZ412/PokerBench"
REVISION = "591f646fcb5d822f1e234fe09ad2526f50635d93"
FILES = {
    "preflop": (
        "preflop_1k_test_set_prompt_and_label.json",
        "preflop_1k_test_set_game_scenario_information.csv",
    ),
    "postflop": (
        "postflop_10k_test_set_prompt_and_label.json",
        "postflop_10k_test_set_game_scenario_information.csv",
    ),
}
SPLITS = tuple(FILES)
STREETS = ("Preflop", "Flop", "Turn", "River")

_VERBS = {
    "fold": "fold",
    "check": "check",
    "call": "call",
    "bet": "bet",
    "raise": "raise",
    "allin": "all in",
    "shove": "all in",
    "jam": "all in",
}
_ACTION = re.compile(
    r"\b(fold|check|call|bet|raise|all[\s_-]?in|shove|jam)\b"
    r"(?:\s*(?:to|of|by)?\s*\$?(\d+(?:\.\d+)?)\s*(?:bb|chips?)?\b)?",
    re.IGNORECASE,
)
# Pre-flop scenario files write a raise as its size alone, e.g. ``13.0bb``.
_BARE_RAISE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*bb\s*$", re.IGNORECASE)


def _canonical(verb: str, amount: str | None) -> str:
    verb = _VERBS[re.sub(r"[\s_-]", "", verb.lower())]
    if verb in ("bet", "raise") and amount is not None:
        return f"{verb} {float(amount):g}"
    return verb


def parse_action(text: str) -> str | None:
    """Return the canonical action one move label names, or None.

    Canonical actions are ``fold``, ``check``, ``call``, ``all in`` and
    ``bet N`` / ``raise N`` (``bet`` / ``raise`` when no size is given).
    """
    bare = _BARE_RAISE.match(text)
    if bare:
        return _canonical("raise", bare.group(1))
    match = _ACTION.search(text)
    return _canonical(match.group(1), match.group(2)) if match else None


def extract_action(response: str) -> str | None:
    """Return the action a response commits to, or None.

    The first action on the last line wins, since upstream asks for the action
    alone and a model that explains anyway tends to end on its answer.
    Otherwise the last action anywhere in the response is taken.
    """
    text = strip_think_tags(response).strip()
    lines = [line for line in text.splitlines() if line.strip()]
    if lines:
        match = _ACTION.search(lines[-1])
        if match:
            return _canonical(match.group(1), match.group(2))
    matches = list(_ACTION.finditer(text))
    if matches:
        return _canonical(matches[-1].group(1), matches[-1].group(2))
    return None


def verb(action: str | None) -> str | None:
    """The action without its size: ``raise 13`` -> ``raise``."""
    return action.rsplit(" ", 1)[0] if action and action[-1].isdigit() else action


def _compatible(left: str, right: str) -> bool:
    """Same action, and the same size whenever both name one."""
    if verb(left) != verb(right):
        return False
    return left == right or left in ("bet", "raise") or right in ("bet", "raise")


def _json_rows(path: Path) -> list[dict]:
    """Rows of a prompt file, written either as one JSON array or as JSONL."""
    text = path.read_text(encoding="utf-8")
    try:
        rows = json.loads(text)
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    return rows if isinstance(rows, list) else [rows]


def build_split(split: str, prompts: Path, scenarios: Path) -> list[Task]:
    """Join one split's prompt file with its scenario CSV, row by row."""
    rows = _json_rows(prompts)
    with open(scenarios, encoding="utf-8", newline="") as f:
        games = list(csv.DictReader(f))
    if len(rows) != len(games):
        raise ValueError(
            f"PokerBench {split}: {len(rows)} prompts but {len(games)} scenarios"
        )
    tasks = []
    for index, (row, game) in enumerate(zip(rows, games, strict=True)):
        where = f"PokerBench {split} row {index}"
        label = parse_action(row["output"])
        solver = parse_action(game["correct_decision"])
        if label is None or solver is None or not _compatible(label, solver):
            raise ValueError(
                f"{where}: prompt label {row['output']!r} does not match "
                f"scenario decision {game['correct_decision']!r}"
            )
        moves = [
            parse_action(str(m)) for m in ast.literal_eval(game["available_moves"])
        ]
        matches = [move for move in moves if move and _compatible(move, label)]
        if None in moves or len(matches) != 1:
            raise ValueError(
                f"{where}: {row['output']!r} is not exactly one of the legal "
                f"moves {game['available_moves']}"
            )
        street = "Preflop" if split == "preflop" else game["evaluation_at"].title()
        tasks.append(
            Task(
                id=f"PokerBench/{split}/{index}",
                prompt=row["instruction"],
                metadata={"answer": matches[0], "choices": moves, "street": street},
            )
        )
    return tasks


def data_dir() -> Path:
    return cache_root() / "pokerbench" / REVISION[:12]


def _installed() -> bool:
    return (data_dir() / ".complete").exists()


def _install(_progress: Progress | None = None) -> None:
    """Fetch the four test files once; the marker is written last."""
    root = data_dir()
    if _installed():
        return
    root.mkdir(parents=True, exist_ok=True)
    with _http_client() as client:
        for name in (name for pair in FILES.values() for name in pair):
            if not (root / name).exists():
                _download_file(client, _url(DATASET, REVISION, name), root / name)
    (root / ".complete").touch()


@cache
def _streets() -> dict[str, str]:
    """Street of every task, by task id."""
    return {task.id: task.metadata["street"] for task in PokerBench().load_tasks()}


class PokerBench:
    name = "pokerbench"
    decision_instructions = "Which legal action is the solver's play in this spot?"
    task_count = 11000
    data_source = Dataset(
        summary="PokerBench test splits: 1k pre-flop and 10k post-flop spots",
        size=None,
        location=data_dir,
        is_installed=_installed,
        install=_install,
    )

    def load_tasks(self) -> list[Task]:
        _install()
        tasks = [
            task
            for split in SPLITS
            for task in build_split(
                split, data_dir() / FILES[split][0], data_dir() / FILES[split][1]
            )
        ]
        # Pre-flop spots are a tenth of the set; keep that share in any slice.
        return spread(tasks, key=lambda task: task.metadata["street"])

    def build_prompt(self, task: Task) -> str:
        return task.prompt

    def decision_criteria(self, task: Task) -> dict[str, str]:
        """The spot's legal moves, keyed by canonical action."""
        return {move: move for move in task.metadata["choices"]}

    def evaluate_with_feedback(self, task: Task, response: str) -> EvaluationResult:
        predicted = extract_action(response)
        expected = task.metadata["answer"]
        if predicted is None:
            feedback = (
                "The answer did not name an action. Reply with exactly one of "
                "fold, check, call, bet N, raise N or all in, and nothing else."
            )
        elif predicted != expected:
            feedback = (
                f"{predicted} is not the solver's play in this spot. Reconsider "
                "and reply with one action only."
            )
        else:
            feedback = ""
        return EvaluationResult(
            score=float(predicted == expected),
            feedback=feedback,
            details={
                "predicted_action": predicted,
                "action_match": verb(predicted) == verb(expected),
            },
        )

    def evaluate(self, task: Task, response: str) -> bool:
        return extract_action(response) == task.metadata["answer"]

    def summary_fields(self, records: list[object]) -> dict:
        """Action accuracy (size ignored) and exact match by street.

        Tasks that never reached the verifier count as a wrong action, exactly
        as they count against exact match.
        """
        streets = _streets()
        scored = [record for record in records if record.task_id in streets]
        if not scored:
            return {}
        matched = sum(
            bool((getattr(record, "workspace", None) or {}).get("action_match"))
            for record in scored
        )
        by_street = category_scores(scored, streets)
        return {
            "poker_action_accuracy": round(100 * matched / len(scored), 1),
            "poker_street_scores": {
                street: by_street[street] for street in STREETS if street in by_street
            },
        }
