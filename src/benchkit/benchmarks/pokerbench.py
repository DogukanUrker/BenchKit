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
structured CSV that sits beside each prompt file, matched to each prompt by
its cards and action (``build_split``). They give decision models their
options and the repair policy its option count; a prompt that matches no
scenario, or whose label is not a legal move, fails loudly instead of scoring.
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
REVISION = "7ac61f961c81a50fc0f667820b2fb0e432dfec0d"
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


_RANKS = {
    "ace": "A",
    "king": "K",
    "queen": "Q",
    "jack": "J",
    "ten": "T",
    "nine": "9",
    "eight": "8",
    "seven": "7",
    "six": "6",
    "five": "5",
    "four": "4",
    "three": "3",
    "two": "2",
}
_CARD = re.compile(r"(\w+) of (\w+)", re.IGNORECASE)
_HOLDING = re.compile(r"your holding is \[([^\]]+)\]")
_POSITION = re.compile(r"your position is (\w+)")
_POT = re.compile(r"current pot size is ([\d.]+) chips")


def _cards(text: str) -> frozenset[str]:
    """``King of Heart and Two Of Club`` -> ``{"Kh", "2c"}``."""
    return frozenset(
        _RANKS[rank.lower()] + suit[0].lower() for rank, suit in _CARD.findall(text)
    )


def _short_cards(text: str) -> frozenset[str]:
    """``Ks7h2d`` -> ``{"Ks", "7h", "2d"}``."""
    return frozenset(re.findall(r"..", text))


def _prompt_spot(split: str, row: dict) -> tuple[tuple, float]:
    """Join key and pot for one prompt, read from its scenario text."""
    text = row["instruction"]
    holding = _cards(_HOLDING.search(text).group(1))
    label = parse_action(row["output"])
    pot = float(_POT.search(text).group(1))
    if split == "preflop":
        return ("Preflop", _POSITION.search(text).group(1), holding, label), pot
    street, board = "Flop", frozenset()
    for name in ("flop", "turn", "river"):
        dealt = re.search(rf"The {name} comes (.*?)(?:, then|\.|\n)", text, re.I)
        if dealt:
            street, board = name.title(), board | _cards(dealt.group(1))
    return (street, board, holding, label), pot


def _scenario_spot(split: str, game: dict) -> tuple[tuple, float]:
    """Join key and pot for one scenario row, in ``_prompt_spot``'s shape."""
    label = parse_action(game["correct_decision"])
    pot = float(game["pot_size"])
    if split == "preflop":
        holding = _short_cards(game["hero_holding"])
        return ("Preflop", game["hero_pos"], holding, label), pot
    street = game["evaluation_at"].title()
    dealt = ["board_flop"] + ["board_turn"] * (street != "Flop")
    dealt += ["board_river"] * (street == "River")
    board = frozenset().union(*(_short_cards(game[key]) for key in dealt))
    return (street, board, _short_cards(game["holding"]), label), pot


def build_split(split: str, prompts: Path, scenarios: Path) -> list[Task]:
    """Join one split's prompts with the scenario rows that give their moves.

    Upstream shuffled the two files independently, so rows are matched on
    content: street, hole cards, the board (post-flop) or hero position
    (pre-flop) and the solver's action, with the pot breaking ties. Every
    scenario is used once, and a prompt with no match, or with candidates
    whose legal moves disagree, raises.
    """
    rows = json.loads(prompts.read_text(encoding="utf-8"))
    with open(scenarios, encoding="utf-8", newline="") as f:
        games = list(csv.DictReader(f))
    if len(rows) != len(games):
        raise ValueError(
            f"PokerBench {split}: {len(rows)} prompts but {len(games)} scenarios"
        )
    index: dict[tuple, list[tuple[float, list]]] = {}
    for game in games:
        key, pot = _scenario_spot(split, game)
        moves = [
            parse_action(str(m)) for m in ast.literal_eval(game["available_moves"])
        ]
        index.setdefault(key, []).append((pot, moves))

    tasks = []
    for number, row in enumerate(rows):
        where = f"PokerBench {split} prompt {number}"
        key, pot = _prompt_spot(split, row)
        candidates = index.get(key, [])
        if len({tuple(moves) for _, moves in candidates}) > 1:
            candidates = [c for c in candidates if abs(c[0] - pot) < 0.05]
        if not candidates or len({tuple(moves) for _, moves in candidates}) > 1:
            raise ValueError(
                f"{where}: {len(candidates)} scenarios match {row['output']!r} "
                "with this hand and board"
            )
        match = candidates[0]
        index[key].remove(match)
        moves = match[1]
        if None in moves or key[-1] not in moves:
            raise ValueError(f"{where}: {row['output']!r} is not a legal move {moves}")
        tasks.append(
            Task(
                id=f"PokerBench/{split}/{number}",
                prompt=row["instruction"],
                metadata={"answer": key[-1], "choices": moves, "street": key[0]},
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
