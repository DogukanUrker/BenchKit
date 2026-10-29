"""PokerBench: action parsing, the prompt/scenario join, scoring and reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchkit import decision
from benchkit.benchmarks import pokerbench
from benchkit.benchmarks.base import Task
from benchkit.benchmarks.pokerbench import (
    PokerBench,
    build_split,
    extract_action,
    parse_action,
    verb,
)
from benchkit.engine import Engine, JobSpec, route_decision_jobs, task_count
from benchkit.report_markdown import render_markdown

RANK_NAMES = {short: name.title() for name, short in pokerbench._RANKS.items()}
SUIT_NAMES = {"h": "Heart", "c": "Club", "d": "Diamond", "s": "Spade"}
BOARD = {"flop": "Ks7h2d", "turn": "Jc", "river": "7c"}

# (hero_pos, hero_holding, correct_decision, prompt label, available_moves, pot)
PREFLOP = [
    ("SB", "KdKc", "call", "call", "['call', 'fold']", 117.0),
    ("BTN", "AhQd", "13.0bb", "raise 13.0", "['fold', '13.0bb', 'allin']", 5.5),
]
# (evaluation_at, holding, correct_decision, prompt label, available_moves, pot)
POSTFLOP = [
    ("Flop", "8h8c", "Check", "check", "['Check', 'Bet 24']", 32),
    ("Turn", "AhQd", "Bet 24", "bet 24", "['Check', 'Bet 10', 'Bet 24']", 40),
    ("River", "Kc5c", "Fold", "fold", "['Fold', 'Call', 'Raise 60']", 20),
    ("River", "9s9d", "Call", "call", "['Fold', 'Call']", 50),
    ("Flop", "AsKd", "Bet 4.5", "bet 4.5", "['Check', 'Bet 4.5']", 9),
    ("Turn", "QhJh", "Check", "check", "['Check', 'Bet 30']", 30),
    ("River", "AcAd", "Raise 60", "raise 60", "['Fold', 'Call', 'Raise 60']", 30),
    # Same hand, board and action: only the pot tells these two apart.
    ("Flop", "6h5h", "Check", "check", "['Check', 'Bet 12']", 12),
    ("Flop", "6h5h", "Check", "check", "['Check', 'Bet 6']", 6),
    ("River", "Tc9c", "Check", "check", "['Check', 'Bet 20']", 20),
]


def _long(card: str, of: str = "of") -> str:
    return f"{RANK_NAMES[card[0]]} {of} {SUIT_NAMES[card[1]]}"


def _prompt(position: str, holding: str, pot: float, street: str = "") -> str:
    """A scenario in upstream's wording, up to the street being decided."""
    hand = f"[{_long(holding[:2])} and {_long(holding[2:])}]"
    board = ""
    if street:
        flop = [_long(BOARD["flop"][i : i + 2], "Of") for i in (0, 2, 4)]
        board = f"The flop comes {flop[0]}, {flop[1]}, and {flop[2]}, then BB check.\n"
        for name in ("turn", "river")[: ("Flop", "Turn", "River").index(street)]:
            board += f"The {name} comes {_long(BOARD[name], 'Of')}, then BB check.\n"
    return (
        "\n\nYou are a specialist in playing 6-handed No Limit Texas Holdem.\n"
        f"In this hand, your position is {position}, and your holding is {hand}.\n"
        "Before the flop, HJ raise 2.0, and BB call.\n"
        f"{board}\nNow it is your turn to make a move.\n"
        f"To remind you, the current pot size is {pot} chips, and your holding is "
        f"{hand}.\n\nDo not explain your answer.\nYour optimal action is:"
    )


def _write(directory: Path, split: str, prompts: list[dict], games: list[dict]):
    """Write both files, the scenarios reversed as upstream's are reshuffled."""
    json_path = directory / f"{split}.json"
    csv_path = directory / f"{split}.csv"
    json_path.write_text(json.dumps(prompts), encoding="utf-8")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["", *games[0]])
        writer.writeheader()
        writer.writerows({"": i, **game} for i, game in enumerate(reversed(games)))
    return json_path, csv_path


def _preflop(directory: Path, spots=PREFLOP):
    prompts = [
        {"instruction": _prompt(pos, holding, pot), "output": label}
        for pos, holding, _dec, label, _moves, pot in spots
    ]
    games = [
        {
            "prev_line": "HJ/2.0bb/BB/call",
            "hero_pos": pos,
            "hero_holding": holding,
            "correct_decision": dec,
            "num_players": 2,
            "num_bets": 1,
            "available_moves": moves,
            "pot_size": pot,
        }
        for pos, holding, dec, _label, moves, pot in spots
    ]
    return _write(directory, "preflop", prompts, games)


def _postflop(directory: Path, spots=POSTFLOP):
    prompts = [
        {"instruction": _prompt("BTN", holding, pot, street), "output": label}
        for street, holding, _dec, label, _moves, pot in spots
    ]
    games = [
        {
            "preflop_action": "HJ/2.0bb/BB/call",
            "board_flop": BOARD["flop"],
            "board_turn": BOARD["turn"],
            "board_river": BOARD["river"],
            "aggressor_position": "OOP",
            "postflop_action": "OOP_CHECK",
            "evaluation_at": street,
            "available_moves": moves,
            "pot_size": pot,
            "hero_position": "IP",
            "holding": holding,
            "correct_decision": dec,
        }
        for street, holding, dec, _label, moves, pot in spots
    ]
    return _write(directory, "postflop", prompts, games)


@pytest.fixture
def installed(tmp_path, monkeypatch):
    """Upstream-shaped files in the dataset cache, as an install leaves them."""
    monkeypatch.setenv("BENCHKIT_DATASET_CACHE", str(tmp_path))
    pokerbench._streets.cache_clear()
    staging = tmp_path / "staging"
    staging.mkdir()
    files = dict(
        zip(("preflop_json", "preflop_csv"), _preflop(staging), strict=True)
    ) | dict(zip(("postflop_json", "postflop_csv"), _postflop(staging), strict=True))
    root = pokerbench.data_dir()
    root.mkdir(parents=True)
    for key, source in files.items():
        split, kind = key.split("_")
        source.replace(root / pokerbench.FILES[split][kind == "csv"])
    (root / ".complete").touch()
    yield
    pokerbench._streets.cache_clear()


@pytest.mark.parametrize(
    ("label", "action"),
    [
        ("check", "check"),
        ("Fold", "fold"),
        ("Bet 24", "bet 24"),
        ("raise 13.0", "raise 13"),
        ("13.0bb", "raise 13"),
        ("Bet 4.50", "bet 4.5"),
        ("allin", "all in"),
        ("All-In", "all in"),
        ("Bet", "bet"),
        ("wait", None),
    ],
)
def test_move_labels_parse_to_canonical_actions(label, action):
    assert parse_action(label) == action


@pytest.mark.parametrize(
    ("response", "action"),
    [
        ("call", "call"),
        ("**Raise to 13bb**", "raise 13"),
        ("I bet 24 chips.", "bet 24"),
        ("<think>fold? no, call.</think>check", "check"),
        ("Folding is too weak with an overpair.\nBet 24", "bet 24"),
        ("Call, don't fold.", "call"),
        ("With top pair I want value, so I would call.\nGood luck!", "call"),
        ("", None),
        ("I am not sure.", None),
    ],
)
def test_responses_commit_to_one_action(response, action):
    assert extract_action(response) == action


def test_split_matches_shuffled_scenarios_by_content(tmp_path):
    tasks = build_split("postflop", *_postflop(tmp_path))
    assert [task.id for task in tasks[:2]] == [
        "PokerBench/postflop/0",
        "PokerBench/postflop/1",
    ]
    turn = tasks[1]
    assert "your holding is [Ace of Heart and Queen of Diamond]" in turn.prompt
    assert turn.metadata == {
        "answer": "bet 24",
        "choices": ["check", "bet 10", "bet 24"],
        "street": "Turn",
    }
    assert [task.metadata["street"] for task in tasks[2:4]] == ["River", "River"]
    # Identical spots apart from the pot each get their own legal moves.
    assert tasks[7].metadata["choices"] == ["check", "bet 12"]
    assert tasks[8].metadata["choices"] == ["check", "bet 6"]

    preflop = build_split("preflop", *_preflop(tmp_path))
    assert preflop[1].metadata["answer"] == "raise 13"
    assert preflop[1].metadata["choices"] == ["fold", "raise 13", "all in"]
    assert {task.metadata["street"] for task in preflop} == {"Preflop"}


def test_unjoinable_files_fail_loudly(tmp_path):
    relabelled = [("Flop", "8h8c", "Check", "bet 24", "['Check', 'Bet 24']", 32)]
    with pytest.raises(ValueError, match="prompt 0: 0 scenarios match"):
        build_split("postflop", *_postflop(tmp_path, relabelled))

    illegal = [("Flop", "8h8c", "Bet 30", "bet 30", "['Check', 'Bet 24']", 32)]
    with pytest.raises(ValueError, match="not a legal move"):
        build_split("postflop", *_postflop(tmp_path, illegal))

    twins = [
        ("Flop", "8h8c", "Check", "check", "['Check', 'Bet 24']", 32),
        ("Flop", "8h8c", "Check", "check", "['Check', 'Bet 12']", 32),
    ]
    with pytest.raises(ValueError, match="2 scenarios match"):
        build_split("postflop", *_postflop(tmp_path, twins))


def test_betting_history_breaks_ties_the_pot_cannot(tmp_path):
    # Same hand, board, action and pot; only the flop bet faced differs.
    twins = [
        ("Flop", "8h8c", "Call", "call", "['Fold', 'Call']", 30),
        ("Flop", "8h8c", "Call", "call", "['Fold', 'Call', 'Raise 60']", 30),
    ]
    prompts, scenarios = _postflop(tmp_path, twins)
    rows = json.loads(prompts.read_text())
    for row, bet in zip(rows, (10, 44), strict=True):
        row["instruction"] = row["instruction"].replace(
            "then BB check.", f"then BB bet {bet} chips."
        )
    prompts.write_text(json.dumps(rows))
    with open(scenarios, newline="") as f:
        games = list(csv.DictReader(f))
    # Scenarios are written reversed: the first row is the second twin.
    for game, bet in zip(games, (44, 10), strict=True):
        game["postflop_action"] = f"OOP_BET_{bet}"
    with open(scenarios, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(games[0]))
        writer.writeheader()
        writer.writerows(games)

    tasks = build_split("postflop", prompts, scenarios)
    assert tasks[0].metadata["choices"] == ["fold", "call"]
    assert tasks[1].metadata["choices"] == ["fold", "call", "raise 60"]


def test_history_sizes_stop_at_the_full_stop():
    text = (
        "Before the flop, HJ raise 2.0, CO call, and BB raise 24.0. Assume\n"
        "The flop comes Ace Of Club, then BB bet 10 chips.\nNow it is your turn"
    )
    assert pokerbench._prompt_history(text) == ("raise 2", "call", "raise 24", "bet 10")


def test_unreadable_rows_are_named(tmp_path):
    prompts, scenarios = _postflop(tmp_path, POSTFLOP[:1])
    rows = json.loads(prompts.read_text())
    rows[0]["instruction"] = rows[0]["instruction"].replace("holding is", "hand is")
    prompts.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="postflop prompt 0: unreadable"):
        build_split("postflop", prompts, scenarios)


def test_counting_never_downloads(monkeypatch):
    monkeypatch.setattr(
        pokerbench, "_download_file", lambda *_a: pytest.fail("downloaded")
    )
    assert task_count("pokerbench") == 11000


def test_slices_keep_preflop_spots(installed):
    tasks = PokerBench().load_tasks()
    assert len(tasks) == 12
    streets = [task.metadata["street"] for task in tasks[:6]]
    assert "Preflop" in streets and {"Flop", "Turn", "River"} <= set(streets)


def test_exact_size_scores_and_feedback_hides_the_answer():
    bench = PokerBench()
    task = Task(
        "t", "spot", {"answer": "bet 24", "choices": ["check", "bet 10", "bet 24"]}
    )
    assert bench.evaluate_with_feedback(task, "Bet 24").score == 1.0

    wrong_size = bench.evaluate_with_feedback(task, "bet 10")
    assert wrong_size.score == 0.0
    assert wrong_size.details == {"predicted_action": "bet 10", "action_match": True}
    assert "24" not in wrong_size.feedback

    wrong = bench.evaluate_with_feedback(task, "check")
    assert wrong.details["action_match"] is False
    assert "bet" not in wrong.feedback.split("reply")[0]

    nothing = bench.evaluate_with_feedback(task, "hmm")
    assert nothing.details["predicted_action"] is None
    assert "bet N" in nothing.feedback


class FakeSolver:
    """Picks a fixed action per spot with probability 0.6."""

    timeout = 5.0
    label = "test"
    host = "test://local"

    def __init__(self, picks: dict[str, str]) -> None:
        self.picks = picks

    def is_decision_model(self, model: str) -> bool:
        return model == "d1"

    def decide(self, model: str, request: dict, cancel_event=None) -> dict:
        keys = list(request["questions"]["answer"]["criteria"])
        probs = dict.fromkeys(keys, 0.4 / (len(keys) - 1))
        probs[self.picks[request["state"]]] = 0.6
        answer = {"type": "choice", "probabilities": probs}
        return {"answers": {"answer": answer}, "usage": {"input_tokens": 9}}


def test_decision_models_choose_among_the_legal_moves(installed):
    bench = PokerBench()
    tasks = bench.load_tasks()[:4]
    question = decision.request_for(bench, tasks[0])["questions"][decision.QUESTION]
    assert question["type"] == "choice"
    assert list(question["criteria"]) == tasks[0].metadata["choices"]

    picks = {task.prompt: task.metadata["answer"] for task in tasks}
    # The last spot gets a legal move that is not the solver's.
    last = tasks[3]
    wrong = next(
        move for move in last.metadata["choices"] if move != last.metadata["answer"]
    )
    picks[last.prompt] = wrong
    jobs = route_decision_jobs([JobSpec("d1", "pokerbench", "4")], FakeSolver(picks))
    [result] = Engine(FakeSolver(picks), jobs).run()

    assert result["harness"] == "decision"
    assert result["score"] == 75.0
    same_verb = verb(wrong) == verb(last.metadata["answer"])
    assert result["poker_action_accuracy"] == (100.0 if same_verb else 75.0)
    assert set(result["poker_street_scores"]) <= {"Preflop", "Flop", "Turn", "River"}
    assert result["tasks"][0]["decision"]["correct"]
    assert "## Poker decisions" in render_markdown([result], "now")


def test_summary_counts_unanswered_spots_as_wrong_actions(installed):
    tasks = PokerBench().load_tasks()[:2]
    records = [
        SimpleNamespace(
            task_id=tasks[0].id,
            score=1.0,
            workspace={"predicted_action": "x", "action_match": True},
        ),
        # A timeout never reaches the verifier, so it has no workspace.
        SimpleNamespace(task_id=tasks[1].id, score=0.0, workspace={}),
    ]
    fields = PokerBench().summary_fields(records)
    assert fields["poker_action_accuracy"] == 50.0
    streets = [task.metadata["street"] for task in tasks]
    assert set(fields["poker_street_scores"]) == set(streets)
