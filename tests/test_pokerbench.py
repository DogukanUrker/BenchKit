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

PREFLOP = [
    ("SB", "call", "['call', 'fold']", "Preflop spot one. Your optimal action is:"),
    ("BTN", "13.0bb", "['fold', '13.0bb', 'allin']", "Preflop spot two."),
]
PREFLOP_LABELS = ["call", "raise 13.0"]
# (evaluation_at, correct_decision, available_moves, label)
POSTFLOP = [
    ("Flop", "Check", "['Check', 'Bet 24']", "check"),
    ("Turn", "Bet", "['Check', 'Bet 10', 'Bet 24']", "bet 24"),
    ("River", "Fold", "['Fold', 'Call', 'Raise 60']", "fold"),
    ("River", "Call", "['Fold', 'Call']", "call"),
    ("Flop", "Bet", "['Check', 'Bet 4.5']", "bet 4.5"),
    ("Turn", "Check", "['Check', 'Bet 30']", "check"),
    ("River", "Raise", "['Fold', 'Call', 'Raise 60']", "raise 60"),
    ("Flop", "Check", "['Check', 'Bet 12']", "check"),
    ("Turn", "Call", "['Fold', 'Call']", "call"),
    ("River", "Check", "['Check', 'Bet 20']", "check"),
]


def _write(directory: Path, split: str, prompts: list[dict], games: list[dict]):
    json_path = directory / f"{split}.json"
    csv_path = directory / f"{split}.csv"
    json_path.write_text(json.dumps(prompts), encoding="utf-8")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(games[0]))
        writer.writeheader()
        writer.writerows(games)
    return json_path, csv_path


def _preflop(directory: Path):
    prompts = [
        {"instruction": prompt, "output": label}
        for (_pos, _dec, _moves, prompt), label in zip(
            PREFLOP, PREFLOP_LABELS, strict=True
        )
    ]
    games = [
        {
            "prev_line": "UTG/2.0bb/BTN/call",
            "hero_pos": pos,
            "hero_holding": "KdKc",
            "correct_decision": dec,
            "num_players": 3,
            "num_bets": 2,
            "available_moves": moves,
            "pot_size": 5.5,
        }
        for pos, dec, moves, _prompt in PREFLOP
    ]
    return _write(directory, "preflop", prompts, games)


def _postflop(directory: Path, rows=POSTFLOP):
    prompts = [
        {"instruction": f"Postflop spot {i} on the {street}.", "output": label}
        for i, (street, _dec, _moves, label) in enumerate(rows)
    ]
    games = [
        {
            "preflop_action": "HJ/2.0bb/BB/call",
            "board_flop": "Ks7h2d",
            "board_turn": "Jc",
            "board_river": "7c",
            "aggressor_position": "OOP",
            "postflop_action": "OOP_CHECK/IP_CHECK",
            "evaluation_at": street,
            "available_moves": moves,
            "pot_size": 32,
            "hero_position": "IP",
            "holding": "8h8c",
            "correct_decision": dec,
        }
        for street, dec, moves, _label in rows
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


def test_split_joins_prompts_with_their_legal_moves(tmp_path):
    tasks = build_split("postflop", *_postflop(tmp_path))
    assert [task.id for task in tasks[:2]] == [
        "PokerBench/postflop/0",
        "PokerBench/postflop/1",
    ]
    turn = tasks[1]
    assert turn.prompt == "Postflop spot 1 on the Turn."
    assert turn.metadata == {
        "answer": "bet 24",
        "choices": ["check", "bet 10", "bet 24"],
        "street": "Turn",
    }
    preflop = build_split("preflop", *_preflop(tmp_path))
    assert preflop[1].metadata["answer"] == "raise 13"
    assert preflop[1].metadata["choices"] == ["fold", "raise 13", "all in"]
    assert {task.metadata["street"] for task in preflop} == {"Preflop"}


def test_misaligned_files_fail_loudly(tmp_path):
    swapped = [POSTFLOP[1], POSTFLOP[0]]
    shifted = [(*row[:3], POSTFLOP[0][3]) for row in swapped]
    with pytest.raises(ValueError, match="row 0"):
        build_split("postflop", *_postflop(tmp_path, shifted))

    illegal = [("Flop", "Bet", "['Check', 'Bet 24']", "bet 30")]
    with pytest.raises(ValueError, match="legal moves"):
        build_split("postflop", *_postflop(tmp_path, illegal))


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
