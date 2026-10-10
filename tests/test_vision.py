"""Vision suites: image transport, dataset cache, slicing and scoring."""

from __future__ import annotations

import base64
import json
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from benchkit import engine as engine_module
from benchkit.benchmarks import REGISTRY, vision
from benchkit.benchmarks.base import Task
from benchkit.benchmarks.chartqa import extract_answer, relaxed_match
from benchkit.benchmarks.ocrbench import matches
from benchkit.client import InferenceClient
from benchkit.engine import Engine, JobSpec

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


def _generation(response: str) -> dict:
    return {
        "thinking": "",
        "response": response,
        "trace_status": "unavailable",
        "tok_s": 10.0,
        "eval_count": 3,
        "eval_duration_ns": 300_000_000,
        "response_time_s": 0.3,
        "done_reason": "stop",
        "timed_out": False,
        "cancelled": False,
        "input_tokens": 5,
    }


class FakeVision:
    name = "fake-vision"
    vision = True
    path = ""

    def load_tasks(self) -> list[Task]:
        return [Task("v/0", "what does it say?", {"images": [FakeVision.path]})]

    def build_prompt(self, task: Task) -> str:
        return task.prompt

    def evaluate(self, task: Task, response: str) -> bool:
        return response == "hello"


class ImageClient:
    timeout = 10.0
    label = "test"
    host = "test://local"
    provider = "openai"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def max_parallel_requests(self, _model: str) -> int:
        return 1

    def generate(self, _model: str, prompt: str, **kwargs: object) -> dict:
        self.calls.append({"prompt": prompt, **kwargs})
        return _generation("hello")


@pytest.fixture
def fake_suite(tmp_path):
    image = tmp_path / "0.png"
    image.write_bytes(PNG)
    FakeVision.path = str(image)
    # tasks_for caches by key, and each test has its own image path.
    with (
        patch.dict(REGISTRY, {"fake-vision": FakeVision}),
        patch.dict(engine_module._TASKS, clear=False),
    ):
        engine_module._TASKS.pop("fake-vision", None)
        yield str(image)


def test_engine_hands_task_images_to_the_client(fake_suite) -> None:
    client = ImageClient()
    result = Engine(client=client, jobs=[JobSpec("m", "fake-vision")]).run()[0]

    assert client.calls[0]["images"] == [fake_suite]
    assert result["score"] == 100.0


def test_repair_attempts_resend_the_images(fake_suite) -> None:
    client = ImageClient()
    with patch.object(FakeVision, "evaluate", return_value=False):
        Engine(
            client=client, jobs=[JobSpec("m", "fake-vision", repair_attempts=1)]
        ).run()

    assert len(client.calls) == 2
    assert all(call["images"] == [fake_suite] for call in client.calls)


def test_vision_suites_refuse_the_pi_harness(fake_suite) -> None:
    engine = Engine(client=ImageClient(), jobs=[JobSpec("m", "fake-vision")])
    with pytest.raises(ValueError, match="direct"):
        engine._run_job(0, JobSpec("m", "fake-vision", harness="pi"), 1)


def test_openai_request_puts_images_before_the_text(tmp_path) -> None:
    image = tmp_path / "chart.png"
    image.write_bytes(PNG)
    client = InferenceClient("http://local/v1", "openai")
    client.provider = "openai"

    with patch.object(client, "_retry_stream", return_value={}) as stream:
        client.generate("m", "read it", images=[str(image)])
        builder = stream.call_args.args[0]
    with patch.object(client, "_stream_openai", return_value={}) as inner:
        builder(None)

    content = inner.call_args.args[0]["messages"][0]["content"]
    encoded = base64.b64encode(PNG).decode()
    assert content == [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
        {"type": "text", "text": "read it"},
    ]


def test_text_only_requests_keep_a_plain_string(tmp_path) -> None:
    client = InferenceClient("http://local/v1", "openai")
    client.provider = "openai"
    with patch.object(client, "_retry_stream", return_value={}) as stream:
        client.generate("m", "hi")
        builder = stream.call_args.args[0]
    with patch.object(client, "_stream_openai", return_value={}) as inner:
        builder(None)

    assert inner.call_args.args[0]["messages"][0]["content"] == "hi"


def test_dataset_cache_extracts_images_once(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("BENCHKIT_VISION_CACHE", str(tmp_path))
    table = pa.table(
        {
            "image": [
                {"bytes": PNG, "path": None},
                {"bytes": b"\xff\xd8x", "path": None},
            ],
            "question": ["a?", "b?"],
        }
    )

    def fake_download(_client, _url: str, target) -> None:
        pq.write_table(table, target)

    def convert(index: int, row: dict) -> dict:
        return {"task_id": f"t/{index}", "question": row["question"]}

    with patch.object(vision, "_download_file", side_effect=fake_download) as dl:
        rows = vision.load_rows("demo", "org/demo", "x.parquet", "a" * 40, convert)
        again = vision.load_rows("demo", "org/demo", "x.parquet", "a" * 40, convert)

    dl.assert_called_once()
    assert rows == again
    assert [row["question"] for row in rows] == ["a?", "b?"]
    assert rows[0]["image"].endswith("0.png") and rows[1]["image"].endswith("1.jpg")
    root = tmp_path / "demo" / ("a" * 12)
    assert not (root / "source.parquet").exists()
    stored = [json.loads(line) for line in (root / "rows.jsonl").open()]
    assert stored[0]["image"] == "images/0.png"


def test_spread_keeps_each_groups_share_in_any_prefix() -> None:
    tasks = [Task(f"a{i}", "", {"c": "a"}) for i in range(6)]
    tasks += [Task(f"b{i}", "", {"c": "b"}) for i in range(3)]
    ordered = vision.spread(tasks, lambda task: task.metadata["c"])

    assert [task.metadata["c"] for task in ordered[:3]] == ["a", "b", "a"]
    assert sorted(task.id for task in ordered) == sorted(task.id for task in tasks)


@pytest.mark.parametrize(
    ("response", "answers", "source", "expected"),
    [
        ("The sign says CENTRE.", ["centre"], "IIIT5K", True),
        ("<think>maybe FRIEND</think>FIEND", ["FRIEND"], "IIIT5K", False),
        ("x ^ { 2 } + 1", ["x^{2}+1"], "HME100k", True),
        ("X^{2}+1", ["x^{2}+1"], "HME100k", False),
    ],
)
def test_ocrbench_matches_like_the_official_evaluator(
    response, answers, source, expected
) -> None:
    assert matches(response, answers, source) is expected


@pytest.mark.parametrize(
    ("response", "answer"),
    [
        ("The bars add up.\nAnswer: 14", "14"),
        ("**Answer:** Yes.", "Yes"),
        ("I think it is 1,234.", "I think it is 1,234"),
        ("final\n\n`0.57`", "0.57"),
    ],
)
def test_chartqa_extracts_the_final_answer(response, answer) -> None:
    assert extract_answer(response) == answer


def test_chartqa_relaxed_accuracy() -> None:
    assert relaxed_match("104", "100")
    assert not relaxed_match("106", "100")
    assert relaxed_match("57%", "0.57")
    assert relaxed_match("6.8%", "6.8")
    assert not relaxed_match("6.8%", "0.5")
    assert relaxed_match("1,000", "1000")
    assert relaxed_match("yes", "Yes")
    assert not relaxed_match("0", "0.0")


def test_decision_requests_record_paths_and_send_data_uris(tmp_path) -> None:
    from benchkit import decision

    image = tmp_path / "pair.png"
    image.write_bytes(PNG)
    bench = type("B", (), {"decision_instructions": "Which?", "name": "x"})()
    task = Task("t", "open or closed?", {"choices": ["Open", "Closed"]})
    task.metadata["images"] = [str(image)]

    prompt = decision.render_request(bench, task)
    assert json.loads(prompt)["images"] == [str(image)]

    class Client:
        def decide(self, _model, request, _cancel=None):
            self.request = request
            probs = {"A": 0.8, "B": 0.2}
            return {"answers": {"answer": {"probabilities": probs}}}

    client = Client()
    result = decision.Decider(client).generate("d1", prompt)
    encoded = base64.b64encode(PNG).decode()
    assert client.request["images"] == [f"data:image/png;base64,{encoded}"]
    assert result["response"] == "A"


def test_mmvp_manifest_pairs_and_letters() -> None:
    from benchkit.benchmarks.mmvp import parse_manifest

    csv_text = (
        "Index,Question,Options,Correct Answer\n"
        "1,Open or closed?,(a) Open (b) Closed,(a)\n"
        "2,Open or closed?,(a) Open (b) Closed,(b)\n"
    )
    rows = parse_manifest(csv_text.encode())
    assert [row["pair"] for row in rows] == [1, 1]
    assert rows[1]["choices"] == ["Open", "Closed"]
    assert [row["answer"] for row in rows] == ["A", "B"]
    assert rows[0]["image"] == "MMVP Images/1.jpg"


def test_mmvp_pair_accuracy_needs_both_questions(monkeypatch) -> None:
    from types import SimpleNamespace

    from benchkit.benchmarks import mmvp

    rows = tuple({"task_id": f"mmvp/{i}", "pair": (i + 1) // 2} for i in range(1, 7))
    monkeypatch.setattr(mmvp, "_rows", lambda: rows)

    def record(index: int, score: float):
        return SimpleNamespace(task_id=f"mmvp/{index}", score=score)

    records = [record(1, 1), record(2, 1), record(3, 1), record(4, 0), record(5, 1)]
    # Pair 3 is half scored (a slice ended mid-pair), so it is left out.
    assert mmvp.pair_accuracy(records) == {"pair_accuracy": 50.0, "pairs": 2}


def test_vstar_manifest_keeps_two_and_four_option_questions() -> None:
    from benchkit.benchmarks.vstar import parse_manifest

    lines = [
        {
            "image": "direct_attributes/a.jpg",
            "text": "What colour is the cup?\n(A) red\n(B) blue\n(C) green\n"
            "(D) white\nAnswer with the option's letter from the given choices "
            "directly.",
            "category": "direct_attributes",
            "question_id": "0",
            "label": "C",
        },
        {
            "image": "relative_position/b.jpg",
            "text": "Is the cat left or right of the dog?\n(A) left\n(B) right\n"
            "Answer with the option's letter from the given choices directly.",
            "category": "relative_position",
            "question_id": "1",
            "label": "B",
        },
    ]
    rows = parse_manifest("\n".join(json.dumps(line) for line in lines).encode())
    assert rows[0]["choices"] == ["red", "blue", "green", "white"]
    assert rows[0]["question"] == "What colour is the cup?"
    assert rows[1]["choices"] == ["left", "right"]
    assert [row["answer"] for row in rows] == ["C", "B"]


def test_vstar_two_option_answers_ignore_other_letters() -> None:
    from benchkit.benchmarks.vstar import VStar

    task = Task("v", "left or right?", {"choices": ["left", "right"], "answer": "B"})
    assert VStar().evaluate(task, "B")
    assert not VStar().evaluate(task, "C")
