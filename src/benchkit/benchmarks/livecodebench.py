"""LiveCodeBench - contamination-resistant competitive programming tasks.

Every problem carries the date its contest was published, so a run can be
restricted to problems released *after* a model's training cutoff. That date
window is the entire point of the suite: without it LiveCodeBench is just
another set a model may already have memorized.

The dataset is the ``code_generation_lite`` release (pruned test cases, scoring
that stays within a point of the full release, far less to download). It is not
bundled with BenchKit - v6 is 1055 problems with their test cases attached - so
it is fetched once and cached, in the same spirit as the EvalPlus suites.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import pickle
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import httpx

from benchkit.benchmarks.base import Task
from benchkit.benchmarks.utils import strip_think_tags
from benchkit.executor import execute

RELEASE = "v6"
CACHE_FORMAT = 1
SHARDS = (
    "test.jsonl",
    "test2.jsonl",
    "test3.jsonl",
    "test4.jsonl",
    "test5.jsonl",
    "test6.jsonl",
)
DEFAULT_BASE_URL = (
    "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main"
)

# Problems released from August 2024 onwards. Most of the 4-35B models BenchKit
# targets have a training cutoff before this, so the default window is already
# contamination-resistant; move it with --lcb-after for a newer model.
DEFAULT_AFTER = "2024-08"

# The lite release is already pruned, but a handful of problems still ship
# dozens of large cases. Capping keeps a full run and the on-disk cache sane;
# public (sample) cases are always kept first.
DEFAULT_MAX_TESTS = 25

# Wall-clock budget handed to the sandbox, per test case and in total. A
# competitive-programming solution that needs longer than this is treated as a
# failure, which is what the official harness does too.
PER_TEST_TIMEOUT_S = 6
MAX_TOTAL_TIMEOUT_S = 90

DIFFICULTIES = ("easy", "medium", "hard")

SYSTEM = (
    "You are an expert Python programmer. You will be given a question "
    "(problem specification) and will generate a correct Python program that "
    "matches the specification and passes all tests."
)
FUNCTIONAL_FORMAT = (
    "You will use the following starter code to write the solution to the "
    "problem and enclose your code within delimiters."
)
STDIN_FORMAT = (
    "Read the inputs from stdin solve the problem and write the answer to "
    "stdout (do not directly test on the sample inputs). Enclose your code "
    "within delimiters as follows."
)


class DatasetUnavailable(RuntimeError):
    """The LiveCodeBench dataset is neither cached nor reachable."""


# Configuration ---------------------------------------------------------


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _cache_root() -> Path:
    configured = _env("BENCHKIT_LCB_CACHE")
    if configured:
        return Path(configured).expanduser()
    base = _env("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base).expanduser() / "benchkit" / "livecodebench"


def _max_tests() -> int:
    raw = _env("BENCHKIT_LCB_MAX_TESTS")
    try:
        value = int(raw) if raw else DEFAULT_MAX_TESTS
    except ValueError:
        return DEFAULT_MAX_TESTS
    return max(1, value)


def _parse_day(value: str, *, end_of_month: bool) -> date:
    """Parse ``YYYY-MM`` or ``YYYY-MM-DD`` into an inclusive boundary date."""
    text = value.strip()
    for fmt in ("%Y-%m-%d", "%Y-%m"):
        try:
            parsed = datetime.strptime(text, fmt).date()
        except ValueError:
            continue
        if fmt == "%Y-%m" and end_of_month:
            # A bare month as an upper bound covers the whole month.
            month = parsed.month
            year = parsed.year + (month == 12)
            next_month = 1 if month == 12 else month + 1
            return date(year, next_month, 1) - timedelta(days=1)
        return parsed
    raise ValueError(f"Invalid date: {value!r} (use YYYY-MM or YYYY-MM-DD)")


@dataclass(frozen=True)
class Window:
    """Inclusive contest-date window applied to the problem set."""

    after: date | None
    before: date | None

    def contains(self, day: date | None) -> bool:
        if day is None:
            return False
        if self.after is not None and day < self.after:
            return False
        return not (self.before is not None and day > self.before)

    @property
    def label(self) -> str:
        if self.after is None and self.before is None:
            return "all contest dates"
        if self.before is None:
            return f"{self.after.isoformat()} onwards"
        if self.after is None:
            return f"up to {self.before.isoformat()}"
        return f"{self.after.isoformat()} to {self.before.isoformat()}"


def window() -> Window:
    """Resolve the configured contest-date window.

    ``BENCHKIT_LCB_AFTER`` / ``BENCHKIT_LCB_BEFORE`` (set directly or through
    ``--lcb-after`` / ``--lcb-before``) move the window. Setting ``after`` to
    ``all`` removes the lower bound and scores the whole release, contamination
    included.
    """
    after_raw = _env("BENCHKIT_LCB_AFTER") or DEFAULT_AFTER
    before_raw = _env("BENCHKIT_LCB_BEFORE")

    after = (
        None
        if after_raw.lower() in {"all", "none", "-"}
        else _parse_day(after_raw, end_of_month=False)
    )
    before = _parse_day(before_raw, end_of_month=True) if before_raw else None
    if after is not None and before is not None and before < after:
        raise ValueError(
            f"Empty LiveCodeBench window: {after.isoformat()} is after "
            f"{before.isoformat()}"
        )
    return Window(after, before)


# Dataset preparation ---------------------------------------------------


class _NoGlobalsUnpickler(pickle.Unpickler):
    """Unpickler that refuses to resolve any global, class or callable.

    LiveCodeBench ships its hidden tests as a zlib+base64+pickle blob whose
    payload is a plain JSON string, so nothing legitimate needs a global. This
    keeps a downloaded file from executing code while it is decoded.
    """

    def find_class(self, module: str, name: str):
        raise pickle.UnpicklingError(
            f"refusing to load {module}.{name} from LiveCodeBench test data"
        )


def _decode_tests(raw: object) -> list[dict]:
    """Decode public or private test cases from any shipped encoding."""
    if not raw:
        return []
    if isinstance(raw, list):
        return [case for case in raw if isinstance(case, dict)]
    if not isinstance(raw, str):
        return []

    text = raw.strip()
    if text.startswith("["):
        return json.loads(text)
    if text.startswith('"'):
        text = json.loads(text)

    payload = _NoGlobalsUnpickler(
        io.BytesIO(zlib.decompress(base64.b64decode(text)))
    ).load()
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8")
    if isinstance(payload, str):
        payload = json.loads(payload)
    return [case for case in payload if isinstance(case, dict)]


def _contest_date(row: dict) -> str:
    raw = str(row.get("contest_date") or "")
    return raw[:10]


def _raw_sources() -> list[Path]:
    """Local shard files, downloading them into the cache when needed."""
    configured = _env("BENCHKIT_LCB_DATASET")
    if configured:
        path = Path(configured).expanduser()
        if path.is_dir():
            shards = sorted(path.glob("*.jsonl"))
            if not shards:
                raise DatasetUnavailable(
                    f"No .jsonl shards found in BENCHKIT_LCB_DATASET={path}"
                )
            return shards
        if path.is_file():
            return [path]
        raise DatasetUnavailable(f"BENCHKIT_LCB_DATASET does not exist: {path}")

    raw_dir = _cache_root() / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    base_url = _env("BENCHKIT_LCB_BASE_URL") or DEFAULT_BASE_URL
    return [_download(f"{base_url}/{shard}", raw_dir / shard) for shard in SHARDS]


def _download(url: str, target: Path) -> Path:
    if target.exists() and target.stat().st_size > 0:
        return target

    partial = target.with_suffix(target.suffix + ".part")
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=120) as response:
            response.raise_for_status()
            with open(partial, "wb") as handle:
                for chunk in response.iter_bytes(1 << 20):
                    handle.write(chunk)
    except httpx.HTTPError as exc:
        partial.unlink(missing_ok=True)
        raise DatasetUnavailable(
            f"Could not download the LiveCodeBench dataset from {url}: {exc}. "
            "Download it separately and point BENCHKIT_LCB_DATASET at the "
            "directory holding its test*.jsonl shards."
        ) from exc
    partial.replace(target)
    return target


def _prepared_dir() -> Path:
    # CACHE_FORMAT is part of the path, so a change to the prepared layout
    # invalidates an older cache instead of half-reading it.
    return _cache_root() / "prepared" / f"{RELEASE}-f{CACHE_FORMAT}-max{_max_tests()}"


def _manifest_path() -> Path:
    return _prepared_dir() / "manifest.json"


def prepare(force: bool = False) -> dict:
    """Normalize the release into the local cache and return its manifest.

    The prepared form keeps problem statements in one small JSONL file and one
    JSON file of test cases per problem, so loading tasks stays cheap and only
    the problems actually scored are read back.
    """
    manifest_path = _manifest_path()
    if manifest_path.exists() and not force:
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except ValueError:
            pass

    prepared = _prepared_dir()
    tests_dir = prepared / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    limit = _max_tests()

    problems: list[dict] = []
    seen: set[str] = set()
    with open(prepared / "problems.jsonl", "w", encoding="utf-8") as out:
        for shard in _raw_sources():
            with open(shard, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    question_id = str(
                        row.get("question_id") or row.get("task_id") or ""
                    )
                    if not question_id or question_id in seen:
                        continue
                    seen.add(question_id)

                    public = _decode_tests(row.get("public_test_cases"))
                    private = _decode_tests(row.get("private_test_cases"))
                    total = len(public) + len(private)
                    cases = (public + private)[:limit]
                    if not cases:
                        continue

                    metadata = row.get("metadata")
                    if isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except ValueError:
                            metadata = {}
                    metadata = metadata if isinstance(metadata, dict) else {}

                    problem = {
                        "question_id": question_id,
                        "title": row.get("question_title") or question_id,
                        "platform": row.get("platform") or "",
                        "contest_date": _contest_date(row),
                        "difficulty": str(row.get("difficulty") or "").lower(),
                        "question_content": row.get("question_content") or "",
                        "starter_code": row.get("starter_code") or "",
                        "func_name": str(metadata.get("func_name") or ""),
                        "tests": len(cases),
                        "tests_total": total,
                    }
                    problems.append(problem)
                    out.write(json.dumps(problem, ensure_ascii=False) + "\n")
                    (tests_dir / f"{_safe_name(question_id)}.json").write_text(
                        json.dumps(cases, ensure_ascii=False),
                        encoding="utf-8",
                    )

    if not problems:
        raise DatasetUnavailable(
            "The LiveCodeBench source files contained no usable problems"
        )

    manifest = {
        "release": RELEASE,
        "problems": len(problems),
        "max_tests": limit,
        "first_contest_date": min(item["contest_date"] for item in problems),
        "last_contest_date": max(item["contest_date"] for item in problems),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _safe_name(question_id: str) -> str:
    """File name for one problem's tests.

    Question ids are contest slugs, so they are sanitized for the filesystem.
    The digest keeps two ids that sanitize to the same text from sharing (and
    silently scoring against) one another's test cases.
    """
    stem = "".join(
        char if char.isalnum() or char in "-_" else "_" for char in question_id
    )
    digest = hashlib.sha1(question_id.encode("utf-8")).hexdigest()[:8]
    return f"{stem[:64]}-{digest}"


def is_prepared() -> bool:
    """Whether the dataset is already cached locally."""
    return _manifest_path().exists()


def _problem_rows() -> Iterator[dict]:
    prepare()
    with open(_prepared_dir() / "problems.jsonl", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _tests_for(question_id: str) -> list[dict]:
    path = _prepared_dir() / "tests" / f"{_safe_name(question_id)}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DatasetUnavailable(
            f"Missing cached test cases for {question_id}. Rebuild the cache "
            "with: benchkit lcb-prepare"
        ) from exc


# Prompting and extraction ----------------------------------------------


def build_question_prompt(problem: dict) -> str:
    starter = problem.get("starter_code") or ""
    body = starter if starter.strip() else "# YOUR CODE HERE"
    fmt = FUNCTIONAL_FORMAT if starter.strip() else STDIN_FORMAT
    return (
        f"{SYSTEM}\n\n"
        f"### Question:\n{problem['question_content'].rstrip()}\n\n"
        f"### Format: {fmt}\n"
        f"```python\n{body}\n```\n\n"
        "### Answer: (use the provided format with backticks)\n"
    )


def extract_code(response: str) -> str:
    """Take the last fenced block, which is where the final solution lands."""
    text = strip_think_tags(response).strip()
    if "```" not in text:
        return text

    # Odd indices are the fenced regions; a trailing unterminated fence lands
    # on an odd index too, so a truncated answer still yields its code.
    parts = text.split("```")
    blocks: list[str] = []
    for index in range(1, len(parts), 2):
        block = parts[index]
        for tag in ("python3", "python", "py"):
            if block.startswith(tag):
                block = block[len(tag) :]
                break
        blocks.append(block.strip("\n"))

    blocks = [block for block in blocks if block.strip()]
    return blocks[-1] if blocks else text


# Sandbox harness -------------------------------------------------------

_HARNESS = """
import io
import json
import sys
import time
from contextlib import redirect_stdout

sys.setrecursionlimit(20000)

_SOLUTION = json.loads({solution})
_TESTS = json.loads({tests})
_FUNC = json.loads({func})
_DEADLINE = time.monotonic() + {budget}


def _normalize(value):
    if isinstance(value, tuple):
        return [_normalize(item) for item in value]
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, dict):
        return {{str(key): _normalize(item) for key, item in value.items()}}
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _same_value(got, expected):
    if _normalize(got) == _normalize(expected):
        return True
    return _same_text(str(got), str(expected))


def _same_text(got, expected):
    got_lines = [line.rstrip() for line in got.strip().splitlines()]
    expected_lines = [line.rstrip() for line in expected.strip().splitlines()]
    if got_lines == expected_lines:
        return True
    return got.split() == expected.split()


def _fail(index, detail):
    print("test {{}} failed: {{}}".format(index, detail), file=sys.stderr)
    raise SystemExit(1)


def _check_budget(index):
    if time.monotonic() > _DEADLINE:
        _fail(index, "time budget exhausted")


def _run_functional():
    namespace = {{"__name__": "__benchkit_solution__"}}
    exec(compile(_SOLUTION, "<solution>", "exec"), namespace)
    factory = namespace.get("Solution")
    for index, test in enumerate(_TESTS):
        _check_budget(index)
        args = [
            json.loads(line)
            for line in test["input"].split("\\n")
            if line.strip()
        ]
        expected = json.loads(test["output"])
        target = factory() if factory is not None else None
        method = (
            getattr(target, _FUNC, None)
            if target is not None
            else namespace.get(_FUNC)
        )
        if method is None:
            _fail(index, "no callable named " + _FUNC)
        got = method(*args)
        if not _same_value(got, expected):
            _fail(index, "expected {{!r}}, got {{!r}}".format(expected, got))


def _run_stdin():
    code = compile(_SOLUTION, "<solution>", "exec")
    original_stdin = sys.stdin
    for index, test in enumerate(_TESTS):
        _check_budget(index)
        buffer = io.StringIO()
        sys.stdin = io.StringIO(test["input"])
        try:
            with redirect_stdout(buffer):
                exec(code, {{"__name__": "__main__"}})
        except SystemExit:
            pass
        finally:
            sys.stdin = original_stdin
        if not _same_text(buffer.getvalue(), test["output"]):
            _fail(
                index,
                "expected {{!r}}, got {{!r}}".format(
                    test["output"], buffer.getvalue()
                ),
            )


if _FUNC:
    _run_functional()
else:
    _run_stdin()
"""


def _harness(solution: str, tests: list[dict], func_name: str, budget: int) -> str:
    return _HARNESS.format(
        solution=repr(json.dumps(solution)),
        tests=repr(json.dumps(tests)),
        func=repr(json.dumps(func_name)),
        budget=budget,
    )


# Benchmark -------------------------------------------------------------


class LiveCodeBench:
    name = "livecodebench"
    evaluation_activity = "executing code and running hidden tests"
    list_note = "downloads on first use · benchkit lcb-prepare"
    group_order = DIFFICULTIES

    def __init__(self) -> None:
        self._cache: tuple[str, list[Task]] | None = None

    def load_tasks(self) -> list[Task]:
        active = window()
        if self._cache is not None and self._cache[0] == active.label:
            return self._cache[1]
        tasks: list[Task] = []
        for problem in _problem_rows():
            try:
                day = date.fromisoformat(problem["contest_date"])
            except ValueError:
                continue
            if not active.contains(day):
                continue
            difficulty = problem["difficulty"] or "unknown"
            tasks.append(
                Task(
                    id=problem["question_id"],
                    prompt=build_question_prompt(problem),
                    metadata={
                        "group": difficulty,
                        "difficulty": difficulty,
                        "contest_date": problem["contest_date"],
                        "platform": problem["platform"],
                        "entry_point": problem["func_name"],
                        "func_name": problem["func_name"],
                        "tests": problem["tests"],
                        "tests_total": problem["tests_total"],
                    },
                )
            )
        if not tasks:
            raise DatasetUnavailable(
                f"No LiveCodeBench problems fall in the window {active.label}"
            )
        # Oldest first, so a slice like `livecodebench:-50` is the freshest tail.
        tasks.sort(key=lambda task: (task.metadata["contest_date"], task.id))
        self._cache = (active.label, tasks)
        return tasks

    @property
    def task_count(self) -> int:
        """Problems in the current window, counted only from a ready cache.

        Task counts are read while browsing benchmarks, so this must never be
        the thing that starts a multi-hundred-megabyte download. Callers report
        an unknown count; the run itself fetches what it needs.
        """
        if not is_prepared():
            raise DatasetUnavailable("not cached yet — run: benchkit lcb-prepare")
        return len(self.load_tasks())

    @property
    def report_note(self) -> str:
        active = window()
        return (
            f"LiveCodeBench {RELEASE} (code_generation_lite) · contest window "
            f"{active.label} · up to {_max_tests()} test cases per problem"
        )

    def build_prompt(self, task: Task) -> str:
        return task.prompt

    def evaluate(self, task: Task, response: str) -> bool:
        code = extract_code(response)
        if not code.strip():
            return False

        tests = _tests_for(task.id)
        if not tests:
            return False

        budget = min(MAX_TOTAL_TIMEOUT_S, PER_TEST_TIMEOUT_S * len(tests))
        harness = _harness(code, tests, task.metadata.get("func_name", ""), budget)
        # The sandbox gets a little slack over the in-harness budget so a test
        # that overruns is reported as a failed budget rather than a hard kill.
        return execute(harness, timeout=budget + 5)
