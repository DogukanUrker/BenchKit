"""Build `src/benchkit/datasets/code_verifier.jsonl` from local HumanEval+ runs.

Each row is one candidate solution to a HumanEval+ problem, written by a model
during a past BenchKit run, and the question "does this pass every test?":

    uv run python scripts/build_code_verifier.py [--results results]

Candidates come from `results/*/results.json`, direct-harness humaneval-plus
jobs only, and only from models in `MODELS`, which is an allowlist: a model
that is not listed is never read, so a new or unreleased model cannot slip in
on a rebuild. Rows that timed out, were loop-killed, hit a length limit or
were harness errors are skipped, and so are responses with a `</think>` that
has no opening tag, because the extractor would grade a draft from the
reasoning.

Labels are never taken from the stored `passed` field. Every distinct program
is executed against the base and plus tests with the current evaluator. A
failure is then re-run under generous time limits, and one that passes there
failed only on time and is dropped, since nothing in the problem states a time
budget. Programs that do not compile (usually prose instead of code) are
dropped as too easy to reject.

Selection is deterministic: 250 failing and 250 passing programs, at most
`PER_TASK` of each per task and at most `FAMILY_CAP` of each side from one
model family. Failing programs that still pass the base tests (near misses)
are taken first. Passing programs are matched to the tasks the failing ones
came from before filling elsewhere, so a task's label mix carries no signal.
Rows are interleaved fail/pass in a seeded order, so any slice is balanced.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from benchkit.benchmarks import evalplus as ep  # noqa: E402

DATASET_VERSION = "v1"
OUTPUT = ROOT / "src" / "benchkit" / "datasets" / "code_verifier.jsonl"
SEED = 53
PER_SIDE = 250
PER_TASK = 4
FAMILY_CAP = 62  # 25% of one side
EXCLUDED_RUNS = {
    # lfm2.5-2.6b with thinking on, interrupted after 42 tasks.
    "2026-10-08_14-03-09",
}
GENEROUS = {"min_time_limit": 10.0, "gt_time_limit_factor": 50.0}

# Publishable models and the family each one counts against. Default-deny.
# fmt: off
MODELS = {
    "bonsai2-27b-pq2": "bonsai2-27b",
    "bonsai2-27b-pq2-np4": "bonsai2-27b",
    "btl4-35b-iq2": "btl4-35b",
    "gemma4-12b": "gemma4-12b",
    "gemma4-e2b": "gemma4-e2b",
    "gemma4-e4b": "gemma4-e4b",
    "granite4.2-3b": "granite4.2-3b",
    "lfm2.5-1.2b-instruct": "lfm2.5-1.2b",
    "lfm2.5-2.6b": "lfm2.5-2.6b",
    "lfm2.5-8b-a1b-control": "lfm2.5-8b-a1b",
    "lfm2.5-8b-a1b-dspark": "lfm2.5-8b-a1b",
    "minicpm5-2b-q4-bench": "minicpm5-2b",
    "minicpm5-2b-q4-bench-rp": "minicpm5-2b",
    "minicpm5-2b-q8-bench": "minicpm5-2b",
    "minicpm5-2b-q8-bench-rp105": "minicpm5-2b",
    "minicpm5-2b-q8-bench-rp110": "minicpm5-2b",
    "minicpm5-2b-q8-bench-rp115": "minicpm5-2b",
    "minicpm5-2b-q8-bench-rp120": "minicpm5-2b",
    "muse-glimmer-30b": "muse-glimmer-30b",
    "nemotron-3.5-lightning": "nemotron-3.5",
    "ornith-35b": "ornith-35b",
    "qwen3.6-35b-a3b-bench": "qwen3.6-35b-a3b",
    "qwen3.8-27b": "qwen3.8-27b",
    "qwen3.8-27b-ad": "qwen3.8-27b",
    "qwen3.8-27b-mtp": "qwen3.8-27b",
    "qwen3.8-27b-mtp-low": "qwen3.8-27b",
    "qwen3.8-27b-mtp-medium": "qwen3.8-27b",
    "qwen3.8-27b-mtp-nothink": "qwen3.8-27b",
    "qwen3.8-27b-mtp-xhigh": "qwen3.8-27b",
    "qwen3.8-27b-q4-low": "qwen3.8-27b",
    "qwen3.8-27b-q4-xhigh": "qwen3.8-27b",
    "qwen3.8-35b-a3b": "qwen3.8-35b-a3b",
    "spark-x2.5-4b-bench": "spark-x2.5-4b",
    "spark-x2.5-4b-bench-rp105": "spark-x2.5-4b",
    "spark-x2.5-4b-bench-rp110": "spark-x2.5-4b",
    "spark-x2.5-4b-bench-rp115": "spark-x2.5-4b",
    "spark-x2.5-4b-bench-rp120": "spark-x2.5-4b",
    "spark-x2.5-4b-bench-t07": "spark-x2.5-4b",
    "spark-x2.5-4b-bench-topk20": "spark-x2.5-4b",
}
# fmt: on


def _skip(task: dict) -> bool:
    response = task.get("response") or ""
    return bool(
        task.get("timed_out")
        or task.get("loop_killed")
        or task.get("length_exceeded")
        or task.get("harness_error")
        or task.get("error")
        or task.get("done_reason") == "length"
        or ("</think>" in response and "<think>" not in response.split("</think>")[0])
    )


def _key(program: str) -> str:
    """Identity of a program: trailing spaces and blank lines don't count."""
    return "\n".join(line.rstrip() for line in program.splitlines() if line.strip())


def collect(results: Path) -> dict[tuple[str, str], dict]:
    """Return distinct candidate programs keyed by (task_id, program key)."""
    problems = ep._problems("humaneval")
    pool: dict[tuple[str, str], dict] = {}
    for path in sorted(results.glob("*/results.json")):
        if path.parent.name in EXCLUDED_RUNS:
            continue
        try:
            jobs = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        for job in jobs if isinstance(jobs, list) else []:
            model = job.get("model")
            if (
                job.get("benchmark") != "humaneval-plus"
                or job.get("harness") != "direct"
                or model not in MODELS
            ):
                continue
            for task in job.get("tasks", []):
                task_id = task.get("task_id")
                if task_id not in problems or _skip(task):
                    continue
                if not ep._extract_code(task.get("response") or "").strip():
                    continue
                program = ep._human_solution(problems[task_id], task["response"])
                entry = pool.setdefault(
                    (task_id, _key(program)),
                    {"task_id": task_id, "program": program, "models": set()},
                )
                entry["models"].add(model)
    return pool


def _check(task_id: str, program: str, **limits: float) -> tuple[bool, bool]:
    problem = ep._problems("humaneval")[task_id]
    result = ep._evalplus_api()[4](
        "humaneval",
        completion_id=0,
        problem=problem,
        solution=program,
        expected_output=ep._expected_output("humaneval", task_id),
        fast_check=True,
        **limits,
    )
    return result["base"][0] == "pass", result["plus"][0] == "pass"


def label(entry: dict) -> dict:
    """Execute one program and record its label, or why it is unusable."""
    try:
        compile(entry["program"], "<candidate>", "exec")
    except (SyntaxError, ValueError):
        return {**entry, "status": "no_compile"}
    base, plus = _check(entry["task_id"], entry["program"])
    if base and plus:
        return {**entry, "status": "pass", "base_pass": True}
    if all(_check(entry["task_id"], entry["program"], **GENEROUS)):
        return {**entry, "status": "slow_only"}
    return {**entry, "status": "fail", "base_pass": base}


def _pick(
    candidates: dict[str, list[dict]],
    quota: dict[str, int],
    family_used: Counter,
    want: int,
    cap: dict[str, int] | None = None,
) -> list[dict]:
    """Round-robin over tasks, one program per task per round, within caps.

    Without ``cap`` every family may reach ``FAMILY_CAP`` and each task's queue
    is taken in order. With ``cap`` each family has its own limit and a task
    offers the family furthest below its limit first.
    """

    def room(entry: dict) -> int:
        limit = FAMILY_CAP if cap is None else cap.get(entry["family"], 0)
        return limit - family_used[entry["family"]]

    chosen: list[dict] = []
    while len(chosen) < want:
        progressed = False
        for task_id in sorted(candidates):
            queue = candidates[task_id]
            eligible = [e for e in queue if room(e) > 0]
            if quota[task_id] <= 0 or not eligible:
                continue
            entry = eligible[0] if cap is None else max(eligible, key=room)
            queue.remove(entry)
            family_used[entry["family"]] += 1
            quota[task_id] -= 1
            chosen.append(entry)
            progressed = True
            if len(chosen) == want:
                break
        if not progressed:
            break
    return chosen


def select(labelled: list[dict]) -> tuple[list[dict], list[dict]]:
    rng = random.Random(SEED)
    by_task: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: {"fail": [], "pass": []}
    )
    for entry in sorted(labelled, key=lambda e: (e["task_id"], e["program"])):
        if entry["status"] in ("pass", "fail"):
            entry["model"] = min(entry["models"])
            entry["family"] = MODELS[entry["model"]]
            by_task[entry["task_id"]][entry["status"]].append(entry)
    for sides in by_task.values():
        for side in sides.values():
            rng.shuffle(side)
        sides["fail"].sort(key=lambda e: not e["base_pass"])  # near misses first

    fails = _pick(
        {t: s["fail"] for t, s in by_task.items() if s["fail"]},
        defaultdict(lambda: PER_TASK),
        Counter(),
        PER_SIDE,
    )
    # Passing programs mirror the failing side: same tasks first, and per
    # family no more than that family failed, so neither the task nor the
    # model's coding style predicts the label. The global cap is a last resort.
    per_task_fails = Counter(e["task_id"] for e in fails)
    fail_families = Counter(e["family"] for e in fails)
    pass_used: Counter = Counter()
    passes = _pick(
        {t: by_task[t]["pass"] for t in per_task_fails},
        defaultdict(int, per_task_fails),
        pass_used,
        PER_SIDE,
        cap=fail_families,
    )
    quota = defaultdict(
        lambda: PER_TASK,
        {t: PER_TASK - sum(e["task_id"] == t for e in passes) for t in by_task},
    )
    pass_queues = {t: s["pass"] for t, s in by_task.items()}
    passes += _pick(
        pass_queues, quota, pass_used, PER_SIDE - len(passes), cap=fail_families
    )
    passes += _pick(pass_queues, quota, pass_used, PER_SIDE - len(passes))
    return fails, passes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2))
    parser.add_argument(
        "--label-cache",
        type=Path,
        help="JSON file of earlier labels, reused and updated (keep it out of git)",
    )
    args = parser.parse_args()

    pool = collect(args.results)
    print(f"{len(pool)} distinct programs over {len({t for t, _ in pool})} tasks")
    cache: dict[str, dict] = {}
    if args.label_cache and args.label_cache.exists():
        cache = json.loads(args.label_cache.read_text(encoding="utf-8"))
    labelled, todo = [], []
    for (task_id, key), entry in pool.items():
        hit = cache.get(f"{task_id}\n{key}")
        if hit:
            labelled.append({**entry, **hit})
        else:
            todo.append(entry)
    for task_id in sorted({e["task_id"] for e in todo}):
        ep._expected_output("humaneval", task_id)

    with ProcessPoolExecutor(args.workers) as executor:
        for done, entry in enumerate(executor.map(label, todo, chunksize=8), 1):
            labelled.append(entry)
            if done % 250 == 0:
                print(f"  labelled {done}/{len(todo)}", flush=True)
    print("labels:", dict(Counter(e["status"] for e in labelled)))
    if args.label_cache:
        for entry in labelled:
            cache[f"{entry['task_id']}\n{_key(entry['program'])}"] = {
                k: entry[k] for k in ("status", "base_pass") if k in entry
            }
        args.label_cache.write_text(json.dumps(cache), encoding="utf-8")

    fails, passes = select(labelled)
    if len(fails) < PER_SIDE or len(passes) < PER_SIDE:
        raise SystemExit(f"only {len(fails)} fail / {len(passes)} pass selectable")

    rng = random.Random(SEED)
    rng.shuffle(fails)
    rng.shuffle(passes)
    evaluator = f"evalplus=={version('evalplus')} base+plus, fast_check"
    problems = ep._problems("humaneval")
    with open(args.output, "w", encoding="utf-8") as out:
        for index, entry in enumerate(
            e for pair in zip(fails, passes, strict=True) for e in pair
        ):
            row = {
                "id": f"code-verifier/{index:03d}",
                "task_id": entry["task_id"],
                "entry_point": problems[entry["task_id"]]["entry_point"],
                "prompt": problems[entry["task_id"]]["prompt"],
                "solution": entry["program"],
                "passes": entry["status"] == "pass",
                "base_pass": entry["base_pass"],
                "model": entry["model"],
                "version": DATASET_VERSION,
                "evaluator": evaluator,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    chosen = fails + passes
    print(f"wrote {len(chosen)} rows to {args.output}")
    print("near-miss fails:", sum(e["base_pass"] for e in fails))
    print("tasks:", len({e["task_id"] for e in chosen}))
    print("families (fail/pass):")
    fail_fam = Counter(e["family"] for e in fails)
    pass_fam = Counter(e["family"] for e in passes)
    for family in sorted(set(fail_fam) | set(pass_fam)):
        print(f"  {family:20} {fail_fam[family]:4} {pass_fam[family]:4}")


if __name__ == "__main__":
    main()
