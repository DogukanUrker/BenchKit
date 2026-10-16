"""Result shaping: from finished tasks to the rows reports serialize.

The per-task record, the running sums a job collects as its tasks finish, the
job result built from them, and the paired direct-versus-Pi annotation applied
once a run is over. Nothing here talks to a model, so it works on plain records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Literal

from benchkit import decision
from benchkit.benchmarks.base import Task
from benchkit.jobs import JobSpec, benchmark, option_count
from benchkit.metrics import throughput_metrics


@dataclass
class TaskRecord:
    """Outcome of a single task."""

    index: int
    task_id: str
    passed: bool
    tok_s: float
    response_time_s: float
    prompt: str
    response: str
    outcome: Literal[
        "pass",
        "fail",
        "loop_killed",
        "timeout",
        "length_exceeded",
        "harness_error",
        "contaminated",
    ] = "fail"
    perturbation: dict = field(default_factory=dict)
    score: float = 0.0
    error: str = ""
    entry_point: str = ""
    thinking: str = ""
    output_tokens: int = 0
    tokens_recovered: bool = False
    done_reason: str = ""
    timed_out: bool = False
    loop_killed: bool = False
    length_exceeded: bool = False
    harness_error: bool = False
    contaminated: bool = False
    loop_kill_score: float = 0.0
    loop_killed_at_s: float | None = None
    trace_status: str = "unavailable"
    thinking_time_s: float = 0.0
    time_to_first_answer_s: float | None = None
    loop_state: str = "unavailable"
    loop_score: float = 0.0
    loop_source: str = "none"
    loop_detected_at_s: float | None = None
    repeated_ngram_coverage: float = 0.0
    max_window_similarity: float = 0.0
    low_novelty_windows: int = 0
    max_repeated_block: int = 0
    loop_evidence: str = "none"
    active_cycle: bool = False
    recovered_cycle: bool = False
    cycle_period_tokens: int = 0
    cycle_repetitions: int = 0
    repeated_suffix_tokens: int = 0
    harness: str = "direct"
    harness_version: str = ""
    input_tokens: int = 0
    model_turns: int = 1
    tool_calls: int = 0
    tool_trace: list[dict] = field(default_factory=list)
    pi_scaffold: dict = field(default_factory=dict)
    attempts: list[dict] = field(default_factory=list)
    repair_attempts_used: int = 0
    repair_feedback: list[str] = field(default_factory=list)
    first_attempt_score: float = 0.0
    repaired: bool = False
    workspace: dict = field(default_factory=dict)
    decision: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return (
            f"{self.task_id} ({self.entry_point})" if self.entry_point else self.task_id
        )


def _repair_score_curve(records: list[TaskRecord], repair_attempts: int) -> list[dict]:
    """Return aggregate score and incremental gains after each repair round."""
    if not records or repair_attempts <= 0:
        return []

    def score_at(record: TaskRecord, round_number: int) -> float:
        if record.attempts:
            attempt = record.attempts[min(round_number, len(record.attempts) - 1)]
            return float(attempt.get("score", 0.0))
        if round_number == 0:
            return record.first_attempt_score * 100
        return record.score * 100

    curve: list[dict] = []
    previous_score = 0.0
    for round_number in range(repair_attempts + 1):
        scores = [score_at(record, round_number) for record in records]
        aggregate = sum(scores) / len(scores)
        previous_scores = (
            [score_at(record, round_number - 1) for record in records]
            if round_number
            else scores
        )
        curve.append(
            {
                "round": round_number,
                "score": round(aggregate, 1),
                "delta_pp": round(aggregate - previous_score, 1)
                if round_number
                else 0.0,
                "cumulative_delta_pp": round(aggregate - curve[0]["score"], 1)
                if curve
                else 0.0,
                "attempted": sum(
                    len(record.attempts) > round_number for record in records
                )
                if round_number
                else len(records),
                "new_passes": sum(
                    before < 100 <= after
                    for before, after in zip(previous_scores, scores, strict=True)
                )
                if round_number
                else sum(score >= 100 for score in scores),
            }
        )
        previous_score = aggregate
    return curve


@dataclass
class TaskOutcome:
    """One finalized task plus the raw metrics used by its job aggregate."""

    record: TaskRecord
    errors: int
    eval_count: int
    eval_duration_ns: int
    response_time_s: float


@dataclass
class JobTally:
    """Running sums for one job, folded in as its tasks finish."""

    records_by_position: dict[int, TaskRecord] = field(default_factory=dict)
    passed: int = 0
    score_points: float = 0.0
    scored_total: int = 0
    errors: int = 0
    total_tokens: int = 0
    total_response_time: float = 0.0
    throughput_tokens: int = 0
    throughput_eval_ns: int = 0
    throughput_response_time: float = 0.0
    throughput_items: int = 0

    def add(self, outcome: TaskOutcome) -> None:
        record = outcome.record
        self.records_by_position[record.index] = record
        self.passed += int(record.passed)
        self.score_points += record.score
        self.scored_total += int(not record.harness_error and not record.contaminated)
        self.errors += outcome.errors
        self.total_tokens += outcome.eval_count
        self.total_response_time += outcome.response_time_s
        complete_terminal_metrics = not (
            record.length_exceeded or record.timed_out
        ) or (
            outcome.eval_count > 0
            and outcome.eval_duration_ns > 0
            and outcome.response_time_s > 0
            and not record.tokens_recovered
        )
        if (
            not (record.loop_killed or record.harness_error)
            and complete_terminal_metrics
        ):
            self.throughput_tokens += outcome.eval_count
            self.throughput_eval_ns += outcome.eval_duration_ns
            self.throughput_response_time += outcome.response_time_s
            self.throughput_items += 1


def job_result(
    job: JobSpec,
    bench: object,
    tasks: list[Task],
    tally: JobTally,
    *,
    slice_spec: str | None,
    concurrency: int,
    wall_time_s: float,
    loop_kill_enabled: bool,
    loop_kill_percent: float,
    loop_kill_seconds: float,
) -> dict:
    """Build the result row for a job that finished at least one task."""
    records = [
        tally.records_by_position[position]
        for position in sorted(tally.records_by_position)
    ]
    total_time = round(wall_time_s, 1)
    completed = len(records)
    scored_records = [
        record
        for record in records
        if not record.harness_error and not record.contaminated
    ]
    scored_total = len(scored_records)
    throughput_coverage = (
        tally.throughput_response_time / tally.total_response_time
        if tally.total_response_time > 0
        else tally.throughput_items / completed
        if completed
        else 0.0
    )
    throughput_wall_time = wall_time_s * throughput_coverage
    throughput = throughput_metrics(
        output_tokens=tally.throughput_tokens,
        generation_time_s=tally.throughput_eval_ns / 1e9,
        request_time_s=tally.throughput_response_time,
        wall_time_s=throughput_wall_time,
    )
    tok_s_per_stream = round(throughput["tok_s_per_stream"], 1)
    tok_s_aggregate = round(throughput["tok_s_aggregate"], 1)
    concurrency_eff = round(
        min(float(concurrency), throughput["concurrency_eff"]),
        2,
    )
    score = (
        sum(record.score for record in scored_records) / scored_total * 100
        if scored_total
        else 0.0
    )
    loops = sum(record.loop_state == "looping" for record in records)
    suspected = sum(
        record.loop_state == "suspected" and not record.recovered_cycle
        for record in records
    )
    recovered = sum(record.recovered_cycle for record in records)
    timeouts = sum(record.timed_out for record in records)
    loop_kills = sum(record.loop_killed for record in records)
    length_exceeded = sum(record.length_exceeded for record in records)
    harness_errors = sum(record.harness_error for record in records)
    contaminated_records = [record for record in records if record.contaminated]
    failures = sum(record.outcome == "fail" for record in records)
    traced = sum(record.trace_status != "unavailable" for record in records)
    latency_records = [
        record
        for record in records
        if not (
            record.loop_killed
            or record.timed_out
            or record.length_exceeded
            or record.harness_error
        )
        and record.time_to_first_answer_s is not None
    ]
    thinking_times = [record.thinking_time_s for record in latency_records]
    answer_times = [float(record.time_to_first_answer_s) for record in latency_records]
    pi_scaffold = next(
        (record.pi_scaffold for record in records if record.pi_scaffold), {}
    )
    pi_metadata = {f"pi_{key}": value for key, value in pi_scaffold.items()}
    agentic = [
        record.workspace.get("agentic_metrics", {})
        for record in records
        if isinstance(record.workspace.get("agentic_metrics"), dict)
    ]
    schema_valid = sum(int(item.get("tool_schema_valid_calls", 0)) for item in agentic)
    schema_invalid = sum(
        int(item.get("tool_schema_invalid_calls", 0)) for item in agentic
    )
    errored_calls = sum(int(item.get("errored_tool_calls", 0)) for item in agentic)
    recoveries = sum(int(item.get("post_error_recoveries", 0)) for item in agentic)
    redundant_calls = sum(int(item.get("redundant_tool_calls", 0)) for item in agentic)
    destructive_actions = sum(
        int(item.get("destructive_action_count", 0)) for item in agentic
    )
    observed_calls = schema_valid + schema_invalid
    solved_records = [record for record in records if record.passed]
    repair_score_curve = _repair_score_curve(scored_records, job.repair_attempts)
    repair_round_columns = {
        key: value
        for point in repair_score_curve[1:]
        for key, value in {
            f"repair_round_{point['round']}_score": point["score"],
            f"repair_round_{point['round']}_delta_pp": point["delta_pp"],
            f"repair_round_{point['round']}_cumulative_delta_pp": point[
                "cumulative_delta_pp"
            ],
            f"repair_round_{point['round']}_attempted": point["attempted"],
            f"repair_round_{point['round']}_new_passes": point["new_passes"],
        }.items()
    }

    result = {
        "model": job.model,
        "benchmark": bench.name,
        "benchmark_label": job.benchmark_label,
        "variant": job.variant,
        "harness": job.harness,
        "harness_label": job.harness_label,
        "harness_version": (
            next(
                (
                    record.harness_version
                    for record in records
                    if record.harness_version
                ),
                "",
            )
        ),
        "score": round(score, 1),
        "passed": tally.passed,
        "total": completed,
        "scored_total": scored_total,
        # Keep ``tok_s`` as a compatibility alias for consumers of older
        # BenchKit JSON. New displays use the explicitly named metrics.
        "tok_s": tok_s_per_stream,
        "tok_s_per_stream": tok_s_per_stream,
        "tok_s_aggregate": tok_s_aggregate,
        "concurrency_eff": concurrency_eff,
        "throughput_coverage": round(throughput_coverage * 100, 1),
        "throughput_items": tally.throughput_items,
        "throughput_wall_time": round(throughput_wall_time, 3),
        "total_output_tokens": tally.total_tokens,
        "throughput_output_tokens": tally.throughput_tokens,
        "total_input_tokens": sum(record.input_tokens for record in records),
        "model_turns": sum(record.model_turns for record in records),
        "tool_calls": sum(record.tool_calls for record in records),
        "tool_schema_valid_calls": schema_valid,
        "tool_schema_invalid_calls": schema_invalid,
        "tool_schema_validity_rate": round(schema_valid / observed_calls * 100, 1)
        if observed_calls
        else None,
        "errored_tool_calls": errored_calls,
        "post_error_recoveries": recoveries,
        "post_error_recovery_rate": round(recoveries / errored_calls * 100, 1)
        if errored_calls
        else None,
        "redundant_tool_calls": redundant_calls,
        "redundant_action_rate": round(redundant_calls / observed_calls * 100, 1)
        if observed_calls
        else None,
        "destructive_action_count": destructive_actions,
        "destructive_action_rate": round(destructive_actions / observed_calls * 100, 1)
        if observed_calls
        else None,
        "avg_turns_to_solve": round(
            sum(record.model_turns for record in solved_records) / len(solved_records),
            1,
        )
        if solved_records
        else None,
        "avg_tokens_to_solve": round(
            sum(record.input_tokens + record.output_tokens for record in solved_records)
            / len(solved_records),
            1,
        )
        if solved_records
        else None,
        "first_attempt_score": round(
            sum(record.first_attempt_score for record in scored_records)
            / scored_total
            * 100,
            1,
        )
        if scored_total
        else 0.0,
        "repair_delta_pp": round(
            score
            - sum(record.first_attempt_score for record in scored_records)
            / scored_total
            * 100,
            1,
        )
        if scored_total
        else 0.0,
        "repair_attempted": sum(record.repair_attempts_used > 0 for record in records),
        "repair_turns": sum(record.repair_attempts_used for record in records),
        "repair_successes": sum(record.repaired for record in records),
        "repair_score_curve": repair_score_curve,
        **repair_round_columns,
        "sum_generation_time": round(tally.throughput_eval_ns / 1e9, 3),
        "sum_request_time": round(tally.throughput_response_time, 3),
        "avg_response_time": round(
            tally.throughput_response_time / tally.throughput_items, 1
        )
        if tally.throughput_items
        else 0.0,
        "total_time": total_time,
        "slice": slice_spec,
        "concurrency": concurrency,
        "errors": tally.errors,
        "harness_errors": harness_errors,
        "contaminated": len(contaminated_records),
        "contamination_by_language": {
            language: sum(
                record.workspace.get("language") == language
                for record in contaminated_records
            )
            for language in sorted(
                {
                    str(record.workspace.get("language"))
                    for record in contaminated_records
                    if record.workspace.get("language")
                }
            )
        },
        "contaminated_tasks": [
            {
                "task_id": record.task_id,
                "language": record.workspace.get("language", ""),
                "guard_hits": record.workspace.get("answer_key_guard_hits", []),
            }
            for record in contaminated_records
        ],
        "length_exceeded": length_exceeded,
        "failures": failures,
        "timeouts": timeouts,
        "loop_kills": loop_kills,
        "loops": loops,
        "suspected_loops": suspected,
        "recovered_loops": recovered,
        "loop_rate": round(loops / completed * 100, 1),
        "trace_coverage": round(traced / completed * 100, 1),
        "median_thinking_time": round(median(thinking_times), 1)
        if thinking_times
        else 0.0,
        "median_time_to_answer": round(median(answer_times), 1)
        if answer_times
        else 0.0,
        "loop_kill_enabled": loop_kill_enabled,
        "loop_kill_percent": loop_kill_percent,
        "loop_kill_seconds": loop_kill_seconds,
        "tasks": [
            {
                "task_id": record.task_id,
                "passed": record.passed,
                "outcome": record.outcome,
                "score": round(record.score * 100, 1),
                "tok_s": record.tok_s,
                "response_time_s": record.response_time_s,
                "prompt": record.prompt,
                "response": record.response,
                "perturbation": record.perturbation or None,
                "error": record.error,
                "entry_point": record.entry_point,
                "thinking": record.thinking,
                "output_tokens": record.output_tokens,
                "tokens_recovered": record.tokens_recovered,
                "done_reason": record.done_reason,
                "timed_out": record.timed_out,
                "loop_killed": record.loop_killed,
                "length_exceeded": record.length_exceeded,
                "harness_error": record.harness_error,
                "contaminated": record.contaminated,
                "contamination_verdict": record.workspace.get(
                    "contamination_verdict", "CLEAR"
                ),
                "loop_kill_score": record.loop_kill_score,
                "loop_killed_at_s": record.loop_killed_at_s,
                "trace_status": record.trace_status,
                "thinking_time_s": record.thinking_time_s,
                "time_to_first_answer_s": record.time_to_first_answer_s,
                "loop_state": record.loop_state,
                "loop_score": record.loop_score,
                "loop_source": record.loop_source,
                "loop_detected_at_s": record.loop_detected_at_s,
                "repeated_ngram_coverage": (record.repeated_ngram_coverage),
                "max_window_similarity": record.max_window_similarity,
                "low_novelty_windows": record.low_novelty_windows,
                "max_repeated_block": record.max_repeated_block,
                "loop_evidence": record.loop_evidence,
                "active_cycle": record.active_cycle,
                "recovered_cycle": record.recovered_cycle,
                "cycle_period_tokens": record.cycle_period_tokens,
                "cycle_repetitions": record.cycle_repetitions,
                "repeated_suffix_tokens": record.repeated_suffix_tokens,
                "harness": record.harness,
                "harness_version": record.harness_version,
                "input_tokens": record.input_tokens,
                "model_turns": record.model_turns,
                "tool_calls": record.tool_calls,
                "tool_trace": record.tool_trace,
                "attempts": record.attempts,
                "repair_attempts_used": record.repair_attempts_used,
                "repair_feedback": record.repair_feedback,
                "first_attempt_score": round(record.first_attempt_score * 100, 1),
                "repaired": record.repaired,
                "workspace": record.workspace or None,
                "agentic_metrics": record.workspace.get("agentic_metrics") or None,
                "decision": record.decision or None,
            }
            for record in records
        ],
        **decision.summary([record.decision for record in scored_records], wall_time_s),
        **pi_metadata,
        **_result_metadata(job),
    }
    task_statistics = getattr(bench, "task_statistics", None)
    if callable(task_statistics):
        result["task_statistics"] = task_statistics(records)
    # Multiple choice: a repair is a second guess at a small option set,
    # so the headline is the first attempt and the repair gain sits beside
    # it. Code and open-answer suites keep the repaired score.
    if (
        job.repair_attempts
        and tasks
        and all(option_count(bench, task) for task in tasks)
    ):
        result.update(
            headline="first_attempt",
            repaired_score=result["score"],
            repaired_passed=result["passed"],
            score=result["first_attempt_score"],
            passed=sum(record.first_attempt_score >= 1.0 for record in scored_records),
        )
    summary_fields = getattr(bench, "summary_fields", None)
    if callable(summary_fields):
        result.update(summary_fields(scored_records))
    return result


def _result_metadata(job: JobSpec) -> dict:
    bench = benchmark(job.benchmark)
    metadata = getattr(bench, "result_metadata", None)
    value = metadata(job.variant) if callable(metadata) else None
    result = dict(value) if isinstance(value, dict) else {}
    result.update(
        harness=job.harness,
        harness_label=job.harness_label,
        repair_attempts=job.repair_attempts,
    )
    if job.perturbation:
        result.update(
            perturbation=job.perturbation,
            perturbation_seed=job.perturbation_seed,
            include_in_overall=False,
        )
    return result


def empty_result(job: JobSpec) -> dict:
    """Result shape for a job that finished without scoring a task."""
    return {
        "model": job.model,
        "benchmark": job.benchmark,
        "benchmark_label": job.benchmark_label,
        "variant": job.variant,
        "harness": job.harness,
        "harness_label": job.harness_label,
        "score": 0.0,
        "passed": 0,
        "total": 0,
        "scored_total": 0,
        "tok_s": 0.0,
        "tok_s_per_stream": 0.0,
        "tok_s_aggregate": 0.0,
        "concurrency_eff": 0.0,
        "throughput_coverage": 0.0,
        "throughput_items": 0,
        "throughput_wall_time": 0.0,
        "total_output_tokens": 0,
        "total_input_tokens": 0,
        "model_turns": 0,
        "tool_calls": 0,
        "first_attempt_score": 0.0,
        "repair_delta_pp": 0.0,
        "repair_attempted": 0,
        "repair_successes": 0,
        "sum_generation_time": 0.0,
        "sum_request_time": 0.0,
        "avg_response_time": 0.0,
        "total_time": 0.0,
        "slice": job.slice_spec,
        "concurrency": 1,
        "errors": 0,
        "harness_errors": 0,
        "contaminated": 0,
        "contamination_by_language": {},
        "contaminated_tasks": [],
        "length_exceeded": 0,
        "timeouts": 0,
        "loop_kills": 0,
        "failures": 0,
        "loops": 0,
        "suspected_loops": 0,
        "loop_rate": 0.0,
        "trace_coverage": 0.0,
        "median_thinking_time": 0.0,
        "median_time_to_answer": 0.0,
        "tasks": [],
        **_result_metadata(job),
    }


def _harness_pair_key(result: dict) -> tuple[object, ...]:
    return (
        result.get("model"),
        result.get("benchmark"),
        result.get("variant"),
        result.get("slice"),
        result.get("perturbation"),
        result.get("perturbation_seed") if result.get("perturbation") else None,
        result.get("repair_attempts", 0),
    )


def annotate_harness_effect(results: list[dict]) -> None:
    """Attach paired direct-versus-agent deltas to Pi result rows."""
    direct = {
        _harness_pair_key(result): result
        for result in results
        if result.get("harness", "direct") == "direct"
    }
    for result in results:
        if result.get("harness") != "pi":
            continue
        baseline = direct.get(_harness_pair_key(result))
        if baseline is None:
            continue
        baseline_tasks = {task["task_id"]: task for task in baseline.get("tasks", [])}
        pairs = [
            (baseline_tasks[task["task_id"]], task)
            for task in result.get("tasks", [])
            if task["task_id"] in baseline_tasks
            and not task.get("harness_error")
            and not baseline_tasks[task["task_id"]].get("harness_error")
        ]
        if not pairs:
            continue
        total = len(pairs)
        direct_points = sum(float(before.get("score", 0)) for before, _ in pairs)
        pi_points = sum(float(after.get("score", 0)) for _, after in pairs)
        direct_first_points = sum(
            float(before.get("first_attempt_score", before.get("score", 0)))
            for before, _ in pairs
        )
        pi_first_points = sum(
            float(after.get("first_attempt_score", after.get("score", 0)))
            for _, after in pairs
        )
        gains = sum(
            not bool(before.get("passed")) and bool(after.get("passed"))
            for before, after in pairs
        )
        regressions = sum(
            bool(before.get("passed")) and not bool(after.get("passed"))
            for before, after in pairs
        )
        direct_loop_kills = sum(bool(before.get("loop_killed")) for before, _ in pairs)
        harness_loop_kills = sum(bool(after.get("loop_killed")) for _, after in pairs)
        result.update(
            harness_paired_total=total,
            direct_score=round(direct_points / total, 1),
            harness_score=round(pi_points / total, 1),
            harness_score_delta_pp=round((pi_points - direct_points) / total, 1),
            direct_first_score=round(direct_first_points / total, 1),
            harness_first_score=round(pi_first_points / total, 1),
            harness_first_score_delta_pp=round(
                (pi_first_points - direct_first_points) / total,
                1,
            ),
            harness_gains=gains,
            harness_regressions=regressions,
            direct_loop_kill_rate=round(direct_loop_kills / total * 100, 1),
            harness_loop_kill_rate=round(harness_loop_kills / total * 100, 1),
            loop_kill_delta_pp=round(
                (harness_loop_kills - direct_loop_kills) / total * 100,
                1,
            ),
        )
