"""Save benchmark results to disk."""

import csv
import json
import os
from datetime import datetime
from importlib.resources import files
from pathlib import Path

from benchkit.metrics import aggregate_tok_s, effective_concurrency, stream_tok_s


def _fmt_time(s: float) -> str:
    s = round(s)
    if s >= 60:
        return f"{s // 60}m {s % 60}s"
    return f"{s}s"


def _safe_json(data: object) -> str:
    """Serialize data for an HTML script element without allowing tag breakout."""
    return (
        json.dumps(data, ensure_ascii=False)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def render_html(
    results: list[dict],
    generated_at: str,
    provider: str = "",
    host: str = "",
    hardware: str = "",
) -> str:
    """Render the packaged HTML template with embedded report data."""
    payload = _safe_json(
        {
            "generated_at": generated_at,
            "provider": provider,
            "host": host,
            "hardware": hardware,
            "results": results,
        }
    )
    template = (
        files("benchkit").joinpath("templates/report.html").read_text(encoding="utf-8")
    )
    return template.replace("__BENCHKIT_REPORT_DATA__", payload)


def save(
    results: list[dict],
    provider: str = "",
    host: str = "",
    hardware: str | None = None,
) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = Path("results") / ts
    out.mkdir(parents=True, exist_ok=True)
    hardware = (
        hardware if hardware is not None else os.environ.get("BENCHKIT_HARDWARE", "")
    )

    # Full JSON (includes per-task details)
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary CSV (one row per model×benchmark, no tasks column)
    summary = [{k: v for k, v in r.items() if k != "tasks"} for r in results]
    if summary:
        with open(out / "results.csv", "w", newline="") as f:
            fieldnames = list(
                dict.fromkeys(key for result in summary for key in result)
            )
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary)

    # Markdown table and task-level details.
    with open(out / "results.md", "w") as f:
        f.write("# BenchKit Results\n\n")
        f.write(f"**Date:** {ts}\n\n")
        f.write(
            "| Model | Harness | Benchmark | Score | Harness score Δ | Loop-kill Δ | Repair Δ | Fixed | Delta | Regressions | Passed | Scored | Total | Fail | Loop killed | Timeout | Length exceeded | Harness error | Tools used/available | Parallel | Trace | Throughput coverage | Agg tok/s | Stream tok/s | Effective | Avg Resp | Wall Time |\n"
        )
        f.write(
            "|-------|---------|-----------|------:|----------------:|------------:|---------:|------:|------:|------------:|-------:|-------:|------:|-----:|------------:|--------:|----------------:|--------------:|---------------------:|---------:|------:|--------------------:|----------:|-------------:|----------:|---------:|----------:|\n"
        )
        for result in results:
            benchmark_label = result.get("benchmark_label", result["benchmark"])
            delta = result.get("score_delta_pp")
            delta_text = f"{float(delta):+.1f}pp" if delta is not None else "—"
            harness_delta = result.get("harness_score_delta_pp")
            harness_delta_text = (
                f"{float(harness_delta):+.1f}pp" if harness_delta is not None else "—"
            )
            repair_delta_text = (
                f"{float(result.get('repair_delta_pp', 0)):+.1f}pp"
                if result.get("repair_attempts")
                else "—"
            )
            repaired_text = (
                f"{result.get('repair_successes', 0)}/"
                f"{result.get('repair_attempted', 0)}"
                if result.get("repair_attempts")
                else "—"
            )
            regressions = (
                f"{result.get('regressions', 0)}/{result.get('paired_total', 0)}"
                if result.get("perturbation")
                else "—"
            )
            loop_kill_delta = result.get("loop_kill_delta_pp")
            loop_kill_delta_text = (
                f"{float(loop_kill_delta):+.1f}pp"
                if loop_kill_delta is not None
                else "—"
            )
            tools_available = len(result.get("pi_tools_available") or [])
            tools_text = (
                f"{result.get('tool_calls', 0)}/{tools_available}"
                if result.get("harness") == "pi"
                else "—"
            )
            f.write(
                f"| {result['model']} | {result.get('harness_label', 'Direct')} "
                f"| {benchmark_label} | {result['score']}% "
                f"| {harness_delta_text} | {loop_kill_delta_text} "
                f"| {repair_delta_text} | {repaired_text} "
                f"| {delta_text} | {regressions} "
                f"| {result['passed']} | {result.get('scored_total', result['total'])} "
                f"| {result['total']} | {result.get('failures', 0)} "
                f"| {result.get('loop_kills', 0)} | {result.get('timeouts', 0)} "
                f"| {result.get('length_exceeded', 0)} "
                f"| {result.get('harness_errors', 0)} | {tools_text} "
                f"| {result.get('concurrency', 1)} "
                f"| {result.get('trace_coverage', 0)}% "
                f"| {result.get('throughput_coverage', 100)}% "
                f"| {aggregate_tok_s(result):.1f} "
                f"| {stream_tok_s(result):.1f} "
                f"| {effective_concurrency(result):.2f}x "
                f"| {result['avg_response_time']}s "
                f"| {_fmt_time(result['total_time'])} |\n"
            )

        f.write(
            "\n**Throughput:** Agg tok/s is covered output tokens divided by job "
            "covered wall time. Stream tok/s uses covered server decode time. "
            "Effective concurrency is summed request time divided by covered wall "
            "time. Throughput coverage is the share of request wall time represented "
            "by completed generations; loop kills, timeouts and harness errors retain "
            "their task token counts but are excluded when complete timing is not "
            "available. Length-exceeded items are included when their token and "
            "timing payload is complete.\n"
        )

        harness_pairs = [
            result
            for result in results
            if result.get("harness_paired_total") is not None
        ]
        if harness_pairs:
            f.write("\n## Harness score comparison\n\n")
            f.write(
                "| Model | Benchmark | Feedback | Paired | Direct first | Pi first | Initial score Δ | Direct final | Pi final | Final score Δ | Direct loop kill | Pi loop kill | Loop-kill Δ | Gains | Regressions | Pi version |\n"
            )
            f.write(
                "|-------|-----------|----------|-------:|-------------:|---------:|----------------:|-------------:|---------:|--------------:|-----------------:|-------------:|------------:|------:|------------:|------------|\n"
            )
            for result in harness_pairs:
                f.write(
                    f"| {result['model']} "
                    f"| {result.get('benchmark_label', result['benchmark'])} "
                    f"| {'one repair' if result.get('repair_attempts') else 'none'} "
                    f"| {result.get('harness_paired_total', 0)} "
                    f"| {result.get('direct_first_score', 0):.1f}% "
                    f"| {result.get('harness_first_score', 0):.1f}% "
                    f"| {result.get('harness_first_score_delta_pp', 0):+.1f}pp "
                    f"| {result.get('direct_score', 0):.1f}% "
                    f"| {result.get('harness_score', 0):.1f}% "
                    f"| {result.get('harness_score_delta_pp', 0):+.1f}pp "
                    f"| {result.get('direct_loop_kill_rate', 0):.1f}% "
                    f"| {result.get('harness_loop_kill_rate', 0):.1f}% "
                    f"| {result.get('loop_kill_delta_pp', 0):+.1f}pp "
                    f"| {result.get('harness_gains', 0)} "
                    f"| {result.get('harness_regressions', 0)} "
                    f"| {result.get('harness_version', 'latest')} |\n"
                )

        repair_rows = [result for result in results if result.get("repair_attempts")]
        if repair_rows:
            f.write("\n## Verifier repair effect\n\n")
            f.write(
                "| Model | Harness | Benchmark | First attempt | Final | Delta | Retried | Repaired |\n"
            )
            f.write(
                "|-------|---------|-----------|--------------:|------:|------:|--------:|---------:|\n"
            )
            for result in repair_rows:
                f.write(
                    f"| {result['model']} "
                    f"| {result.get('harness_label', 'Direct')} "
                    f"| {result.get('benchmark_label', result['benchmark'])} "
                    f"| {result.get('first_attempt_score', 0):.1f}% "
                    f"| {result.get('score', 0):.1f}% "
                    f"| {result.get('repair_delta_pp', 0):+.1f}pp "
                    f"| {result.get('repair_attempted', 0)} "
                    f"| {result.get('repair_successes', 0)} |\n"
                )

        perturbed = [result for result in results if result.get("perturbation")]
        if perturbed:
            f.write("\n## Choice-order robustness\n\n")
            f.write(
                "| Model | Harness | Benchmark | Seed | Paired | Clean | Perturbed | Delta | Regressions | Recoveries |\n"
            )
            f.write(
                "|-------|---------|-----------|-----:|-------:|------:|----------:|------:|------------:|-----------:|\n"
            )
            for result in perturbed:
                f.write(
                    f"| {result['model']} "
                    f"| {result.get('harness_label', 'Direct')} "
                    f"| {result.get('baseline_benchmark_label', result['benchmark'])} "
                    f"| {result.get('perturbation_seed', 42)} "
                    f"| {result.get('paired_total', 0)} "
                    f"| {result.get('baseline_score', 0):.1f}% "
                    f"| {result.get('perturbed_score', 0):.1f}% "
                    f"| {result.get('score_delta_pp', 0):+.1f}pp "
                    f"| {result.get('regressions', 0)} "
                    f"| {result.get('recoveries', 0)} |\n"
                )

        f.write("\n---\n\n")
        for result in results:
            benchmark_label = result.get("benchmark_label", result["benchmark"])
            harness = result.get("harness_label", "Direct")
            version = result.get("harness_version")
            version_text = f" {version}" if version else ""
            f.write(
                f"## {result['model']} / {harness}{version_text} / {benchmark_label}\n\n"
            )
            if result.get("harness") == "pi":
                tools = ", ".join(result.get("pi_tools_available") or []) or "none"
                f.write(
                    "**Pi scaffold:** "
                    f"system prompt `{result.get('pi_system_prompt_sha256', 'unavailable')}` "
                    f"· {result.get('pi_system_prompt_tokens', 'unknown')} tokens "
                    f"· {result.get('pi_system_prompt_chars', 0)} chars "
                    f"· native tools available: {tools}\n\n"
                )
                if result.get("pi_max_output_tokens"):
                    f.write(
                        "**Pi output limit:** "
                        f"{result['pi_max_output_tokens']} tokens via "
                        f"`{result.get('pi_max_output_tokens_field', 'unknown')}`\n\n"
                    )
                if result.get("pi_system_prompt"):
                    f.write("**Pi system prompt:**\n\n~~~text\n")
                    f.write(str(result["pi_system_prompt"]).rstrip())
                    f.write("\n~~~\n\n")
            for task in result["tasks"]:
                task_id = task["task_id"]
                entry = task.get("entry_point")
                label = f"{task_id} ({entry})" if entry else task_id
                status = (
                    "🛑 LOOP KILLED"
                    if task.get("loop_killed")
                    else "⏱️ TIMEOUT"
                    if task.get("timed_out")
                    else "📏 LENGTH EXCEEDED"
                    if task.get("length_exceeded")
                    else "⚠️ HARNESS ERROR"
                    if task.get("harness_error")
                    else "⚠️ ERROR"
                    if task.get("error")
                    else "✅ PASS"
                    if task["passed"]
                    else f"🟨 PARTIAL ({task.get('score', 0):.1f}%)"
                    if task.get("score", 0) > 0
                    else "❌ FAIL"
                )
                loop = (
                    "RECOVERED"
                    if task.get("recovered_cycle")
                    else task.get("loop_state", "unavailable").upper()
                )
                if (
                    loop == "CLEAR"
                    and task.get("loop_source") == "answer"
                    and task.get("trace_status") == "unavailable"
                ):
                    loop = "NO TRACE"
                f.write(f"### {label} — {status} · {loop}\n\n")
                f.write(
                    f"**Generation:** {task.get('output_tokens', 0)} tokens · "
                    f"{task.get('input_tokens', 0)} input · "
                    f"{task.get('model_turns', 1)} model turn(s) · "
                    f"{task.get('tool_calls', 0)} tool call(s) · "
                    f"{task.get('tok_s', 0)} stream tok/s · "
                    f"{task.get('response_time_s', 0)}s · "
                    f"trace {task.get('trace_status', 'unavailable')} · "
                    f"loop score {task.get('loop_score', 0):.1%}\n\n"
                )
                if task.get("error"):
                    f.write(f"**Error:** {task['error']}\n\n")
                if workspace := task.get("workspace"):
                    command = workspace.get("test_command") or []
                    f.write(
                        "**Workspace verifier:** "
                        f"`{' '.join(map(str, command))}` · exit "
                        f"{workspace.get('test_exit_code', '—')}\n\n"
                    )
                    if workspace.get("test_output"):
                        f.write(
                            "~~~text\n"
                            + str(workspace["test_output"]).rstrip()
                            + "\n~~~\n\n"
                        )
                    if workspace.get("patch"):
                        f.write(
                            "**Workspace patch:**\n\n~~~diff\n"
                            + str(workspace["patch"]).rstrip()
                            + "\n~~~\n\n"
                        )
                if attempts := task.get("attempts"):
                    f.write("**Verifier-feedback attempts:**\n\n")
                    for attempt in attempts:
                        state = "pass" if attempt.get("passed") else "fail"
                        f.write(
                            f"#### Attempt {attempt.get('attempt', 1)} — "
                            f"{state} · {attempt.get('score', 0):.1f}%\n\n"
                        )
                        if attempt.get("feedback"):
                            f.write(f"Verifier feedback: {attempt['feedback']}\n\n")
                        f.write("~~~text\n")
                        f.write(str(attempt.get("response") or "").rstrip())
                        f.write("\n~~~\n\n")
                if tool_trace := task.get("tool_trace"):
                    f.write("**Pi native tools:**\n\n")
                    for call in tool_trace:
                        state = "error" if call.get("is_error") else "ok"
                        duration = float(call.get("duration_s") or 0)
                        f.write(
                            f"- `{call.get('name', 'tool')}` · {state} · "
                            f"{duration:.3f}s\n\n"
                        )
                        f.write(
                            "~~~json\n"
                            + json.dumps(
                                call.get("arguments") or {},
                                ensure_ascii=False,
                                indent=2,
                            ).rstrip()
                            + "\n~~~\n\n"
                        )
                        if call.get("output"):
                            suffix = (
                                " (last 8,000 chars)"
                                if call.get("output_truncated")
                                else ""
                            )
                            f.write(f"Tool output{suffix}:\n\n")
                            f.write(f"~~~\n{str(call['output']).rstrip()}\n~~~\n\n")
                if perturbation := task.get("perturbation"):
                    order = ", ".join(perturbation.get("choice_order", []))
                    f.write(
                        "**Perturbation:** choice-order "
                        f"(seed {perturbation.get('seed')}) · original labels in "
                        f"new order: {order} · answer "
                        f"{perturbation.get('original_answer')} → "
                        f"{perturbation.get('perturbed_answer')}\n\n"
                    )
                f.write("**Prompt:**\n\n")
                f.write(f"~~~\n{task['prompt'].rstrip()}\n~~~\n\n")
                if task.get("thinking"):
                    f.write("**Thinking:**\n\n")
                    f.write(f"~~~\n{task['thinking'].rstrip()}\n~~~\n\n")
                f.write("**Response:**\n\n")
                f.write(f"~~~\n{task['response'].rstrip()}\n~~~\n\n")
                f.write("---\n\n")

    with open(out / "results.html", "w") as f:
        f.write(render_html(results, ts, provider, host, hardware))

    return out
