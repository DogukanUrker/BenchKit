"""Save benchmark results to disk."""

import csv
import json
import os
from datetime import datetime
from importlib.resources import files
from pathlib import Path

from benchkit import artifacts
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


def arena_results(results: list[dict]) -> list[dict]:
    """Result rows scored by rendering, whether or not a page ever loaded.

    Selection is by suite, not by outcome: a run whose tasks all timed out
    still gets a gallery, and it still shows those tasks.
    """
    return [
        result
        for result in results
        if result.get("scoring") in {"render-only", "build-metrics"}
        or any(
            isinstance(task.get("workspace"), dict)
            and (
                task["workspace"].get("render_status")
                or task["workspace"].get("build_status")
            )
            for task in result.get("tasks") or []
        )
    ]


def render_arena_html(
    results: list[dict],
    generated_at: str,
    provider: str = "",
    host: str = "",
    hardware: str = "",
) -> str:
    """Render the screenshot gallery for the render-scored suites.

    The gallery lives beside the screenshots and generated pages it links to,
    so a preview opens the model's own page in a new tab instead of a picture
    of it.
    """
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
        files("benchkit").joinpath("templates/arena.html").read_text(encoding="utf-8")
    )
    return template.replace("__BENCHKIT_ARENA_DATA__", payload)


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

    # Screenshots and generated pages staged during the run move in here first,
    # so every path written below is relative to the report directory.
    artifacts.collect(out, results)

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
        contaminated_rows = [result for result in results if result.get("contaminated")]
        if contaminated_rows:
            f.write("## ⚠️ Contamination detected\n\n")
            language_counts: dict[str, int] = {}
            offending: list[dict] = []
            for result in contaminated_rows:
                for language, count in result.get(
                    "contamination_by_language", {}
                ).items():
                    language_counts[language] = language_counts.get(language, 0) + int(
                        count
                    )
                offending.extend(result.get("contaminated_tasks", []))
            f.write(
                "**Headline score excludes "
                f"{sum(language_counts.values())} contaminated task(s).**\n\n"
            )
            f.write(
                "**Per language:** "
                + " · ".join(
                    f"{language} {count}"
                    for language, count in sorted(language_counts.items())
                )
                + "\n\n"
            )
            for task in offending:
                hits = task.get("guard_hits") or []
                targets = sorted(
                    {str(match) for hit in hits for match in (hit.get("matches") or [])}
                )
                f.write(
                    f"- `{task.get('task_id', '')}`: "
                    + ", ".join(f"`{target}`" for target in targets)
                    + "\n"
                )
            f.write("\n")
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

        agentic_rows = [
            result
            for result in results
            if result.get("tool_schema_validity_rate") is not None
        ]
        if agentic_rows:
            f.write("\n## Agentic metrics\n\n")
            f.write(
                "| Model | Benchmark | Schema valid | Avg turns to solve | "
                "Avg tokens to solve | Post-error recovery | Redundant actions | "
                "Destructive actions |\n"
            )
            f.write(
                "|-------|-----------|-------------:|-------------------:|"
                "--------------------:|--------------------:|------------------:|"
                "--------------------:|\n"
            )
            for result in agentic_rows:
                recovery = result.get("post_error_recovery_rate")
                f.write(
                    f"| {result['model']} "
                    f"| {result.get('benchmark_label', result['benchmark'])} "
                    f"| {result['tool_schema_validity_rate']:.1f}% "
                    f"| {result.get('avg_turns_to_solve') or '—'} "
                    f"| {result.get('avg_tokens_to_solve') or '—'} "
                    f"| {f'{recovery:.1f}%' if recovery is not None else '—'} "
                    f"| {result.get('redundant_action_rate', 0):.1f}% "
                    f"| {result.get('destructive_action_count', 0)} |\n"
                )

        ruler_rows = [result for result in results if result.get("task_statistics")]
        if ruler_rows:
            f.write("\n## RULER per-task confidence intervals\n\n")
            f.write("| Model | Context | Task | Samples | Score | 95% CI |\n")
            f.write("|-------|---------|------|--------:|------:|-------:|\n")
            for result in ruler_rows:
                for row in result["task_statistics"]:
                    f.write(
                        f"| {result['model']} | {result.get('context_label', '')} "
                        f"| {row['task']} | {row['samples']} | {row['score']:.1f}% "
                        f"| [{row['ci95_low']:.1f}%, {row['ci95_high']:.1f}%] |\n"
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
                    f"| {result.get('repair_attempts', 0)} repair turn(s) "
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
                "| Model | Harness | Benchmark | Max repairs | First attempt | Final | Delta | Retried tasks | Repair turns | Repaired |\n"
            )
            f.write(
                "|-------|---------|-----------|------------:|--------------:|------:|------:|--------------:|-------------:|---------:|\n"
            )
            for result in repair_rows:
                f.write(
                    f"| {result['model']} "
                    f"| {result.get('harness_label', 'Direct')} "
                    f"| {result.get('benchmark_label', result['benchmark'])} "
                    f"| {result.get('repair_attempts', 0)} "
                    f"| {result.get('first_attempt_score', 0):.1f}% "
                    f"| {result.get('score', 0):.1f}% "
                    f"| {result.get('repair_delta_pp', 0):+.1f}pp "
                    f"| {result.get('repair_attempted', 0)} "
                    f"| {result.get('repair_turns', result.get('repair_attempted', 0))} "
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
                    "☣️ CONTAMINATED"
                    if task.get("contaminated")
                    else "🛑 LOOP KILLED"
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
                    if checkpoints := workspace.get("checkpoints"):
                        f.write(
                            "**Checkpoints:** "
                            f"{workspace.get('positive_points', 0)}/"
                            f"{workspace.get('max_points', 0)} points"
                        )
                        if workspace.get("penalty_points"):
                            f.write(f" · −{workspace.get('penalty_points', 0)} penalty")
                        f.write("\n\n")
                        f.write("| Checkpoint | Weight | Awarded | Evidence |\n")
                        f.write("|------------|-------:|--------:|----------|\n")
                        for checkpoint in checkpoints:
                            evidence = str(checkpoint.get("evidence", "")).replace(
                                "|", "\\|"
                            )
                            f.write(
                                f"| {checkpoint.get('id', '')} "
                                f"| {checkpoint.get('weight', 0):+} "
                                f"| {checkpoint.get('awarded', 0):+} "
                                f"| {evidence} |\n"
                            )
                        f.write("\n")
                    command = workspace.get("test_command") or []
                    if command:
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
                    if workspace.get("build_status"):
                        f.write(
                            f"**Build:** {workspace['build_status']} · "
                            f"{workspace.get('blocks_total', 0)} block(s) · "
                            f"{workspace.get('palette_size', 0)} block type(s) · "
                            f"{workspace.get('hallucinated_blocks', 0)} "
                            "non-existent id(s) · "
                            f"{workspace.get('blocks_out_of_bounds', 0)} "
                            "out of bounds · "
                            f"{workspace.get('floating_fraction', 0)} floating\n\n"
                        )
                        for view in ("iso", "side", "top"):
                            if shot := workspace.get(f"screenshot_{view}"):
                                f.write(
                                    f"![{task.get('task_id', 'build')} {view}]"
                                    f"({shot})\n\n"
                                )
                        if script := workspace.get("script_py"):
                            f.write(f"Build script: [{script}]({script})\n\n")
                    elif workspace.get("render_status"):
                        f.write(
                            f"**Render:** {workspace['render_status']} · "
                            f"{len(workspace.get('console_errors') or [])} console "
                            f"error(s) · {len(workspace.get('page_errors') or [])} "
                            "uncaught · "
                            f"{workspace.get('animation_frames', 0)} frame(s)\n\n"
                        )
                        if shot := workspace.get("screenshot"):
                            f.write(
                                f"![{task.get('task_id', 'screenshot')}]({shot})\n\n"
                            )
                        if page := workspace.get("page_html"):
                            f.write(f"Generated page: [{page}]({page})\n\n")
                        diagnostics = [
                            *(
                                f"[uncaught] {item}"
                                for item in workspace.get("page_errors") or []
                            ),
                            *(
                                f"[console] {item}"
                                for item in workspace.get("console_errors") or []
                            ),
                            *(
                                f"[blocked] {item}"
                                for item in workspace.get("blocked_requests") or []
                            ),
                        ]
                        if diagnostics:
                            f.write(
                                "~~~text\n"
                                + "\n".join(map(str, diagnostics))
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

    # Rendered suites get their own gallery: big previews that open the
    # generated page itself, next to the pass rate.
    if gallery := arena_results(results):
        with open(out / "arena.html", "w") as f:
            f.write(render_arena_html(gallery, ts, provider, host, hardware))

    artifacts.cleanup()
    return out
