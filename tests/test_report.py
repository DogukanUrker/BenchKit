"""Report semantics regressions."""

import csv
import json

from benchkit.report import save


def test_markdown_exposes_score_denominator_failure_modes_and_pi_scaffold(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = {
        "model": "model",
        "harness": "pi",
        "harness_label": "Pi agent",
        "harness_version": "0.84.1",
        "benchmark": "gpqa",
        "benchmark_label": "gpqa",
        "score": 50.0,
        "passed": 1,
        "scored_total": 2,
        "total": 3,
        "failures": 0,
        "loop_kills": 0,
        "timeouts": 0,
        "length_exceeded": 1,
        "harness_errors": 1,
        "concurrency": 1,
        "trace_coverage": 50.0,
        "throughput_coverage": 50.0,
        "tok_s_aggregate": 10.0,
        "tok_s_per_stream": 10.0,
        "concurrency_eff": 1.0,
        "avg_response_time": 1.0,
        "total_time": 2.0,
        "tool_calls": 0,
        "pi_system_prompt": "Exact stock Pi scaffold",
        "pi_system_prompt_sha256": "abc123",
        "pi_system_prompt_chars": 23,
        "pi_system_prompt_tokens": 5,
        "pi_tools_available": ["read", "bash", "edit", "write"],
        "pi_max_output_tokens": 16384,
        "pi_max_output_tokens_field": "max_completion_tokens",
        "tasks": [
            {
                "task_id": "GPQA/0",
                "passed": True,
                "score": 100.0,
                "prompt": "question",
                "response": "A",
            },
            {
                "task_id": "GPQA/1",
                "passed": False,
                "score": 0.0,
                "prompt": "question",
                "response": "",
                "thinking": "truncated reasoning",
                "error": "generation reached the model output-token limit",
                "length_exceeded": True,
                "output_tokens": 16384,
                "response_time_s": 610.0,
            },
            {
                "task_id": "GPQA/2",
                "passed": False,
                "score": 0.0,
                "prompt": "question",
                "response": "",
                "error": "Pi agent exited without a response",
                "harness_error": True,
            },
        ],
    }

    output = save([result])
    markdown = (output / "results.md").read_text()
    report_json = json.loads((output / "results.json").read_text())
    with open(output / "results.csv", newline="") as report_file:
        report_csv = next(csv.DictReader(report_file))
    report_html = (output / "results.html").read_text()

    assert "Harness score Δ" in markdown
    assert "Fail | Loop killed | Timeout | Length exceeded | Harness error" in markdown
    assert "| 1 | 2 | 3 | 0 | 0 | 0 | 1 | 1 | 0/4 |" in markdown
    assert "native tools available: read, bash, edit, write" in markdown
    assert "abc123` · 5 tokens · 23 chars" in markdown
    assert "Exact stock Pi scaffold" in markdown
    assert "16384 tokens via `max_completion_tokens`" in markdown
    assert "LENGTH EXCEEDED" in markdown
    assert "HARNESS ERROR" in markdown
    assert report_json[0]["pi_system_prompt_tokens"] == 5
    assert report_csv["pi_system_prompt_tokens"] == "5"
    assert '"pi_system_prompt_tokens": 5' in report_html
    assert "Pi scaffold · ${number(row.pi_system_prompt_tokens)} tokens" in report_html
