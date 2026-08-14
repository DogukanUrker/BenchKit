"""Historical result archive and dashboard regressions."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from benchkit import cli
from benchkit.history import build_archive, create_history_server, render_history_html


def _write_report(root, run_id, name, payload) -> None:
    output = root / run_id
    output.mkdir(parents=True, exist_ok=True)
    (output / name).write_text(json.dumps(payload), encoding="utf-8")


def test_archive_normalizes_versions_deduplicates_and_preserves_unknowns(
    tmp_path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    legacy = [
        {
            "model": "gemma",
            "benchmark": "humaneval",
            "score": 50.0,
            "passed": 1,
            "total": 2,
            "tok_s": 10.0,
            "tasks": [{"task_id": "HumanEval/0", "passed": True}],
        }
    ]
    current = [
        {
            "model": "gemma",
            "benchmark": "humaneval",
            "score": 75.0,
            "passed": 3,
            "scored_total": 4,
            "total": 4,
            "harness": "direct",
            "harness_label": "Direct + repair",
            "trace_coverage": 100.0,
            "loops": 1,
            "suspected_loops": 2,
            "loop_kills": 1,
            "repair_attempted": 1,
            "repair_successes": 1,
            "tasks": [
                {
                    "task_id": "HumanEval/1",
                    "passed": False,
                    "score": 0,
                    "loop_state": "confirmed",
                    "loop_killed": True,
                }
            ],
        }
    ]
    _write_report(first, "2026-01-01_00-00-00", "results.json", legacy)
    _write_report(first, "2026-01-02_00-00-00", "results.json", current)
    _write_report(second, "duplicate", "results.json", current)
    bad = second / "broken"
    bad.mkdir(parents=True)
    (bad / "results.json").write_text("{broken", encoding="utf-8")

    archive = build_archive([first, second, tmp_path / "missing"])

    assert archive["counts"] == {
        "discovered_reports": 4,
        "unique_reports": 2,
        "benchmark_results": 2,
        "performance_points": 0,
        "models": 1,
    }
    legacy_row, current_row = archive["benchmark_rows"]
    assert legacy_row["schema_generation"] == "legacy"
    assert legacy_row["loops"] is None
    assert current_row["schema_generation"] == "current"
    assert current_row["loops"] == 1
    assert current_row["task_diagnostics"][0]["loop_killed"] is True
    assert len(current_row["sources"]) == 2
    assert {warning["kind"] for warning in archive["warnings"]} == {
        "invalid_report",
        "missing_directory",
    }


def test_archive_reads_perf_profiles(tmp_path) -> None:
    profile = {
        "model": "model-a",
        "version": "1",
        "cases": [
            {
                "depth_label": "4k",
                "actual_input_tokens": 4090,
                "successful_reps": 3,
                "pp_tps": {"median": 500.0},
                "tg_tps": {"median": 42.0},
                "ttft_s": {"median": 8.0},
                "wall_time_s": {"median": 12.0},
            }
        ],
    }
    _write_report(tmp_path, "2026-01-01_00-00-00", "perf.json", profile)

    archive = build_archive([tmp_path])

    assert archive["counts"]["performance_points"] == 1
    assert archive["performance_rows"][0]["depth"] == 4090
    assert archive["performance_rows"][0]["tg_tps"] == 42


def test_history_html_is_self_contained_and_escapes_script_breakout() -> None:
    archive = {
        "roots": ["</script><script>alert(1)</script>"],
        "counts": {},
        "benchmark_rows": [],
        "performance_rows": [],
        "warnings": [],
    }

    html = render_history_html(archive)

    assert "BenchKit History" in html
    assert "__BENCHKIT_HISTORY_DATA__" not in html
    assert "</script><script>alert(1)</script>" not in html
    assert "\\u003c/script\\u003e" in html


def test_history_server_serves_dashboard_and_rejects_other_paths(tmp_path) -> None:
    _write_report(
        tmp_path,
        "2026-01-01_00-00-00",
        "results.json",
        [{"model": "m", "benchmark": "b", "score": 100, "total": 1}],
    )
    server = create_history_server([tmp_path])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/"
        with urllib.request.urlopen(url) as response:
            body = response.read().decode()
        assert response.status == 200
        assert '"model": "m"' in body
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(f"{url}missing")
        assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_history_cli_forwards_repeatable_directories(monkeypatch) -> None:
    captured = {}

    def fake_serve(results_dirs, *, port, open_browser):
        captured.update(
            results_dirs=list(results_dirs), port=port, open_browser=open_browser
        )

    monkeypatch.setattr(cli, "serve_history", fake_serve)

    cli.main(
        [
            "history",
            "--results-dir",
            "one",
            "--results-dir",
            "two",
            "--port",
            "8765",
            "--no-open",
        ]
    )

    assert captured == {
        "results_dirs": ["one", "two"],
        "port": 8765,
        "open_browser": False,
    }
