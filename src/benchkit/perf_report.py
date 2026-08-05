"""Write portable performance-profile reports."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from importlib.resources import files
from pathlib import Path


def _safe_json(data: object) -> str:
    return (
        json.dumps(data, ensure_ascii=False)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def render_perf_html(profile: dict) -> str:
    template = (
        files("benchkit").joinpath("templates/perf.html").read_text(encoding="utf-8")
    )
    return template.replace("__BENCHKIT_PERF_DATA__", _safe_json(profile))


def _number(case: dict, metric: str, field: str = "median") -> float:
    return float((case.get(metric) or {}).get(field) or 0.0)


def _markdown(profile: dict) -> str:
    environment = profile.get("environment") or {}
    settings = profile.get("settings") or {}
    note = environment.get("config_note") or environment.get("hardware") or ""
    lines = [
        "# BenchKit Performance Profile",
        "",
        f"**Model:** `{profile['model']}`",
        f"**Provider:** {profile.get('provider') or 'unknown'}",
        f"**Host:** `{profile.get('host') or ''}`",
        f"**Generated:** {profile.get('generated_at') or ''}",
        (
            f"**Settings:** {settings.get('gen_tokens', 0)} output tokens · "
            f"{settings.get('reps', 0)} repetitions · "
            f"{settings.get('warmups', 0)} warmup(s)"
        ),
    ]
    if note:
        lines.append(f"**Configuration:** {note}")
    startup = profile.get("startup") or {}
    if startup.get("ttft_s") is not None:
        lines.append(
            f"**First request TTFT:** {startup['ttft_s']:.3f}s "
            f"({startup.get('label', 'startup probe')})"
        )
    drift = profile.get("drift_check") or {}
    if drift and not drift.get("error"):
        warning = " ⚠ possible warmup/drift anomaly" if not drift["stable"] else ""
        lines.append(
            f"**Drift check:** {drift['depth_label']} TG "
            f"{drift['initial_tg_tps']:.1f} → {drift['final_tg_tps']:.1f} tok/s "
            f"({drift['change']:+.1%}){warning}"
        )
    sweep = profile.get("sweep_summary") or {}
    lines.append(
        f"**Sweep median:** "
        f"TG {float(sweep.get('tg_tps_median') or 0):.1f} tok/s · "
        f"PP {float(sweep.get('pp_tps_median') or 0):.1f} tok/s"
    )

    lines.extend(
        [
            "",
            "| Depth | Actual input | PP tok/s | TG tok/s | TTFT | Wall | Overhead | Output |",
            "|------:|-------------:|---------:|---------:|-----:|-----:|---------:|-------:|",
        ]
    )
    for case in profile.get("cases", []):
        actual = case.get("actual_input_tokens") or "—"
        marker = "!" if not case.get("depth_within_tolerance", True) else ""
        pp = f"{_number(case, 'pp_tps'):.1f}" if case.get("pp_reliable") else "n/a"
        overhead = (
            f"{_number(case, 'overhead_s'):.3f}s"
            if case.get("overhead_available")
            else "n/a"
        )
        lines.append(
            f"| {case['depth_label']}{marker} | {actual} "
            f"| {pp} "
            f"| {_number(case, 'tg_tps'):.1f} "
            f"| {_number(case, 'ttft_s'):.3f}s "
            f"| {_number(case, 'wall_time_s'):.2f}s "
            f"| {overhead} "
            f"| {_number(case, 'output_tokens'):.0f} |"
        )

    calibrated = any(
        case.get("depth_method") == "calibrated" for case in profile.get("cases", [])
    )
    inaccurate = [
        case["depth_label"]
        for case in profile.get("cases", [])
        if not case.get("depth_within_tolerance", True)
    ]
    estimated = [
        case["depth_label"]
        for case in profile.get("cases", [])
        if case.get("depth_method") == "estimated"
    ]
    lines.extend(
        [
            "",
            "> Values are medians. Min/max and every raw repetition are available "
            "in `perf.json`.",
        ]
    )
    if calibrated:
        lines.extend(
            [
                "",
                "Depth method: calibrated. The server does not expose its "
                "native tokenizer route, so BenchKit calibrated deterministic "
                "text against the reported input-token count.",
            ]
        )
    if inaccurate:
        lines.extend(
            [
                "",
                f"⚠ Depth calibration remained outside ±1% for: "
                f"{', '.join(inaccurate)}. Compare using Actual input.",
            ]
        )
    if estimated:
        lines.extend(
            [
                "",
                f"⚠ Token-depth calibration was unavailable for: "
                f"{', '.join(estimated)}. Compare using Actual input.",
            ]
        )
    lines.extend(
        [
            "",
            "PP is server-reported prompt processing throughput; TG is "
            "server-reported token generation throughput. PP is shown as n/a "
            "for tiny prompts where fixed overhead dominates the value.",
            "",
        ]
    )
    return "\n".join(lines)


def save_profile(profile: dict) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output = Path("results") / timestamp
    output.mkdir(parents=True, exist_ok=True)

    with open(output / "perf.json", "w", encoding="utf-8") as handle:
        json.dump(profile, handle, indent=2, ensure_ascii=False)

    rows = []
    for case in profile.get("cases", []):
        rows.append(
            {
                "model": profile["model"],
                "depth": case["depth_label"],
                "actual_input_tokens": case.get("actual_input_tokens", 0),
                "depth_method": case.get("depth_method", ""),
                "depth_deviation": case.get("depth_deviation", 0),
                "depth_within_tolerance": case.get("depth_within_tolerance", False),
                "pp_reliable": case.get("pp_reliable", False),
                "overhead_available": case.get("overhead_available", False),
                "pp_tps_median": _number(case, "pp_tps"),
                "pp_tps_min": _number(case, "pp_tps", "min"),
                "pp_tps_max": _number(case, "pp_tps", "max"),
                "tg_tps_median": _number(case, "tg_tps"),
                "tg_tps_min": _number(case, "tg_tps", "min"),
                "tg_tps_max": _number(case, "tg_tps", "max"),
                "ttft_s_median": _number(case, "ttft_s"),
                "ttft_s_min": _number(case, "ttft_s", "min"),
                "ttft_s_max": _number(case, "ttft_s", "max"),
                "wall_time_s_median": _number(case, "wall_time_s"),
                "overhead_s_median": _number(case, "overhead_s"),
                "overhead_s_min": _number(case, "overhead_s", "min"),
                "overhead_s_max": _number(case, "overhead_s", "max"),
                "output_tokens_median": _number(case, "output_tokens"),
                "successful_reps": case.get("successful_reps", 0),
            }
        )
    if rows:
        with open(output / "perf.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    with open(output / "perf.md", "w", encoding="utf-8") as handle:
        handle.write(_markdown(profile))

    with open(output / "perf.html", "w", encoding="utf-8") as handle:
        handle.write(render_perf_html(profile))

    return output
