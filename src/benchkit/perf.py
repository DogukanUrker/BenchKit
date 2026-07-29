"""Controlled model-speed profiling for llama.cpp-compatible servers."""

from __future__ import annotations

import os
import platform
import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime

from benchkit.client import InferenceClient

DEFAULT_DEPTHS = (0, 4096, 16384, 32768, 65536)
DEPTH_TOLERANCE = 0.01
DRIFT_TOLERANCE = 0.10
MIN_RELIABLE_PP_TOKENS = 256
FILLER = (
    "A local inference server processes this deterministic BenchKit workload. "
    "The passage is intentionally ordinary so tokenization remains stable "
    "across repetitions while context depth changes. "
)


@dataclass(frozen=True)
class PerfConfig:
    """Inputs that make a performance profile reproducible."""

    model: str
    depths: tuple[int, ...] = DEFAULT_DEPTHS
    gen_tokens: int = 128
    reps: int = 5
    warmups: int = 1
    config_note: str = ""


def parse_depths(value: str) -> tuple[int, ...]:
    """Parse comma-separated token depths such as ``minimal,4k,16k``."""
    depths: list[int] = []
    for raw in value.split(","):
        item = raw.strip().lower()
        if item in {"minimal", "min", "0"}:
            depth = 0
        else:
            multiplier = 1
            if item.endswith("k"):
                multiplier, item = 1024, item[:-1]
            elif item.endswith("m"):
                multiplier, item = 1024 * 1024, item[:-1]
            try:
                depth = int(float(item) * multiplier)
            except ValueError as exc:
                raise ValueError(f"Invalid depth: {raw.strip()!r}") from exc
            if depth < 1:
                raise ValueError(f"Invalid depth: {raw.strip()!r}")
        if depth not in depths:
            depths.append(depth)
    if not depths:
        raise ValueError("At least one context depth is required")
    return tuple(depths)


def depth_label(depth: int) -> str:
    if depth == 0:
        return "minimal"
    if depth % 1024 == 0:
        return f"{depth // 1024}k"
    return f"{depth:,}"


def _filler_text(length: int) -> str:
    repetitions = max(1, length // len(FILLER) + 1)
    return (FILLER * repetitions)[:length]


def _prompt_for(
    client: InferenceClient,
    model: str,
    target_depth: int,
    announce: Callable[[str], None],
) -> tuple[str | list[int], str, list[dict]]:
    if target_depth == 0:
        return (
            "BenchKit speed profile. Continue with concise factual prose:",
            "minimal",
            [],
        )

    content = _filler_text(target_depth * 8)
    tokens = client.tokenize(model, content)
    if tokens is not None:
        while len(tokens) < target_depth:
            content += _filler_text(target_depth * 2)
            tokens = client.tokenize(model, content)
            if tokens is None:
                break
        if tokens is not None and len(tokens) >= target_depth:
            return tokens[:target_depth], "tokenizer", []

    # llama-swap commonly proxies the OpenAI routes but not llama.cpp's native
    # tokenizer. Calibrate text length against the completion response instead
    # of silently trusting a characters-per-token estimate.
    prompt = _filler_text(target_depth * 5)
    calibration: list[dict] = []
    for attempt in range(3):
        announce(
            f"{depth_label(target_depth)} · calibrating input "
            f"{attempt + 1}/3"
        )
        try:
            measured = client.profile_completion(model, prompt, 1)
        except Exception as exc:
            calibration.append(
                {
                    "attempt": attempt + 1,
                    "characters": len(prompt),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            break

        actual = int(measured.get("input_tokens") or 0)
        calibration.append(
            {
                "attempt": attempt + 1,
                "characters": len(prompt),
                "actual_input_tokens": actual,
            }
        )
        if not actual:
            break
        error_ratio = abs(actual - target_depth) / target_depth
        if error_ratio <= DEPTH_TOLERANCE:
            return prompt, "calibrated", calibration
        prompt = _filler_text(max(1, round(len(prompt) * target_depth / actual)))

    calibrated = any(item.get("actual_input_tokens") for item in calibration)
    return prompt, "calibrated" if calibrated else "estimated", calibration


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "median": round(statistics.median(values), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }


def _summarize_case(
    requested_depth: int,
    depth_method: str,
    calibration: list[dict],
    runs: list[dict],
    errors: list[str],
) -> dict:
    successful = [run for run in runs if not run.get("error")]
    actual_depths = [run["input_tokens"] for run in successful if run["input_tokens"]]
    actual_depth = int(statistics.median(actual_depths)) if actual_depths else 0
    depth_deviation = (
        (actual_depth - requested_depth) / requested_depth
        if requested_depth and actual_depth
        else 0.0
    )
    return {
        "depth": requested_depth,
        "depth_label": depth_label(requested_depth),
        "actual_input_tokens": actual_depth,
        "depth_method": depth_method,
        "exact_depth": depth_method == "tokenizer",
        "depth_deviation": round(depth_deviation, 5),
        "depth_within_tolerance": (
            requested_depth == 0
            or (
                actual_depth > 0
                and abs(depth_deviation) <= DEPTH_TOLERANCE
            )
        ),
        "calibration": calibration,
        "pp_reliable": actual_depth >= MIN_RELIABLE_PP_TOKENS,
        "successful_reps": len(successful),
        "errors": errors,
        "pp_tps": _summary([run["pp_tps"] for run in successful]),
        "tg_tps": _summary([run["tg_tps"] for run in successful]),
        "ttft_s": _summary([run["ttft_s"] for run in successful]),
        "wall_time_s": _summary([run["wall_time_s"] for run in successful]),
        "wall_tg_tps": _summary([run["wall_tg_tps"] for run in successful]),
        "overhead_available": any(
            run.get("server_time_s", 0) > 0 for run in successful
        ),
        "overhead_s": _summary([run["overhead_s"] for run in successful]),
        "output_tokens": _summary(
            [float(run["output_tokens"]) for run in successful]
        ),
        "runs": runs,
    }


def run_profile(
    client: InferenceClient,
    config: PerfConfig,
    *,
    model_meta: dict | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Run a serial context-depth sweep and return a complete raw profile."""

    def announce(message: str) -> None:
        if progress is not None:
            progress(message)

    cases: list[dict] = []
    startup: dict | None = None
    first_prompt: str | list[int] | None = None

    # Run this before tokenization because llama-swap may need to load the model
    # even for its native tokenizer endpoint.
    if config.warmups:
        announce("First request · startup probe")
        try:
            measured = client.profile_completion(
                config.model,
                "BenchKit startup probe. Continue with concise factual prose:",
                min(config.gen_tokens, 32),
            )
            startup = {
                "ttft_s": round(measured["ttft_s"], 3),
                "wall_time_s": round(measured["wall_time_s"], 3),
                "label": "may include llama-swap model loading",
            }
        except Exception as exc:
            startup = {
                "error": f"{type(exc).__name__}: {exc}",
                "label": "first request",
            }

    for depth in config.depths:
        label = depth_label(depth)
        announce(f"Preparing {label} context")
        prompt, depth_method, calibration = _prompt_for(
            client, config.model, depth, announce
        )
        if first_prompt is None:
            first_prompt = prompt

        for warmup in range(config.warmups):
            announce(f"{label} · warmup {warmup + 1}/{config.warmups}")
            try:
                client.profile_completion(
                    config.model, prompt, min(config.gen_tokens, 32)
                )
            except Exception:
                pass

        runs: list[dict] = []
        errors: list[str] = []
        for repetition in range(config.reps):
            announce(f"{label} · run {repetition + 1}/{config.reps}")
            try:
                measured = client.profile_completion(
                    config.model, prompt, config.gen_tokens
                )
                runs.append(
                    {
                        "repetition": repetition + 1,
                        **{
                            key: round(value, 4)
                            if isinstance(value, float)
                            else value
                            for key, value in measured.items()
                            if key != "response"
                        },
                    }
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                errors.append(error)
                runs.append({"repetition": repetition + 1, "error": error})

        cases.append(
            _summarize_case(
                depth,
                depth_method,
                calibration,
                runs,
                errors,
            )
        )

    drift_check: dict | None = None
    if first_prompt is not None and cases:
        announce(f"{cases[0]['depth_label']} · end-of-sweep drift check")
        try:
            measured = client.profile_completion(
                config.model, first_prompt, config.gen_tokens
            )
            initial_tg = float(cases[0]["tg_tps"]["median"])
            final_tg = float(measured["tg_tps"])
            change = (final_tg - initial_tg) / initial_tg if initial_tg else 0.0
            drift_check = {
                "depth": cases[0]["depth"],
                "depth_label": cases[0]["depth_label"],
                "initial_tg_tps": round(initial_tg, 3),
                "final_tg_tps": round(final_tg, 3),
                "change": round(change, 5),
                "stable": bool(initial_tg)
                and abs(change) <= DRIFT_TOLERANCE,
                "tolerance": DRIFT_TOLERANCE,
            }
        except Exception as exc:
            drift_check = {
                "depth": cases[0]["depth"],
                "depth_label": cases[0]["depth_label"],
                "error": f"{type(exc).__name__}: {exc}",
                "stable": False,
                "tolerance": DRIFT_TOLERANCE,
            }

    return {
        "kind": "benchkit-performance-profile",
        "version": 1,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": config.model,
        "provider": client.label,
        "host": client.host,
        "environment": {
            "hardware": os.environ.get("BENCHKIT_HARDWARE", ""),
            "system": platform.system(),
            "machine": platform.machine(),
            "config_note": config.config_note
            or os.environ.get("BENCHKIT_PERF_CONFIG", ""),
        },
        "model_metadata": model_meta or {},
        "settings": asdict(config),
        "startup": startup,
        "drift_check": drift_check,
        "cases": cases,
    }
