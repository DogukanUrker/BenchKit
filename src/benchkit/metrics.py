"""Throughput metrics shared by execution, front-ends and reports."""

from __future__ import annotations

from collections.abc import Mapping


def throughput_metrics(
    *,
    output_tokens: int,
    generation_time_s: float,
    request_time_s: float,
    wall_time_s: float,
) -> dict[str, float]:
    """Calculate stream speed, total throughput and occupied request slots.

    ``generation_time_s`` is the sum of server-reported decode times, while
    ``request_time_s`` is the sum of complete per-task request durations.
    Their denominators intentionally differ: stream speed describes decoder
    performance and effective concurrency describes client/server saturation.
    """
    return {
        "tok_s_per_stream": (
            output_tokens / generation_time_s if generation_time_s > 0 else 0.0
        ),
        "tok_s_aggregate": (output_tokens / wall_time_s if wall_time_s > 0 else 0.0),
        "concurrency_eff": (request_time_s / wall_time_s if wall_time_s > 0 else 0.0),
    }


def _number(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def stream_tok_s(result: Mapping[str, object]) -> float:
    """Read stream throughput from new or legacy result data."""
    if "tok_s_per_stream" in result:
        return _number(result["tok_s_per_stream"])
    return _number(result.get("tok_s"))


def effective_concurrency(result: Mapping[str, object]) -> float:
    """Read or derive occupied request slots from a result."""
    if "concurrency_eff" in result:
        return _number(result["concurrency_eff"])
    wall_time = _number(result.get("total_time"))
    if wall_time <= 0:
        return 0.0
    derived = (
        _number(result.get("avg_response_time"))
        * _number(result.get("total"))
        / wall_time
    )
    configured = _number(result.get("concurrency"))
    return min(configured, derived) if configured > 0 else derived


def aggregate_tok_s(result: Mapping[str, object]) -> float:
    """Read or derive wall-clock output throughput from a result."""
    if "tok_s_aggregate" in result:
        return _number(result["tok_s_aggregate"])

    wall_time = _number(result.get("total_time"))
    tasks = result.get("tasks")
    if wall_time > 0 and isinstance(tasks, list):
        token_values = [
            _number(task.get("output_tokens"))
            for task in tasks
            if isinstance(task, Mapping) and "output_tokens" in task
        ]
        if token_values:
            return sum(token_values) / wall_time

    # Legacy CSV summaries do not contain tasks. Stream throughput multiplied
    # by measured slot occupancy is the closest derivation available.
    return stream_tok_s(result) * effective_concurrency(result)
