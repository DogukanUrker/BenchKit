"""Small formatting helpers shared across TUI screens."""

from __future__ import annotations


def fmt_duration(seconds: float | None) -> str:
    """Compact duration: ``1h 04m``, ``4m 12s``, ``42s`` or ``0.8s``."""
    if seconds is None:
        return "--"
    if seconds < 0:
        return "--"
    if seconds < 10:
        return f"{seconds:.1f}s"

    total = int(round(seconds))
    if total < 60:
        return f"{total}s"
    if total < 3600:
        return f"{total // 60}m {total % 60:02d}s"
    return f"{total // 3600}h {(total % 3600) // 60:02d}m"


def fmt_size(size: object) -> str:
    """Render a model size in GB when the server reports one."""
    if isinstance(size, (int, float)) and size > 0:
        return f"{size / (1024**3):.1f} GB"
    return ""


def fmt_count(value: int) -> str:
    return f"{value:,}"


def score_class(score: float) -> str:
    """CSS class name matching a score band."""
    if score >= 80:
        return "score-high"
    if score >= 50:
        return "score-mid"
    return "score-low"


def score_color(score: float) -> str:
    if score >= 80:
        return "#3fb950"
    if score >= 50:
        return "#d29922"
    return "#f85149"


def bar(fraction: float, width: int = 12) -> str:
    """A tiny inline bar for table cells."""
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(fraction * width))
    return "█" * filled + "░" * (width - filled)
