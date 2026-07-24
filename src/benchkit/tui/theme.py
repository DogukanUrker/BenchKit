"""Dogi themes for BenchKit.

Ported from the Dogi editor theme by Doğukan Ürker
(https://github.com/DogukanUrker/DogiZed). Dogi is a flat, high-contrast
palette: pure black or pure white everywhere, no tinted panels or borders, and
colour reserved for state. The TUI gets its separation from translucent
overlays instead of solid panel colours, which is what `bk-raise` and
`bk-border` carry.
"""

from __future__ import annotations

from textual.theme import Theme

# Dogi Dark
DARK_BACKGROUND = "#000000"
DARK_FOREGROUND = "#FFFFFF"
DARK_BLUE = "#60A5FA"
DARK_PINK = "#F472B6"
DARK_CYAN = "#22D3EE"
DARK_GREEN = "#34D399"
DARK_YELLOW = "#FACC15"
DARK_RED = "#EF4444"
DARK_MUTED = "#6B7280"
DARK_PLACEHOLDER = "#4A4A4A"

# Dogi Light
LIGHT_BACKGROUND = "#FFFFFF"
LIGHT_FOREGROUND = "#0B0B0B"
LIGHT_BLUE = "#2563EB"
LIGHT_PINK = "#DB2777"
LIGHT_CYAN = "#0891B2"
LIGHT_GREEN = "#059669"
LIGHT_YELLOW = "#D97706"
LIGHT_RED = "#DC2626"
LIGHT_MUTED = "#6B7280"
LIGHT_PLACEHOLDER = "#9CA3AF"

DOGI_DARK = Theme(
    name="dogi",
    primary=DARK_BLUE,
    secondary=DARK_PINK,
    accent=DARK_CYAN,
    success=DARK_GREEN,
    warning=DARK_YELLOW,
    error=DARK_RED,
    foreground=DARK_FOREGROUND,
    background=DARK_BACKGROUND,
    surface=DARK_BACKGROUND,
    panel=DARK_BACKGROUND,
    # Solid stand-ins for Dogi's translucent white overlay (#FFFFFF08).
    boost="#141414",
    dark=True,
    variables={
        # Dogi draws no hard borders; these translucent whites stand in.
        "bk-border": "#FFFFFF 14%",
        "bk-raise": "#FFFFFF 6%",
        "bk-selection": "#FFFFFF 20%",
        "text-muted": DARK_MUTED,
        "text-disabled": DARK_PLACEHOLDER,
        "input-cursor-background": DARK_BLUE,
        "input-cursor-foreground": DARK_BACKGROUND,
        "input-selection-background": f"{DARK_BLUE} 35%",
        "block-cursor-background": DARK_BLUE,
        "block-cursor-foreground": DARK_BACKGROUND,
        "block-cursor-text-style": "bold",
        "footer-key-foreground": DARK_BLUE,
        "footer-description-foreground": DARK_FOREGROUND,
        "scrollbar": "#FFFFFF 12%",
        "scrollbar-hover": "#FFFFFF 20%",
        "scrollbar-active": DARK_BLUE,
        "scrollbar-background": DARK_BACKGROUND,
        "link-color-hover": DARK_BLUE,
        "markdown-table-border": "#FFFFFF 14%",
    },
)

DOGI_LIGHT = Theme(
    name="dogi-light",
    primary=LIGHT_BLUE,
    secondary=LIGHT_PINK,
    accent=LIGHT_CYAN,
    success=LIGHT_GREEN,
    warning=LIGHT_YELLOW,
    error=LIGHT_RED,
    foreground=LIGHT_FOREGROUND,
    background=LIGHT_BACKGROUND,
    surface=LIGHT_BACKGROUND,
    panel=LIGHT_BACKGROUND,
    # Dogi's #00000010 hover overlay, flattened.
    boost="#F0F0F0",
    dark=False,
    variables={
        "bk-border": "#000000 14%",
        "bk-raise": "#000000 5%",
        "bk-selection": "#000000 18%",
        "text-muted": LIGHT_MUTED,
        "text-disabled": LIGHT_PLACEHOLDER,
        "input-cursor-background": LIGHT_BLUE,
        "input-cursor-foreground": LIGHT_BACKGROUND,
        "input-selection-background": f"{LIGHT_BLUE} 25%",
        "block-cursor-background": LIGHT_BLUE,
        "block-cursor-foreground": LIGHT_BACKGROUND,
        "block-cursor-text-style": "bold",
        "footer-key-foreground": LIGHT_BLUE,
        "footer-description-foreground": LIGHT_FOREGROUND,
        "scrollbar": "#000000 15%",
        "scrollbar-hover": "#000000 25%",
        "scrollbar-active": LIGHT_BLUE,
        "scrollbar-background": LIGHT_BACKGROUND,
        "link-color-hover": LIGHT_BLUE,
        "markdown-table-border": "#000000 14%",
    },
)

THEMES = [DOGI_DARK, DOGI_LIGHT]

DEFAULT_THEME = DOGI_DARK.name
ALTERNATE_THEME = DOGI_LIGHT.name

# Fallbacks so the stylesheet still parses under Textual's built-in themes,
# which know nothing about the Dogi variables. Variable values are substituted
# literally, so these have to be real colours rather than other variables.
VARIABLE_DEFAULTS = {
    "bk-border": "#808080 40%",
    "bk-raise": "#808080 15%",
    "bk-selection": "#808080 30%",
}


def score_palette(dark: bool) -> dict[str, str]:
    """Pass/warn/fail colours that stay readable on the active background."""
    if dark:
        return {"high": DARK_GREEN, "mid": DARK_YELLOW, "low": DARK_RED}
    return {"high": LIGHT_GREEN, "mid": LIGHT_YELLOW, "low": LIGHT_RED}
