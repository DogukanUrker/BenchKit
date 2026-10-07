#!/usr/bin/env bash
# Reference hand solution, run from the task workspace.
set -euo pipefail
fix() { git log release-2.x --format=%H --grep="^Fixes: $1\$" -1; }
git cherry-pick -x "$(fix BK-101)" || true
cat > ledger/money.py <<'PY'
"""Amount parsing and formatting in integer cents."""


def parse_amount(text: str) -> int:
    value = text.strip()
    if not value.isdigit():
        raise ValueError(f"bad amount: {text!r}")
    return int(value)


def format_amount(cents: int) -> str:
    return f"{cents // 100}.{cents % 100:02d}"
PY
git add ledger/money.py && git cherry-pick --continue
git cherry-pick -x "$(fix BK-104)" || true
cat > ledger/money.py <<'PY'
"""Amount parsing and formatting in integer cents."""


def parse_amount(text: str) -> int:
    value = text.strip()
    sign = -1 if value.startswith("-") else 1
    value = value.removeprefix("-")
    if not value.isdigit():
        raise ValueError(f"bad amount: {text!r}")
    return sign * int(value)


def format_amount(cents: int) -> str:
    return f"{cents // 100}.{cents % 100:02d}"
PY
git add ledger/money.py && git cherry-pick --continue
git cherry-pick -x "$(fix BK-107)"
git cherry-pick -x "$(fix BK-112)" || git cherry-pick --skip
git cherry-pick -x "$(fix BK-115)"
