#!/usr/bin/env bash
set -euo pipefail
seed="${1:?usage: setup.sh SEED [WORKSPACE]}"
workspace="${2:-/workspace/backport-release-stack}"
slug="$(printf '%s' "$seed:backport" | sha256sum | cut -c1-12)"
mkdir -p "$workspace"
cd "$workspace"
git init -q -b release-2.x
git config user.name "BenchKit Generator"
git config user.email "generator@benchkit.invalid"
git config commit.gpgSign false
git config core.autocrlf false
export GIT_AUTHOR_NAME="BenchKit Generator" GIT_AUTHOR_EMAIL="generator@benchkit.invalid"
export GIT_COMMITTER_NAME="$GIT_AUTHOR_NAME" GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"
export GIT_CONFIG_GLOBAL=/dev/null
clock=1700500000
commit() {
    clock=$((clock + 60))
    git add -A
    GIT_AUTHOR_DATE="@$clock +0000" GIT_COMMITTER_DATE="@$clock +0000" git commit -q "$@"
}
replace() {
    python3 - "$@" <<'PY'
import sys
from pathlib import Path

path, old, new = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
text = path.read_text()
if old not in text:
    raise SystemExit(f"setup: {old!r} not found in {path}")
path.write_text(text.replace(old, new, 1))
PY
}

mkdir -p ledger docs
printf '__pycache__/\n' > .gitignore
: > ledger/__init__.py
cat > ledger/money.py <<'PY'
"""Amount parsing and formatting in integer cents."""


def parse_amount(text: str) -> int:
    value = text
    if not value.isdigit():
        raise ValueError(f"bad amount: {text!r}")
    return int(value)


def format_amount(cents: int) -> str:
    return f"{cents // 100}.{cents % 100:02d}"
PY
cat > ledger/discount.py <<'PY'
"""Discounts applied to order totals."""


def apply_discount(total: int, discount: int) -> int:
    return total - discount
PY
cat > ledger/config.py <<'PY'
"""Runtime configuration."""

DEFAULTS = {
    "timeout": 30,
    "retries": 3,
}

VERSION = "1.0.0"
PY
cat > test_core.py <<'PY'
import unittest

from ledger.config import DEFAULTS
from ledger.discount import apply_discount
from ledger.money import format_amount, parse_amount


class CoreTests(unittest.TestCase):
    def test_parse_amount(self):
        self.assertEqual(parse_amount("1250"), 1250)

    def test_parse_amount_rejects_text(self):
        with self.assertRaises(ValueError):
            parse_amount("ten")

    def test_format_amount(self):
        self.assertEqual(format_amount(1205), "12.05")

    def test_apply_discount(self):
        self.assertEqual(apply_discount(500, 120), 380)

    def test_retries(self):
        self.assertEqual(DEFAULTS["retries"], 3)
PY
cat > CHANGELOG.md <<EOF
# Ledger changelog ($slug)

EOF
printf '# Ledger %s\n\nRun tests with `python3 -m unittest discover -v`.\n' "$slug" > README.md
commit -m "bootstrap ledger $slug"
fork="$(git rev-parse HEAD)"

noise() {
    local index="$1"
    case $((index % 4)) in
        0)
            printf -- '- %s: maintenance entry %02d\n' "$slug" "$index" >> CHANGELOG.md
            commit -m "chore: update changelog ($index)"
            ;;
        1)
            if [[ ! -f ledger/export.py ]]; then
                cat > ledger/export.py <<'PY'
"""CSV export, new in 2.x."""

COLUMNS = ["id"]


def header() -> str:
    return ",".join(COLUMNS)
PY
                cat > test_export.py <<'PY'
import unittest

from ledger.export import COLUMNS, header


class ExportTests(unittest.TestCase):
    def test_header(self):
        self.assertEqual(header(), ",".join(COLUMNS))
PY
                commit -m "export: introduce CSV export"
            else
                replace ledger/export.py ']' ", \"col$index\"]"
                commit -m "export: add column col$index"
            fi
            ;;
        2)
            printf 'Release note %02d for %s.\n' "$index" "$slug" >> "docs/release-$((index % 3)).md"
            commit -m "docs: expand release notes ($index)"
            ;;
        3)
            printf 'step-%02d\n' "$index" >> docs/pipeline.txt
            commit -m "ci: adjust pipeline step $index"
            ;;
    esac
}

for index in $(seq 1 70); do
    case "$index" in
        6)
            python3 - <<'PY'
from pathlib import Path

path = Path("ledger/money.py")
text = path.read_text()
text = text.replace("    value = text\n    if not value.isdigit():", "    raw = text\n    if not raw.isdigit():")
text = text.replace("return int(value)", "return int(raw)")
path.write_text(text)
PY
            commit -m "refactor: rename parse_amount locals"
            ;;
        11)
            replace ledger/money.py 'def parse_amount(text: str) -> int:
    raw = text
' 'def parse_amount(text: str, allow_commas: bool = False) -> int:
    raw = text
    if allow_commas:
        raw = raw.replace(",", "")
'
            cat >> test_export.py <<'PY'


class SeparatorTests(unittest.TestCase):
    def test_thousands_separators(self):
        from ledger.money import parse_amount

        self.assertEqual(parse_amount("1,250", allow_commas=True), 1250)
PY
            commit -m "feat: accept thousands separators"
            ;;
        17)
            replace ledger/money.py '    raw = text
' '    raw = text.strip()
'
            cat > test_bk101.py <<'PY'
import unittest

from ledger.money import parse_amount


class WhitespaceTests(unittest.TestCase):
    def test_surrounding_whitespace_is_ignored(self):
        self.assertEqual(parse_amount("  42\n"), 42)
PY
            commit -m "Fix whitespace handling in parse_amount" -m "Fixes: BK-101"
            ;;
        19)
            printf -- '- Fix typo in the BK-104 changelog entry\n' >> CHANGELOG.md
            commit -m "Fix typo in BK-104 changelog entry"
            ;;
        22)
            printf '# Parsing\n\nSurrounding whitespace is ignored (BK-101).\n' > docs/parsing.md
            commit -m "BK-101: document whitespace parsing" -m "Refs: BK-101"
            ;;
        28)
            replace ledger/money.py '    raw = text.strip()
' '    raw = text.strip()
    sign = -1 if raw.startswith("-") else 1
    raw = raw.removeprefix("-")
'
            replace ledger/money.py 'return int(raw)' 'return sign * int(raw)'
            cat > test_bk104.py <<'PY'
import unittest

from ledger.money import parse_amount


class NegativeAmountTests(unittest.TestCase):
    def test_negative_amounts(self):
        self.assertEqual(parse_amount("-75"), -75)
PY
            commit -m "Accept negative amounts" -m "Fixes: BK-104"
            ;;
        34)
            replace ledger/money.py '    if not raw.isdigit():' '    if len(raw) > 18:
        raise ValueError(f"amount too long: {text!r}")
    if not raw.isdigit():'
            commit -m "Reject overlong amounts" -m "Fixes: BK-1040"
            ;;
        39)
            replace ledger/money.py '    return f"{cents // 100}.{cents % 100:02d}"' '    return f"{cents / 100:.2f}"'
            cat > test_bk107.py <<'PY'
import unittest

from ledger.money import format_amount


class NegativeFormatTests(unittest.TestCase):
    def test_negative_cents(self):
        self.assertEqual(format_amount(-5), "-0.05")
PY
            commit -m "Format negative amounts" -m "Fixes: BK-107"
            attempt="$(git rev-parse HEAD)"
            ;;
        43)
            git revert --no-edit --no-commit "$attempt" >/dev/null
            commit -m "Revert \"Format negative amounts\"" \
                -m "This reverts commit $attempt. Float formatting loses cents above 2**53." \
                -m "Refs: BK-107"
            ;;
        48)
            replace ledger/money.py '    return f"{cents // 100}.{cents % 100:02d}"' '    sign = "-" if cents < 0 else ""
    cents = abs(cents)
    return f"{sign}{cents // 100}.{cents % 100:02d}"'
            cat > test_bk107.py <<'PY'
import unittest

from ledger.money import format_amount


class NegativeFormatTests(unittest.TestCase):
    def test_negative_cents(self):
        self.assertEqual(format_amount(-5), "-0.05")

    def test_large_amounts_stay_exact(self):
        self.assertEqual(format_amount(123456789012345678), "1234567890123456.78")
PY
            commit -m "Format negative amounts without floats" -m "Fixes: BK-107"
            ;;
        54)
            replace ledger/discount.py 'return total - discount' 'return max(0, total - discount)'
            cat > test_bk112.py <<'PY'
import unittest

from ledger.discount import apply_discount


class ClampTests(unittest.TestCase):
    def test_discount_never_goes_negative(self):
        self.assertEqual(apply_discount(100, 250), 0)
PY
            commit -m "Clamp discounted totals at zero" -m "Fixes: BK-112"
            ;;
        57)
            replace ledger/config.py 'VERSION = "1.0.0"' 'VERSION = "2.0.0"
RELEASE_CHANNEL = "2.x"'
            commit -m "Bump version to 2.0.0"
            ;;
        60)
            replace ledger/config.py '"timeout": 30,' '"timeout": 10,'
            cat > test_bk115.py <<'PY'
import unittest

from ledger.config import DEFAULTS


class TimeoutTests(unittest.TestCase):
    def test_default_timeout(self):
        self.assertEqual(DEFAULTS["timeout"], 10)
PY
            commit -m "Lower default request timeout" -m "Fixes: BK-115"
            ;;
        65)
            printf -- '- Follow-up for BK-115 timeouts in docs\n' >> CHANGELOG.md
            commit -m "Lower default request timeout in docs" -m "Refs: BK-115"
            ;;
        *)
            noise "$index"
            ;;
    esac
done

git switch -q -c maint-1.x "$fork"
replace ledger/discount.py 'return total - discount' 'return max(0, total - discount)'
cat > test_bk112.py <<'PY'
import unittest

from ledger.discount import apply_discount


class ClampTests(unittest.TestCase):
    def test_discount_never_goes_negative(self):
        self.assertEqual(apply_discount(100, 250), 0)
PY
commit -m "Clamp discounted totals at zero (BK-112)"
replace ledger/config.py 'VERSION = "1.0.0"' 'VERSION = "1.0.3"'
printf -- '- 1.0.3 maintenance release\n' >> CHANGELOG.md
commit -m "Release 1.0.3"
git status --porcelain | grep -q . && exit 1 || true
