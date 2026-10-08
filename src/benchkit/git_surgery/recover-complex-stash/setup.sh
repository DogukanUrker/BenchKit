#!/usr/bin/env bash
set -euo pipefail
seed="${1:?usage: setup.sh SEED [WORKSPACE]}"
workspace="${2:-/workspace/recover-complex-stash}"
slug="$(printf '%s' "$seed:stash" | sha256sum | cut -c1-12)"
mkdir -p "$workspace"
cd "$workspace"
git init -q -b main
git config user.name "BenchKit Generator"
git config user.email "generator@benchkit.invalid"
git config commit.gpgSign false
git config core.autocrlf false
export GIT_AUTHOR_NAME="BenchKit Generator" GIT_AUTHOR_EMAIL="generator@benchkit.invalid"
export GIT_COMMITTER_NAME="$GIT_AUTHOR_NAME" GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"
export GIT_CONFIG_GLOBAL=/dev/null
clock=1700700000
tick() {
    clock=$((clock + 60))
    export GIT_AUTHOR_DATE="@$clock +0000" GIT_COMMITTER_DATE="@$clock +0000"
}
commit() {
    tick
    git add -A
    git commit -q "$@"
}
stash() {
    tick
    git stash push -q --include-untracked -m "$1"
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
fixture() {
    mkdir -p fixtures
    printf '{\n  "currency": "EUR",\n  "rate_bp": %s,\n  "source": "%s"\n}\n' "$1" "$2" > fixtures/eur.json
}

printf '__pycache__/\n' > .gitignore
cat > legacy_rates.py <<'PY'
"""Exchange rates in basis points of one USD."""

RATES = {
    "USD": 10000,
    "EUR": 9200,
}


def convert(amount: int, currency: str) -> int:
    return amount * RATES[currency] // 10000
PY
cat > settings.py <<'PY'
"""Quote settings."""

DEFAULT_CURRENCY = "USD"
PRECISION = 2
PY
cat > app.py <<'PY'
from legacy_rates import convert
from settings import DEFAULT_CURRENCY, PRECISION


def quote(amount: int) -> str:
    value = convert(amount, DEFAULT_CURRENCY)
    return f"{value:.{PRECISION}f} {DEFAULT_CURRENCY}"
PY
cat > test_app.py <<'PY'
import unittest

from app import quote
from settings import DEFAULT_CURRENCY, PRECISION


class QuoteTests(unittest.TestCase):
    def test_quote_follows_settings(self):
        self.assertTrue(quote(100).endswith(f" {DEFAULT_CURRENCY}"))
        self.assertEqual(len(quote(100).split()[0].split(".")[1]), PRECISION)
PY
printf '# Quote service %s\n\nRun tests with `python3 -m unittest discover -v`.\n' "$slug" > README.md
commit -m "bootstrap quote service $slug"
printf '\nQuotes are rendered in the configured default currency.\n' >> README.md
commit -m "Document quote currency"

# First attempt: the rename is staged, but the fixture is for the wrong currency.
git mv legacy_rates.py rates.py
replace app.py 'from legacy_rates import convert' 'from rates import convert'
git add app.py
mkdir -p fixtures
printf '{\n  "currency": "GBP",\n  "rate_bp": 7900\n}\n' > fixtures/gbp.json
stash "rates migration (first try)"

# The wanted stash: staged rename and import, unstaged edits, untracked files.
git mv legacy_rates.py rates.py
replace app.py 'from legacy_rates import convert' 'from rates import convert'
git add app.py
replace rates.py '    "EUR": 9200,
' '    "EUR": 9200,
    "GBP": 7900,
'
replace settings.py 'DEFAULT_CURRENCY = "USD"' 'DEFAULT_CURRENCY = "EUR"'
fixture 9200 "$slug"
cat > test_rates.py <<'PY'
import json
import unittest
from pathlib import Path

from rates import RATES, convert


class RatesTests(unittest.TestCase):
    def test_fixture_matches_rates(self):
        fixture = json.loads(Path("fixtures/eur.json").read_text())
        self.assertEqual(RATES[fixture["currency"]], fixture["rate_bp"])

    def test_gbp_conversion(self):
        self.assertEqual(convert(1000, "GBP"), 790)
PY
stash "rates migration"

# Newest attempt: copies instead of renaming and ships a stale fixture.
cp legacy_rates.py rates.py
replace app.py 'from legacy_rates import convert' 'from rates import convert'
replace settings.py 'DEFAULT_CURRENCY = "USD"' 'DEFAULT_CURRENCY = "EUR"'
fixture 9100 "stale-$slug"
stash "rates migration v2"

# A stray snapshot commit of the migration with another stale fixture.
git switch -q -c snapshot
git mv legacy_rates.py rates.py
replace app.py 'from legacy_rates import convert' 'from rates import convert'
fixture 9150 "snapshot-$slug"
commit -m "WIP on main: rates migration snapshot"
git switch -q main
git branch -q -D snapshot
git stash clear

# main moves on after the stash was dropped.
replace settings.py 'PRECISION = 2' 'PRECISION = 4'
commit -m "Quote with four decimal places"
printf '# Changelog\n\n- Quotes now use four decimal places.\n' > CHANGELOG.md
commit -m "Start a changelog"
unset GIT_AUTHOR_DATE GIT_COMMITTER_DATE
git reflog expire --expire=now --all
git status --porcelain | grep -q . && exit 1 || true
