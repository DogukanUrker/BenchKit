#!/usr/bin/env bash
set -euo pipefail
seed="${1:?usage: setup.sh SEED [WORKSPACE]}"
workspace="${2:-/workspace/revert-merge-with-followups}"
slug="$(printf '%s' "$seed:revert-merge" | sha256sum | cut -c1-12)"
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
clock=1700600000
tick() {
    clock=$((clock + 60))
    export GIT_AUTHOR_DATE="@$clock +0000" GIT_COMMITTER_DATE="@$clock +0000"
}
commit() {
    tick
    git add -A
    git commit -q "$@"
}
merge() {
    tick
    git merge -q --no-ff "$@"
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

printf '__pycache__/\n' > .gitignore
cat > pricing.py <<'PY'
"""Order pricing in integer cents."""

PRICES = {"apple": 120, "pear": 95, "plum": 60}


def price(sku: str, qty: int) -> int:
    unit = PRICES[sku]
    return unit * qty


def total(lines: list[tuple[str, int]]) -> int:
    return sum(price(sku, qty) for sku, qty in lines)
PY
cat > test_pricing.py <<'PY'
import unittest

from pricing import price, total


class PricingTests(unittest.TestCase):
    def test_single_line(self):
        self.assertEqual(price("pear", 3), 285)

    def test_total(self):
        self.assertEqual(total([("apple", 1), ("plum", 2)]), 240)
PY
printf '# Pricing service %s\n\nRun tests with `python3 -m unittest discover -v`.\n' "$slug" > README.md
commit -m "bootstrap pricing service $slug"

git switch -q -c catalog-search
cat > search.py <<'PY'
from pricing import PRICES


def search(prefix: str) -> list[str]:
    return sorted(sku for sku in PRICES if sku.startswith(prefix))
PY
cat > test_search.py <<'PY'
import unittest

from search import search


class SearchTests(unittest.TestCase):
    def test_prefix(self):
        self.assertEqual(search("p"), ["pear", "plum"])
PY
commit -m "Add catalog prefix search"
git switch -q main
merge catalog-search -m "Merge branch 'catalog-search'"

git switch -q -c pricing-engine
replace pricing.py 'PRICES = {"apple": 120, "pear": 95, "plum": 60}
' 'PRICES = {"apple": 120, "pear": 95, "plum": 60}
BULK_QTY = 10
'
replace pricing.py '    unit = PRICES[sku]
    return unit * qty' '    unit = PRICES[sku]
    if qty >= BULK_QTY:
        unit = unit * 9 // 10
    return unit * qty'
cat > test_bulk.py <<'PY'
import unittest

from pricing import price


class BulkTests(unittest.TestCase):
    def test_bulk_discount(self):
        self.assertEqual(price("apple", 10), 1080)

    def test_below_threshold(self):
        self.assertEqual(price("apple", 9), 1080)
PY
commit -m "Add bulk discount"
replace pricing.py '    return unit * qty' '    return unit * min(qty, 99)'
commit -m "Cap line quantities to protect inventory"

git switch -q main
cat >> pricing.py <<'PY'


def total_with_tax(lines: list[tuple[str, int]]) -> int:
    return (total(lines) * 108 + 50) // 100
PY
cat >> test_pricing.py <<'PY'


class TaxTests(unittest.TestCase):
    def test_total_with_tax(self):
        from pricing import total_with_tax

        self.assertEqual(total_with_tax([("pear", 1)]), 103)
PY
commit -m "Add tax-inclusive totals"
merge pricing-engine -m "Merge branch 'pricing-engine'"

replace pricing.py 'BULK_QTY = 10
' 'BULK_QTY = 8
BULK_RATE = 90
'
replace pricing.py '        unit = unit * 9 // 10' '        unit = unit * BULK_RATE // 100'
cat > test_bulk.py <<'PY'
import unittest

from pricing import price


class BulkTests(unittest.TestCase):
    def test_bulk_discount(self):
        self.assertEqual(price("apple", 8), 864)

    def test_below_threshold(self):
        self.assertEqual(price("apple", 7), 840)
PY
commit -m "Make the bulk discount configurable"

cat >> pricing.py <<'PY'


def receipt(lines: list[tuple[str, int]]) -> str:
    return f"net={total(lines)} gross={total_with_tax(lines)}"
PY
cat >> test_pricing.py <<'PY'


class ReceiptTests(unittest.TestCase):
    def test_receipt_uses_bulk_and_tax(self):
        from pricing import receipt

        self.assertEqual(receipt([("apple", 8)]), "net=864 gross=933")
PY
commit -m "Add itemised receipts"

replace pricing.py 'def price(sku: str, qty: int) -> int:
' 'def price(sku: str, qty: int) -> int:
    if qty <= 0:
        raise ValueError("quantity must be positive")
'
cat >> test_pricing.py <<'PY'


class QuantityTests(unittest.TestCase):
    def test_rejects_empty_lines(self):
        with self.assertRaises(ValueError):
            price("pear", 0)

    def test_large_orders_charge_every_unit(self):
        self.assertEqual(price("plum", 150), 8100)
PY
commit -m "Validate quantities and cover large orders"
git branch -q -D catalog-search pricing-engine
unset GIT_AUTHOR_DATE GIT_COMMITTER_DATE
git status --porcelain | grep -q . && exit 1 || true
