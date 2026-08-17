#!/usr/bin/env bash
set -euo pipefail
seed="${1:?usage: setup.sh SEED [WORKSPACE]}"
workspace="${2:-/workspace/split-the-mega-commit}"
slug="$(printf '%s' "$seed:split" | sha256sum | cut -c1-12)"
mkdir -p "$workspace"
cd "$workspace"
git init -q -b main
git config user.name "BenchKit Generator"
git config user.email "generator@benchkit.invalid"
git config commit.gpgSign false
git config core.autocrlf false
export GIT_AUTHOR_NAME="BenchKit Generator" GIT_AUTHOR_EMAIL="generator@benchkit.invalid"
export GIT_COMMITTER_NAME="$GIT_AUTHOR_NAME" GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"
cat > invoice.py <<'PY'
from decimal import Decimal, ROUND_DOWN


def cents(value: str) -> int:
    return int((Decimal(value) * 100).quantize(Decimal("1"), rounding=ROUND_DOWN))
PY
cat > test_invoice.py <<'PY'
import unittest
from invoice import cents


class InvoiceTests(unittest.TestCase):
    def test_cents_rounds_half_up(self):
        self.assertEqual(cents("10.235"), 1024)
PY
printf '# Invoice tools %s\n\nRun tests with `python3 -m unittest discover -v`.\n' "$slug" > README.md
git add .
GIT_AUTHOR_DATE='@1700200000 +0000' GIT_COMMITTER_DATE='@1700200000 +0000' git commit -q -m "bootstrap invoice tools $slug"
git tag "benchkit-split-baseline-$slug"
python3 -c 'from pathlib import Path; p=Path("invoice.py"); p.write_text(p.read_text().replace("ROUND_DOWN", "ROUND_HALF_UP"))'
cat > labels.py <<'PY'
def invoice_label(number: int) -> str:
    return f"INV-{number:06d}"
PY
cat > test_labels.py <<'PY'
import unittest
from labels import invoice_label


class LabelTests(unittest.TestCase):
    def test_invoice_label(self):
        self.assertEqual(invoice_label(42), "INV-000042")
PY
git add .
GIT_AUTHOR_DATE='@1700200060 +0000' GIT_COMMITTER_DATE='@1700200060 +0000' git commit -q -m "ship invoice patch bundle"
