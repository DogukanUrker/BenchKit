#!/usr/bin/env bash
set -euo pipefail
seed="${1:?usage: setup.sh SEED [WORKSPACE]}"
workspace="${2:-/workspace/rebase-conflict-chain}"
slug="$(printf '%s' "$seed:rebase" | sha256sum | cut -c1-12)"
mkdir -p "$workspace"
cd "$workspace"
git init -q -b main
git config user.name "BenchKit Generator"
git config user.email "generator@benchkit.invalid"
git config commit.gpgSign false
git config core.autocrlf false
export GIT_AUTHOR_NAME="BenchKit Generator" GIT_AUTHOR_EMAIL="generator@benchkit.invalid"
export GIT_COMMITTER_NAME="$GIT_AUTHOR_NAME" GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"
commit_at() { local t="$1"; shift; GIT_AUTHOR_DATE="@$t +0000" GIT_COMMITTER_DATE="@$t +0000" git commit -q "$@"; }
cat > auth.py <<'PY'
def identify(user: str) -> str:
    return user.strip()
PY
cat > billing.py <<'PY'
def total(subtotal: int) -> int:
    return subtotal
PY
cat > report.py <<'PY'
def render(items: list[str]) -> str:
    return ",".join(items)
PY
cat > test_behavior.py <<'PY'
import unittest
from auth import identify
from billing import total
from report import render


class BehaviorTests(unittest.TestCase):
    def test_auth_keeps_audit_and_normalization(self):
        self.assertEqual(identify(" Alice "), "audit:alice")

    def test_billing_keeps_discount_and_tax(self):
        self.assertEqual(total(100), 105)

    def test_report_keeps_header_and_uppercase(self):
        self.assertEqual(render(["one", "two"]), "items:ONE,TWO")
PY
printf '# Conflict chain %s\n' "$slug" > README.md
git add .
commit_at 1700400000 -m "bootstrap conflict chain $slug"
git tag "benchkit-rebase-baseline-$slug"
git switch -q -c feature
python3 -c 'from pathlib import Path; p=Path("auth.py"); p.write_text(p.read_text().replace("return user.strip()", "return f\"audit:{user.strip()}\""))'
git add auth.py
commit_at 1700400060 -m "add authentication audit marker"
python3 -c 'from pathlib import Path; p=Path("billing.py"); p.write_text(p.read_text().replace("return subtotal", "return subtotal - 5"))'
git add billing.py
commit_at 1700400120 -m "add loyalty discount"
python3 -c 'from pathlib import Path; p=Path("report.py"); p.write_text(p.read_text().replace("return \",\".join(items)", "return \",\".join(item.upper() for item in items)"))'
git add report.py
commit_at 1700400180 -m "add uppercase report format"
git switch -q main
python3 -c 'from pathlib import Path; p=Path("auth.py"); p.write_text(p.read_text().replace("return user.strip()", "return user.strip().lower()"))'
python3 -c 'from pathlib import Path; p=Path("billing.py"); p.write_text(p.read_text().replace("return subtotal", "return subtotal + 10"))'
python3 -c 'from pathlib import Path; p=Path("report.py"); p.write_text(p.read_text().replace("return \",\".join(items)", "return \"items:\" + \",\".join(items)"))'
git add auth.py billing.py report.py
commit_at 1700400240 -m "move main behavior forward"
git switch -q feature
