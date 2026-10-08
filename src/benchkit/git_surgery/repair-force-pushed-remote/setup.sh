#!/usr/bin/env bash
set -euo pipefail
seed="${1:?usage: setup.sh SEED [WORKSPACE]}"
workspace="${2:-/workspace/repair-force-pushed-remote}"
slug="$(printf '%s' "$seed:force-push" | sha256sum | cut -c1-12)"
mkdir -p "$workspace"
cd "$workspace"
export GIT_AUTHOR_NAME="BenchKit Generator" GIT_AUTHOR_EMAIL="generator@benchkit.invalid"
export GIT_COMMITTER_NAME="$GIT_AUTHOR_NAME" GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"
export GIT_CONFIG_GLOBAL=/dev/null
clock=1700800000
tick() {
    clock=$((clock + 60))
    export GIT_AUTHOR_DATE="@$clock +0000" GIT_COMMITTER_DATE="@$clock +0000"
}
configure() {
    git -C "$1" config user.name "$2"
    git -C "$1" config user.email "$(tr 'A-Z' 'a-z' <<<"$2")@benchkit.invalid"
    git -C "$1" config commit.gpgSign false
    git -C "$1" config core.autocrlf false
    git -C "$1" config core.logAllRefUpdates always
}
commit() {
    local clone="$1"
    shift
    tick
    git -C "$clone" add -A
    git -C "$clone" commit -q "$@"
}
push() {
    tick
    git -C "$1" push -q "${@:2}" 2>/dev/null
}

git init -q --bare -b main origin.git
git -C origin.git config core.logAllRefUpdates always
git -C origin.git config receive.denyNonFastForwards false

tick
git clone -q origin.git bob 2>/dev/null
configure bob Bob
printf '__pycache__/\n' > bob/.gitignore
cat > bob/invoice.py <<'PY'
"""Invoice helpers in integer cents."""


def subtotal(lines: list[int]) -> int:
    return sum(lines)
PY
cat > bob/test_invoice.py <<'PY'
import unittest

from invoice import subtotal


class InvoiceTests(unittest.TestCase):
    def test_subtotal(self):
        self.assertEqual(subtotal([100, 250]), 350)
PY
printf '# Invoicing %s\n\nRun tests with `python3 -m unittest discover -v`.\n' "$slug" > bob/README.md
printf '1.0.0\n' > bob/VERSION
commit bob -m "bootstrap invoicing $slug"
tick
git -C bob tag -a v1.0 -m "Invoicing 1.0"
push bob origin main v1.0

git -C bob switch -q -c release
cat >> bob/invoice.py <<'PY'


def tax(amount: int) -> int:
    return amount * 20 // 100
PY
commit bob -m "release: add tax helper"
cat >> bob/test_invoice.py <<'PY'


class TaxTests(unittest.TestCase):
    def test_tax(self):
        from invoice import tax

        self.assertEqual(tax(1000), 200)
PY
commit bob -m "release: cover tax helper"
printf '1.1.0\n' > bob/VERSION
commit bob -m "release: prepare 1.1"
tick
git -C bob tag -a v1.1 -m "Invoicing 1.1"
push bob -u origin release v1.1

git -C bob switch -q main
printf '\nSee CONTRIBUTING for the release process.\n' >> bob/README.md
commit bob -m "Document the release process"
push bob origin main

tick
git clone -q origin.git alice 2>/dev/null
configure alice Alice
git -C alice switch -q release
cat >> alice/invoice.py <<'PY'


def discount(amount: int, percent: int) -> int:
    return amount - amount * percent // 100
PY
cat > alice/test_discount.py <<'PY'
import unittest

from invoice import discount


class DiscountTests(unittest.TestCase):
    def test_discount(self):
        self.assertEqual(discount(1000, 15), 850)
PY
commit alice -m "release: add percentage discounts"
cat >> alice/invoice.py <<'PY'


def grand_total(lines: list[int], percent: int = 0) -> int:
    net = discount(subtotal(lines), percent)
    return net + tax(net)
PY
cat > alice/test_grand_total.py <<'PY'
import unittest

from invoice import grand_total


class GrandTotalTests(unittest.TestCase):
    def test_grand_total(self):
        self.assertEqual(grand_total([600, 400], 10), 1080)
PY
printf '1.2.0-rc1\n' > alice/VERSION
commit alice -m "release: prepare 1.2"
printf '1.2.0\n' > alice/VERSION
tick
git -C alice add -A
git -C alice commit -q --amend -m "release: prepare 1.2"
push alice origin release

# A candidate that was pushed for review and then deleted again.
git -C alice switch -q -c candidate
printf '1.2.0-final\n' > alice/VERSION
commit alice -m "release: prepare 1.2 final"
push alice origin candidate
push alice origin --delete candidate
git -C alice switch -q release
git -C alice branch -q -D candidate

# Bob rewords his stale tip and force-pushes over Alice's work.
tick
git -C bob switch -q release
git -C bob commit -q --amend -m "release: prepare 1.1 (reworded)"
push bob --force origin release

# Alice syncs to the rewritten branch and pushes a legitimate fix on top.
tick
git -C alice fetch -q origin
tick
git -C alice reset -q --hard origin/release
printf '\nInvoices round tax down to whole cents.\n' >> alice/README.md
cat > alice/CHANGES.md <<'PY'
# Changes

- Tax is rounded down to whole cents.
PY
commit alice -m "release: document tax rounding"
push alice origin release
git -C bob fetch -q origin
unset GIT_AUTHOR_DATE GIT_COMMITTER_DATE
