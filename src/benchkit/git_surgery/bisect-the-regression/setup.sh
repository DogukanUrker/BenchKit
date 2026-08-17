#!/usr/bin/env bash
set -euo pipefail
seed="${1:?usage: setup.sh SEED [WORKSPACE]}"
workspace="${2:-/workspace/bisect-the-regression}"
slug="$(printf '%s' "$seed:bisect" | sha256sum | cut -c1-12)"
bad_index=$((10 + seed % 19))
mkdir -p "$workspace"
cd "$workspace"
git init -q -b main
git config user.name "BenchKit Generator"
git config user.email "generator@benchkit.invalid"
git config commit.gpgSign false
git config core.autocrlf false
export GIT_AUTHOR_NAME="BenchKit Generator" GIT_AUTHOR_EMAIL="generator@benchkit.invalid"
export GIT_COMMITTER_NAME="$GIT_AUTHOR_NAME" GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"
commit_at() {
    local timestamp="$1"
    shift
    GIT_AUTHOR_DATE="@$timestamp +0000" GIT_COMMITTER_DATE="@$timestamp +0000" git commit -q "$@"
}
cat > calculator.py <<'PY'
def shipping_total(items: int) -> int:
    """Return the shipping charge for a number of items."""
    return items * 3
PY
cat > test_calculator.py <<'PY'
import unittest
from calculator import shipping_total


class ShippingTests(unittest.TestCase):
    def test_shipping_total(self):
        self.assertEqual(shipping_total(7), 21)
PY
printf '# Regression history %s\n' "$slug" > README.md
git add .
commit_at 1700100000 -m "bootstrap shipping calculator $slug"
git tag "benchkit-bisect-baseline-$slug"
for index in $(seq 1 39); do
    printf 'note-%02d=%s\n' "$index" "$(printf '%s' "$seed:$index" | sha256sum | cut -c1-8)" >> changelog.txt
    if [[ "$index" == "$bad_index" ]]; then
        python3 -c 'from pathlib import Path; p=Path("calculator.py"); p.write_text(p.read_text().replace("items * 3", "items * 4"))'
        message="optimize shipping coefficient"
    else
        message="record maintenance note $index"
    fi
    git add .
    commit_at "$((1700100000 + index * 60))" -m "$message"
done
