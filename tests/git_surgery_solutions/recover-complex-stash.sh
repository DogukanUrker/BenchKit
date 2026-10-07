#!/usr/bin/env bash
# Reference hand solution, run from the task workspace.
set -euo pipefail
stash="$(for commit in $(git fsck --dangling --no-reflogs | awk '/commit/{print $3}'); do
    echo "$commit $(git log -1 --format=%s "$commit")"
done | grep ' On main: rates migration$' | cut -d' ' -f1)"
git diff --binary "$stash^1" "$stash^2" | git apply --index
git commit -q -m "Migrate rates module"
git diff --binary "$stash^2" "$stash" | git apply -3 || true
printf '"""Quote settings."""\n\nDEFAULT_CURRENCY = "EUR"\nPRECISION = 4\n' > settings.py
git checkout "$stash^3" -- .
git add -A
git commit -q -m "Finish rates migration"
