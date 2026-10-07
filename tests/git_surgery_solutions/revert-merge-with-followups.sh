#!/usr/bin/env bash
# Reference hand solution, run from the task workspace.
set -euo pipefail
merge="$(git log --merges --format=%H --grep="^Merge branch 'pricing-engine'$" -1)"
git revert -m 1 "$merge" || true
git show HEAD:pricing.py | sed 's/return unit \* min(qty, 99)/return unit * qty/' > pricing.py
git add pricing.py test_bulk.py
git revert --continue
