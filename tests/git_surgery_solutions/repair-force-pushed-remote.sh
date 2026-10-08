#!/usr/bin/env bash
# Reference hand solution, run from the task workspace.
set -euo pipefail
published="$(git -C origin.git reflog show --format=%H release | sed -n 3p)"
cd alice
fix="$(git rev-parse origin/release)"
git switch -q release
git reset -q --hard "$published"
git cherry-pick "$fix"
git push -q --force-with-lease="release:$fix" origin release
