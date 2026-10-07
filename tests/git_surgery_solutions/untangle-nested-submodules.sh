#!/usr/bin/env bash
# Reference hand solution, run from the task workspace.
set -euo pipefail
cd app
codec=libs/engine/vendor/codec
git -C "$codec" commit -q -am "Fix codec frame whitespace"
git -C "$codec" push -q origin HEAD:main
git -C libs/engine checkout -q "$(git -C libs/engine rev-parse origin/main)"
git add libs/engine
git commit -q --amend --no-edit
git -C libs/engine add vendor/codec
git -C libs/engine commit -q -m "Pick up codec whitespace fix"
git -C libs/engine push -q origin HEAD:main
git add libs/engine
git commit -q -m "Pick up codec whitespace fix"
