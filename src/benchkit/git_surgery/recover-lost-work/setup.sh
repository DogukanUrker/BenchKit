#!/usr/bin/env bash
set -euo pipefail
seed="${1:?usage: setup.sh SEED [WORKSPACE]}"
workspace="${2:-/workspace/recover-lost-work}"
slug="$(printf '%s' "$seed:recover" | sha256sum | cut -c1-12)"
mkdir -p "$workspace"
cd "$workspace"
git init -q -b main
git config user.name "BenchKit Generator"
git config user.email "generator@benchkit.invalid"
git config commit.gpgSign false
git config core.autocrlf false
export GIT_AUTHOR_NAME="BenchKit Generator" GIT_AUTHOR_EMAIL="generator@benchkit.invalid"
export GIT_COMMITTER_NAME="$GIT_AUTHOR_NAME" GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"
printf '# Recovery exercise %s\n' "$slug" > README.md
git add README.md
GIT_AUTHOR_DATE='@1700300000 +0000' GIT_COMMITTER_DATE='@1700300000 +0000' git commit -q -m "bootstrap recovery workspace $slug"
git tag "benchkit-recover-baseline-$slug"
git switch -q -c abandoned-work
cat > recovery.py <<EOF
RECOVERY_TOKEN = "$slug"


def recovered_value() -> str:
    return f"recovered:{RECOVERY_TOKEN}"
EOF
cat > test_recovery.py <<EOF
import unittest
from recovery import recovered_value


class RecoveryTests(unittest.TestCase):
    def test_recovered_value(self):
        self.assertEqual(recovered_value(), "recovered:$slug")
EOF
git add .
GIT_AUTHOR_DATE='@1700300060 +0000' GIT_COMMITTER_DATE='@1700300060 +0000' git commit -q -m "preserve recovered payload $slug"
git switch -q main
git branch -D abandoned-work >/dev/null
git reflog expire --expire=now --all
