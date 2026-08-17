#!/usr/bin/env bash
set -euo pipefail

seed="${1:?usage: setup.sh SEED [WORKSPACE]}"
workspace="${2:-/workspace/secret-in-history}"

digest() {
    printf '%s' "$seed:$1" | sha256sum | cut -c1-16
}

slug="$(digest file)"
secret="AKIA$(printf '%s' "$seed:secret" | sha256sum | tr 'a-f' 'A-F' | cut -c1-16)"
module="service_${slug}"

mkdir -p "$workspace"
cd "$workspace"
git init -q -b main
git config user.name "BenchKit Generator"
git config user.email "generator@benchkit.invalid"
git config commit.gpgSign false
git config core.autocrlf false

export GIT_AUTHOR_NAME="BenchKit Generator"
export GIT_AUTHOR_EMAIL="generator@benchkit.invalid"
export GIT_COMMITTER_NAME="$GIT_AUTHOR_NAME"
export GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"

commit_at() {
    timestamp="$1"
    shift
    GIT_AUTHOR_DATE="@$timestamp +0000" GIT_COMMITTER_DATE="@$timestamp +0000" \
        git commit -q "$@"
}

cat > "$module.py" <<EOF
"""Generated deployment configuration."""

REGION = "eu-central-1"
RETRIES = 2


def endpoint() -> str:
    return f"https://{REGION}.example.invalid"
EOF
cat > test_config.py <<EOF
import unittest

import $module as config


class ConfigTests(unittest.TestCase):
    def test_endpoint(self):
        self.assertEqual(config.endpoint(), "https://eu-central-1.example.invalid")

    def test_retry_policy(self):
        self.assertEqual(config.RETRIES, 5)

    def test_credential_source(self):
        self.assertEqual(config.CREDENTIAL_ENV, "BENCHKIT_DEPLOY_KEY")
EOF
cat > README.md <<EOF
# Deployment helper ${slug}

Run tests with \`python3 -m unittest -v\`.
EOF
git add .
commit_at 1700000000 -m "bootstrap deployment helper ${slug}"
baseline_sha="$(git rev-parse HEAD)"

printf '\nDEPLOY_KEY = "%s"\n' "$secret" >> "$module.py"
git add "$module.py"
commit_at 1700000060 -m "temporarily configure deployment credentials"

python3 - "$module.py" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
text = text.replace('DEPLOY_KEY = "', 'CREDENTIAL_ENV = "BENCHKIT_DEPLOY_KEY"  # replaced: ')
text = text.replace('"\n', '"\n', 1)
# Remove the leaked value from the current tree while deliberately changing the
# same line, so dropping/editing the earlier commit creates a rebase conflict.
lines = text.splitlines()
lines[-1] = 'CREDENTIAL_ENV = "BENCHKIT_DEPLOY_KEY"'
path.write_text("\n".join(lines) + "\n")
PY
git add "$module.py"
commit_at 1700000120 -m "read deployment credentials from environment"

python3 - "$module.py" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text().replace("RETRIES = 2", "RETRIES = 5")
path.write_text(text)
PY
git add "$module.py"
commit_at 1700000180 -m "harden deployment retry policy"

printf '\nHistory must remain reviewable after credential cleanup.\n' >> README.md
git add README.md
commit_at 1700000240 -m "document credential cleanup policy"

git tag "benchkit-baseline-$slug" "$baseline_sha"
git status --porcelain | grep -q . && exit 1 || true
