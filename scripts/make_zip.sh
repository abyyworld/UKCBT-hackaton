#!/usr/bin/env bash
# Build the code ZIP for the portal: source, docs and dependency list only. No data, no secrets.
set -euo pipefail
out="${1:-submissions/code.zip}"
mkdir -p "$(dirname "$out")"
rm -f "$out"
zip -q -r "$out" src run.py requirements.txt README.md SUBMISSION.md LICENSE ATTRIBUTION.md TEAM_RULES.md scripts \
  -x '*__pycache__*' 2>/dev/null || zip -q -r "$out" src requirements.txt README.md scripts -x '*__pycache__*'
unzip -l "$out" | tail -1
ls -la "$out"
