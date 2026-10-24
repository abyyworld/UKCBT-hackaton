#!/usr/bin/env bash
# Build the code ZIP for the portal: source, docs, slides, team details, validation results and the predictions.
# No data, no secrets, no virtual environment.
#   bash scripts/make_zip.sh [out.zip] [predictions.csv]
set -euo pipefail
out="${1:-submissions/code.zip}"
preds="${2:-submissions/final.csv}"
mkdir -p "$(dirname "$out")"
rm -f "$out"
files=(src run.py requirements.txt README.md SUBMISSION.md LICENSE ATTRIBUTION.md TEAM.md TEAM_RULES.md NOW.md scripts report)
for f in explore/*.md; do files+=("$f"); done
[ -f "$preds" ] && files+=("$preds")
existing=()
for f in "${files[@]}"; do [ -e "$f" ] && existing+=("$f"); done
zip -q -r "$out" "${existing[@]}" -x '*__pycache__*'
unzip -l "$out" | tail -1
ls -la "$out"
