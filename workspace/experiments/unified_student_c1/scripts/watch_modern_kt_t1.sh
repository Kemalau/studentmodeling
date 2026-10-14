#!/usr/bin/env bash
set -euo pipefail

experiment="${1:-modern-kt-t1-five-source-20260824}"
state_dir="${2:-results/modern_kt_t1_five_source_20260824}"
export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure-studentmodeling-feeds-v2
export AMLT_PROJECT_PATH=/home/v-yuejiali
work_dir="$(pwd)"
mkdir -p "$state_dir/status" "$state_dir/download"

while true; do
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  (cd "$AMLT_PROJECT_PATH" && amlt status "$experiment" --hide-urls) > "$state_dir/status/$timestamp.txt" 2>&1 || true
  latest="$state_dir/status/latest.txt"
  cp "$state_dir/status/$timestamp.txt" "$latest"
  if grep -Eq '(^|[[:space:]])(queued|running|preparing|finalizing)([[:space:]]|$)' "$latest"; then
    sleep 30
    continue
  fi
  break
done

if grep -Eq '(^|[[:space:]])(failed|canceled|stopped|paused)([[:space:]]|$)' "$state_dir/status/latest.txt"; then
  touch "$state_dir/needs_attention"
  exit 1
fi

for model in simplekt dtransformer rekt ukt; do
  for seed in 42 97 123; do
    destination="$state_dir/download/${model}_seed${seed}"
    mkdir -p "$destination"
    (cd "$AMLT_PROJECT_PATH" && amlt results download "$experiment" ":modern-kt-t1-${model}-seed${seed}" -o "$work_dir/$destination")
  done
done

python scripts/summarize_modern_kt_t1.py \
  --input-dir "$state_dir/download" \
  --output-json "$state_dir/summary.json" \
  --output-md "$state_dir/summary.md" \
  > "$state_dir/summarize.log" 2>&1
touch "$state_dir/complete"
