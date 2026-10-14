#!/usr/bin/env bash
set -euo pipefail

base_experiment="${1:-modern-kt-t1-five-source-padfix-20260824}"
recovery_experiment="${2:-modern-kt-t1-dependency-recovery-20260824}"
state_dir="${3:-results/modern_kt_t1_combined_20260824}"
export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure-studentmodeling-feeds-v2
export AMLT_PROJECT_PATH=/home/v-yuejiali
work_dir="$(pwd)"
mkdir -p "$state_dir/status" "$state_dir/download"

while true; do
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  for experiment in "$base_experiment" "$recovery_experiment"; do
    destination="$state_dir/status/${experiment}_${timestamp}.txt"
    (cd "$AMLT_PROJECT_PATH" && amlt status "$experiment" --hide-urls) > "$destination" 2>&1 || true
    cp "$destination" "$state_dir/status/${experiment}_latest.txt"
  done
  # The base experiment contains superseded simpleKT/UKT jobs that are allowed
  # to fail or be canceled. Only DTransformer/ReKT gate base completion.
  base_selected="$state_dir/status/base_selected_${timestamp}.txt"
  grep -E 'modern-kt-t1-(dtransformer|rekt)-seed' "$state_dir/status/${base_experiment}_latest.txt" > "$base_selected" || true
  recovery_selected="$state_dir/status/recovery_selected_${timestamp}.txt"
  grep -E 'modern-kt-t1-(simplekt|ukt)-seed' "$state_dir/status/${recovery_experiment}_latest.txt" > "$recovery_selected" || true
  if grep -Eq '(^|[[:space:]])(queued|running|preparing|finalizing)([[:space:]]|$)' "$base_selected" \
      || grep -Eq '(^|[[:space:]])(queued|running|preparing|finalizing)([[:space:]]|$)' "$recovery_selected"; then
    sleep 30
    continue
  fi
  if grep -Eq '(^|[[:space:]])(failed|canceled|stopped|paused)([[:space:]]|$)' "$base_selected" \
      || grep -Eq '(^|[[:space:]])(failed|canceled|stopped|paused)([[:space:]]|$)' "$recovery_selected"; then
    touch "$state_dir/needs_attention"
    exit 1
  fi
  break
done

# DTransformer/ReKT are authoritative in the base experiment. simpleKT/UKT
# are authoritative in the dependency-recovery experiment.
for model in dtransformer rekt; do
  for seed in 42 97 123; do
    destination="$state_dir/download/${model}_seed${seed}"
    mkdir -p "$destination"
    (cd "$AMLT_PROJECT_PATH" && amlt results download "$base_experiment" ":modern-kt-t1-${model}-seed${seed}" -o "$work_dir/$destination")
  done
done
for model in simplekt ukt; do
  for seed in 42 97 123; do
    destination="$state_dir/download/${model}_seed${seed}"
    mkdir -p "$destination"
    (cd "$AMLT_PROJECT_PATH" && amlt results download "$recovery_experiment" ":modern-kt-t1-${model}-seed${seed}" -o "$work_dir/$destination")
  done
done

python scripts/summarize_modern_kt_t1.py \
  --input-dir "$state_dir/download" \
  --output-json "$state_dir/summary.json" \
  --output-md "$state_dir/summary.md" \
  > "$state_dir/summarize.log" 2>&1
touch "$state_dir/complete"
