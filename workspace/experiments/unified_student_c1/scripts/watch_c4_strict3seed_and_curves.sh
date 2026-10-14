#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
train_experiment="c4-cf1-visionpad-strict3seed-v2-20260824"
curve_config="amlt/feeds_c4_cf1_visionpad_checkpoint_development_curve_20260824.yaml"
run_dir="$project_dir/results/c4_strict3seed_and_curves_monitor_20260824"
download_dir="$run_dir/download"
mkdir -p "$run_dir" "$download_dir"

export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure-studentmodeling-feeds-v2
cd "$project_dir"

status_of() {
  amlt status "$1" 2>&1
}

submit_curve_if_ready() {
  local seed="$1"
  local train_status="$2"
  local train_job="c4-cf1-visionpad-strict-seed${seed}"
  local curve_experiment="c4-cf1-visionpad-checkpoint-development-seed${seed}-v1-20260824"
  local curve_job="c4-cf1-visionpad-checkpoint-curve-seed${seed}"
  local receipt="$run_dir/${curve_experiment}.submitted"
  local curve_status

  if [[ -f "$receipt" ]]; then
    return 0
  fi
  curve_status="$(status_of "$curve_experiment")"
  if grep -q "Experiment:" <<<"$curve_status"; then
    touch "$receipt"
    return 0
  fi
  if ! grep -q "Cannot find an experiment" <<<"$curve_status"; then
    return 0
  fi
  if ! grep -Eq ":${train_job}[[:space:]].*(running|pass)" <<<"$train_status"; then
    return 0
  fi

  if amlt run "$curve_config" ":${curve_job}" "$curve_experiment" -t Feeds \
    --description "Strict fixed-seed CF1 vision-pad development curve for seed${seed}: frozen six checkpoints, cross-template accuracy, probe, and frozen C4 controls" \
    >"$run_dir/${curve_experiment}.submission.log" 2>&1; then
    touch "$receipt"
  fi
}

while true; do
  timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  train_status="$(status_of "$train_experiment")"
  {
    echo "[$timestamp]"
    echo "=== $train_experiment ==="
    echo "$train_status"
  } >>"$run_dir/status.log"

  for seed in 13 42 97; do
    submit_curve_if_ready "$seed" "$train_status" || true
  done

  all_curves_terminal=true
  all_curves_pass=true
  for seed in 13 42 97; do
    curve_experiment="c4-cf1-visionpad-checkpoint-development-seed${seed}-v1-20260824"
    curve_status="$(status_of "$curve_experiment")"
    {
      echo "=== $curve_experiment ==="
      echo "$curve_status"
    } >>"$run_dir/status.log"
    if grep -Eqi 'running|queued|preparing|paused|resuming' <<<"$curve_status" \
        || ! grep -q "Experiment:" <<<"$curve_status"; then
      all_curves_terminal=false
    fi
    if ! grep -Eqi '1 job.*1 pass' <<<"$curve_status"; then
      all_curves_pass=false
    fi
  done

  train_terminal=false
  train_pass=false
  if ! grep -Eqi 'running|queued|preparing|paused|resuming' <<<"$train_status"; then
    train_terminal=true
  fi
  if grep -Eqi '3 jobs.*3 pass' <<<"$train_status"; then
    train_pass=true
  fi

  if $train_pass && $all_curves_pass; then
    amlt results download "$train_experiment" --status pass \
      -I 'gate/gate_summary.json' -I 'audit/trainer_state_step*.json' \
      -o "$download_dir" >>"$run_dir/download.log" 2>&1 || true
    for seed in 13 42 97; do
      curve_experiment="c4-cf1-visionpad-checkpoint-development-seed${seed}-v1-20260824"
      amlt results download "$curve_experiment" --status pass \
        -I 'curve_summary.json' -I 'curve_summary.md' -I 'curve/step_*/step_summary.json' \
        -o "$download_dir" >>"$run_dir/download.log" 2>&1 || true
    done
    touch "$run_dir/complete"
    exit 0
  fi

  if $train_terminal && ! $train_pass; then
    echo "$train_status" >"$run_dir/needs_attention"
    exit 1
  fi
  if $all_curves_terminal && ! $all_curves_pass; then
    touch "$run_dir/needs_attention"
    exit 1
  fi
  sleep 120
done
