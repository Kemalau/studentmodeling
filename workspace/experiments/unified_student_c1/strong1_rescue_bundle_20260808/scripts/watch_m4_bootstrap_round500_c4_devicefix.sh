#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
state_dir="$project_dir/results/recurrent_m4_pipeline_20260730/bootstrap_round500_c4_devicefix"
experiment="recurrent-m4-bootstrap-round500-c4-devicefix-seed42-20260731"
base_experiment="$experiment"
config="$project_dir/amlt/feeds_m4_bootstrap_round500_c4_devicefix_20260731.yaml"
remote_prefix="studentmodeling/diagnostics/world_model_v1_m4_causal"
account="csnewsandfeeds4150361735"
container="amulet"
mkdir -p "$state_dir/download"
cd "$project_dir"

exec 9>"$state_dir/controller.lock"
flock -n 9 || exit 0
echo "$$" > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT
rm -f "$state_dir/complete" "$state_dir/needs_attention"

heartbeat() { date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" \
    | tee -a "$state_dir/controller.log"
}

passed=false
for recovery_wave in 0 1 2 3; do
  while true; do
    heartbeat
    status="$(amlt status "$experiment" 2>&1)"
    printf '[%s] %s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      "$experiment" "$status" >> "$state_dir/status.log"
    if grep -Eqi 'No authentication method succeeded|refreshing your credentials|connection|timed out' <<< "$status"; then
      sleep 60
      continue
    fi
    if grep -Eqi 'running|queued|preparing|paused|resuming|pre-queued' <<< "$status"; then
      sleep 120
      continue
    fi
    if python - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1], 2) else 1)
PY
    then
      passed=true
      break
    fi
    break
  done
  [[ "$passed" != true ]] || break
  printf '%s\n' "$status" > "$state_dir/${experiment}.terminal_status.txt"
  timeout 180 amlt logs view -n 300 "$experiment" \
    > "$state_dir/${experiment}.failure.log" 2>&1 || true
  (( recovery_wave < 3 )) || break
  experiment="${base_experiment}-recovery$((recovery_wave + 1))"
  exec 8>"$project_dir/results/feeds_submission_global.lock"; flock 8
  output="$(amlt run "$config" "$experiment" -t Feeds -y \
    --description "C4-only bootstrap device-fix infrastructure recovery $((recovery_wave + 1)); no retraining" 2>&1)"
  rc=$?
  printf '%s\n' "$output" > "$state_dir/${experiment}.submission.txt"
  flock -u 8
  (( rc == 0 )) || break
done
if [[ "$passed" == true ]]; then
    printf '%s\n' "$status" > "$state_dir/terminal_status.txt"
    amlt results download "$experiment" --status pass \
      --include recurrent_causal_gate.json --include probe_scope.txt \
      --output "$state_dir/download" >> "$state_dir/controller.log" 2>&1 || {
        log "ERROR: result download failed"
        touch "$state_dir/needs_attention"
        exit 1
      }
    for branch in bootstrap_balanced_sum_w1 bootstrap_balanced_sum_w4; do
      branch_dir="$state_dir/training/$branch"
      mkdir -p "$branch_dir"
      remote_root="${remote_prefix}_${branch}_seed42_round500_20260731"
      for name in branch_training_summary.json validation_metrics.json \
          causal_validation_metrics.json; do
        az storage blob download --account-name "$account" \
          --container-name "$container" --auth-mode login \
          --name "$remote_root/$name" --file "$branch_dir/$name" \
          --overwrite >> "$state_dir/controller.log" 2>&1 || {
            log "ERROR: could not download $branch/$name"
            touch "$state_dir/needs_attention"
            exit 1
          }
      done
    done
    python - "$state_dir/download" "$state_dir/training" \
      "$state_dir/comparison.json" "$state_dir/comparison.md" <<'PY'
import json
import pathlib
import sys

root, training_root, output, output_md = map(pathlib.Path, sys.argv[1:])
paths = sorted(root.rglob("recurrent_causal_gate.json"))
if len(paths) != 2:
    raise SystemExit(f"expected two C4 results, found {len(paths)}")
branches = []
for path in paths:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload["summary"]
    job = next(part for part in path.parts if part.startswith("m4-bootstrap-"))
    branch_name = (
        "bootstrap_balanced_sum_w1" if "-w1-" in job
        else "bootstrap_balanced_sum_w4"
    )
    train_dir = training_root / branch_name
    branch = json.loads((train_dir / "branch_training_summary.json").read_text())
    validation = json.loads((train_dir / "validation_metrics.json").read_text())
    causal = json.loads((train_dir / "causal_validation_metrics.json").read_text())
    clean = summary["clean"]
    branches.append({
        **branch,
        "C4_D_specific": float(summary["D_specific"]),
        "C4_D_damage": float(summary["D_damage"]),
        "C4_clean_n": int(summary["clean_n"]),
        "C4_swap_flip": float(clean["swap"]["flip_to_A_rate"]),
        "C4_random_flip": float(clean["random"]["flip_to_A_rate"]),
        "C4_noop_flip": float(clean["noop"]["flip_to_A_rate"]),
        "C4_same_rule_flip": float(clean["samemal"]["flip_to_A_rate"]),
        "validation_recurrent_accuracy": float(validation["recurrent_accuracy"]),
        "validation_direct_accuracy": float(validation["direct_accuracy"]),
        "validation_accuracy_gap": float(validation["accuracy_gap_recurrent_minus_direct"]),
        "causal_native_accuracy": float(causal["native_accuracy"]),
        "causal_donor_accuracy": float(causal["donor_accuracy"]),
        "causal_directed_flip": float(causal["directed_flip_to_A_on_clean"]),
        "source": str(path),
    })
eligible = [
    row for row in branches
    if row["C4_clean_n"] >= 50
    and row["C4_noop_flip"] <= 0.05
    and row["C4_same_rule_flip"] <= 0.10
    and row["validation_accuracy_gap"] >= -0.05
]
selected = max(eligible, key=lambda row: row["C4_D_specific"], default=None)
payload = {
    "stage": "M4_causal_strength_round500",
    "branches": branches,
    "selected_branch": selected["branch"] if selected and selected["C4_D_specific"] >= 0.15 else None,
    "scientific_gate_pass": bool(selected and selected["C4_D_specific"] >= 0.15),
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
lines = [
    "# Bootstrap directional round-500 C4 (device-fixed evaluation)", "",
    "| branch | C4 D_specific | C4 damage | swap | random | clean n | recurrent acc | direct acc |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
for row in branches:
    lines.append(
        f"| {row['branch']} | {row['C4_D_specific']:.3f} | {row['C4_D_damage']:.3f} | "
        f"{row['C4_swap_flip']:.3f} | {row['C4_random_flip']:.3f} | {row['C4_clean_n']} | "
        f"{row['validation_recurrent_accuracy']:.3f} | {row['validation_direct_accuracy']:.3f} |"
    )
output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
    cp "$state_dir/comparison.json" \
      "$project_dir/results/recurrent_m4_pipeline_20260730/causal_bootstrap_directional_tail/comparison.json"
    cp "$state_dir/comparison.md" \
      "$project_dir/results/recurrent_m4_pipeline_20260730/causal_bootstrap_directional_tail/comparison.md"
    touch "$project_dir/results/recurrent_m4_pipeline_20260730/causal_bootstrap_directional_tail/complete"
    rm -f "$project_dir/results/recurrent_m4_pipeline_20260730/causal_bootstrap_directional_tail/needs_attention"
    touch "$state_dir/complete"
    log "C4 device-fix comparison complete"
    exit 0
fi
touch "$state_dir/needs_attention"
log "ERROR: C4 device-fix evaluation exhausted infrastructure recovery"
exit 1
