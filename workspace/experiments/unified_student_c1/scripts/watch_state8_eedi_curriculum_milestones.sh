#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
source "$project/scripts/use_studentmodeling_feeds_auth.sh"
parent_state="$project/results/state8_diagnosis_first_eedi_curriculum_20260812"
state="$parent_state/milestones"
experiment_file="$parent_state/active_experiment.txt"
default_experiment=state8-diagnosis-first-eedi-curriculum-recovery1-20260812
job=state8-diagnosis-first-eedi-curriculum-seed83
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
account=csnewsandfeeds4150361735
container=amulet
source_root=studentmodeling/checkpoints/state8_qualified_teacher_curriculum_seed42_exact5000_rounds1250_kd1_v2_20260805_taxonomy_semantic_v1_diagnosis_first_eedi_curriculum_seed83_single_ga4
snapshot_root=studentmodeling/checkpoint_snapshots/state8_diagnosis_first_eedi_curriculum_20260812
milestones=(1000 2000 3000 4000 5000)

mkdir -p "$state"
cd "$project" || exit 1
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"
trap 'rm -f "$state/controller.active.pid"' EXIT
[[ -s "$experiment_file" ]] || printf '%s\n' "$default_experiment" > "$experiment_file"
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
sleep_unlocked(){ sleep "$1" 9>&-; }

follow_parent_recovery(){
  local terminal_experiment="$1" next parent_pid
  while :; do
    heartbeat
    if [[ -f "$parent_state/needs_attention" ]]; then
      log "parent curriculum controller requires attention after terminal experiment=$terminal_experiment"
      return 1
    fi
    next="$(<"$experiment_file")"
    if [[ -n "$next" && "$next" != "$terminal_experiment" ]]; then
      log "following parent exact-resume recovery old=$terminal_experiment new=$next"
      return 0
    fi
    if [[ -s "$parent_state/controller.active.pid" ]]; then
      parent_pid="$(<"$parent_state/controller.active.pid")"
      if [[ -d "/proc/$parent_pid" ]]; then
        sleep_unlocked 20
        continue
      fi
    fi
    log "parent curriculum controller exited before replacing terminal experiment=$terminal_experiment"
    return 1
  done
}

blob_exists(){
  az storage blob exists --account-name "$account" --container-name "$container" \
    --name "$1" --auth-mode login --query exists -o tsv 2>/dev/null
}

checkpoint_source(){
  local step="$1" candidate receipt exists
  local matches=()
  for candidate in "checkpoint-$step" "checkpoint-round-$step"; do
    receipt="$source_root/$candidate/checkpoint_completion.json"
    exists="$(blob_exists "$receipt")" || return 1
    [[ "$exists" == true ]] && matches+=("$source_root/$candidate")
  done
  if [[ ${#matches[@]} -gt 1 ]]; then
    log "ambiguous checkpoint-$step source formats: ${matches[*]}"
    return 2
  fi
  [[ ${#matches[@]} -eq 1 ]] || return 1
  printf '%s\n' "${matches[0]}"
}

snapshot_one(){
  local step="$1" source="$2" destination receipt output
  destination="$snapshot_root/checkpoint-$step"
  receipt="$state/checkpoint-$step.completion.json"
  output="$state/checkpoint-$step.snapshot.json"
  az storage blob download --account-name "$account" --container-name "$container" \
    --name "$source/checkpoint_completion.json" --file "$receipt" \
    --auth-mode login --no-progress --overwrite -o none || return 1
  "$python_bin" - "$receipt" "$step" <<'PY' || return 1
import json,sys
x=json.load(open(sys.argv[1])); step=int(sys.argv[2])
position=x.get('step',x.get('round',-1))
assert x.get('status')=='complete' and int(position)==step,x
assert set(x.get('sha256') or {})=={'adapter_model','recurrent_state'},x
PY
  "$python_bin" scripts/snapshot_state8_checkpoint_blob.py \
    --source-prefix "$source" --destination-prefix "$destination" \
    --receipt "$receipt" --output "$output" >> "$state/controller.log" 2>&1 \
    || return 1
  touch "$state/checkpoint-$step.snapshot_complete"
  log "checkpoint-$step receipt-last snapshot complete"
}

while :; do
  heartbeat
  complete=0
  for step in "${milestones[@]}"; do
    if [[ -f "$state/checkpoint-$step.snapshot_complete" ]]; then
      complete=$((complete + 1)); continue
    fi
    source="$(checkpoint_source "$step")" || continue
    if snapshot_one "$step" "$source"; then
      # Count a milestone captured in this iteration immediately.  Otherwise
      # the final receipt can be snapshotted successfully and the watcher can
      # still fall through to a now-terminal parent job with a stale count.
      complete=$((complete + 1))
    else
      rm -f "$state/checkpoint-$step.completion.json.partial" \
        "$state/checkpoint-$step.snapshot.json.partial"
      log "checkpoint-$step exists but snapshot is not complete; retrying"
    fi
  done
  [[ "$complete" -eq "${#milestones[@]}" ]] && {
    touch "$state/complete"; heartbeat; log "all curriculum milestones preserved"; exit 0;
  }

  experiment="$(<"$experiment_file")"
  status="$(timeout 90 amlt --json-tables status "$experiment" --hide-urls \
    --force-status-update 2>/dev/null)" || { sleep_unlocked 20; continue; }
  [[ -n "$status" && "$status" != '[]' ]] || { sleep_unlocked 20; continue; }
  printf '%s\n' "$status" > "$state/latest_status.json"
  phase="$(STATUS="$status" "$python_bin" - <<'PY'
import json,os
states={str(row.get('status') or '').casefold() for row in json.loads(os.environ['STATUS'])}
if states & {'preparing','pre-queued','queued','running','resuming','paused'}: print('active')
elif states=={'pass'}: print('pass')
else: print('terminal')
PY
)" || phase=unknown
  if [[ "$phase" == terminal ]]; then
    timeout 180 amlt logs view -n 3000 "$experiment" ":$job" \
      > "$state/$experiment.failure.log" 2>&1 || true
    if follow_parent_recovery "$experiment"; then
      continue
    fi
    touch "$state/needs_attention"
    log "training became terminal before all milestones were preserved and no parent recovery remained"
    exit 1
  fi
  sleep_unlocked 20
done
