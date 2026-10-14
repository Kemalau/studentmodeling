#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
source "$project/scripts/use_studentmodeling_feeds_auth.sh"
state="$project/results/state8_causal_teacher_binding_milestones_20260812"
shared="$project/results/state8_curriculum_repair_milestones_20260812"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
account=csnewsandfeeds4150361735
container=amulet
snapshot_root=studentmodeling/checkpoint_snapshots/state8_curriculum_repairs_20260812
milestones=(20 100 240 500)
specs=(8201:causal_teacher_low 8202:causal_teacher_high)

mkdir -p "$state" "$shared"; cd "$project" || exit 1
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"
trap 'rm -f "$state/controller.active.pid"' EXIT
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
sleep_unlocked(){ sleep "$1" 9>&-; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }

blob_exists(){
  az storage blob exists --auth-mode login --account-name "$account" \
    --container-name "$container" --name "$1" --query exists -o tsv 2>/dev/null
}

snapshot_one(){
  local seed="$1" branch="$2" step="$3" source destination receipt output
  source="studentmodeling/checkpoints/state8_stateonly_repair_seed${seed}_taskanchor_source_anchor_full_mse1_eedi_same_question_v1_${branch}_seed${seed}/checkpoint-${step}"
  destination="$snapshot_root/seed${seed}_${branch}/checkpoint-${step}"
  receipt="$shared/seed${seed}.${branch}.checkpoint-${step}.completion.json"
  output="$shared/seed${seed}.${branch}.checkpoint-${step}.snapshot.json"
  [[ "$(blob_exists "$source/checkpoint_completion.json")" == true ]] || return 1
  az storage blob download --auth-mode login --account-name "$account" \
    --container-name "$container" --name "$source/checkpoint_completion.json" \
    --file "$receipt" --overwrite --no-progress -o none || return 1
  "$python_bin" - "$receipt" "$step" <<'PY' || return 1
import json,sys
x=json.load(open(sys.argv[1])); step=int(sys.argv[2])
assert x.get('status')=='complete' and int(x.get('step',-1))==step,x
assert set(x.get('sha256') or {})=={'adapter_model','recurrent_state'},x
PY
  "$python_bin" scripts/snapshot_state8_checkpoint_blob.py \
    --source-prefix "$source" --destination-prefix "$destination" \
    --receipt "$receipt" --output "$output" >> "$state/controller.log" 2>&1 \
    || return 1
  touch "$shared/seed${seed}.${branch}.checkpoint-${step}.snapshot_complete"
  log "captured causal-teacher checkpoint seed=$seed branch=$branch step=$step"
}

while :; do
  heartbeat; captured=0
  for spec in "${specs[@]}"; do
    seed="${spec%%:*}"; branch="${spec#*:}"
    for step in "${milestones[@]}"; do
      marker="$shared/seed${seed}.${branch}.checkpoint-${step}.snapshot_complete"
      if [[ -f "$marker" ]]; then captured=$((captured+1)); continue; fi
      snapshot_one "$seed" "$branch" "$step" || true
    done
  done
  if (( captured == ${#specs[@]} * ${#milestones[@]} )); then
    touch "$state/complete"; heartbeat
    log "all causal-teacher milestones captured receipt-last"
    exit 0
  fi
  sleep_unlocked 10
done
