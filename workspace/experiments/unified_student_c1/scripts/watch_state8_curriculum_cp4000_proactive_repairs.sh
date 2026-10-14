#!/usr/bin/env bash
set -euo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
results="$project/results"
checkpoint_step="${CHECKPOINT_STEP:-4000}"
seed_base="${SEED_BASE:-5201}"
case "$checkpoint_step" in 3000|4000) ;; *) exit 2 ;; esac
[[ "$seed_base" =~ ^[1-9][0-9]*$ ]] || exit 2
selection="${SELECTION_DIR:-$results/state8_curriculum_cp4000_proactive_selection_20260812}"
state="${REPAIR_STATE_DIR:-$results/state8_curriculum_cp4000_proactive_repairs_20260812}"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
mkdir -p "$selection" "$state"; cd "$project"
exec 9>"$selection/controller.lock"; flock -n 9 || exit 0
echo $$ > "$selection/controller.active.pid"
trap 'rm -f "$selection/controller.active.pid"' EXIT
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$selection/controller.heartbeat"; }
sleep_unlocked(){ sleep "$1" 9>&-; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$selection/controller.log"; }

receipts=(
  "$results/state8_diagnosis_first_eedi_curriculum_20260812/milestones/checkpoint-${checkpoint_step}.completion.json"
  "$results/state8_diagnosis_first_eedi_current_query_curriculum_20260812/milestones/checkpoint-${checkpoint_step}.completion.json"
)
while :; do
  heartbeat; ready=1
  for receipt in "${receipts[@]}"; do [[ -s "$receipt" ]] || ready=0; done
  (( ready )) && break
  sleep_unlocked 30
done

"$python_bin" scripts/materialize_state8_curriculum_cp4000_proactive_plan.py \
  --results-root "$results" --checkpoint-step "$checkpoint_step" \
  --seed-base "$seed_base" --output "$selection/repair_plan.json" \
  > "$selection/plan.log" 2>&1
touch "$selection/repair_required" "$selection/complete"
heartbeat
log "checkpoint-$checkpoint_step receipts complete; launching three proactive state-only repairs"

export STATE8_REPAIR_SELECTION_DIR="$selection"
export STATE8_REPAIR_STATE_DIR="$state"
export STATE8_REPAIR_BASE_EXPERIMENT="${REPAIR_BASE_EXPERIMENT:-state8-curriculum-cp4000-proactive-repairs-20260812}"
exec bash "$project/scripts/watch_state8_eedi_same_question_repair_branches.sh"
