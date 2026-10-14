#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
model="$project/results/state8_strong1_widen8_bridge_20260808/complete_pipeline"
c3="$project/results/c3_exposure_matched_world2_20260809"
state="$project/results/complete_unified_objective_20260809"
model="${OBJECTIVE_MODEL_DIR:-$model}"
c3="${OBJECTIVE_C3_DIR:-$c3}"
state="${OBJECTIVE_STATE_DIR:-$state}"
final_eval_dir="${OBJECTIVE_FINAL_EVAL_DIR:-$model/final_eval}"
final_fit_dir="${OBJECTIVE_FINAL_FIT_DIR:-$model/final_fit}"
selection="${OBJECTIVE_SELECTION:-$model/selection.json}"
final_gate="${OBJECTIVE_FINAL_GATE:-$final_eval_dir/final_gate.json}"
c3_summary="${OBJECTIVE_C3_SUMMARY:-$c3/summary.json}"
model_attention_dir="${OBJECTIVE_MODEL_ATTENTION_DIR:-$model}"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python

mkdir -p "$state"
cd "$project" || exit 1
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
sleep_unlocked(){ sleep "$1" 9>&-; }
echo $$ > "$state/controller.active.pid"
cleanup(){ rm -f "$state/controller.active.pid"; }
trap cleanup EXIT
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }

while [[ ! -f "$final_eval_dir/complete" || ! -f "$c3/complete" ]]; do
  heartbeat
  if [[ -f "$model_attention_dir/needs_attention" || -f "$final_eval_dir/needs_attention" \
        || -f "$final_fit_dir/needs_attention" || -f "$c3/needs_attention" ]]; then
    fail "downstream model or C3 controller requires attention"
  fi
  sleep_unlocked 60
done

for path in "$selection" "$final_gate" "$c3_summary"; do
  [[ -s "$path" ]] || fail "missing terminal evidence path=$path"
done
mapfile -t publish < <(find "$final_fit_dir/download" -type f \
  -name canonical_publish_receipt.json)
[[ ${#publish[@]} -eq 1 ]] \
  || fail "canonical publish receipt cardinality drift observed=${#publish[@]}"

set +e
"$python_bin" scripts/validate_complete_unified_objective.py \
  --selection "$selection" --final-gate "$final_gate" \
  --publish-receipt "${publish[0]}" --c3-summary "$c3_summary" \
  --output "$state/final_objective.json" > "$state/validation.log" 2>&1
rc=$?
set -e
if (( rc == 0 )); then
  touch "$state/scientific_gate_pass"
elif (( rc == 2 )); then
  touch "$state/scientific_gate_failed"
else
  fail "top-level objective validator crashed rc=$rc"
fi
touch "$state/complete"; heartbeat
log "complete unified objective reached terminal audited state rc=$rc"
