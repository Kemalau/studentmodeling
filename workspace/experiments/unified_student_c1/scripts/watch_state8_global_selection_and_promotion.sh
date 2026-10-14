#!/usr/bin/env bash
set -euo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
results="$project/results"
curriculum="$results/state8_curriculum_selection_20260812"
regular="$results/state8_curriculum_repair_branches_20260812"
proactive="$results/state8_curriculum_cp4000_proactive_repairs_20260812"
proactive_cp3000="$results/state8_curriculum_cp3000_proactive_repairs_20260812"
repair_checkpoints="$results/state8_curriculum_repair_checkpoint_selection_20260812"
control_robust="$results/state8_current_query_control_robust_repairs_20260812"
causal_teacher="$results/state8_causal_teacher_checkpoint_selection_20260812"
early_label="$results/state8_early_label_checkpoint_selection_20260813"
interleaved="$results/state8_interleaved_retention_selection_20260813"
state="$results/state8_curriculum_global_selection_20260812"
promotion="$results/state8_curriculum_promotion_20260812"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
mkdir -p "$state"; cd "$project"
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"
trap 'rm -f "$state/controller.active.pid"' EXIT
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
sleep_unlocked(){ sleep "$1" 9>&-; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }

# The base curriculum is immutable evidence.  The causal-teacher and
# early-label lineages are independent repairs under the same gates: promote
# as soon as either one passes, but if the first terminal lineage fails, keep
# waiting for the other.  This preserves every floor without waiting for a
# slower comparator after a complete model is already available.
while [[ ! -f "$curriculum/complete" ]]; do
  heartbeat
  [[ ! -f "$curriculum/needs_attention" ]] \
    || fail "source needs attention: $curriculum"
  sleep_unlocked 60
done
repair_sources=("$causal_teacher" "$early_label" "$interleaved")
while :; do
  heartbeat; repair_pass=0; repair_terminal=0
  for source in "${repair_sources[@]}"; do
    if [[ -f "$source/complete" ]]; then
      repair_terminal=$((repair_terminal+1))
      if [[ -s "$source/selection.json" ]] && "$python_bin" - "$source/selection.json" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1])).get('pass') is True else 2)
PY
      then repair_pass=1; fi
    elif [[ -f "$source/needs_attention" ]]; then
      repair_terminal=$((repair_terminal+1))
      marker="$state/excluded.$(basename "$source")"
      if [[ ! -f "$marker" ]]; then
        log "required repair source needs attention and will be excluded only if its sibling passes: $source"
        touch "$marker"
      fi
    fi
  done
  (( repair_pass == 1 || repair_terminal == ${#repair_sources[@]} )) && break
  sleep_unlocked 30
done

optional_sources=("$regular" "$proactive_cp3000" "$proactive" "$repair_checkpoints" "$control_robust")
for source in "${optional_sources[@]}"; do
  [[ ! -f "$source/needs_attention" ]] \
    || log "optional source needs attention and will be excluded: $source"
done

args=(--selection curriculum "$curriculum/selection.json")
[[ -s "$regular/selection.json" ]] \
  && args+=(--selection regular_repair "$regular/selection.json")
[[ -s "$proactive/selection.json" ]] \
  && args+=(--selection proactive_cp4000 "$proactive/selection.json")
[[ -s "$proactive_cp3000/selection.json" ]] \
  && args+=(--selection proactive_cp3000 "$proactive_cp3000/selection.json")
[[ -s "$repair_checkpoints/selection.json" ]] \
  && args+=(--selection repair_checkpoint_cp3000 "$repair_checkpoints/selection.json")
[[ -s "$control_robust/selection.json" ]] \
  && args+=(--selection current_query_control_robust "$control_robust/selection.json")
[[ -f "$causal_teacher/complete" && ! -f "$causal_teacher/needs_attention" \
   && -s "$causal_teacher/selection.json" ]] \
  && args+=(--selection causal_teacher_binding "$causal_teacher/selection.json")
[[ -f "$early_label/complete" && ! -f "$early_label/needs_attention" \
   && -s "$early_label/selection.json" ]] \
  && args+=(--selection early_label_binding "$early_label/selection.json")
[[ -f "$interleaved/complete" && ! -f "$interleaved/needs_attention" \
   && -s "$interleaved/selection.json" ]] \
  && args+=(--selection interleaved_retention "$interleaved/selection.json")
output="$state/selection.json"
set +e
"$python_bin" scripts/select_state8_global_candidate.py "${args[@]}" \
  --output "$output" > "$state/selection.log" 2>&1
rc=$?
set -e
if (( rc == 2 )); then
  mkdir -p "$promotion"
  touch "$promotion/diagnostic_gate_failed" "$promotion/complete"
  touch "$state/scientific_gate_failed" "$state/complete"; heartbeat
  log "all curriculum and repair lineages failed the shared global gate; propagated terminal failure to promotion"
  exit 0
elif (( rc != 0 )); then
  fail "global selector crashed rc=$rc"
fi
touch "$state/scientific_gate_pass" "$state/complete"; heartbeat
log "global candidate selected; starting frozen 150-pair answer-C4 confirmation"

variant="$("$python_bin" - "$output" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); print((x.get('selected') or {}).get('finalfit_variant',''))
PY
)"
case "$variant" in
  same_question_curriculum_promoted)
    gate_blob=studentmodeling/gates/state8_curriculum_v1/promotion.json ;;
  same_question_repair_promoted)
    gate_blob=studentmodeling/gates/state8_eedi_same_question_v1/promotion.json ;;
  same_question_repair_checkpoint_promoted)
    gate_blob=studentmodeling/gates/state8_eedi_same_question_v1/promotion.json ;;
  *) fail "global winner has unsupported final-fit variant: $variant" ;;
esac

export STATE8_PROMOTION_STATE_DIR="$promotion"
export STATE8_PROMOTION_SELECTION_FILE="$output"
export STATE8_PROMOTION_BASE_EXPERIMENT=state8-global-answer-c4-20260812
export STATE8_PROMOTION_GATE_BLOB="$gate_blob"
exec bash "$project/scripts/watch_state8_eedi_same_question_promotion.sh"
