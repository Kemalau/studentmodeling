#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
pipeline="$project_dir/results/layerwise_unified_final_pipeline_20260731"
c4_dir="$pipeline/checkpoint1000_c4"
formal_dir="$pipeline/checkpoint1000_formal_screen"
t3_semantic_dir="$pipeline/checkpoint1000_t3_semantic_judge"
state_dir="$pipeline/checkpoint1000_decision"
auth_required="$project_dir/results/recurrent_unified_supervisor_20260731/auth_required"
mkdir -p "$state_dir"
cd "$project_dir"
exec 9>"$state_dir/controller.lock"
flock -n 9 || exit 0
echo $$ > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT
rm -f "$state_dir/needs_attention"

heartbeat() { date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"; }
fail() { log "ERROR: $*"; touch "$state_dir/needs_attention"; exit 1; }
attention_is_current() {
  local owner="$1" marker="$1/needs_attention" active_pid="$1/controller.active.pid"
  [[ -f "$marker" ]] || return 1
  if [[ -s "$active_pid" && "$active_pid" -nt "$marker" ]]; then
    local pid
    pid="$(<"$active_pid")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
      return 1
    fi
  fi
  if [[ -s "$owner/active_experiment.txt" \
        && "$owner/active_experiment.txt" -nt "$marker" ]]; then
    return 1
  fi
  return 0
}

while [[ ! -f "$c4_dir/complete" || ! -f "$formal_dir/complete" \
        || ! -f "$t3_semantic_dir/complete" ]]; do
  heartbeat
  if [[ -f "$auth_required" ]]; then sleep 60; continue; fi
  attention_is_current "$c4_dir" \
    && fail "round-1000 C4 owner needs attention"
  attention_is_current "$formal_dir" \
    && fail "round-1000 formal-screen owner needs attention"
  attention_is_current "$t3_semantic_dir" \
    && fail "round-1000 T3 semantic owner needs attention"
  sleep 60
done

python scripts/summarize_layerwise_checkpoint1000_decision.py \
  --c4 "$c4_dir/summary.json" --formal "$formal_dir/summary.json" \
  --t3-semantic "$t3_semantic_dir/t3_semantic_gate.json" \
  --output "$state_dir/decision.json" >> "$state_dir/controller.log" 2>&1 \
  || fail "round-1000 joint decision failed"
python - "$state_dir/decision.json" <<'PY' || fail "round-1000 decision receipt is invalid"
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
assert x['stage']=='layerwise_unified_round1000_joint_decision',x
assert x['checkpoint_round']==1000 and x['diagnostic_only'] is True,x
assert x['main_fit_must_continue'] is True,x
assert x['formal_final_evaluation_required'] is True,x
PY
if python - "$state_dir/decision.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
raise SystemExit(0 if x['rescue_preparation_required'] else 1)
PY
then touch "$state_dir/rescue_preparation_required"; else touch "$state_dir/healthy"; fi
touch "$state_dir/complete"
log "round-1000 joint diagnostic decision complete"
