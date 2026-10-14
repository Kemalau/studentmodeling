#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
primary="$project/results/state8_qualified_promotion_eval_20260805"
balanced="$project/results/state8_qualified_balanced_promotion_eval_20260805"
highkd="$project/results/state8_qualified_highkd_promotion_eval_20260805"
state="$project/results/state8_qualified_promotion_selection_20260805"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python

mkdir -p "$state"
cd "$project" || exit 1
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"
trap 'rm -f "$state/controller.active.pid"' EXIT
rm -f "$state/needs_attention" "$state/complete" "$state/scientific_gate_pass" "$state/scientific_gate_failed"
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
fail(){ printf '%s\n' "$*" >> "$state/controller.log"; touch "$state/needs_attention"; exit 1; }

wait_upstream(){
  local upstream="$1"
  while [[ ! -f "$upstream/complete" ]]; do
    heartbeat
    [[ ! -f "$upstream/needs_attention" ]] || fail "promotion upstream needs attention: $upstream"
    sleep 60
  done
}
gate_passes(){
  "$python_bin" - "$1" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
raise SystemExit(0 if x.get('pass') is True else 1)
PY
}

# Preserve the preregistered priority while removing hedges from the critical
# path.  A complete primary joint gate selects immediately; high-KD is needed
# only after primary failure, and balanced only after both higher-priority
# arms fail.  Unselected arms keep running for the controlled paper evidence.
wait_upstream "$primary"
selector_args=(--primary "$primary/promotion_gate.json")
if ! gate_passes "$primary/promotion_gate.json"; then
  wait_upstream "$highkd"
  selector_args+=(--highkd "$highkd/promotion_gate.json")
  if ! gate_passes "$highkd/promotion_gate.json"; then
    wait_upstream "$balanced"
    selector_args+=(--balanced "$balanced/promotion_gate.json")
  fi
fi
"$python_bin" scripts/select_state8_qualified_promotion.py \
  "${selector_args[@]}" \
  --output "$state/selection.json" >> "$state/controller.log" 2>&1
rc=$?
if (( rc == 0 )); then touch "$state/scientific_gate_pass"
elif (( rc == 2 )); then touch "$state/scientific_gate_failed"
else fail "promotion selector failed operationally rc=$rc"; fi
touch "$state/complete"
heartbeat
