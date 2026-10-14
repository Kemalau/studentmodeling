#!/usr/bin/env bash
set -euo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
state="$project/results/state8_large_evaluation_live_status_20260813"
poll_seconds="${STATE8_LARGE_EVAL_STATUS_POLL_SECONDS:-60}"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
mkdir -p "$state"; cd "$project"
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"
trap 'rm -f "$state/controller.active.pid"' EXIT

while true; do
  date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"
  "$python_bin" scripts/summarize_state8_large_evaluation_live_status.py \
    --project "$project" --output "$state/status.json" \
    > "$state/status.log" 2>&1 || true
  if "$python_bin" - "$state/status.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); raise SystemExit(0 if x.get('evaluation_flow_complete') else 1)
PY
  then
    touch "$state/complete"
    exit 0
  fi
  sleep "$poll_seconds"
done
