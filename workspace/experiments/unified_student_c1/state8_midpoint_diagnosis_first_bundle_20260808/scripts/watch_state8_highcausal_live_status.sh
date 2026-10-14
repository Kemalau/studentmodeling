#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
state="$project/results/state8_highcausal_pipeline_status"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python

mkdir -p "$state"
cd "$project" || exit 1
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"
trap 'rm -f "$state/controller.active.pid"' EXIT
rm -f "$state/needs_attention"

while true; do
  date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"
  if ! "$python_bin" scripts/summarize_state8_highcausal_pipeline.py \
      --results results --output results/state8_highcausal_pipeline_live_status.json \
      > "$state/latest.json.tmp" 2> "$state/latest.error.tmp"; then
    mv "$state/latest.error.tmp" "$state/latest.error"
    touch "$state/needs_attention"
    exit 1
  fi
  mv "$state/latest.json.tmp" "$state/latest.json"
  rm -f "$state/latest.error.tmp"
  sleep 60
done
