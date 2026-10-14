#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
state="$project/results/taskonly_qwen_formal_eval_20260813/unified_t4_paired_rescore"
experiment=studentmodeling-unified-t4-paired-rescore-20260830
job=qwen3-32b-unified-t4-paired-rescore-seed42
mkdir -p "$state"; cd "$project" || exit 1
exec 9>"$state/watcher.lock"; flock -n 9 || exit 0
trap 'rm -f "$state/watcher.pid"' EXIT
echo $$ > "$state/watcher.pid"

while true; do
  date -u +%Y-%m-%dT%H:%M:%SZ > "$state/heartbeat"
  source scripts/use_studentmodeling_feeds_auth.sh
  status="$(timeout 90 amlt --json-tables status "$experiment" ":$job" --no-calculate-result-size --force-status-update 2>/dev/null)" || { sleep 60; continue; }
  printf '%s\n' "$status" > "$state/latest_status.json"
  job_status="$(STATUS="$status" python - <<'PY'
import json,os
x=json.loads(os.environ['STATUS']); print(str(x[0].get('status') or '').casefold() if len(x)==1 else 'unknown')
PY
)"
  case "$job_status" in
    pass) break ;;
    preparing|pre-queued|queued|running|resuming|paused) sleep 60 ;;
    failed|killed|cancelled|canceled|stopped)
      timeout 180 amlt logs view -n 2000 "$experiment" ":$job" > "$state/failure.log" 2>&1 || true
      touch "$state/needs_attention"; exit 1 ;;
    *) sleep 60 ;;
  esac
done

rm -rf "$state/download"; mkdir -p "$state/download"
timeout --foreground 600 amlt results download "$experiment" ":$job" \
  --include '*.json' --include '*.jsonl' --output "$state/download" </dev/null \
  >> "$state/watcher.log" 2>&1 || { touch "$state/needs_attention"; exit 1; }
python scripts/finalize_unified_t4_paired_rescore.py \
  --rescore-dir "$state/download" --root "$project" \
  > "$state/finalize.log" 2>&1 || { touch "$state/needs_attention"; exit 1; }
touch "$state/complete"
