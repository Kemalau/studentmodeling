#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
experiment="recurrent-teacher-cache-batch-benchmark-b1-b4-v3-recurrentdata-20260731"
state_dir="$project_dir/results/recurrent_final_pipeline_20260730/cache_batch_benchmark_v3"
mkdir -p "$state_dir"
cd "$project_dir"

exec 9>"$state_dir/controller.lock"
flock -n 9 || exit 0
echo "$$" > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT
rm -f "$state_dir/complete" "$state_dir/needs_attention"

while true; do
  date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"
  status="$(amlt status "$experiment" 2>&1)"
  printf '[%s]\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" >> "$state_dir/status.log"
  if grep -Eqi 'No authentication method succeeded|refreshing your credentials|management\.azure\.com|connection|timed out|permission' <<< "$status"; then
    sleep 120
    continue
  fi
  if grep -Eqi 'running|queued|preparing|paused|resuming' <<< "$status"; then
    sleep 120
    continue
  fi
  if python - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1], 2) else 1)
PY
  then
    break
  fi
  printf '%s\n' "$status" > "$state_dir/terminal_status.txt"
  amlt show "$experiment" > "$state_dir/show.txt" 2>&1 || true
  amlt logs view -n 1000 "$experiment" :cache-bench-b1-seed42 \
    > "$state_dir/b1.failure.log" 2>&1 || true
  amlt logs view -n 1000 "$experiment" :cache-bench-b4-seed42 \
    > "$state_dir/b4.failure.log" 2>&1 || true
  touch "$state_dir/needs_attention"
  exit 1
done

printf '%s\n' "$status" > "$state_dir/terminal_status.txt"
download="$state_dir/download"
mkdir -p "$download"
amlt results download "$experiment" --status pass \
  --include 'timing.json' \
  --include 'manifest.json' \
  --include 'integrity.json' \
  --include 'index.json' \
  --include 'shard-*.pt' \
  --output "$download" > "$state_dir/download.log" 2>&1 || {
    touch "$state_dir/needs_attention"
    exit 1
  }

mapfile -t b1_indexes < <(find "$download" -type f -path '*cache-bench-b1-seed42*/bench/index.json')
mapfile -t b4_indexes < <(find "$download" -type f -path '*cache-bench-b4-seed42*/bench/index.json')
if [[ "${#b1_indexes[@]}" -ne 1 || "${#b4_indexes[@]}" -ne 1 ]]; then
  find "$download" -type f | sort > "$state_dir/download_inventory.txt"
  touch "$state_dir/needs_attention"
  exit 1
fi
b1_root="$(dirname "${b1_indexes[0]}")"
b4_root="$(dirname "${b4_indexes[0]}")"
python scripts/compare_teacher_cache_equivalence.py \
  --left "$b1_root" \
  --right "$b4_root" \
  --tolerance 0.0009765625 \
  --output "$state_dir/equivalence.json" \
  > "$state_dir/equivalence.log" 2>&1 || {
    touch "$state_dir/needs_attention"
    exit 1
  }
python - "$b1_root/timing.json" "$b4_root/timing.json" "$state_dir/summary.json" <<'PY'
import json, sys
b1 = json.load(open(sys.argv[1], encoding='utf-8'))
b4 = json.load(open(sys.argv[2], encoding='utf-8'))
assert b1['batch_size'] == 1 and b4['batch_size'] == 4, (b1, b4)
result = {
    'status': 'complete',
    'b1': b1,
    'b4': b4,
    'speedup_b4_over_b1': b4['rows_per_second'] / b1['rows_per_second'],
}
with open(sys.argv[3], 'w', encoding='utf-8') as handle:
    json.dump(result, handle, indent=2, sort_keys=True)
    handle.write('\n')
PY
touch "$state_dir/complete"
