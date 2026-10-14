#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
state_dir="$project_dir/results/recurrent_final_pipeline_20260730/cache_v2_16way"
experiment="recurrent-finalfit-cache-v2-merge-seed42-20260731"
mkdir -p "$state_dir/merge_download"
cd "$project_dir"
exec 9>"$state_dir/merge_controller.lock"
flock -n 9 || exit 0
echo "$$" > "$state_dir/merge_controller.active.pid"
trap 'rm -f "$state_dir/merge_controller.active.pid"' EXIT
# This controller is a targeted recovery after the previous local merge failed.
# Once the recovery owns the lock, the old terminal marker is stale; retaining
# it makes the final-fit controller incorrectly activate the slower legacy
# four-way fallback while this verified 16-way merge is still queued/running.
rm -f "$state_dir/needs_attention"

log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/merge_controller.log"; }
fail() { log "ERROR: $*"; touch "$state_dir/needs_attention"; exit 1; }

while true; do
  date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/merge_controller.heartbeat"
  status="$(amlt status "$experiment" 2>&1)"
  printf '[%s]\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" >> "$state_dir/merge_status.log"
  if grep -Eqi 'No authentication method succeeded|refreshing your credentials|management\.azure\.com|connection|timed out|permission' <<< "$status"; then
    sleep 60
    continue
  fi
  if grep -Eqi 'running|queued|preparing|paused|resuming|pre-queued' <<< "$status"; then
    sleep 60
    continue
  fi
  if python - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1], 1) else 1)
PY
  then
    break
  fi
  printf '%s\n' "$status" > "$state_dir/merge_terminal_status.txt"
  amlt show "$experiment" > "$state_dir/merge_show.txt" 2>&1 || true
  timeout 180s amlt logs view -n 1200 "$experiment" > "$state_dir/merge_failure.log" 2>&1 || true
  fail "remote 16-way cache merge did not pass"
done

rm -rf "$state_dir/merge_download"
mkdir -p "$state_dir/merge_download"
amlt results download "$experiment" --status pass \
  --include 'merge_receipt.json' --include 'final_verification.json' \
  --include 'manifest.json' --include 'integrity.json' \
  --output "$state_dir/merge_download" >> "$state_dir/merge_controller.log" 2>&1 \
  || fail "could not download remote merge receipts"
python - "$state_dir/merge_download" "$state_dir/final_verification.json" <<'PY' \
  || fail "downloaded remote merge receipts are invalid"
import json, pathlib, shutil, sys
root = pathlib.Path(sys.argv[1])
matches = {name: list(root.rglob(name)) for name in (
    'merge_receipt.json', 'final_verification.json', 'manifest.json', 'integrity.json'
)}
assert all(len(paths) == 1 for paths in matches.values()), matches
merge = json.load(open(matches['merge_receipt.json'][0], encoding='utf-8'))
verify = json.load(open(matches['final_verification.json'][0], encoding='utf-8'))
manifest = json.load(open(matches['manifest.json'][0], encoding='utf-8'))
integrity = json.load(open(matches['integrity.json'][0], encoding='utf-8'))
assert merge['status'] == 'complete' and merge['rows'] == 259845, merge
assert merge['partition_count'] == 16 and len(merge['partition_manifests']) == 16, merge
assert merge['source_sha256'] == '9961f746cb17e6c314e182d3453b6157022fb48b8de44d2e68bc8e87a1131ccd', merge
assert merge['batch_size'] == 1 and merge['attn_implementation'] == 'eager', merge
assert verify['status'] == 'verified' and verify['rows'] == 259845, verify
assert manifest == merge, (manifest, merge)
assert integrity['status'] == 'verified' and integrity['rows'] == 259845, integrity
shutil.copy2(matches['final_verification.json'][0], sys.argv[2])
PY
printf '%s\n' "$status" > "$state_dir/merge_terminal_status.txt"
rm -f "$state_dir/needs_attention"
touch "$state_dir/complete"
log "remote 16-way exact teacher cache merge verified; final-fit cache is ready"
