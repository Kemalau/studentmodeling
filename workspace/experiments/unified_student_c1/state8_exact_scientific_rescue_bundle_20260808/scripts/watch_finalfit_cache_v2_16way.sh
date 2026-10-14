#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
experiment="recurrent-finalfit-cache-v2-16way-seed42-20260731"
config="amlt/feeds_recurrent_finalfit_cache_v2_16way_20260731.yaml"
state_dir="$project_dir/results/recurrent_final_pipeline_20260730/cache_v2_16way"
cache_root="/mnt/amulet/studentmodeling/world_model_v1/final_fit/cache_full259845_top256_seed42_v2_16way"
mkdir -p "$state_dir"
cd "$project_dir"

exec 9>"$state_dir/controller.lock"
flock -n 9 || exit 0
echo "$$" > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT
rm -f "$state_dir/complete" "$state_dir/needs_attention"

fail() {
  printf '[%s] ERROR: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"
  touch "$state_dir/needs_attention"
  exit 1
}

experiment_class() {
  local name="$1" output rc
  output="$(amlt status --no-update-db "$name" 2>&1)"
  rc=$?
  python - "$rc" "$output" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]), sys.argv[2]))
PY
}

wait_quota() {
  local required="$1"
  while true; do
    quota="$(amlt --json-tables ti sing -t Feeds 2>&1)" || { sleep 120; continue; }
    if python - "$quota" "$required" <<'PY'
import sys
from scripts.feeds_quota_utils import parse_json_table, select_a100_80gb
row = select_a100_80gb(parse_json_table(sys.argv[1]))
quota = row['quotas_(used/limit)']
required = int(sys.argv[2])
available = int(quota['standard']['limit']) - int(quota['standard']['used'])
user_available = int(quota['user_max']['limit']) - int(quota['user_max'].get('used') or 0)
assert available >= required and user_available >= required
PY
    then return 0; fi
    sleep 120
  done
}

submit_recovery() {
  local recovery="$1"; shift
  local jobs=("$@") selectors=() job
  for job in "${jobs[@]}"; do selectors+=(":$job"); done
  local dump="$state_dir/${recovery}.dump.yaml"
  exec 8>"$project_dir/results/feeds_submission_global.lock"
  flock 8
  amlt run "$config" "${selectors[@]}" --dump -t Feeds > "$dump" 2>&1 \
    || { flock -u 8; fail "cache-v2 recovery dump failed"; }
  python scripts/validate_finalfit_cache_v2_amlt_config.py \
    --config "$dump" --jobs "${jobs[@]}" >> "$state_dir/controller.log" 2>&1 \
    || { flock -u 8; fail "cache-v2 recovery preflight failed"; }
  class="$(experiment_class "$recovery")" \
    || { flock -u 8; fail "cache-v2 recovery status is ambiguous"; }
  if [[ "$class" == missing ]]; then
    amlt run "$config" "${selectors[@]}" "$recovery" -t Feeds -y \
      --description "Resume exact final-fit teacher-cache failed ranks from atomic remote shards: ${jobs[*]}" \
      > "$state_dir/${recovery}.submission.txt" 2>&1 \
      || { flock -u 8; fail "cache-v2 recovery submission failed"; }
  elif [[ "$class" != exists ]]; then
    flock -u 8
    fail "cache-v2 recovery pre-existing state is ambiguous"
  fi
  flock -u 8
}

active_experiment="$experiment"
expected_jobs=16
recovery_wave=0
while true; do
  date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"
  status="$(amlt status "$active_experiment" 2>&1)"
  printf '[%s]\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" >> "$state_dir/status.log"
  if grep -Eqi 'No authentication method succeeded|refreshing your credentials|management\.azure\.com|connection|timed out|permission' <<< "$status"; then
    sleep 120
    continue
  fi
  if grep -Eqi 'running|queued|preparing|paused|resuming|pre-queued' <<< "$status"; then
    sleep 120
    continue
  fi
  if python - "$status" "$expected_jobs" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1], int(sys.argv[2])) else 1)
PY
  then
    break
  fi
  mapfile -t failed_jobs < <(python - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import terminal_nonpass_job_names
print(*terminal_nonpass_job_names(sys.argv[1]), sep='\n')
PY
  )
  printf '%s\n' "$status" > "$state_dir/${active_experiment}.terminal_status.txt"
  amlt show "$active_experiment" > "$state_dir/${active_experiment}.show.txt" 2>&1 || true
  amlt logs view -n 1500 "$active_experiment" > "$state_dir/${active_experiment}.failure.log" 2>&1 || true
  (( ${#failed_jobs[@]} > 0 )) || fail "terminal cache-v2 state has no recoverable failed ranks"
  (( recovery_wave < 3 )) || fail "cache-v2 failed ranks exhausted three recovery waves"
  recovery_wave=$((recovery_wave + 1))
  wait_quota "${#failed_jobs[@]}"
  active_experiment="recurrent-finalfit-cache-v2-recovery${recovery_wave}-seed42-20260731"
  expected_jobs="${#failed_jobs[@]}"
  submit_recovery "$active_experiment" "${failed_jobs[@]}"
  printf '[%s] recovery wave %s submitted for %s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$recovery_wave" "${failed_jobs[*]}" \
    | tee -a "$state_dir/controller.log"
done

printf '%s\n' "$status" > "$state_dir/terminal_status.txt"
python scripts/merge_teacher_cache_partitions.py \
  --parts-root "$cache_root" \
  --output-dir "$cache_root" \
  --expected-rows 259845 \
  > "$state_dir/merge_receipt.json" 2> "$state_dir/merge_error.log" \
  || fail "16-way cache merge/integrity verification failed"
python - "$cache_root/manifest.json" "$cache_root/integrity.json" <<'PY' \
  > "$state_dir/final_verification.json" || fail "merged cache-v2 receipt is invalid"
import json, sys
manifest = json.load(open(sys.argv[1], encoding='utf-8'))
integrity = json.load(open(sys.argv[2], encoding='utf-8'))
assert manifest['status'] == 'complete' and manifest['rows'] == 259845, manifest
assert manifest['partition_count'] == 16, manifest
assert manifest['source_sha256'] == '9961f746cb17e6c314e182d3453b6157022fb48b8de44d2e68bc8e87a1131ccd', manifest
assert manifest['order_seed'] == 42 and manifest['top_k'] == 256, manifest
assert manifest['shard_size'] == 512 and manifest['state_token_alias'] == '<|vision_pad|>', manifest
assert manifest['batch_size'] == 1 and manifest['attn_implementation'] == 'eager', manifest
assert len(manifest['partition_manifests']) == 16, manifest
assert integrity['status'] == 'verified' and integrity['rows'] == 259845, integrity
print(json.dumps({'manifest': manifest, 'integrity_summary': {
    'status': integrity['status'], 'rows': integrity['rows'],
    'shards': len(integrity['shards'])}}, indent=2, sort_keys=True))
PY
touch "$state_dir/complete"
printf '[%s] 16-way exact final-fit teacher cache verified and merged\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  | tee -a "$state_dir/controller.log"
