#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
export AZURE_CONFIG_DIR="${AZURE_CONFIG_DIR:-/home/v-yuejiali/.azure-studentmodeling-feeds}"
pipeline="$project_dir/results/layerwise_unified_final_pipeline_20260731"
state_dir="$pipeline/checkpoint750_formal_full_feeds"
local_root="$pipeline/checkpoint750_local_audit/formal_full"
config="$project_dir/amlt/feeds_layerwise_checkpoint750_formal_full_20260802.yaml"
active_file="$state_dir/active_experiment.txt"
jobs_file="$state_dir/active_jobs.txt"
experiment="layerwise-checkpoint750-formal-full-20260802"
[[ -s "$active_file" ]] && experiment="$(<"$active_file")"
expected_jobs=3
[[ -s "$jobs_file" ]] && expected_jobs="$(wc -l < "$jobs_file")"
recovery_wave=0
[[ "$experiment" =~ -recovery([0-9]+)-20260802$ ]] && recovery_wave="${BASH_REMATCH[1]}"
mkdir -p "$state_dir/preserved" "$local_root"
cd "$project_dir"
exec 9>"$state_dir/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT
rm -f "$state_dir/needs_attention"

heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state_dir/needs_attention"; exit 1; }

preserve_passed(){
  local exp="$1"
  amlt results download "$exp" --status pass \
    --include '*.json' --include '*.jsonl' --include '*.txt' \
    --output "$state_dir/preserved" >> "$state_dir/controller.log" 2>&1 || true
}

while true; do
  heartbeat
  status="$(amlt status "$experiment" 2>&1)"; rc=$?
  printf '[%s] experiment=%s rc=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$experiment" "$rc" "$status" >> "$state_dir/status.log"
  if grep -Eqi 'No authentication method succeeded|refreshing your credentials|connection|timed out|permission|TooManyRequests|InternalServerError|ServiceUnavailable' <<< "$status"; then
    sleep 120; continue
  fi
  if grep -Eqi 'running|queued|preparing|paused|resuming|pre-queued' <<< "$status"; then
    sleep 120; continue
  fi
  if python - "$status" "$expected_jobs" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1],int(sys.argv[2])) else 1)
PY
  then
    preserve_passed "$experiment"
    break
  fi
  preserve_passed "$experiment"
  mapfile -t failed_jobs < <(python - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import terminal_nonpass_job_names
print(*terminal_nonpass_job_names(sys.argv[1]),sep='\n')
PY
  )
  (( ${#failed_jobs[@]} > 0 )) || fail "terminal remote full evaluation has no failed sibling"
  (( recovery_wave < 6 )) || fail "remote full evaluation exhausted six recovery waves"
  recovery_wave=$((recovery_wave+1))
  next="layerwise-checkpoint750-formal-full-recovery${recovery_wave}-20260802"
  selectors=(); for job in "${failed_jobs[@]}"; do selectors+=(":$job"); done
  exec 8>"$project_dir/results/feeds_submission_global.lock"; flock 8
  amlt run "$config" "${selectors[@]}" --dump -t Feeds \
    > "$state_dir/${next}.preflight.dump.yaml" \
    2> "$state_dir/${next}.preflight.stderr" \
    || { flock -u 8; fail "remote full recovery dump failed"; }
  python scripts/validate_layerwise_checkpoint750_formal_full_amlt.py \
    --config "$state_dir/${next}.preflight.dump.yaml" --jobs "${failed_jobs[@]}" \
    >> "$state_dir/controller.log" 2>&1 \
    || { flock -u 8; fail "remote full recovery preflight failed"; }
  output="$(amlt run "$config" "${selectors[@]}" "$next" -t Feeds -y \
    --description "Read-only round-750 full-support evaluation sibling recovery=${recovery_wave}; jobs=${failed_jobs[*]}" 2>&1)"; submit_rc=$?
  printf '[%s] rc=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$submit_rc" "$output" \
    > "$state_dir/${next}.submission.txt"
  flock -u 8
  (( submit_rc == 0 )) || fail "remote full recovery submission failed"
  experiment="$next"; expected_jobs="${#failed_jobs[@]}"
  printf '%s\n' "$experiment" > "$active_file"
  printf '%s\n' "${failed_jobs[@]}" > "$jobs_file"
  log "monitoring remote full recovery=$recovery_wave jobs=${failed_jobs[*]}"
done

for group in t2 t3 t5; do
  mapfile -t metrics < <(find "$state_dir/preserved" -type f -path "*/eval/$group/metrics.json" | sort)
  [[ ${#metrics[@]} -eq 1 ]] || fail "expected one preserved full $group metrics file, found ${#metrics[@]}"
  source_dir="$(dirname "${metrics[0]}")"
  test -s "$source_dir/predictions.jsonl" || fail "$group unredacted predictions missing"
  test -s "$source_dir/predictions_redacted.jsonl" || fail "$group redacted predictions missing"
  test -s "$source_dir/scope.txt" || fail "$group scope receipt missing"
  grep -qx 'raw_history_visible_to_decoder=false' "$source_dir/scope.txt" \
    || fail "$group decoder closure missing"
  grep -qx 'persistent_history_kv_cache=false' "$source_dir/scope.txt" \
    || fail "$group KV closure missing"
  temporary="$local_root/.${group}.remote.partial"
  rm -rf "$temporary"; mkdir -p "$temporary"
  cp -a "$source_dir/." "$temporary/"
  rm -rf "$local_root/$group"
  mv "$temporary" "$local_root/$group"
done
touch "$state_dir/complete"
heartbeat
log "round-750 remote full T2/T3/T5 evaluation installed locally"
