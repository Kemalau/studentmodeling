#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
export AZURE_CONFIG_DIR="${AZURE_CONFIG_DIR:-/home/v-yuejiali/.azure-studentmodeling-feeds}"
fit_dir="$project_dir/results/layerwise_unified_final_pipeline_20260731/final_fit"
auth_required="$project_dir/results/recurrent_unified_supervisor_20260731/auth_required"
checkpoint_round="${CHECKPOINT_ROUND:-1000}"
[[ "$checkpoint_round" == 500 || "$checkpoint_round" == 1000 ]] || exit 2
state_dir="$project_dir/results/layerwise_unified_final_pipeline_20260731/checkpoint${checkpoint_round}_c4"
if [[ "$checkpoint_round" == 1000 ]]; then
  config="$project_dir/amlt/feeds_layerwise_finalfit_checkpoint250_c4_20260801.yaml"
else
  config="$project_dir/amlt/feeds_layerwise_finalfit_checkpoint500_c4_20260801.yaml"
fi
experiment="layerwise-finalfit-checkpoint${checkpoint_round}-c4-seed42-20260801"
source_prefix="studentmodeling/checkpoints/layerwise_unified_world_model_seed42_full259845_v1_20260731/checkpoint-round-${checkpoint_round}"
snapshot_root="studentmodeling/diagnostics/layerwise_unified_finalfit_seed42_20260801_snapshot"
snapshot_prefix="$snapshot_root/checkpoint-round-${checkpoint_round}"
account="csnewsandfeeds4150361735"
container="amulet"
mkdir -p "$state_dir"
cd "$project_dir"
exec 9>"$state_dir/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT
rm -f "$state_dir/needs_attention"

heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state_dir/needs_attention"; exit 1; }
attention_is_current(){
  local owner="$1" marker="$1/needs_attention" active_pid="$1/controller.active.pid"
  [[ -f "$marker" ]] || return 1

  # A supervisor restart or a resilient recovery can leave a failure marker
  # from the previous controller/experiment for a few seconds.  Do not let
  # that stale marker poison the immutable checkpoint branch.  A marker
  # written by the currently live controller is newer than its pid receipt and
  # therefore remains authoritative.
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
blob_exists(){
  az storage blob exists --auth-mode login --account-name "$account" \
    --container-name "$container" --name "$1" --query exists -o tsv 2>/dev/null
}

copy_blob_verified(){
  local file="$1" source="$source_prefix/$file" destination="$snapshot_prefix/$file"
  local source_info source_size source_etag destination_info destination_size copy_status
  source_info="$(az storage blob show --auth-mode login --account-name "$account" \
    --container-name "$container" --name "$source" \
    --query '{size:properties.contentLength,etag:properties.etag}' -o json 2>/dev/null)" \
    || fail "snapshot source metadata unavailable: $file"
  mapfile -t source_fields < <(python - "$source_info" <<'PY'
import json, sys
value = json.loads(sys.argv[1])
print(value.get('size', ''))
print(value.get('etag', ''))
PY
)
  source_size="${source_fields[0]:-}"
  source_etag="${source_fields[1]:-}"
  [[ "$source_size" =~ ^[1-9][0-9]*$ && -n "$source_etag" ]] \
    || fail "snapshot source metadata invalid: $file"

  destination_info="$(az storage blob show --auth-mode login --account-name "$account" \
    --container-name "$container" --name "$destination" \
    --query '{size:properties.contentLength,status:properties.copy.status}' -o json 2>/dev/null || true)"
  mapfile -t destination_fields < <(python - "$destination_info" <<'PY'
import json, sys
try: value = json.loads(sys.argv[1])
except Exception: value = {}
print(value.get('size', ''))
print(value.get('status') or '')
PY
)
  destination_size="${destination_fields[0]:-}"
  copy_status="${destination_fields[1]:-}"
  if [[ "$destination_size" == "$source_size" \
        && ( "$copy_status" == success || -z "$copy_status" || "$copy_status" == None ) ]]; then
    return 0
  fi

  if [[ "$copy_status" != pending ]]; then
    # Large block blobs cannot use synchronous server-side Copy Blob. Start an
    # asynchronous service-side copy pinned to the source ETag, then poll its
    # status. Keep command output suppressed because Azure may return a
    # temporary source URL containing delegated query parameters.
    az storage blob copy start --auth-mode login --account-name "$account" \
      --destination-container "$container" --destination-blob "$destination" \
      --source-account-name "$account" --source-container "$container" \
      --source-blob "$source" --source-if-match "$source_etag" -o none \
      >/dev/null 2>&1 || fail "atomic checkpoint snapshot copy start failed: $file"
  fi

  local deadline=$(( $(date +%s) + 7200 ))
  while true; do
    heartbeat
    destination_info="$(az storage blob show --auth-mode login --account-name "$account" \
      --container-name "$container" --name "$destination" \
      --query '{size:properties.contentLength,status:properties.copy.status}' -o json 2>/dev/null || true)"
    mapfile -t destination_fields < <(python - "$destination_info" <<'PY'
import json, sys
try: value = json.loads(sys.argv[1])
except Exception: value = {}
print(value.get('size', ''))
print(value.get('status') or '')
PY
)
    destination_size="${destination_fields[0]:-}"
    copy_status="${destination_fields[1]:-}"
    if [[ "$copy_status" == success ]]; then
      [[ "$destination_size" == "$source_size" ]] \
        || fail "snapshot size mismatch after copy: $file"
      return 0
    fi
    [[ "$copy_status" != failed && "$copy_status" != aborted ]] \
      || fail "atomic checkpoint snapshot copy ended status=$copy_status: $file"
    (( $(date +%s) < deadline )) \
      || fail "atomic checkpoint snapshot copy timed out: $file"
    sleep 10
  done
}

while true; do
  heartbeat
  if [[ -f "$auth_required" ]]; then sleep 60; continue; fi
  ready=true
  for file in adapter/adapter_config.json adapter/adapter_model.safetensors recurrent_state.pt checkpoint_completion.json; do
    [[ "$(blob_exists "$source_prefix/$file")" == true ]] || ready=false
  done
  [[ "$ready" == true ]] && break
  attention_is_current "$fit_dir" \
    && fail "formal final fit failed before round-${checkpoint_round} checkpoint became available"
  sleep 60
done
log "atomic layerwise round-${checkpoint_round} checkpoint is available"

while ! az account get-access-token \
    --subscription b6dc87f3-c479-49c8-8cb5-7896da3ff895 \
    --query expiresOn -o tsv >/dev/null 2>&1; do
  heartbeat
  sleep 60
done

# Freeze the selected early checkpoint with service-side synchronous copies
# before queueing the read-only probes, so allocation delay cannot race later
# checkpoint cleanup in a recovery attempt.
if [[ ! -f "$state_dir/snapshot_complete" ]]; then
  for file in adapter/adapter_config.json adapter/adapter_model.safetensors recurrent_state.pt checkpoint_completion.json; do
    copy_blob_verified "$file"
  done
  az storage blob download --auth-mode login --account-name "$account" \
    --container-name "$container" --name "$snapshot_prefix/checkpoint_completion.json" \
    --file "$state_dir/snapshot_checkpoint_completion.json" --overwrite -o none \
    || fail "snapshot receipt download failed"
  python - "$state_dir/snapshot_checkpoint_completion.json" "$account" "$container" \
    "$snapshot_prefix" "$checkpoint_round" <<'PY' || fail "snapshot receipt validation failed"
import json, subprocess, sys
receipt = json.load(open(sys.argv[1], encoding='utf-8'))
expected_round = int(sys.argv[5])
assert receipt.get('status') == 'complete' and receipt.get('round') == expected_round, receipt
assert receipt.get('world_size') == 16, receipt
source_root = f'studentmodeling/checkpoints/layerwise_unified_world_model_seed42_full259845_v1_20260731/checkpoint-round-{expected_round}'
for filename in ('adapter/adapter_config.json', 'adapter/adapter_model.safetensors', 'recurrent_state.pt', 'checkpoint_completion.json'):
    def show(name):
        output = subprocess.check_output([
            'az', 'storage', 'blob', 'show', '--auth-mode', 'login',
            '--account-name', sys.argv[2], '--container-name', sys.argv[3],
            '--name', name, '--output', 'json',
        ], text=True)
        return json.loads(output)
    source = show(f'{source_root}/{filename}')
    snapshot = show(f'{sys.argv[4]}/{filename}')
    source_size = int(source['properties']['contentLength'])
    snapshot_size = int(snapshot['properties']['contentLength'])
    assert source_size > 0 and snapshot_size == source_size, (filename, source_size, snapshot_size)
    copy_status = (snapshot['properties'].get('copy') or {}).get('status')
    assert copy_status in (None, 'success'), (filename, copy_status)
PY
  touch "$state_dir/snapshot_complete"
fi
log "immutable layerwise round-${checkpoint_round} snapshot is verified"

while true; do
  heartbeat
  quota="$(amlt --json-tables ti sing -t Feeds 2>&1)" || { sleep 120; continue; }
  if python - "$quota" "$state_dir/quota_before_submission.json" <<'PY'
import json, sys
from scripts.feeds_quota_utils import parse_json_table, select_a100_80gb
row = select_a100_80gb(parse_json_table(sys.argv[1]))
quota = row['quotas_(used/limit)']
available = int(quota['standard']['limit']) - int(quota['standard']['used'])
user_available = int(quota['user_max']['limit']) - int(quota['user_max'].get('used') or 0)
open(sys.argv[2], 'w', encoding='utf-8').write(json.dumps({'selected_row': row, 'required_units': 1, 'standard_available': available, 'user_available': user_available}, indent=2, sort_keys=True) + '\n')
assert available >= 1 and user_available >= 1
PY
  then break; fi
  sleep 120
done

exec 8>"$project_dir/results/feeds_submission_global.lock"; flock 8
amlt run "$config" --dump -t Feeds > "$state_dir/preflight.dump.yaml" \
  2> "$state_dir/preflight.stderr" || { flock -u 8; fail "checkpoint C4 dump failed"; }
python scripts/validate_layerwise_finalfit_checkpoint_c4_amlt.py \
  --config "$state_dir/preflight.dump.yaml" --round "$checkpoint_round" \
  --job-name "layerwise-finalfit-checkpoint${checkpoint_round}-c4-seed42" \
  --run-dir "/mnt/amulet/$snapshot_root" >> "$state_dir/controller.log" 2>&1 \
  || { flock -u 8; fail "checkpoint C4 preflight failed"; }
flock -u 8

base_experiment="$experiment"
passed=false
for recovery_wave in 0 1 2 3; do
  if (( recovery_wave > 0 )); then experiment="${base_experiment}-recovery${recovery_wave}"; fi
  current="$(amlt status --no-update-db "$experiment" 2>&1)"; rc=$?
  class="$(python - "$rc" "$current" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]), sys.argv[2]))
PY
)" || fail "checkpoint C4 experiment state is ambiguous"
  if [[ "$class" == missing ]]; then
    exec 8>"$project_dir/results/feeds_submission_global.lock"; flock 8
    amlt run "$config" "$experiment" -t Feeds -y \
    --description "Diagnostic-only frozen C4 at formal layerwise fit round ${checkpoint_round}; recovery=$recovery_wave; final C4 remains mandatory" \
      > "$state_dir/${experiment}.submission.txt" 2>&1 \
      || { flock -u 8; fail "checkpoint C4 submission failed"; }
    flock -u 8
  elif [[ "$class" != exists ]]; then
    fail "checkpoint C4 pre-existing experiment state is ambiguous"
  fi
  while true; do
    heartbeat
    status="$(amlt status "$experiment" 2>&1)"
    printf '[%s] experiment=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$experiment" "$status" >> "$state_dir/status.log"
    if grep -Eqi 'No authentication method succeeded|refreshing your credentials|connection|timed out|permission' <<< "$status"; then sleep 120; continue; fi
    if grep -Fqi 'No jobs found in the current experiment' <<< "$status"; then sleep 30; continue; fi
    if grep -Eqi 'running|queued|preparing|paused|resuming|pre-queued' <<< "$status"; then sleep 120; continue; fi
    if python - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1], 1) else 1)
PY
    then passed=true; break; fi
    printf '%s\n' "$status" > "$state_dir/${experiment}.terminal_status.txt"
    amlt show "$experiment" > "$state_dir/${experiment}.show.txt" 2>&1 || true
    timeout 180 amlt logs view -n 3000 "$experiment" > "$state_dir/${experiment}.failure.log" 2>&1 || true
    log "checkpoint C4 attempt failed experiment=$experiment"
    break
  done
  [[ "$passed" == true ]] && break
done
[[ "$passed" == true ]] || fail "checkpoint C4 probe exhausted three recovery waves"

rm -rf "$state_dir/download"; mkdir -p "$state_dir/download"
amlt results download "$experiment" --status pass \
  --include 'recurrent_causal_gate.json' --include 'checkpoint_resume_audit.json' \
  --include 'probe_scope.txt' \
  --output "$state_dir/download" >> "$state_dir/controller.log" 2>&1 \
  || fail "checkpoint C4 result download failed"
mapfile -t gates < <(find "$state_dir/download" -type f -name recurrent_causal_gate.json)
[[ ${#gates[@]} -eq 1 ]] || fail "expected one checkpoint C4 result"
mapfile -t audits < <(find "$state_dir/download" -type f -name checkpoint_resume_audit.json)
[[ ${#audits[@]} -eq 1 ]] || fail "expected one checkpoint recovery audit"
python - "${gates[0]}" "${audits[0]}" "$state_dir/summary.json" "$checkpoint_round" <<'PY' || fail "checkpoint C4 result invalid"
import json, sys
value = json.load(open(sys.argv[1], encoding='utf-8'))
audit = json.load(open(sys.argv[2], encoding='utf-8'))
checkpoint_round = int(sys.argv[4])
assert audit.get('status') == 'passed' and audit.get('exact_resume_ready') is True, audit
assert audit.get('checkpoint_round') == checkpoint_round and audit.get('world_size') == 16, audit
assert value.get('state_architecture') == 'layerwise', value
assert value.get('state_shape') == [64, 5120], value
assert value.get('raw_history_visible_to_decoder') is False, value
assert value.get('persistent_history_kv_cache') is False, value
summary = value['summary']; clean = summary['clean']
payload = {
    'stage': f'layerwise_finalfit_checkpoint{checkpoint_round}_c4',
    'diagnostic_only': True,
    'formal_final_c4_required': True,
    'exact_resume_ready': True,
    'checkpoint_recovery_audit': audit,
    'clean_n': summary['clean_n'],
    'swap': clean['swap']['flip_to_A_rate'],
    'random': clean['random']['flip_to_A_rate'],
    'same_rule': clean['samemal']['flip_to_A_rate'],
    'noop': clean['noop']['flip_to_A_rate'],
    'D_specific': summary['D_specific'],
    'scientific_gate_pass_at_checkpoint': summary['scientific_gate_pass'],
}
open(sys.argv[3], 'w', encoding='utf-8').write(json.dumps(payload, indent=2, sort_keys=True) + '\n')
print(json.dumps(payload, indent=2, sort_keys=True))
PY
touch "$state_dir/complete"
log "layerwise round-${checkpoint_round} C4 diagnostic complete"
