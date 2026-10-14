#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
export AZURE_CONFIG_DIR="${AZURE_CONFIG_DIR:-/home/v-yuejiali/.azure-studentmodeling-feeds}"
pilot_dir="$project_dir/results/layerwise_trajectory_curriculum_20260731"
state_dir="$project_dir/results/layerwise_unified_final_pipeline_20260731/final_fit"
progress_dir="$project_dir/results/layerwise_unified_final_pipeline_20260731/progress"
config="$project_dir/amlt/feeds_layerwise_unified_final_fit_20260731.yaml"
finalize_config="$project_dir/amlt/feeds_layerwise_unified_final_validation_recovery_20260801.yaml"
final_checkpoint_prefix="studentmodeling/checkpoints/layerwise_unified_world_model_seed42_full259845_v1_20260731/checkpoint-round-16241"
early_checkpoint_prefix="studentmodeling/checkpoints/layerwise_unified_world_model_seed42_full259845_v1_20260731/checkpoint-round-1000"
early_snapshot_dir="$project_dir/results/layerwise_unified_final_pipeline_20260731/checkpoint1000_c4"
finalize_base_experiment="layerwise-unified-finalize-seed42-20260801"
active_experiment_file="$state_dir/active_experiment.txt"
if [[ -n "${FINAL_FIT_EXPERIMENT:-}" ]]; then
  experiment="$FINAL_FIT_EXPERIMENT"
elif [[ -s "$active_experiment_file" ]]; then
  experiment="$(<"$active_experiment_file")"
else
  experiment="layerwise-unified-finalfit-seed42-20260731"
fi
[[ "$experiment" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || exit 2
printf '%s\n' "$experiment" > "$active_experiment_file"
account="csnewsandfeeds4150361735"
container="amulet"
selection_blob="studentmodeling/layerwise_world_model_v1/selection/seed42.json"
receipt_blob="studentmodeling/layerwise_world_model_v1/final_fit/seed42.json"
max_resilient_recoveries="${MAX_RESILIENT_RECOVERIES:-12}"
mkdir -p "$state_dir"
cd "$project_dir"
exec 9>"$state_dir/controller.lock"
flock -n 9 || exit 0
echo $$ > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT
rm -f "$state_dir/complete" "$state_dir/needs_attention"

heartbeat() { date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"; }
fail() { log "ERROR: $*"; touch "$state_dir/needs_attention"; exit 1; }
blob_exists() {
  az storage blob exists --auth-mode login --account-name "$account" \
    --container-name "$container" --name "$1" --query exists -o tsv 2>/dev/null
}
final_checkpoint_ready() {
  local file receipt="$state_dir/final_checkpoint_completion.json"
  for file in adapter/adapter_model.safetensors recurrent_state.pt checkpoint_completion.json; do
    [[ "$(blob_exists "$final_checkpoint_prefix/$file")" == true ]] || return 1
  done
  az storage blob download --auth-mode login --account-name "$account" \
    --container-name "$container" --name "$final_checkpoint_prefix/checkpoint_completion.json" \
    --file "$receipt" --overwrite -o none >/dev/null 2>&1 || return 1
  python - "$receipt" <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
assert x.get('status')=='complete' and x.get('round')==16241,x
assert x.get('world_size')==16,x
PY
}
wait_quota() {
  local required="$1" output="$2" quota
  while true; do
    heartbeat
    quota="$(amlt --json-tables ti sing -t Feeds 2>&1)" || { sleep 120; continue; }
    if python - "$quota" "$required" "$output" <<'PY'
import json,sys
from scripts.feeds_quota_utils import parse_json_table,select_a100_80gb
row=select_a100_80gb(parse_json_table(sys.argv[1])); n=int(sys.argv[2]); q=row['quotas_(used/limit)']
available=int(q['standard']['limit'])-int(q['standard']['used'])
user_available=int(q['user_max']['limit'])-int(q['user_max'].get('used') or 0)
open(sys.argv[3],'w',encoding='utf-8').write(json.dumps({'selected_row':row,'required_units':n,'standard_available':available,'user_available':user_available},indent=2,sort_keys=True)+'\n')
assert available>=n and user_available>=n,(available,user_available,n)
PY
    then return 0; fi
    sleep 120
  done
}
submit_finalizer() {
  local next_experiment="$1" description="$2" current rc class output
  wait_quota 1 "$state_dir/${next_experiment}.quota.json"
  exec 8>"$project_dir/results/feeds_submission_global.lock"; flock 8
  amlt run "$finalize_config" --dump -t Feeds \
    > "$state_dir/${next_experiment}.preflight.dump.yaml" \
    2> "$state_dir/${next_experiment}.preflight.stderr" \
    || { flock -u 8; return 1; }
  python scripts/validate_layerwise_unified_final_validation_recovery_amlt.py \
    --config "$state_dir/${next_experiment}.preflight.dump.yaml" \
    >> "$state_dir/controller.log" 2>&1 || { flock -u 8; return 1; }
  current="$(amlt status --no-update-db "$next_experiment" 2>&1)"; rc=$?
  class="$(python - "$rc" "$current" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]),sys.argv[2]))
PY
)" || { flock -u 8; return 1; }
  if [[ "$class" == missing ]]; then
    output="$(amlt run "$finalize_config" "$next_experiment" -t Feeds -y \
      --description "$description" 2>&1)"; rc=$?
    printf '[%s] submit rc=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$output" \
      > "$state_dir/${next_experiment}.submission.txt"
    (( rc == 0 )) || { flock -u 8; return 1; }
  elif [[ "$class" == exists ]]; then
    printf 'status=preexisting_not_resubmitted\n%s\n' "$current" \
      > "$state_dir/${next_experiment}.submission.txt"
  else
    flock -u 8; return 1
  fi
  flock -u 8
  experiment="$next_experiment"
  printf '%s\n' "$experiment" > "$active_experiment_file"
  log "monitoring single-A100 exact-final-checkpoint finalizer experiment=$experiment"
}

while [[ ! -f "$pilot_dir/scientific_gate_pass" ]]; do
  heartbeat
  [[ ! -f "$pilot_dir/needs_attention" ]] || fail "layerwise trajectory curriculum needs attention"
  sleep 60
done
python - "$pilot_dir/selection.json" <<'PY' || fail "pilot selection is invalid"
import json, sys
x=json.load(open(sys.argv[1], encoding='utf-8'))
assert x['scientific_gate_pass'] is True and x['selected_branch'], x
assert x['selected_C4']['D_specific'] >= 0.15, x
assert x['stage'] == 'layerwise_trajectory_curriculum_selection', x
PY

# This upload is an idempotent pre-submission gate. On controller restarts the
# byte-identical downloaded verification is sufficient evidence that it
# already completed; do not turn a temporary monitoring-token expiry into a
# scientific failure marker or rewrite the same Blob unnecessarily.
if ! cmp -s "$pilot_dir/selection.json" "$state_dir/selection_verification.json"; then
  while ! az account get-access-token \
      --subscription b6dc87f3-c479-49c8-8cb5-7896da3ff895 \
      --query expiresOn -o tsv >/dev/null 2>&1; do
    heartbeat
    sleep 120
  done
  az account set --subscription b6dc87f3-c479-49c8-8cb5-7896da3ff895 >/dev/null \
    || fail "could not select fixed subscription after authentication"
  az storage blob upload --auth-mode login --account-name "$account" \
    --container-name "$container" --name "$selection_blob" \
    --file "$pilot_dir/selection.json" --overwrite \
    >> "$state_dir/controller.log" 2>&1 || fail "selection upload failed"
  az storage blob download --auth-mode login --account-name "$account" \
    --container-name "$container" --name "$selection_blob" \
    --file "$state_dir/selection_verification.json" --overwrite \
    >> "$state_dir/controller.log" 2>&1 || fail "selection verification download failed"
  cmp -s "$pilot_dir/selection.json" "$state_dir/selection_verification.json" \
    || fail "uploaded selection differs"
fi

recovery_wave=0
if [[ "$experiment" =~ resilient-recovery([0-9]+)-seed42 ]]; then
  recovery_wave="${BASH_REMATCH[1]}"
fi
finalize_retry=0
if [[ "$experiment" =~ ^layerwise-unified-finalize-retry([0-9]+)-seed42-20260801$ ]]; then
  finalize_retry="${BASH_REMATCH[1]}"
fi
if [[ ! -f "$state_dir/submission.txt" ]]; then
  wait_quota 16 "$state_dir/quota_before_submission.json"
  exec 8>"$project_dir/results/feeds_submission_global.lock"; flock 8
  amlt run "$config" --dump -t Feeds > "$state_dir/preflight.dump.yaml" \
    2> "$state_dir/preflight.dump.stderr" \
    || { flock -u 8; fail "final-fit dump failed"; }
  python scripts/validate_layerwise_unified_final_fit_amlt.py \
    --config "$state_dir/preflight.dump.yaml" >> "$state_dir/controller.log" 2>&1 \
    || { flock -u 8; fail "final-fit preflight failed"; }
  current="$(amlt status --no-update-db "$experiment" 2>&1)"; rc=$?
  class="$(python - "$rc" "$current" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]), sys.argv[2]))
PY
)" || { flock -u 8; fail "could not classify final fit"; }
  if [[ "$class" == missing ]]; then
    output="$(amlt run "$config" "$experiment" -t Feeds -y \
      --description 'Selected layerwise recurrent unified model; exact one epoch over 259845 examples with KD and CF4 causal rehearsal' 2>&1)"; rc=$?
    printf '[%s] submit rc=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$output" > "$state_dir/submission.txt"
    (( rc == 0 )) || { flock -u 8; fail "final-fit submission failed"; }
  elif [[ "$class" == exists ]]; then
    printf 'status=preexisting_not_resubmitted\n%s\n' "$current" > "$state_dir/submission.txt"
  else flock -u 8; fail "ambiguous final-fit state"; fi
  flock -u 8
fi

while true; do
  heartbeat
  if ! status="$(timeout 90 amlt status "$experiment" 2>&1)"; then
    # A monitoring transport timeout is not evidence of remote job failure.
    # Keep the active experiment and retry without touching the training job.
    sleep 120
    continue
  fi
  printf '[%s]\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" >> "$state_dir/status.log"
  if grep -Eqi 'No authentication method succeeded|refreshing your credentials|connection|timed out|permission' <<< "$status"; then sleep 120; continue; fi
  if grep -Eqi 'running|queued|preparing|paused|resuming|pre-queued' <<< "$status"; then
    # A live service state is not sufficient evidence of forward progress.
    # Cancel only after two hours without a structured training round and only
    # before final validation; the normal path below then resumes the identical
    # 16-rank recipe from the newest atomic checkpoint.
    if [[ -f "$progress_dir/stalled.json" ]] && python - "$progress_dir/stalled.json" "$experiment" <<'PY'
import json, sys
from datetime import datetime, timezone
x = json.load(open(sys.argv[1], encoding='utf-8'))
same = x.get('experiment') == sys.argv[2]
training = int(x.get('round', 0)) < 16000
stalled = float(x.get('seconds_since_progress', 0)) >= 7200
observed = datetime.fromisoformat(str(x.get('observed_at')))
if observed.tzinfo is None:
    observed = observed.replace(tzinfo=timezone.utc)
fresh = 0 <= (datetime.now(timezone.utc) - observed).total_seconds() <= 900
raise SystemExit(0 if same and training and stalled and fresh else 1)
PY
    then
      cp "$progress_dir/stalled.json" "$state_dir/${experiment}.stalled.json"
      timeout 90 amlt show "$experiment" > "$state_dir/${experiment}.stalled.show.txt" 2>&1 || true
      timeout 180 amlt logs view -n 3000 "$experiment" \
        > "$state_dir/${experiment}.stalled.log" 2>&1 || true
      log "two-hour training stall detected; cancelling for atomic-checkpoint recovery experiment=$experiment"
      amlt cancel "$experiment" --status running -y \
        > "$state_dir/${experiment}.stalled.cancel.txt" 2>&1 \
        || { sleep 120; continue; }
      sleep 120
      continue
    fi
    sleep 120
    continue
  fi
  if python - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1],1) else 1)
PY
  then break; fi
  printf '%s\n' "$status" > "$state_dir/${experiment}.terminal_status.txt"
  timeout 90 amlt show "$experiment" > "$state_dir/${experiment}.show.txt" 2>&1 || true
  timeout 180 amlt logs view -n 3000 "$experiment" \
    > "$state_dir/${experiment}.failure.log" 2>&1 || true
  if [[ "$experiment" == "$finalize_base_experiment" \
        || "$experiment" =~ ^layerwise-unified-finalize-retry[0-9]+-seed42-20260801$ ]]; then
    (( finalize_retry < 3 )) || fail "exact-final-checkpoint finalizer exhausted three retries"
    final_checkpoint_ready || fail "finalizer failed and exact final checkpoint is no longer valid"
    finalize_retry=$((finalize_retry + 1))
    next_experiment="layerwise-unified-finalize-retry${finalize_retry}-seed42-20260801"
    submit_finalizer "$next_experiment" \
      "Single-A100 exact-final-checkpoint validation retry=${finalize_retry}; no optimizer steps" \
      || fail "single-A100 finalizer retry submission failed"
    continue
  fi
  if final_checkpoint_ready; then
    submit_finalizer "$finalize_base_experiment" \
      "Single-A100 validation and artifact finalization from exact distributed round 16241; no optimizer steps" \
      || fail "single-A100 exact-final-checkpoint finalizer submission failed"
    continue
  fi
  if (( recovery_wave >= max_resilient_recoveries )); then
    fail "layerwise final fit exhausted $max_resilient_recoveries resilient recoveries"
  fi
  if [[ "$(blob_exists "$early_checkpoint_prefix/checkpoint_completion.json")" == true ]]; then
    log "waiting for immutable round-1000 snapshot before checkpoint-pruning recovery"
    while [[ ! -f "$early_snapshot_dir/snapshot_complete" ]]; do
      heartbeat
      [[ ! -f "$early_snapshot_dir/needs_attention" ]] \
        || fail "round-1000 snapshot owner needs attention before training recovery"
      sleep 60
    done
  fi
  recovery_wave=$((recovery_wave + 1))
  next_experiment="layerwise-unified-finalfit-resilient-recovery${recovery_wave}-seed42-20260801"
  wait_quota 16 "$state_dir/${next_experiment}.quota.json"
  exec 8>"$project_dir/results/feeds_submission_global.lock"; flock 8
  amlt run "$config" --dump -t Feeds \
    > "$state_dir/${next_experiment}.preflight.dump.yaml" \
    2> "$state_dir/${next_experiment}.preflight.stderr" \
    || { flock -u 8; fail "resilient recovery dump failed"; }
  python scripts/validate_layerwise_unified_final_fit_amlt.py \
    --config "$state_dir/${next_experiment}.preflight.dump.yaml" \
    >> "$state_dir/controller.log" 2>&1 \
    || { flock -u 8; fail "resilient recovery preflight failed"; }
  current="$(amlt status --no-update-db "$next_experiment" 2>&1)"; rc=$?
  class="$(python - "$rc" "$current" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]), sys.argv[2]))
PY
)" || { flock -u 8; fail "could not classify resilient recovery"; }
  if [[ "$class" == missing ]]; then
    output="$(amlt run "$config" "$next_experiment" -t Feeds -y \
      --description "Resilient recovery ${recovery_wave}: resume selected r256-scale exact one-epoch fit from newest atomic checkpoint; checkpoint every 250 rounds, retain newest two" 2>&1)"; rc=$?
    printf '[%s] submit rc=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$output" \
      > "$state_dir/${next_experiment}.submission.txt"
    (( rc == 0 )) || { flock -u 8; fail "resilient recovery submission failed"; }
  elif [[ "$class" == exists ]]; then
    printf 'status=preexisting_not_resubmitted\n%s\n' "$current" \
      > "$state_dir/${next_experiment}.submission.txt"
  else
    flock -u 8
    fail "ambiguous resilient recovery state"
  fi
  flock -u 8
  experiment="$next_experiment"
  printf '%s\n' "$experiment" > "$active_experiment_file"
  log "monitoring resilient recovery experiment=$experiment"
done

rm -rf "$state_dir/download"; mkdir -p "$state_dir/download"
amlt results download "$experiment" --status pass \
  --include '*.json' --include '*.safetensors' --include '*.pt' \
  --output "$state_dir/download" \
  >> "$state_dir/controller.log" 2>&1 || fail "final-fit result download failed"
readarray -t receipts < <(find "$state_dir/download" -name final_fit_receipt.json -type f)
[[ ${#receipts[@]} -eq 1 ]] || fail "expected one final-fit receipt, found ${#receipts[@]}"
receipt="${receipts[0]}"
python - "$receipt" <<'PY' || fail "final-fit receipt invalid"
import json, sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
assert x['status']=='final_fit_complete' and x['stage']=='layerwise_unified_final_fit', x
assert x['real_examples']==259845 and x['one_epoch_exact'] is True, x
assert x['state_architecture']=='layerwise' and x['state_shape']==[64,5120], x
assert x['raw_history_visible_to_decoder'] is False, x
assert x['persistent_history_kv_cache'] is False, x
PY
python - "$state_dir/download" "$state_dir/artifact_manifest.json" <<'PY' \
  || fail "final-fit artifact manifest failed"
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
def one(name):
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise ValueError(f'expected one {name}: {matches}')
    return matches[0]
def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()
artifacts = {}
for name in ('adapter_model.safetensors', 'recurrent_state.pt'):
    path = one(name); size = path.stat().st_size
    if size <= 0:
        raise ValueError(f'empty artifact: {path}')
    artifacts[name] = {'path': str(path), 'bytes': size, 'sha256': digest(path)}
payload = {'status': 'verified', 'algorithm': 'sha256', 'artifacts': artifacts}
path = pathlib.Path(sys.argv[2]); temporary = path.with_suffix(path.suffix + '.partial')
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
temporary.replace(path)
print(json.dumps(payload, sort_keys=True))
PY
az storage blob upload --auth-mode login --account-name "$account" \
  --container-name "$container" --name "$receipt_blob" --file "$receipt" --overwrite \
  >> "$state_dir/controller.log" 2>&1 || fail "final-fit receipt upload failed"
touch "$state_dir/scientific_gate_pass" "$state_dir/complete"
log "layerwise unified final fit complete"
