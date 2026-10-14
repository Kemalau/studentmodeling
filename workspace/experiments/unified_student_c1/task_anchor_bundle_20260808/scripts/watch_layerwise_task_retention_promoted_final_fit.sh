#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
export AZURE_CONFIG_DIR="${AZURE_CONFIG_DIR:-/home/v-yuejiali/.azure-studentmodeling-feeds}"
pipeline="$project_dir/results/layerwise_unified_final_pipeline_20260731"
rehearsal_dir="$pipeline/task_retention_rehearsal"
eedi_dir="$pipeline/eedi_state_path_ab"
state_kd_dir="$pipeline/state_kd"
staged_dir="$pipeline/staged_task_retention"
staged_state_path_dir="$pipeline/staged_state_path"
equal_state_path_dir="$pipeline/equal_state_path"
equal_state_kd_dir="$pipeline/equal_state_kd"
equal_state_kd_open_t3_dir="$pipeline/equal_state_kd_open_t3"
equal_state_path_open_t3_dir="$pipeline/equal_state_path_open_t3"
equal_state_kd_open_t3_lr_scan_dir="$pipeline/equal_state_kd_open_t3_lr_scan"
equal_state_kd_open_t3_lr1e6_dir="$pipeline/equal_state_kd_open_t3_lr1e6"
open_t3_consolidation_dir="$pipeline/open_t3_consolidation"
t6_calibration_dir="$pipeline/t6_calibration/eedi_full_open_t3_lr2e6"
eedi_full_dir="$pipeline/eedi_full_lr_scan"
eedi_full_state_kd_dir="$pipeline/eedi_full_state_kd_lr_scan"
eedi_full_open_t3_dir="$pipeline/eedi_full_open_t3_lr_scan"
eedi_full_open_t3_selective_kd_dir="$pipeline/eedi_full_open_t3_selective_kd_lr_scan"
promotion_dir="$pipeline/task_retention_promotion"
state_dir="$pipeline/task_retention_promoted_final_fit"
config="$project_dir/amlt/feeds_layerwise_task_retention_promoted_final_fit_20260801.yaml"
max_recovery_waves="${MAX_RECOVERY_WAVES:-32}"
account=csnewsandfeeds4150361735
container=amulet
promotion_blob=studentmodeling/layerwise_world_model_v1/task_retention_promotion/selected.json
receipt_blob=studentmodeling/layerwise_world_model_v1/task_retention_promotion/final_fit_receipt.json
calibration_blob=studentmodeling/layerwise_world_model_v1/task_retention_promotion/t6_calibration.json
mkdir -p "$state_dir"
cd "$project_dir"
[[ "$max_recovery_waves" =~ ^[1-9][0-9]*$ ]] \
  || { printf 'invalid MAX_RECOVERY_WAVES=%s\n' "$max_recovery_waves" >&2; exit 2; }
exec 9>"$state_dir/controller.lock"
flock -n 9 || exit 0
echo $$ > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT
rm -f "$state_dir/needs_attention"

heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state_dir/needs_attention"; exit 1; }

selection="$state_dir/candidate_selection.json"
promotion_barrier_dirs=(
  "$equal_state_kd_open_t3_lr_scan_dir"
  "$equal_state_kd_open_t3_lr1e6_dir"
  "$open_t3_consolidation_dir"
)
candidate_dirs=(
  "$rehearsal_dir" "$eedi_dir" "$state_kd_dir" "$staged_dir"
  "$staged_state_path_dir" "$equal_state_path_dir" "$equal_state_kd_dir"
  "$equal_state_kd_open_t3_dir"
  "$equal_state_path_open_t3_dir"
  "$equal_state_kd_open_t3_lr_scan_dir"
  "$equal_state_kd_open_t3_lr1e6_dir"
  "$open_t3_consolidation_dir"
  "$eedi_full_dir" "$eedi_full_state_kd_dir" "$eedi_full_open_t3_dir"
  "$eedi_full_open_t3_selective_kd_dir"
)
while true; do
  heartbeat
  if python scripts/select_layerwise_task_retention_candidate.py --available \
      --rehearsal "$rehearsal_dir/comparison.json" \
      --eedi "$eedi_dir/comparison.json" \
      --state-kd "$state_kd_dir/summary.json" \
      --staged "$staged_dir/summary.json" \
      --staged-state-path "$staged_state_path_dir/summary.json" \
      --equal-state-path "$equal_state_path_dir/summary.json" \
      --equal-state-kd "$equal_state_kd_dir/summary.json" \
      --equal-state-kd-open-t3 "$equal_state_kd_open_t3_dir/summary.json" \
      --equal-state-path-open-t3 "$equal_state_path_open_t3_dir/summary.json" \
      --equal-state-kd-open-t3-lr-scan "$equal_state_kd_open_t3_lr_scan_dir/comparison.json" \
      --equal-state-kd-open-t3-lr1e6 "$equal_state_kd_open_t3_lr1e6_dir/summary.json" \
      --open-t3-consolidation "$open_t3_consolidation_dir/selection.json" \
      --t6-calibrated-candidate "$t6_calibration_dir/calibrated_candidate.json" \
      --eedi-full "$eedi_full_dir/comparison.json" \
      --eedi-full-state-kd "$eedi_full_state_kd_dir/comparison.json" \
      --eedi-full-open-t3 "$eedi_full_open_t3_dir/comparison.json" \
      --eedi-full-open-t3-selective-kd "$eedi_full_open_t3_selective_kd_dir/comparison.json" \
      --output "$selection" >> "$state_dir/controller.log" 2>&1; then
    # Compare every currently bounded open-T3 arm before promotion.  This
    # prevents a fast but weaker passing arm from winning only because the
    # lr=1e-6 midpoint or two-stage consolidation was still running.
    promotion_barrier_complete=true
    for barrier_dir in "${promotion_barrier_dirs[@]}"; do
      if [[ ! -f "$barrier_dir/complete" \
          && ! -f "$barrier_dir/needs_attention" \
          && ! -f "$barrier_dir/deferred" \
          && ! -f "$barrier_dir/superseded" ]]; then
        promotion_barrier_complete=false
      fi
    done
    if [[ "$promotion_barrier_complete" == true ]]; then
      break
    fi
  fi
  pending=false
  for required_dir in "${candidate_dirs[@]}"; do
    # A deferred/superseded diagnostic is intentionally removed from the
    # scheduling critical path.  Its existing evidence remains selectable only
    # when a complete summary exists; otherwise select_available() omits it.
    # Do not keep the promoted fit waiting forever for an arm whose compute was
    # explicitly retired after a hash-verified checkpoint was preserved.
    if [[ ! -f "$required_dir/complete" \
        && ! -f "$required_dir/needs_attention" \
        && ! -f "$required_dir/deferred" \
        && ! -f "$required_dir/superseded" ]]; then
      pending=true
    fi
  done
  [[ "$pending" == true ]] \
    || fail "all candidate diagnostics terminated without a fully gated promotion candidate"
  sleep 60
done
readarray -t selected_values < <(python - "$selection" <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
assert x['stage']=='layerwise_task_retention_unified_candidate_selection',x
assert x['selection_scope']=='completed_candidates_only',x
assert 1 <= x['candidate_count'] <= x['total_candidate_count']==24,x
assert x['any_all_gates_pass'] is True,x
row=x['selected_candidate']; assert row,x
assert row['all_gates_pass'] is True and float(row['causal_weight'])==2.0,row
assert row['branch'] in {'equal_t3_t6','equal_t3_t6_open_t3','equal_t3_t6_open_t3_two_stage','t3_heavy','eedi_dense','staged_t3_t6','eedi_full','eedi_full_open_t3'},row
assert row['loss_profile'] in {'standard','state_kd','state_path'},row
if row['branch'] in {'eedi_full','eedi_full_open_t3'} or float(row['learning_rate']) != 5e-7:
    assert row['candidate_id']==f"{row['branch']}__{row['loss_profile']}__lr{float(row['learning_rate']):.0e}",row
    assert float(row['learning_rate']) in {1e-6,2e-6,5e-6},row
else:
    assert row['candidate_id']==f"{row['branch']}__{row['loss_profile']}",row
    assert float(row['learning_rate'])==5e-7,row
print(row['candidate_id']); print(row['branch']); print(row['loss_profile']); print(row['learning_rate']); print(row['causal_weight'])
PY
)
[[ ${#selected_values[@]} -eq 5 ]] || fail "selected candidate metadata is incomplete"
selected_candidate="${selected_values[0]}"
selected_branch="${selected_values[1]}"
selected_loss_profile="${selected_values[2]}"
selected_learning_rate="${selected_values[3]}"
selected_causal_weight="${selected_values[4]}"
source_promotion="$promotion_dir/$selected_branch.json"
if [[ "$selected_branch" == eedi_full_open_t3 ]]; then
  source_promotion="$promotion_dir/eedi_full_open_t3.json"
elif [[ "$selected_branch" == equal_t3_t6_open_t3 ]]; then
  source_promotion="$promotion_dir/equal_t3_t6_open_t3.json"
elif [[ "$selected_branch" == equal_t3_t6_open_t3_two_stage ]]; then
  source_promotion="$promotion_dir/equal_t3_t6_open_t3_two_stage.json"
fi
test -f "$source_promotion" || fail "selected promotion manifest is missing"
python - "$source_promotion" "$selected_branch" "$selected_candidate" "$selected_loss_profile" "$selected_learning_rate" "$selected_causal_weight" "$selection" "$state_dir/selected.json" <<'PY' \
  || fail "selected promotion manifest is invalid"
import hashlib,json,pathlib,sys
x=json.load(open(sys.argv[1],encoding='utf-8')); selection=json.load(open(sys.argv[7],encoding='utf-8'))
assert x['stage']=='layerwise_task_retention_promoted_order_v1',x
assert x['branch']==sys.argv[2],x
assert x['source_checkpoint_round']==750 and x['source_world_size']==16,x
assert x['completed_prefix_rows']==12000 and x['epoch_rows']==259845,x
expected_rows=(
  5000 if sys.argv[2] in {'eedi_full','eedi_full_open_t3'}
  else 2000 if sys.argv[2]=='equal_t3_t6_open_t3_two_stage'
  else 1000
)
assert x['promoted_rows']==expected_rows and x['formal_example_id_overlap']==0,x
assert x['one_epoch_permutation'] is True and x['completed_prefix_unchanged'] is True,x
order=x['real_order']; assert len(order)==259845 and len(set(order))==259845,x
assert hashlib.sha256('\n'.join(map(str,order)).encode()).hexdigest()==x['real_order_sha256'],x
x['selected_candidate_id']=sys.argv[3]
x['selected_loss_profile']=sys.argv[4]
x['selected_learning_rate']=float(sys.argv[5])
x['selected_causal_weight']=float(sys.argv[6])
x['training_view']=(
 'open_t3_selective_kd'
 if x['branch'] in {'eedi_full_open_t3','equal_t3_t6_open_t3','equal_t3_t6_open_t3_two_stage'} and (x['selected_loss_profile']=='state_kd' or x['branch']=='equal_t3_t6_open_t3_two_stage')
 else ('open_t3_no_candidate_enumeration' if x['branch'] in {'eedi_full_open_t3','equal_t3_t6_open_t3'} else 'standard')
)
if x['branch']=='equal_t3_t6_open_t3_two_stage':
    assert x.get('curriculum_mode')=='open_t3_formation_then_causal_task_consolidation',x
    assert [p['name'] for p in x.get('curriculum_phases',[])]==[
      'open_t3_formation','causal_task_consolidation'],x
    x.update({
      'formation_loss_profile':'state_kd',
      'formation_learning_rate':5e-6,
      'formation_requested_rows':1000,
      'formation_aligned_rows':1008,
      'formation_transition_global_position':13008,
      'consolidation_loss_profile':x['selected_loss_profile'],
      'consolidation_learning_rate':float(x['selected_learning_rate']),
    })
x['candidate_selection_sha256']=hashlib.sha256(open(sys.argv[7],'rb').read()).hexdigest()
assert selection['selected_candidate']['candidate_id']==sys.argv[3],selection
assert float(selection['selected_candidate']['learning_rate'])==float(sys.argv[5]),selection
assert float(selection['selected_candidate']['causal_weight'])==float(sys.argv[6])==2.0,selection
calibration=selection['selected_candidate'].get('t6_calibration')
if calibration is not None:
    assert calibration['stage']=='layerwise_t6_threshold_calibration_v1',calibration
    assert calibration['selection_split']=='unused_original_validation',calibration
    assert calibration['candidate_training_overlap']==0,calibration
    assert calibration['formal_example_id_overlap']==0,calibration
    threshold=float(calibration['decision_threshold'])
    assert 0.0 < threshold < 1.0 and len(calibration['artifact_sha256'])==64,calibration
    x['t6_decision_threshold']=threshold
    x['t6_calibration_artifact_sha256']=calibration['artifact_sha256']
    x['t6_calibration_selection_split']=calibration['selection_split']
    x['t6_calibration_selection_rows']=int(calibration['selection_rows'])
path=pathlib.Path(sys.argv[8]); temporary=path.with_suffix(path.suffix+'.partial')
temporary.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n'); temporary.replace(path)
PY
printf '%s\n' "$selected_branch" > "$state_dir/selected_branch.txt"
printf '%s\n' "$selected_loss_profile" > "$state_dir/selected_loss_profile.txt"
printf '%s\n' "$selected_candidate" > "$state_dir/selected_candidate.txt"
printf '%s\n' "$selected_learning_rate" > "$state_dir/selected_learning_rate.txt"
printf '%s\n' "$selected_causal_weight" > "$state_dir/selected_causal_weight.txt"

while ! az account get-access-token --subscription b6dc87f3-c479-49c8-8cb5-7896da3ff895 \
  --query expiresOn -o tsv >/dev/null 2>&1; do heartbeat; sleep 120; done
az account set --subscription b6dc87f3-c479-49c8-8cb5-7896da3ff895 >/dev/null \
  || fail "could not select fixed subscription"
az storage blob upload --auth-mode login --account-name "$account" \
  --container-name "$container" --name "$promotion_blob" \
  --file "$state_dir/selected.json" --overwrite \
  > "$state_dir/promotion_upload.log" 2>&1 || fail "promotion upload failed"
az storage blob download --auth-mode login --account-name "$account" \
  --container-name "$container" --name "$promotion_blob" \
  --file "$state_dir/selected.remote.json" --overwrite \
  > "$state_dir/promotion_download.log" 2>&1 || fail "promotion verification download failed"
cmp -s "$state_dir/selected.json" "$state_dir/selected.remote.json" \
  || fail "remote promotion differs from selected local manifest"
if python - "$state_dir/selected.json" <<'PY'
import json,sys
raise SystemExit(0 if 't6_calibration_artifact_sha256' in json.load(open(sys.argv[1])) else 1)
PY
then
  test -f "$t6_calibration_dir/frozen_threshold.json" \
    || fail "selected T6 calibration artifact is missing"
  test "$(sha256sum "$t6_calibration_dir/frozen_threshold.json" | cut -d' ' -f1)" = \
    "$(python - "$state_dir/selected.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['t6_calibration_artifact_sha256'])
PY
)" || fail "selected T6 calibration artifact hash drift"
  az storage blob upload --auth-mode login --account-name "$account" \
    --container-name "$container" --name "$calibration_blob" \
    --file "$t6_calibration_dir/frozen_threshold.json" --overwrite \
    > "$state_dir/calibration_upload.log" 2>&1 \
    || fail "T6 calibration upload failed"
  az storage blob download --auth-mode login --account-name "$account" \
    --container-name "$container" --name "$calibration_blob" \
    --file "$state_dir/t6_calibration.remote.json" --overwrite \
    > "$state_dir/calibration_download.log" 2>&1 \
    || fail "T6 calibration verification download failed"
  cmp -s "$t6_calibration_dir/frozen_threshold.json" \
    "$state_dir/t6_calibration.remote.json" \
    || fail "remote T6 calibration differs from frozen local artifact"
fi

wait_quota(){
  local output="$1" quota
  while true; do
    heartbeat
    quota="$(amlt --json-tables ti sing -t Feeds 2>&1)" || { sleep 120; continue; }
    if python - "$quota" "$output" <<'PY'
import json,sys
from scripts.feeds_quota_utils import parse_json_table,select_a100_80gb
row=select_a100_80gb(parse_json_table(sys.argv[1])); q=row['quotas_(used/limit)']
available=int(q['standard']['limit'])-int(q['standard']['used'])
user_available=int(q['user_max']['limit'])-int(q['user_max'].get('used') or 0)
open(sys.argv[2],'w',encoding='utf-8').write(json.dumps({'selected_row':row,'required_units':16,'standard_available':available,'user_available':user_available},indent=2,sort_keys=True)+'\n')
assert available>=16 and user_available>=16,(available,user_available)
PY
    then return 0; fi
    sleep 120
  done
}

wait_local_result_space(){
  local required_bytes=$((12 * 1024 * 1024 * 1024)) available
  while true; do
    heartbeat
    available="$(df -PB1 "$state_dir" | awk 'NR==2 {print $4}')"
    if [[ "$available" =~ ^[0-9]+$ ]] && (( available >= required_bytes )); then
      printf 'required_bytes=%s\navailable_bytes=%s\nchecked_at=%s\n' \
        "$required_bytes" "$available" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        > "$state_dir/local_result_space.receipt"
      return 0
    fi
    log "waiting for local result space required_bytes=$required_bytes available_bytes=${available:-unknown}"
    sleep 300
  done
}

submit_wave(){
  local wave="$1" experiment output rc current class
  if (( wave == 0 )); then
    experiment="layerwise-task-retention-promoted-${selected_branch}-${selected_loss_profile}-finalfit-seed42-20260802"
  else
    experiment="layerwise-task-retention-promoted-${selected_branch}-${selected_loss_profile}-finalfit-recovery${wave}-seed42-20260802"
  fi
  wait_quota "$state_dir/${experiment}.quota.json"
  exec 8>"$project_dir/results/feeds_submission_global.lock"; flock 8
  amlt run "$config" --dump -t Feeds \
    > "$state_dir/${experiment}.preflight.dump.yaml" \
    2> "$state_dir/${experiment}.preflight.stderr" \
    || { flock -u 8; return 1; }
  python scripts/validate_layerwise_task_retention_promoted_final_fit_amlt.py \
    --config "$state_dir/${experiment}.preflight.dump.yaml" \
    >> "$state_dir/controller.log" 2>&1 || { flock -u 8; return 1; }
  current="$(amlt status --no-update-db "$experiment" 2>&1)"; rc=$?
  class="$(python - "$rc" "$current" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]),sys.argv[2]))
PY
)" || { flock -u 8; return 1; }
  if [[ "$class" == missing ]]; then
    output="$(amlt run "$config" "$experiment" -t Feeds -y \
      --description "Exact-epoch continuation from round 750 with selected candidate=$selected_candidate; data order and loss profile frozen; optimizer, scheduler, RNG and causal schedule preserved" 2>&1)"; rc=$?
    printf '[%s] submit rc=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$output" \
      > "$state_dir/${experiment}.submission.txt"
    (( rc == 0 )) || { flock -u 8; return 1; }
  elif [[ "$class" == exists ]]; then
    printf 'status=preexisting_not_resubmitted\n%s\n' "$current" \
      > "$state_dir/${experiment}.submission.txt"
  else
    flock -u 8; return 1
  fi
  flock -u 8
  printf '%s\n' "$experiment" > "$state_dir/active_experiment.txt"
  log "monitoring promoted exact fit experiment=$experiment candidate=$selected_candidate"
}

wave=0
if [[ -f "$state_dir/active_experiment.txt" ]]; then
  experiment="$(<"$state_dir/active_experiment.txt")"
  if [[ "$experiment" =~ -recovery([0-9]+)-seed42- ]]; then wave="${BASH_REMATCH[1]}"; fi
else
  submit_wave "$wave" || fail "promoted final-fit submission failed"
  experiment="$(<"$state_dir/active_experiment.txt")"
fi

while true; do
  heartbeat
  status="$(timeout 90 amlt status "$experiment" 2>&1)" || { sleep 120; continue; }
  printf '[%s]\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" >> "$state_dir/status.log"
  if grep -Eqi 'No authentication method succeeded|refreshing your credentials|connection|timed out|permission|TooManyRequests|ServiceUnavailable' <<< "$status"; then sleep 120; continue; fi
  if grep -Eqi 'running|queued|preparing|paused|resuming|pre-queued' <<< "$status"; then sleep 120; continue; fi
  if python - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1],1) else 1)
PY
  then break; fi
  printf '%s\n' "$status" > "$state_dir/${experiment}.terminal_status.txt"
  timeout 90 amlt show "$experiment" > "$state_dir/${experiment}.show.txt" 2>&1 || true
  timeout 180 amlt logs view -n 3000 "$experiment" > "$state_dir/${experiment}.failure.log" 2>&1 || true
  (( wave < max_recovery_waves )) \
    || fail "promoted final fit exhausted ${max_recovery_waves} recovery submissions"
  wave=$((wave+1))
  submit_wave "$wave" || fail "promoted final-fit recovery submission failed wave=$wave"
  experiment="$(<"$state_dir/active_experiment.txt")"
done

wait_local_result_space
rm -rf "$state_dir/download"; mkdir -p "$state_dir/download"
amlt results download "$experiment" --status pass \
  --include '*.json' --include '*.safetensors' --include '*.pt' \
  --output "$state_dir/download" >> "$state_dir/controller.log" 2>&1 \
  || fail "promoted final-fit result download failed"
readarray -t receipts < <(find "$state_dir/download" -name promoted_final_fit_receipt.json -type f)
[[ ${#receipts[@]} -eq 1 ]] || fail "expected one promoted final-fit receipt"
python - "${receipts[0]}" "$selected_branch" "$selected_loss_profile" "$selected_candidate" "$selected_learning_rate" "$selected_causal_weight" <<'PY' \
  || fail "promoted final-fit receipt invalid"
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
assert x['status']=='final_fit_complete',x
assert x['stage']=='layerwise_task_retention_promoted_final_fit',x
assert x['branch']==sys.argv[2] and x['one_epoch_exact'] is True,x
assert x['loss_profile']==sys.argv[3] and x['candidate_id']==sys.argv[4],x
assert abs(float(x['learning_rate'])-float(sys.argv[5]))<1e-15,x
assert float(x['causal_weight'])==float(sys.argv[6])==2.0,x
assert x['real_examples']==259845 and x['promoted_source_round']==750,x
assert x['state_architecture']=='layerwise' and x['state_shape']==[64,5120],x
assert x['raw_history_visible_to_decoder'] is False,x
assert x['persistent_history_kv_cache'] is False,x
PY
python - "$state_dir/download" "$state_dir/artifact_manifest.json" <<'PY' \
  || fail "promoted final-fit artifact manifest failed"
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
    path = one(name)
    if path.stat().st_size <= 0:
        raise ValueError(f'empty artifact: {path}')
    artifacts[name] = {
        'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest(path),
    }
payload = {'status': 'verified', 'algorithm': 'sha256', 'artifacts': artifacts}
path = pathlib.Path(sys.argv[2])
temporary = path.with_suffix(path.suffix + '.partial')
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
temporary.replace(path)
print(json.dumps(payload, sort_keys=True))
PY
while ! az account get-access-token --subscription b6dc87f3-c479-49c8-8cb5-7896da3ff895 \
  --query expiresOn -o tsv >/dev/null 2>&1; do heartbeat; sleep 120; done
az account set --subscription b6dc87f3-c479-49c8-8cb5-7896da3ff895 >/dev/null \
  || fail "could not select fixed subscription for receipt publication"
az storage blob upload --auth-mode login --account-name "$account" \
  --container-name "$container" --name "$receipt_blob" \
  --file "${receipts[0]}" --overwrite \
  > "$state_dir/receipt_upload.log" 2>&1 || fail "promoted receipt upload failed"
az storage blob download --auth-mode login --account-name "$account" \
  --container-name "$container" --name "$receipt_blob" \
  --file "$state_dir/promoted_final_fit_receipt.remote.json" --overwrite \
  > "$state_dir/receipt_download.log" 2>&1 || fail "promoted receipt verification download failed"
cmp -s "${receipts[0]}" "$state_dir/promoted_final_fit_receipt.remote.json" \
  || fail "remote promoted receipt differs from local result"
touch "$state_dir/scientific_gate_pass" "$state_dir/complete"
log "promoted task-retention exact fit complete candidate=$selected_candidate"
