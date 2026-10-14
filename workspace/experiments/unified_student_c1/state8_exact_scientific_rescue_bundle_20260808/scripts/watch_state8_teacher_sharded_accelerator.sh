#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
teacher_state="$project/results/state8_qualified_teacher_pipeline_20260805"
variant="${STATE8_TEACHER_EVAL_VARIANT:-clean_v2}"
case "$variant" in
  clean_v2)
    state="$teacher_state/sharded_clean_v2"
    base_experiment=state8-full-history-teacher-eval-clean-v2-sharded-20260805
    recovery_prefix=state8-full-history-teacher-eval-clean-v2-sharded-recovery
    shard_count=4
    config="$project/amlt/feeds_state8_teacher_eval_sharded_20260805.yaml"
    ;;
  clean_v2_16way)
    state="$teacher_state/sharded_clean_v2_16way"
    base_experiment=state8-full-history-teacher-eval-clean-v2-16way-20260805
    recovery_prefix=state8-full-history-teacher-eval-clean-v2-16way-recovery
    shard_count=16
    config="$project/amlt/feeds_state8_teacher_eval_16way_20260805.yaml"
    ;;
  legacy_v1)
    state="$teacher_state/sharded_accelerator"
    base_experiment=state8-full-history-teacher-eval-sharded-20260805
    recovery_prefix=state8-full-history-teacher-eval-sharded-recovery
    shard_count=4
    config="$project/amlt/feeds_state8_teacher_eval_sharded_20260805.yaml"
    ;;
  *) echo "Unknown STATE8_TEACHER_EVAL_VARIANT=$variant" >&2; exit 2 ;;
esac
curriculum_config="$project/amlt/feeds_state8_qualified_teacher_curriculum_20260805.yaml"
curriculum_base=state8-qualified-teacher-curriculum-20260805
curriculum_recovery_prefix=state8-qualified-teacher-curriculum-recovery
curriculum_job=state8-qualified-teacher-curriculum-seed42
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
account=csnewsandfeeds4150361735
container=amulet
teacher_prefix=studentmodeling/diagnostics/state8_full_history_natural_teacher_v1
max_recovery_waves="${MAX_RECOVERY_WAVES:-8}"
all_jobs=()
for ((index=0; index<shard_count; index++)); do
  all_jobs+=("state8-full-history-teacher-eval-shard$index")
done
export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure

mkdir -p "$state/preserved_results" "$state/semantic"
cd "$project" || exit 1
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"
trap 'rm -f "$state/controller.active.pid"' EXIT
rm -f "$state/needs_attention"
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }

wait_quota(){
  local required="$1" quota
  while true; do
    heartbeat
    quota="$(timeout 90 amlt --json-tables ti sing -t Feeds 2>&1)" || { sleep 30; continue; }
    if "$python_bin" - "$quota" "$required" "$state/quota.json" <<'PY'
import json,sys
from scripts.feeds_quota_utils import parse_json_table,select_a100_80gb
row=select_a100_80gb(parse_json_table(sys.argv[1])); required=int(sys.argv[2])
q=row['quotas_(used/limit)']; standard_limit=int(q['standard']['limit'])
available=standard_limit-int(q['standard']['used'])
user_limit=int(q['user_max']['limit']); user=user_limit-int(q['user_max'].get('used') or 0)
open(sys.argv[3],'w').write(json.dumps({'selected_row':row,'required':required,
 'standard_available':available,'user_available':user,
 'scheduler_queue_allowed':True},indent=2,sort_keys=True)+'\n')
assert standard_limit>=required and user_limit>=required
PY
    then return 0; fi
    sleep 30
  done
}

preserve_passed(){
  local experiment="$1" status
  status="$(timeout 90 amlt status "$experiment" 2>&1)" || return 1
  grep -Eqi '(^|[[:space:]])pass([[:space:]]|$)' <<< "$status" || return 0
  amlt results download "$experiment" --status pass \
    --include '*.json' --include '*.jsonl' --output "$state/preserved_results" \
    >> "$state/controller.log" 2>&1
}

submit_eval_jobs(){
  local experiment="$1"; shift
  local names=("$@") selectors=() current class rc output
  for name in "${names[@]}"; do selectors+=(":$name"); done
  exec 8>"$project/results/feeds_submission_global.lock"; flock 8
  amlt run "$config" "${selectors[@]}" --dump -t Feeds \
    > "$state/${experiment}.rendered.yaml" 2> "$state/${experiment}.rendered.stderr" \
    || { flock -u 8; return 1; }
  "$python_bin" scripts/validate_state8_teacher_eval_sharded_amlt.py \
    --config "$state/${experiment}.rendered.yaml" --num-shards "$shard_count" \
    --jobs "${names[@]}" \
    >> "$state/controller.log" 2>&1 \
    || { flock -u 8; return 1; }
  current="$(amlt status --no-update-db "$experiment" 2>&1)"; rc=$?
  class="$("$python_bin" - "$rc" "$current" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]),sys.argv[2]))
PY
  )" || { flock -u 8; return 1; }
  if [[ "$class" == missing ]]; then
    # Existing recovery experiments do not consume new quota.  Check first so
    # an eager/manual recovery can be adopted even while cluster quota is full.
    wait_quota "${#names[@]}"
    output="$(amlt run "$config" "${selectors[@]}" "$experiment" -t Feeds -y \
      --description "Qualified full-history teacher; deterministic failed-sibling shard recovery; frozen adapter/data/prompt/decoding" 2>&1)"; rc=$?
    printf '[%s] rc=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$output" \
      > "$state/${experiment}.submission.txt"
    (( rc == 0 )) || { flock -u 8; return 1; }
  elif [[ "$class" != exists ]]; then
    flock -u 8; return 1
  fi
  flock -u 8
  printf '%s\n' "$experiment" > "$state/active_experiment.txt"
  printf '%s\n' "${names[@]}" > "$state/active_jobs.txt"
}

experiment="$base_experiment"
wave=0
current="$(amlt status --no-update-db "$experiment" 2>&1)"; rc=$?
class="$("$python_bin" - "$rc" "$current" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]),sys.argv[2]))
PY
)" || fail "could not classify initial teacher-shard experiment"
if [[ "$class" == missing ]]; then
  submit_eval_jobs "$experiment" "${all_jobs[@]}" \
    || fail "initial clean teacher-shard submission failed"
elif [[ "$class" != exists ]]; then
  fail "ambiguous initial teacher-shard lookup"
fi
printf '%s\n' "$experiment" > "$state/active_experiment.txt"
printf '%s\n' "${all_jobs[@]}" > "$state/active_jobs.txt"
while true; do
  heartbeat
  status="$(timeout 90 amlt status "$experiment" 2>&1)" || { sleep 60; continue; }
  printf '[%s]\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" >> "$state/status.log"
  expected="$(wc -l < "$state/active_jobs.txt")"
  if "$python_bin" - "$status" "$expected" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1],int(sys.argv[2])) else 1)
PY
  then
    preserve_passed "$experiment" || fail "could not preserve passed teacher shards"
    break
  fi
  if "$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import has_forward_progress_job
raise SystemExit(0 if has_forward_progress_job(sys.argv[1]) else 1)
PY
  then sleep 60; continue; fi
  preserve_passed "$experiment" || fail "could not preserve passed teacher siblings"
  mapfile -t failed < <("$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import terminal_nonpass_job_names
values=terminal_nonpass_job_names(sys.argv[1])
sys.stdout.write(''.join(value+'\n' for value in values))
PY
  )
  mapfile -t paused < <("$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import paused_job_names
values=paused_job_names(sys.argv[1])
sys.stdout.write(''.join(value+'\n' for value in values))
PY
  )
  for job in "${paused[@]}"; do
    log "converting stalled paused teacher shard to explicit recovery: $job"
    timeout 90 amlt cancel "$experiment" ":$job" -y \
      >> "$state/controller.log" 2>&1 \
      || fail "could not cancel paused teacher shard $job"
    failed+=("$job")
  done
  (( ${#failed[@]} > 0 )) || fail "terminal teacher-shard state lacks failed jobs"
  for job in "${failed[@]}"; do
    timeout 180 amlt logs view -n 1200 "$experiment" ":$job" \
      > "$state/${experiment}.${job}.failure.log" 2>&1 || true
  done
  (( wave < max_recovery_waves )) || fail "teacher shards exhausted recovery waves"
  wave=$((wave+1)); experiment="${recovery_prefix}${wave}-20260805"
  submit_eval_jobs "$experiment" "${failed[@]}" || fail "teacher shard recovery submit failed"
done

rm -rf "$state/download"; mkdir -p "$state/download"
cp -a "$state/preserved_results/." "$state/download/"
readarray -t predictions < <(find "$state/download" -type f -path '*/eval/t3/predictions.jsonl' | sort)
readarray -t metrics < <(find "$state/download" -type f -path '*/eval/t3/metrics.json' | sort)
readarray -t clean_receipts < <(find "$state/download" -type f -path '*/eval/t3/clean_prompt_receipt.json' | sort)
readarray -t teacher_evals < <(find "$teacher_state/train_download" -type f -path '*/metrics/teacher_eval.jsonl' | sort)
[[ ${#predictions[@]} -eq "$shard_count" && ${#metrics[@]} -eq "$shard_count" && ${#clean_receipts[@]} -eq "$shard_count" && ${#teacher_evals[@]} -eq 1 ]] \
  || fail "sharded result cardinality drift predictions=${#predictions[@]} metrics=${#metrics[@]} clean_receipts=${#clean_receipts[@]} eval=${#teacher_evals[@]}"
"$python_bin" - "${metrics[@]}" -- "${clean_receipts[@]}" <<'PY' \
  || fail "clean-prompt shard provenance validation failed"
import json,sys
split=sys.argv.index('--'); metrics=sys.argv[1:split]; receipts=sys.argv[split+1:]
assert len(metrics)==len(receipts)>=2,(metrics,receipts)
expected_input='78850c5434f24c3072b7ab89370400347955686e9aac5b99aadb69ae19c7cfa5'
expected_clean='c35fd46e0cfbccd9141eccb524992bd5452416d32720686f74ad9373f57de314'
for path in metrics:
    value=json.load(open(path))
    assert value['artifact_sha256']['evaluation_input']==expected_input,value
for path in receipts:
    value=json.load(open(path))
    assert value['stage']=='state8_teacher_clean_prompt_v2' and value['rows']==1494,value
    assert value['changed_rows']==600 and value['legacy_answer_adjacent_contracts']==0,value
    assert value['source_counts']=={'eedi_mining':894,'malrule':600},value
    assert value['output_sha256']==expected_clean,value
PY
rm -rf "$state/merged"; mkdir -p "$state/merged"
"$python_bin" scripts/merge_state8_teacher_eval_shards.py \
  --predictions "${predictions[@]}" --metrics "${metrics[@]}" \
  --eval-jsonl "${teacher_evals[0]}" --output-dir "$state/merged" \
  >> "$state/controller.log" 2>&1 || fail "teacher shard merge audit failed"
touch "$state/merge_complete"

exec 4>"$teacher_state/teacher_semantic_judge.lock"; flock -x 4
if [[ -f "$teacher_state/teacher_gate_pass" ]]; then
  # The sibling accelerator already judged this immutable teacher/eval pair.
  # Reuse only after checking its published artifact identity against this
  # independently merged generation run.
  "$python_bin" - "$teacher_state/teacher_gate.json" \
    "$teacher_state/teacher_provenance_gate.json" "$state/merged/metrics.json" <<'PY' \
    || { flock -u 4; fail "published teacher gate identity drift"; }
import json,sys
gate=json.load(open(sys.argv[1])); provenance=json.load(open(sys.argv[2])); merged=json.load(open(sys.argv[3]))
assert gate['pass'] is True and gate['support']==1494,gate
assert provenance['teacher_gate_pass'] is True,provenance
assert provenance['teacher_adapter_sha256']==merged['artifact_sha256']['adapter_model'],(provenance,merged)
assert provenance['evaluation_input_sha256']==merged['artifact_sha256']['evaluation_input'],(provenance,merged)
assert provenance['clean_prompt_sha256']=='c35fd46e0cfbccd9141eccb524992bd5452416d32720686f74ad9373f57de314',provenance
PY
  cp "$teacher_state/teacher_gate.json" "$state/teacher_gate.json"
  cp "$teacher_state/teacher_provenance_gate.json" "$state/teacher_provenance_gate.json"
  touch "$state/scientific_gate_pass" "$state/semantic_reused"
  log "reused published semantic teacher gate after immutable identity audit"
else
  attempt=0
  while true; do
    heartbeat; attempt=$((attempt+1))
    if PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
        --predictions-jsonl "$state/merged/predictions.jsonl" \
        --eval-jsonl "${teacher_evals[0]}" --output-dir "$state/semantic" \
        --judge-backend trapi --judge-workers 12 --judge-max-retries 12 \
        --judge-timeout-seconds 60 --judge-cache "$state/semantic/cache.jsonl" \
        --judge-audit-samples 100 --seed 42 >> "$state/semantic.log" 2>&1; then
      break
    fi
    (( attempt < 30 )) || { flock -u 4; fail "TRAPI teacher semantic judge exhausted"; }
    sleep 120
  done
  "$python_bin" scripts/validate_state8_teacher_gate.py \
    --metrics "$state/semantic/metrics.json" --output "$state/teacher_gate.json" \
    >> "$state/controller.log" 2>&1
  gate_rc=$?
  if (( gate_rc != 0 )); then
    touch "$state/scientific_gate_failed" "$state/needs_attention"
    flock -u 4
    log "teacher semantic gate failed rc=$gate_rc; student distillation remains stopped"
    exit "$gate_rc"
  fi
  touch "$state/scientific_gate_pass"
  "$python_bin" - "$state/teacher_gate.json" "$state/merged/metrics.json" \
    "${clean_receipts[0]}" "$state/teacher_provenance_gate.json" <<'PY' \
    || { flock -u 4; fail "teacher provenance gate construction failed"; }
import json,pathlib,sys
gate=json.load(open(sys.argv[1])); merged=json.load(open(sys.argv[2])); clean=json.load(open(sys.argv[3]))
assert gate['pass'] is True and gate['exact_match_is_authoritative'] is False,gate
assert merged['sampled_rows']==1494 and merged['evaluation_shards']>=2,merged
assert merged['artifact_sha256']['evaluation_input']=='78850c5434f24c3072b7ab89370400347955686e9aac5b99aadb69ae19c7cfa5',merged
assert clean['stage']=='state8_teacher_clean_prompt_v2' and clean['legacy_answer_adjacent_contracts']==0,clean
out={'status':'complete','stage':'state8_clean_prompt_teacher_provenance_gate_v2',
 'teacher_gate_pass':True,'teacher_adapter_sha256':merged['artifact_sha256']['adapter_model'],
 'evaluation_input_sha256':merged['artifact_sha256']['evaluation_input'],
 'clean_prompt_sha256':clean['output_sha256'],'changed_prompt_rows':clean['changed_rows'],
 'legacy_answer_adjacent_contracts':clean['legacy_answer_adjacent_contracts'],
 'evaluation_shards':merged['evaluation_shards'],'evaluation_rows':merged['sampled_rows'],
 'output_schema':'label_diagnosis_evidence_v1'}
pathlib.Path(sys.argv[4]).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
PY

  exec 7>"$teacher_state/teacher_gate_publish.lock"; flock -x 7
  az storage blob upload --account-name "$account" --auth-mode login --container-name "$container" \
    --name "$teacher_prefix/metrics/teacher_gate.json" --file "$state/teacher_gate.json" --overwrite true \
    >> "$state/controller.log" 2>&1 || fail "teacher gate publication failed"
  az storage blob upload --account-name "$account" --auth-mode login --container-name "$container" \
    --name "$teacher_prefix/metrics/teacher_provenance_gate.json" --file "$state/teacher_provenance_gate.json" --overwrite true \
    >> "$state/controller.log" 2>&1 || fail "teacher provenance publication failed"
  cp "$state/teacher_gate.json" "$teacher_state/teacher_gate.json"
  cp "$state/teacher_provenance_gate.json" "$teacher_state/teacher_provenance_gate.json"
  touch "$teacher_state/teacher_gate_pass"
  flock -u 7
fi
flock -u 4
log "qualified sharded teacher gate passed and published"

# Submit and own the primary Stage-B run so downstream promotion is not held by
# the slower independent single-job teacher replication.
"$python_bin" scripts/validate_state8_qualified_teacher_curriculum_amlt.py \
  --config "$curriculum_config" >> "$state/controller.log" 2>&1 \
  || fail "primary Stage-B config invalid"
curriculum_wave=0
curriculum_experiment="$curriculum_base"
while true; do
  printf '%s\n' "$curriculum_experiment" > "$state/active_curriculum_experiment.txt"
  # Both the four-way replication and the sixteen-way accelerator can pass the
  # same immutable teacher gate.  Serialize lookup plus submission so both
  # controllers cannot observe a missing Stage-B experiment before either one
  # creates it.
  exec 6>"$project/results/feeds_submission_global.lock"; flock 6
  current="$(amlt status --no-update-db "$curriculum_experiment" 2>&1)"; rc=$?
  class="$("$python_bin" - "$rc" "$current" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]),sys.argv[2]))
PY
)" || { flock -u 6; fail "could not classify primary Stage-B"; }
  if [[ "$class" == missing ]]; then
    # Validate the target and request ceiling while retaining the lock. Current
    # headroom may be zero: Feeds accepts the job into its scheduler queue.
    wait_quota 4
    amlt run "$curriculum_config" ":$curriculum_job" "$curriculum_experiment" -t Feeds -y \
      --description "Qualified frozen teacher to state8 primary curriculum; five-phase dropout; state-only inference; C4 specificity; sharded teacher gate; recovery_wave=$curriculum_wave" \
      > "$state/${curriculum_experiment}.submission.txt" 2>&1
    rc=$?
    (( rc == 0 )) || { flock -u 6; fail "primary Stage-B submission failed"; }
  elif [[ "$class" != exists ]]; then
    flock -u 6
    fail "ambiguous primary Stage-B lookup"
  fi
  flock -u 6
  while true; do
    heartbeat
    status="$(timeout 90 amlt status "$curriculum_experiment" 2>&1)" || { sleep 60; continue; }
    printf '[%s]\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" >> "$state/curriculum_status.log"
    if "$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1],1) else 1)
PY
    then break 2; fi
    if "$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import has_active_job
raise SystemExit(0 if has_active_job(sys.argv[1]) else 1)
PY
    then sleep 60; continue; fi
    timeout 180 amlt logs view -n 1500 "$curriculum_experiment" ":$curriculum_job" \
      > "$state/${curriculum_experiment}.failure.log" 2>&1 || true
    break
  done
  (( curriculum_wave < max_recovery_waves )) || fail "primary Stage-B exhausted recoveries"
  curriculum_wave=$((curriculum_wave+1))
  curriculum_experiment="${curriculum_recovery_prefix}${curriculum_wave}-20260805"
done

exec 5>"$teacher_state/stage_b_finalize.lock"; flock -x 5
if [[ ! -f "$teacher_state/complete" ]]; then
  rm -rf "$teacher_state/stage_b_download"; mkdir -p "$teacher_state/stage_b_download"
  amlt results download "$curriculum_experiment" --status pass \
    --include 'adapter_model.safetensors' --include 'adapter_config.json' \
    --include 'recurrent_state.pt' --include 'completion.json' \
    --include 'qualified_teacher_receipt.json' --include 'immutable_receipt.json' \
    --include 'metrics.json' --include 'predictions.jsonl' \
    --output "$teacher_state/stage_b_download" >> "$state/controller.log" 2>&1 \
    || { flock -u 5; fail "primary Stage-B download failed"; }
  readarray -t receipts < <(find "$teacher_state/stage_b_download" -type f -name qualified_teacher_receipt.json)
  readarray -t immutable < <(find "$teacher_state/stage_b_download" -type f -name immutable_receipt.json)
  readarray -t screens < <(find "$teacher_state/stage_b_download" -type f -path '*/eval/formal_screen/metrics.json')
  [[ ${#receipts[@]} -eq 1 && ${#immutable[@]} -eq 1 && ${#screens[@]} -eq 1 ]] \
    || { flock -u 5; fail "primary Stage-B artifact cardinality drift"; }
  "$python_bin" - "${receipts[0]}" "${immutable[0]}" "${screens[0]}" <<'PY' \
    || { flock -u 5; fail "primary Stage-B artifact validation failed"; }
import json,sys
q=json.load(open(sys.argv[1])); i=json.load(open(sys.argv[2])); m=json.load(open(sys.argv[3]))
assert q['stage']=='state8_qualified_teacher_curriculum_v1' and q['replay_variant']=='t3_heavy',q
assert q['micro_steps']==5000 and q['state_shape']==[64,8,5120],q
assert q['raw_history_visible_to_decoder'] is False and q['persistent_history_kv_cache'] is False,q
assert i['stage']=='state8_qualified_teacher_curriculum_immutable_v1',i
assert i['sha256']=={'adapter_model':q['adapter_sha256'],'recurrent_state':q['recurrent_state_sha256']},(i,q)
assert m['state_shape']==[64,8,5120] and m['sampled_rows']==487,m
assert m['raw_history_visible_to_decoder'] is False and m['persistent_history_kv_cache'] is False,m
PY
  touch "$teacher_state/stage_b_submitted" "$teacher_state/complete"
fi
touch "$state/complete"
flock -u 5
heartbeat
log "sharded accelerator completed qualified teacher plus primary Stage-B"
