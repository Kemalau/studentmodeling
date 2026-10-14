#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
stage_a="$project/results/state8_diagnosis_first_teacher_stage_a_20260810"
state="$project/results/state8_diagnosis_first_teacher_full_eval_20260810"
config="$project/amlt/feeds_state8_diagnosis_first_teacher_full_eval_16way_20260810.yaml"
base=state8-diagnosis-first-teacher-full-eval-16way-20260810
recovery_prefix=state8-diagnosis-first-teacher-full-eval-recovery
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
account=csnewsandfeeds4150361735
container=amulet
remote_prefix=studentmodeling/diagnostics/state8_diagnosis_first_full_history_teacher_v1
max_waves=32
all_jobs=(); for index in $(seq -w 0 15); do all_jobs+=("state8-diagnosis-first-teacher-full-eval-shard$index"); done
export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure

mkdir -p "$state/preserved_results" "$state/semantic"
cd "$project" || exit 1
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"
cleanup(){ rm -f "$state/controller.active.pid"; }
trap cleanup EXIT
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }
sleep_unlocked(){ sleep "$1" 9>&-; }
rm -f "$state/needs_attention"

while [[ ! -f "$stage_a/complete" ]]; do
  heartbeat
  if [[ -f "$stage_a/needs_attention" || -f "$stage_a/scientific_gate_failed" ]]; then
    fail "75-row Teacher gate stopped full Teacher evaluation"
  fi
  sleep_unlocked 60
done
[[ -f "$stage_a/scientific_gate_pass" ]] || fail "Teacher Stage A completed without passing"

available_slots(){
  local audit="$state/active_a100.json"
  "$python_bin" scripts/count_active_a100_requests.py --user-limit 32 --recent 500 \
    --output "$audit" >/dev/null 2>> "$state/controller.log" || return 1
  "$python_bin" - "$audit" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); assert x['auditable'] is True,x
print(int(x['available_user_slots']))
PY
}

preserve_passed(){
  local experiment="$1" status job
  status="$(timeout 90 amlt --json-tables status "$experiment" --hide-urls 2>/dev/null)" || return 1
  mapfile -t passed < <(STATUS="$status" "$python_bin" - <<'PY'
import json,os
for row in json.loads(os.environ['STATUS']):
 if str(row.get('status') or '').casefold()=='pass': print(str(row['job_name']).lstrip(':'))
PY
  )
  (( ${#passed[@]} > 0 )) || return 0
  amlt results download "$experiment" --status pass --include '*.json' --include '*.jsonl' \
    --output "$state/preserved_results" >> "$state/controller.log" 2>&1 || return 1
  for job in "${passed[@]}"; do touch "$state/passed.$job"; done
}

submit_wave(){
  local experiment="$1"; shift
  local requested=("$@") available capacity current rc class output
  local selected=() selectors=()
  exec 8>"$project/results/feeds_submission_global.lock"; flock -x 8
  available="$(available_slots)" || { flock -u 8; return 1; }
  capacity=$(( available < ${#requested[@]} ? available : ${#requested[@]} ))
  (( capacity > 0 )) || { flock -u 8; return 1; }
  selected=("${requested[@]:0:capacity}")
  for job in "${selected[@]}"; do selectors+=(":$job"); done
  amlt run "$config" "${selectors[@]}" --dump -t Feeds \
    > "$state/$experiment.preflight.yaml" 2> "$state/$experiment.preflight.stderr" \
    || { flock -u 8; return 1; }
  "$python_bin" scripts/validate_state8_diagnosis_first_teacher_full_eval_amlt.py \
    --config "$state/$experiment.preflight.yaml" --jobs "${selected[@]}" \
    >> "$state/controller.log" 2>&1 || { flock -u 8; return 1; }
  amlt --json-tables ti sing -t Feeds > "$state/$experiment.quota.json" \
    || { flock -u 8; return 1; }
  current="$(amlt status --no-update-db "$experiment" 2>&1)"; rc=$?
  class="$("$python_bin" - "$rc" "$current" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]),sys.argv[2]))
PY
  )" || { flock -u 8; return 1; }
  if [[ "$class" == missing ]]; then
    output="$(amlt run "$config" "${selectors[@]}" "$experiment" -t Feeds -y \
      --description "Frozen diagnosis-first Teacher full semantic evaluation; disjoint shards; failed siblings recover independently" 2>&1)"; rc=$?
    printf '[%s] rc=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$output" > "$state/$experiment.submission.txt"
    (( rc == 0 )) || { flock -u 8; return 1; }
  elif [[ "$class" != exists ]]; then
    flock -u 8; return 1
  fi
  flock -u 8
  printf '%s\n' "$experiment" > "$state/active_experiment.txt"
  printf '%s\n' "${selected[@]}" > "$state/active_jobs.txt"
  log "submitted full Teacher eval experiment=$experiment shards=${#selected[@]}"
}

missing_jobs(){
  local job
  for job in "${all_jobs[@]}"; do [[ -f "$state/passed.$job" ]] || printf '%s\n' "$job"; done
}

wave=0
while true; do
  mapfile -t missing < <(missing_jobs)
  (( ${#missing[@]} > 0 )) || break
  if [[ -s "$state/active_experiment.txt" && -s "$state/active_jobs.txt" ]]; then
    experiment="$(<"$state/active_experiment.txt")"
    [[ "$experiment" =~ recovery([0-9]+)-20260810$ ]] && wave="${BASH_REMATCH[1]}"
  else
    experiment="$base"
    until submit_wave "$experiment" "${missing[@]}"; do heartbeat; sleep_unlocked 60; done
  fi
  while true; do
    heartbeat
    status="$(timeout 90 amlt status "$experiment" 2>&1)" || { sleep_unlocked 60; continue; }
    printf '[%s]\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" >> "$state/status.log"
    # A multi-job wave can contain healthy running siblings and a paused job
    # at the same time.  Resume paused siblings before treating the aggregate
    # as forward-progressing, otherwise one paused shard can sit idle until
    # every running sibling reaches a terminal state.
    mapfile -t paused < <("$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import paused_job_names
values=paused_job_names(sys.argv[1])
sys.stdout.write(''.join(value+'\n' for value in values))
PY
    )
    if (( ${#paused[@]} > 0 )); then
      # Fail closed on a malformed/empty parser result.  Passing an empty
      # selector to AMLT becomes the literal job reference `:`, which neither
      # resumes a shard nor identifies which row caused the parse drift.
      selectors=(); validated_paused=()
      for job in "${paused[@]}"; do
        [[ -n "$job" ]] || continue
        for expected in "${all_jobs[@]}"; do
          if [[ "$job" == "$expected" ]]; then
            selectors+=(":$job"); validated_paused+=("$job"); break
          fi
        done
      done
      if (( ${#selectors[@]} > 0 )); then
        log "requesting resume for paused full-Teacher shards: ${validated_paused[*]}"
        timeout 90 amlt resume "$experiment" "${selectors[@]}" >> "$state/controller.log" 2>&1 || true
        sleep_unlocked 60; continue
      fi
      log "ignored empty or unknown paused-shard parser output"
    fi
    if "$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import has_forward_progress_job
raise SystemExit(0 if has_forward_progress_job(sys.argv[1]) else 1)
PY
    then sleep_unlocked 60; continue; fi
    preserve_passed "$experiment" || fail "could not preserve passed Teacher shards"
    break
  done
  rm -f "$state/active_experiment.txt" "$state/active_jobs.txt"
  mapfile -t missing < <(missing_jobs)
  (( ${#missing[@]} == 0 )) && break
  (( wave < max_waves )) || fail "full Teacher evaluation exhausted recovery waves"
  wave=$((wave+1)); experiment="${recovery_prefix}${wave}-20260810"
  until submit_wave "$experiment" "${missing[@]}"; do heartbeat; sleep_unlocked 60; done
done

rm -rf "$state/download"; mkdir -p "$state/download"
cp -a "$state/preserved_results/." "$state/download/"
mapfile -t predictions < <(find "$state/download" -type f -path '*/eval/t3/predictions.jsonl' | sort)
mapfile -t metrics < <(find "$state/download" -type f -path '*/eval/t3/metrics.json' | sort)
mapfile -t teacher_evals < <(find "$stage_a/download" -type f -path '*/metrics/teacher_eval.jsonl')
[[ ${#predictions[@]} -eq 16 && ${#metrics[@]} -eq 16 && ${#teacher_evals[@]} -eq 1 ]] \
  || fail "full Teacher artifact cardinality drift predictions=${#predictions[@]} metrics=${#metrics[@]} eval=${#teacher_evals[@]}"
rm -rf "$state/merged"; mkdir -p "$state/merged"
"$python_bin" scripts/merge_state8_teacher_eval_shards.py \
  --predictions "${predictions[@]}" --metrics "${metrics[@]}" \
  --eval-jsonl "${teacher_evals[0]}" --output-dir "$state/merged" \
  >> "$state/controller.log" 2>&1 || fail "full Teacher shard merge failed"

[[ -x "$judge_python" ]] || fail "TRAPI judge environment unavailable"
PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
  --predictions-jsonl "$state/merged/predictions.jsonl" \
  --eval-jsonl "${teacher_evals[0]}" --output-dir "$state/semantic" \
  --judge-backend trapi --trapi-deployment gpt-4o_2024-11-20 \
  --judge-workers 12 --judge-max-retries 12 \
  --judge-timeout-seconds 60 --judge-cache "$state/semantic/cache.jsonl" \
  --judge-audit-samples 100 --seed 66 >> "$state/semantic/judge.log" 2>&1 \
  || fail "full Teacher TRAPI judge failed"
"$python_bin" scripts/validate_state8_teacher_gate.py \
  --metrics "$state/semantic/metrics.json" --output "$state/teacher_full_gate.json" \
  >> "$state/controller.log" 2>&1
gate_rc=$?
if (( gate_rc != 0 )); then
  touch "$state/scientific_gate_failed" "$state/complete"
  log "full 1494-row Teacher semantic gate failed rc=$gate_rc"
  exit "$gate_rc"
fi
"$python_bin" - "$state/teacher_full_gate.json" "$state/merged/metrics.json" \
  "$stage_a/completed_publish_receipt.json" "$state/teacher_full_provenance_gate.json" <<'PY'
import json,pathlib,sys
g=json.load(open(sys.argv[1])); m=json.load(open(sys.argv[2])); p=json.load(open(sys.argv[3]))
assert g['pass'] is True and g['support']==1494 and g['exact_match_is_authoritative'] is False,g
assert m['sampled_rows']==1494 and m['evaluation_shards']==16,m
assert m['artifact_sha256']['adapter_model']==p['adapter_sha256'],(m,p)
out={'stage':'state8_diagnosis_first_teacher_full_provenance_gate_v1','status':'complete',
 'teacher_gate_pass':True,'teacher_adapter_sha256':p['adapter_sha256'],
 'evaluation_input_sha256':m['artifact_sha256']['evaluation_input'],
 'teacher_source_eval_sha256':p['eval_sha256'],'evaluation_shards':16,
 'evaluation_rows':1494,'output_schema':'diagnosis_evidence_label_v1',
 'diagnosis_field_order':['diagnosis','evidence','label'],
 'canonical_label_role':'auxiliary_supervision_and_reporting',
 'semantic_judge_authoritative':True,'exact_match_authoritative':False,
 'teacher_history_visible':True,'student_history_visible_at_final_inference':False}
pathlib.Path(sys.argv[4]).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
PY
for name in teacher_full_gate.json teacher_full_provenance_gate.json; do
  az storage blob upload --auth-mode login --account-name "$account" \
    --container-name "$container" --name "$remote_prefix/metrics/$name" \
    --file "$state/$name" --overwrite >> "$state/controller.log" 2>&1 \
    || fail "full Teacher gate upload failed name=$name"
done
touch "$state/scientific_gate_pass" "$state/complete"
heartbeat
log "full 1494-row diagnosis-first Teacher semantic gate passed and published"
