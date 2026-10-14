#!/usr/bin/env bash
set -uo pipefail

project_dir=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
fit_dir="${FIT_DIR:-$project_dir/results/state8_unified_final_fit_20260805}"
state_dir="${STATE_DIR:-$project_dir/results/state8_complete_final_eval_20260805}"
config="${FINAL_EVAL_CONFIG:-$project_dir/amlt/feeds_state8_final_eval_20260805.yaml}"
eval_jsonl="$project_dir/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project_dir/scripts:/home/v-yuejiali/fkft/src"
base_experiment="${BASE_EXPERIMENT:-state8-qualified-complete-final-eval-seed42-20260805}"
recovery_experiment_prefix="${RECOVERY_EXPERIMENT_PREFIX:-state8-qualified-complete-final-eval-recovery}"
recovery_experiment_suffix="${RECOVERY_EXPERIMENT_SUFFIX:--seed42-20260805}"
max_recovery_waves="${MAX_RECOVERY_WAVES:-32}"
if [[ -z "${AZURE_CONFIG_DIR:-}" ]]; then
  if AZURE_CONFIG_DIR=/home/v-yuejiali/.azure az account get-access-token \
      --subscription b6dc87f3-c479-49c8-8cb5-7896da3ff895 \
      --query expiresOn -o tsv >/dev/null 2>&1; then
    export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure
  else
    export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure-keyflip-feeds
  fi
fi
all_jobs=()
for index in $(seq -w 0 15); do all_jobs+=("state8-final-formal-shard$index"); done
all_jobs+=(state8-final-c4-seed13 state8-final-c4-seed42 state8-final-c4-seed97)

mkdir -p "$state_dir/preserved_results" "$state_dir/t3_semantic/full1494"
cd "$project_dir" || exit 1
exec 9>"$state_dir/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state_dir/controller.active.pid"
heartbeat_pid=""
cleanup(){
  if [[ -n "$heartbeat_pid" ]]; then
    kill "$heartbeat_pid" 2>/dev/null || true
    wait "$heartbeat_pid" 2>/dev/null || true
  fi
  rm -f "$state_dir/controller.active.pid"
}
trap cleanup EXIT
rm -f "$state_dir/needs_attention"

heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state_dir/needs_attention"; exit 1; }

while [[ ! -f "$fit_dir/complete" ]]; do
  heartbeat
  [[ ! -f "$fit_dir/needs_attention" ]] || fail "state8 exact final fit needs attention"
  sleep 120
done
[[ -f "$fit_dir/scientific_gate_pass" ]] || fail "state8 exact final fit gate failed"
readarray -t fit_receipts < <(find "$fit_dir/download" -type f -name state8_final_fit_receipt.json)
[[ ${#fit_receipts[@]} -eq 1 ]] || fail "expected one exact final-fit receipt"
"$python_bin" - "${fit_receipts[0]}" <<'PY' || fail "invalid exact final-fit receipt"
import json,sys
x=json.load(open(sys.argv[1])); assert x['status']=='final_fit_complete',x
assert x['one_epoch_exact'] is True and x['real_examples']==259845,x
assert x['state_shape']==[64,8,5120],x
assert x['teacher_prompt_provenance_stage']=='state8_clean_prompt_teacher_provenance_gate_v2',x
assert x['teacher_evaluation_input_sha256']=='78850c5434f24c3072b7ab89370400347955686e9aac5b99aadb69ae19c7cfa5',x
assert x['teacher_clean_prompt_sha256']=='c35fd46e0cfbccd9141eccb524992bd5452416d32720686f74ad9373f57de314',x
assert x['legacy_answer_adjacent_contracts']==0,x
assert x['raw_history_visible_to_decoder'] is False,x
assert x['persistent_history_kv_cache'] is False,x
assert len(x['adapter_sha256'])==64 and len(x['recurrent_state_sha256'])==64,x
PY

wait_quota(){
  local required="$1" quota
  while true; do
    heartbeat
    quota="$(timeout 90 amlt --json-tables ti sing -t Feeds 2>&1)" || { sleep 30; continue; }
    if "$python_bin" - "$quota" "$required" "$state_dir/quota.json" <<'PY'
import json,sys
from scripts.feeds_quota_utils import parse_json_table,select_a100_80gb
row=select_a100_80gb(parse_json_table(sys.argv[1])); n=int(sys.argv[2]); q=row['quotas_(used/limit)']
standard_limit=int(q['standard']['limit']); available=standard_limit-int(q['standard']['used'])
user_limit=int(q['user_max']['limit']); user_available=user_limit-int(q['user_max'].get('used') or 0)
open(sys.argv[3],'w').write(json.dumps({'selected_row':row,'required':n,'available':available,
 'user_available':user_available,'scheduler_queue_allowed':True},indent=2,sort_keys=True)+'\n')
assert standard_limit>=n and user_limit>=n
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
    --include '*.json' --include '*.jsonl' --include '*.txt' \
    --output "$state_dir/preserved_results" >> "$state_dir/controller.log" 2>&1
}

submit_jobs(){
  local experiment="$1"; shift
  local names=("$@") selectors=() output current class rc
  for name in "${names[@]}"; do selectors+=(":$name"); done
  exec 8>"$project_dir/results/feeds_submission_global.lock"; flock 8
  amlt run "$config" "${selectors[@]}" --dump -t Feeds \
    > "$state_dir/${experiment}.preflight.dump.yaml" \
    2> "$state_dir/${experiment}.preflight.stderr" || { flock -u 8; return 1; }
  "$python_bin" scripts/validate_state8_final_eval_amlt.py \
    --config "$state_dir/${experiment}.preflight.dump.yaml" --jobs "${names[@]}" \
    >> "$state_dir/controller.log" 2>&1 || { flock -u 8; return 1; }
  current="$(amlt status --no-update-db "$experiment" 2>&1)"; rc=$?
  class="$("$python_bin" - "$rc" "$current" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]),sys.argv[2]))
PY
)" || { flock -u 8; return 1; }
  if [[ "$class" == missing ]]; then
    wait_quota "${#names[@]}"
    output="$(amlt run "$config" "${selectors[@]}" "$experiment" -t Feeds -y \
      --description "Complete immutable state8 exact-epoch model: 16 disjoint 21676-row formal shards and three frozen C4 pair seeds; failed siblings recover independently" 2>&1)"; rc=$?
    printf '[%s] submit rc=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$output" \
      > "$state_dir/${experiment}.submission.txt"
    (( rc == 0 )) || { flock -u 8; return 1; }
  elif [[ "$class" == exists ]]; then
    printf 'status=preexisting_not_resubmitted\n%s\n' "$current" \
      > "$state_dir/${experiment}.submission.txt"
  else flock -u 8; return 1; fi
  flock -u 8
  printf '%s\n' "$experiment" > "$state_dir/active_experiment.txt"
  printf '%s\n' "${names[@]}" > "$state_dir/active_jobs.txt"
  log "submitted state8 final evaluation experiment=$experiment jobs=${#names[@]}"
}

experiment="$base_experiment"; wave=0
if [[ -s "$state_dir/active_experiment.txt" ]]; then
  experiment="$(<"$state_dir/active_experiment.txt")"
  [[ "$experiment" =~ recovery([0-9]+) ]] && wave="${BASH_REMATCH[1]}"
else
  submit_jobs "$experiment" "${all_jobs[@]}" || fail "initial state8 final-eval submission failed"
fi

while true; do
  heartbeat
  status="$(timeout 90 amlt status "$experiment" 2>&1)" || { sleep 120; continue; }
  printf '[%s]\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" >> "$state_dir/status.log"
  if "$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import has_forward_progress_job
raise SystemExit(0 if has_forward_progress_job(sys.argv[1]) else 1)
PY
  then
    sleep 120; continue
  fi
  expected_jobs="$(wc -l < "$state_dir/active_jobs.txt")"
  if "$python_bin" - "$status" "$expected_jobs" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1],int(sys.argv[2])) else 1)
PY
  then preserve_passed "$experiment" || fail "could not preserve final-eval results"; break; fi
  mapfile -t paused < <("$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import paused_job_names
values=paused_job_names(sys.argv[1])
sys.stdout.write(''.join(value+'\n' for value in values))
PY
  )
  if (( ${#paused[@]} > 0 )); then
    log "requesting resume for paused final-eval jobs: ${paused[*]}"
    selectors=(); for job in "${paused[@]}"; do selectors+=(":$job"); done
    timeout 90 amlt resume "$experiment" "${selectors[@]}" \
      >> "$state_dir/controller.log" 2>&1 || true
    sleep 120; continue
  fi
  preserve_passed "$experiment" || fail "could not preserve passed final-eval siblings"
  mapfile -t failed < <("$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import terminal_nonpass_job_names
values=terminal_nonpass_job_names(sys.argv[1])
sys.stdout.write(''.join(value+'\n' for value in values))
PY
  )
  (( ${#failed[@]} > 0 )) || fail "terminal final-eval status has no failed jobs"
  for job in "${failed[@]}"; do
    timeout 180 amlt logs view -n 2000 "$experiment" ":$job" \
      > "$state_dir/${experiment}.${job}.failure.log" 2>&1 || true
  done
  (( wave < max_recovery_waves )) || fail "state8 final evaluation exhausted recovery waves"
  wave=$((wave+1)); experiment="${recovery_experiment_prefix}${wave}${recovery_experiment_suffix}"
  submit_jobs "$experiment" "${failed[@]}" || fail "final-eval recovery failed wave=$wave"
done

rm -rf "$state_dir/download"; mkdir -p "$state_dir/download"
cp -a "$state_dir/preserved_results/." "$state_dir/download/"
readarray -t predictions < <(find "$state_dir/download" -type f -path '*/eval/formal_shard*/predictions.jsonl' | sort)
readarray -t metrics < <(find "$state_dir/download" -type f -path '*/eval/formal_shard*/metrics.json' | sort)
readarray -t c4_results < <(find "$state_dir/download" -type f -path '*/eval/c4_seed*/recurrent_causal_gate.json' | sort)
readarray -t diagnosis_predictions < <(find "$state_dir/download" -type f -path '*/eval/diagnosis_c4_seed*/judge_predictions.jsonl' | sort)
readarray -t diagnosis_evals < <(find "$state_dir/download" -type f -path '*/eval/diagnosis_c4_seed*/judge_eval.jsonl' | sort)
readarray -t diagnosis_metrics < <(find "$state_dir/download" -type f -path '*/eval/diagnosis_c4_seed*/metrics.json' | sort)
[[ ${#predictions[@]} -eq 16 && ${#metrics[@]} -eq 16 && ${#c4_results[@]} -eq 3 \
   && ${#diagnosis_predictions[@]} -eq 3 && ${#diagnosis_evals[@]} -eq 3 \
   && ${#diagnosis_metrics[@]} -eq 3 ]] \
  || fail "final result cardinality drift predictions=${#predictions[@]} metrics=${#metrics[@]} c4=${#c4_results[@]} diagnosis_predictions=${#diagnosis_predictions[@]} diagnosis_evals=${#diagnosis_evals[@]} diagnosis_metrics=${#diagnosis_metrics[@]}"
"$python_bin" scripts/merge_state8_final_eval.py \
  --predictions "${predictions[@]}" --metrics "${metrics[@]}" \
  --eval-jsonl "$eval_jsonl" --c4 "${c4_results[@]}" \
  --output-dir "$state_dir/merged" >> "$state_dir/controller.log" 2>&1 \
  || fail "state8 final merge audit failed"

exec 11>"$state_dir/t3_semantic/judge.lock"; flock -x 11
(while true; do heartbeat; sleep 60; done) & heartbeat_pid=$!
[[ -x "$judge_python" ]] || fail "TRAPI judge environment unavailable"
attempt=0
while [[ ! -f "$state_dir/t3_semantic/complete" ]]; do
  heartbeat; attempt=$((attempt+1))
  if PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
      --predictions-jsonl "$state_dir/merged/t3_predictions.jsonl" \
      --eval-jsonl "$eval_jsonl" --output-dir "$state_dir/t3_semantic/full1494" \
      --judge-backend trapi --judge-workers 12 --judge-max-retries 12 \
      --judge-timeout-seconds 60 --judge-cache "$state_dir/t3_semantic/cache.jsonl" \
      --judge-audit-samples 100 --seed 42 >> "$state_dir/t3_semantic/full.log" 2>&1; then
    "$python_bin" scripts/validate_t3_semantic_result.py \
      --metrics "$state_dir/t3_semantic/full1494/metrics.json" \
      --output "$state_dir/t3_semantic/t3_semantic_gate.json" \
      >> "$state_dir/t3_semantic/controller.log" 2>&1
    rc=$?
    if (( rc == 0 )); then touch "$state_dir/t3_semantic/scientific_gate_pass"
    elif (( rc == 2 )); then touch "$state_dir/t3_semantic/scientific_gate_failed"
    else fail "semantic T3 validator failed rc=$rc"; fi
    touch "$state_dir/t3_semantic/complete"; break
  fi
  (( attempt < 30 )) || fail "TRAPI semantic judge exhausted"
  sleep 120
done
kill "$heartbeat_pid" 2>/dev/null || true; wait "$heartbeat_pid" 2>/dev/null || true
heartbeat_pid=""; flock -u 11

mkdir -p "$state_dir/diagnosis_c4_semantic"
diagnosis_summaries=()
for index in 0 1 2; do
  seed="$("$python_bin" - "${diagnosis_metrics[$index]}" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['pair_seed'])
PY
)"
  semantic_dir="$state_dir/diagnosis_c4_semantic/seed$seed"
  mkdir -p "$semantic_dir"
  attempt=0
  while [[ ! -f "$semantic_dir/complete" ]]; do
    heartbeat; attempt=$((attempt+1))
    if PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
        --predictions-jsonl "${diagnosis_predictions[$index]}" \
        --eval-jsonl "${diagnosis_evals[$index]}" \
        --output-dir "$semantic_dir" --judge-backend trapi \
        --judge-workers 12 --judge-max-retries 12 --judge-timeout-seconds 60 \
        --judge-cache "$semantic_dir/cache.jsonl" \
        --judge-audit-samples 100 --seed "$seed" \
        >> "$semantic_dir/judge.log" 2>&1; then
      PYTHONPATH="$project_dir/scripts" "$python_bin" scripts/summarize_state8_diagnosis_c4.py \
        --judge-redacted "$semantic_dir/predictions_redacted.jsonl" \
        --generation-metrics "${diagnosis_metrics[$index]}" \
        --output "$semantic_dir/summary.json" \
        >> "$semantic_dir/controller.log" 2>&1 \
        || fail "diagnosis-C4 semantic summary failed seed=$seed"
      touch "$semantic_dir/complete"; break
    fi
    (( attempt < 30 )) || fail "diagnosis-C4 semantic judge exhausted seed=$seed"
    sleep 120
  done
  diagnosis_summaries+=("$semantic_dir/summary.json")
done
"$python_bin" - "${diagnosis_summaries[@]}" \
  "$state_dir/diagnosis_c4_semantic/multiseed.json" <<'PY'
import json,pathlib,statistics,sys
rows=[json.load(open(path)) for path in sys.argv[1:-1]]
seeds=[int(row['generation']['pair_seed']) for row in rows]
if set(seeds)!={13,42,97}: raise SystemExit(f'diagnosis-C4 seed drift: {seeds}')
result={'stage':'state8_closed_diagnosis_c4_semantic_multiseed_v1',
 'semantic_judge_authoritative':True,
 'exact_label_role':'auxiliary_only_not_used_for_gate','rows':rows,
 'seed_passes':sum(bool(row['pass']) for row in rows),
 'mean_semantic_D_specific':statistics.mean(row['semantic_D_specific'] for row in rows),
 'mean_swap_to_donor':statistics.mean(row['cells']['swap_to_donor']['semantic_equivalence_rate'] for row in rows),
 'mean_random_matches_either':statistics.mean(row['random_matches_either_stable_target_rate'] for row in rows),
 'pass':sum(bool(row['pass']) for row in rows)>=2}
pathlib.Path(sys.argv[-1]).write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
PY

"$python_bin" scripts/validate_state8_final_eval.py \
  --formal "$state_dir/merged/metrics.json" \
  --c4 "$state_dir/merged/c4_multiseed.json" \
  --semantic-gate "$state_dir/t3_semantic/t3_semantic_gate.json" \
  --semantic-metrics "$state_dir/t3_semantic/full1494/metrics.json" \
  --diagnosis-c4 "$state_dir/diagnosis_c4_semantic/multiseed.json" \
  --final-fit-receipt "${fit_receipts[0]}" \
  --output "$state_dir/final_gate.json" >> "$state_dir/controller.log" 2>&1
rc=$?
if (( rc == 0 )); then
  touch "$state_dir/scientific_gate_pass"
elif (( rc == 2 )); then
  touch "$state_dir/scientific_gate_failed"
else fail "state8 joint final validator failed rc=$rc"; fi
touch "$state_dir/complete"
heartbeat
log "state8 complete model final evaluation finished"
