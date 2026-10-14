#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
probe_step="${STATE8_EXACT_PROBE_STEP:-2000}"
case "$probe_step" in 2000|3000|4000|5000) ;; *) exit 2 ;; esac
if [[ "$probe_step" == 2000 ]]; then
  config="$project/amlt/feeds_state8_exact_diagnosis_midpoint_eval_20260809.yaml"
  state="$project/results/state8_exact_diagnosis_midpoint_eval_20260809"
  base_experiment=state8-exact-diagnosis-midpoint-eval-20260809
  recovery_prefix=state8-exact-diagnosis-midpoint-eval-recovery
else
  config="$project/amlt/feeds_state8_exact_diagnosis_checkpoint${probe_step}_eval_20260809.yaml"
  state="$project/results/state8_exact_diagnosis_checkpoint${probe_step}_eval_20260809"
  base_experiment="state8-exact-diagnosis-checkpoint${probe_step}-eval-20260809"
  recovery_prefix="state8-exact-diagnosis-checkpoint${probe_step}-eval-recovery"
fi
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
eval_jsonl="$project/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
recovery_suffix=-20260809
max_recovery_waves=16
jobs=(
  "state8-exact-diagnosis-midpoint-step${probe_step}-t3"
  "state8-exact-diagnosis-midpoint-step${probe_step}-c4"
)

if [[ -z "${AZURE_CONFIG_DIR:-}" ]]; then
  export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure
fi
mkdir -p "$state/preserved_results"
cd "$project" || exit 1
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
sleep_unlocked(){ sleep "$1" 9>&-; }
echo $$ > "$state/controller.active.pid"
heartbeat_pid=""
cleanup(){
  if [[ -n "$heartbeat_pid" ]]; then
    kill "$heartbeat_pid" 2>/dev/null || true
    wait "$heartbeat_pid" 2>/dev/null || true
  fi
  rm -f "$state/controller.active.pid"
}
trap cleanup EXIT
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }
heartbeat
(exec 9>&-; while true; do heartbeat; sleep 30; done) &
heartbeat_pid=$!

# A later checkpoint is produced by the concurrently running exact curriculum.
# Wait for its immutable completion receipt instead of submitting jobs that are
# guaranteed to fail before the remote input exists.
if [[ "$probe_step" != 2000 ]]; then
  checkpoint_blob="studentmodeling/checkpoints/state8_qualified_teacher_curriculum_seed42_exact5000_rounds1250_kd1_v2_20260805_taxonomy_semantic_v1_exact_diagnosis_first_seed64_single_ga4/checkpoint-${probe_step}/checkpoint_completion.json"
  while [[ "$(az storage blob exists --auth-mode login \
      --account-name csnewsandfeeds4150361735 --container-name amulet \
      --name "$checkpoint_blob" --query exists -o tsv 2>/dev/null)" != true ]]; do
    [[ -f "$project/results/state8_exact_diagnosis_pipeline_20260808/needs_attention" ]] \
      && fail "parent exact curriculum needs attention before checkpoint-${probe_step}"
    heartbeat; sleep_unlocked 60
  done
  log "observed immutable checkpoint-${probe_step} completion receipt"
fi

status_class(){
  STATUS="$1" "$python_bin" - <<'PY'
import json,os
rows=json.loads(os.environ['STATUS'])
states={str(row.get('status') or '').casefold() for row in rows}
active={'preparing','pre-queued','queued','running','paused','resuming'}
if states and states <= {'pass'}: print('pass')
elif states & active: print('active')
elif states: print('terminal')
else: print('invalid')
PY
}

submit_jobs(){
  local experiment="$1"; shift
  local requested=("$@") selectors=() rc status phase
  for job in "${requested[@]}"; do selectors+=(":$job"); done
  exec 8>"$project/results/feeds_submission_global.lock"; flock -x 8
  "$python_bin" scripts/validate_state8_exact_diagnosis_midpoint_eval_amlt.py \
    --config "$config" >> "$state/controller.log" 2>&1 \
    || { flock -u 8; return 1; }
  amlt run "$config" "${selectors[@]}" --dump -t Feeds \
    > "$state/$experiment.preflight.yaml" \
    2> "$state/$experiment.preflight.stderr" \
    || { flock -u 8; return 1; }
  "$python_bin" scripts/validate_state8_exact_diagnosis_midpoint_eval_amlt.py \
    --config "$state/$experiment.preflight.yaml" \
    >> "$state/controller.log" 2>&1 \
    || { flock -u 8; return 1; }
  amlt --json-tables ti sing -t Feeds \
    > "$state/$experiment.quota.json" 2>> "$state/controller.log" \
    || { flock -u 8; return 1; }
  amlt run "$config" "${selectors[@]}" "$experiment" -t Feeds -y \
    --description "Read-only exact State8 checkpoint-${probe_step} diagnosis-first T3/C4 screen; strict state-only; semantic judge authoritative" \
    > "$state/$experiment.submission.txt" 2>&1
  rc=$?
  if (( rc != 0 )); then
    status="$(timeout 90 amlt --json-tables status "$experiment" --hide-urls 2>/dev/null)" \
      || { flock -u 8; return "$rc"; }
    phase="$(status_class "$status")" || { flock -u 8; return "$rc"; }
    [[ "$phase" == active || "$phase" == pass ]] \
      || { flock -u 8; return "$rc"; }
    printf '\nsubmission_rc=%s but exact experiment exists phase=%s\n' \
      "$rc" "$phase" >> "$state/$experiment.submission.txt"
  fi
  flock -u 8
}

preserve_passed(){
  local experiment="$1" status="$2" job
  mapfile -t observed < <(STATUS="$status" "$python_bin" - <<'PY'
import json,os
for row in json.loads(os.environ['STATUS']):
 if str(row.get('status') or '').casefold()=='pass':
  print(str(row['job_name']).lstrip(':'))
PY
  )
  for job in "${observed[@]}"; do
    [[ " ${jobs[*]} " == *" $job "* ]] || continue
    [[ -f "$state/passed.$job" ]] && continue
    amlt results download "$experiment" ":$job" \
      --include '*.json' --include '*.jsonl' --include '*.txt' \
      --output "$state/preserved_results" >> "$state/controller.log" 2>&1 \
      || return 1
    touch "$state/passed.$job"
    log "preserved passed midpoint job=$job experiment=$experiment"
  done
}

experiment="$base_experiment"
wave=0
if [[ -s "$state/active_experiment.txt" ]]; then
  experiment="$(<"$state/active_experiment.txt")"
  [[ "$experiment" =~ recovery([0-9]+)-20260809 ]] \
    && wave="${BASH_REMATCH[1]}"
fi

while [[ ! -f "$state/remote_passed" ]]; do
  missing=()
  for job in "${jobs[@]}"; do
    [[ -f "$state/passed.$job" ]] || missing+=("$job")
  done
  if (( ${#missing[@]} == 0 )); then
    touch "$state/remote_passed"
    break
  fi
  if [[ ! -f "$state/submitted.$experiment" ]]; then
    until submit_jobs "$experiment" "${missing[@]}"; do
      heartbeat; log "transient midpoint submission failure experiment=$experiment; retrying"
      sleep_unlocked 120
    done
    printf '%s\n' "$experiment" > "$state/active_experiment.txt"
    printf '%s\n' "${missing[@]}" > "$state/active_jobs.txt"
    touch "$state/submitted.$experiment"
    log "submitted midpoint experiment=$experiment jobs=${missing[*]}"
  fi
  heartbeat
  status="$(timeout 90 amlt --json-tables status "$experiment" --hide-urls 2>/dev/null)" \
    || { sleep_unlocked 120; continue; }
  printf '%s\n' "$status" > "$state/latest_status.json"
  preserve_passed "$experiment" "$status" \
    || fail "could not preserve passed midpoint jobs"
  missing=()
  for job in "${jobs[@]}"; do
    [[ -f "$state/passed.$job" ]] || missing+=("$job")
  done
  if (( ${#missing[@]} == 0 )); then
    touch "$state/remote_passed"
    break
  fi
  phase="$(status_class "$status")"
  [[ "$phase" == active ]] && { sleep_unlocked 120; continue; }
  timeout 90 amlt show "$experiment" > "$state/$experiment.show.txt" 2>&1 || true
  timeout 180 amlt logs view -n 3000 "$experiment" \
    > "$state/$experiment.failure.log" 2>&1 || true
  (( wave < max_recovery_waves )) || fail "midpoint evaluation exhausted recovery waves"
  wave=$((wave+1))
  experiment="${recovery_prefix}${wave}${recovery_suffix}"
  log "recovering failed midpoint jobs wave=$wave jobs=${missing[*]}"
done

mapfile -t t3_roots < <(find "$state/preserved_results" -type d -name "${jobs[0]}")
mapfile -t c4_roots < <(find "$state/preserved_results" -type d -name "${jobs[1]}")
[[ ${#t3_roots[@]} -eq 1 && ${#c4_roots[@]} -eq 1 ]] \
  || fail "preserved artifact root cardinality drift"
mapfile -t t3_predictions < <(find "${t3_roots[0]}" -type f -path '*/eval/formal_t3_75/predictions.jsonl')
mapfile -t t3_metrics < <(find "${t3_roots[0]}" -type f -path '*/eval/formal_t3_75/metrics.json')
mapfile -t answer_c4 < <(find "${c4_roots[0]}" -type f -path '*/eval/answer_c4_50/recurrent_causal_gate.json')
mapfile -t diag_predictions < <(find "${c4_roots[0]}" -type f -path '*/eval/diagnosis_c4_50/judge_predictions.jsonl')
mapfile -t diag_eval < <(find "${c4_roots[0]}" -type f -path '*/eval/diagnosis_c4_50/judge_eval.jsonl')
mapfile -t diag_metrics < <(find "${c4_roots[0]}" -type f -path '*/eval/diagnosis_c4_50/metrics.json')
[[ ${#t3_predictions[@]} -eq 1 && ${#t3_metrics[@]} -eq 1 \
   && ${#answer_c4[@]} -eq 1 && ${#diag_predictions[@]} -eq 1 \
   && ${#diag_eval[@]} -eq 1 && ${#diag_metrics[@]} -eq 1 ]] \
  || fail "evaluation artifact cardinality drift"
[[ -x "$judge_python" ]] || fail "TRAPI judge environment unavailable"

if [[ ! -f "$state/t3_semantic_complete" ]]; then
  mkdir -p "$state/t3_semantic"
  attempt=0
  until PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
      --predictions-jsonl "${t3_predictions[0]}" --eval-jsonl "$eval_jsonl" \
      --output-dir "$state/t3_semantic" --judge-backend trapi \
      --judge-workers 12 --judge-max-retries 12 --judge-timeout-seconds 60 \
      --judge-cache "$state/t3_semantic/cache.jsonl" --judge-audit-samples 75 \
      --seed 42 >> "$state/t3_semantic/judge.log" 2>&1; do
    attempt=$((attempt+1)); (( attempt < 30 )) || fail "midpoint T3 semantic judge exhausted"
    heartbeat; sleep_unlocked 120
  done
  touch "$state/t3_semantic_complete"
fi

if [[ ! -f "$state/c4_semantic_complete" ]]; then
  mkdir -p "$state/c4_semantic"
  attempt=0
  until PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
      --predictions-jsonl "${diag_predictions[0]}" --eval-jsonl "${diag_eval[0]}" \
      --output-dir "$state/c4_semantic" --judge-backend trapi \
      --judge-workers 12 --judge-max-retries 12 --judge-timeout-seconds 60 \
      --judge-cache "$state/c4_semantic/cache.jsonl" --judge-audit-samples 100 \
      --seed 42 >> "$state/c4_semantic/judge.log" 2>&1; do
    attempt=$((attempt+1)); (( attempt < 30 )) || fail "midpoint C4 semantic judge exhausted"
    heartbeat; sleep_unlocked 120
  done
  "$python_bin" scripts/summarize_state8_diagnosis_c4.py \
    --judge-redacted "$state/c4_semantic/predictions_redacted.jsonl" \
    --generation-metrics "${diag_metrics[0]}" \
    --output "$state/c4_semantic/summary.json" \
    >> "$state/c4_semantic/summarize.log" 2>&1 \
    || fail "midpoint C4 semantic summary failed"
  touch "$state/c4_semantic_complete"
fi

"$python_bin" - "${t3_metrics[0]}" "$state/t3_semantic/metrics.json" \
  "${answer_c4[0]}" "$state/c4_semantic/summary.json" "$state/summary.json" \
  "$probe_step" <<'PY'
import json,pathlib,sys
formal,t3,answer,diagnosis=map(lambda p:json.load(open(p)),sys.argv[1:5])
step=int(sys.argv[6])
groups=t3['metrics_by_group']; support=sum(int(r['llm_judge_n']) for r in groups.values())
api=sum(int(r['llm_judge_api_or_cache']) for r in groups.values())
shortcuts=sum(int(r['llm_judge_exact_shortcuts']) for r in groups.values())
assert t3['judge_backend']=='trapi' and t3['judge_errors']==0,t3
assert support==t3['generation_rows']==t3['judged_rows'] and api==support and shortcuts==0,groups
t3_accuracy=sum(float(r['llm_judge_accuracy'])*int(r['llm_judge_n']) for r in groups.values())/support
s=answer['summary']; clean=s['clean']
levels=[0.0,0.25,0.5,0.75,1.0]
counts={str(value):max(0,min(1000,step-index*1000)) for index,value in enumerate(levels)}
out={
 'stage':f'state8_exact_diagnosis_checkpoint{step}_screen_v1','status':'complete',
 'checkpoint_step':step,'state_shape':[64,8,5120],
 'history_dropout_counts':counts,
 'raw_history_visible_to_decoder':False,'persistent_history_kv_cache':False,
 'semantic_judge_authoritative':True,'exact_match_authoritative':False,
 'canonical_label_role':'auxiliary_ordinary_weight',
 't3_semantic_accuracy':t3_accuracy,'t3_support':support,
 't3_formal_metrics':formal['metrics_by_group'],
 'answer_c4':{'clean_n':s['clean_n'],'swap':clean['swap']['flip_to_A_rate'],
              'random':clean['random']['flip_to_A_rate'],
              'same_rule':clean['samemal']['flip_to_A_rate'],
              'noop':clean['noop']['flip_to_A_rate'],'D_specific':s['D_specific'],
              'checks':s['checks'],'scientific_gate_pass':s['scientific_gate_pass']},
 'diagnosis_c4':diagnosis,'artifact_sha256':answer['artifact_sha256'],
 'promotion_eligible':False,'diagnostic_role':'early_curriculum_health_screen',
}
path=pathlib.Path(sys.argv[5]); partial=path.with_suffix('.json.partial')
partial.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); partial.replace(path)
print(json.dumps(out,sort_keys=True))
PY
touch "$state/complete"
heartbeat
log "exact diagnosis-first midpoint semantic T3/C4 screen complete"
