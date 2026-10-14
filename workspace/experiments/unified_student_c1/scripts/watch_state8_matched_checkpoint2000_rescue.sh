#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
state="$project/results/state8_matched_checkpoint2000_rescue_20260810"
baseline="$project/results/state8_matched_checkpoint2000_screen_20260810/summary.json"
config="$project/amlt/feeds_state8_matched_checkpoint2000_rescue_20260810.yaml"
base=state8-matched-cp2000-causal-rescue-fix2-20260810
recovery_prefix=state8-matched-cp2000-causal-rescue-recovery
experiment="$base"
max_recovery_waves=24
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
eval_jsonl="$project/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
jobs=(
  state8-matched-cp2000-rescue-semantic-preserve
  state8-matched-cp2000-rescue-causal-taskanchor
)
modes=(semantic_preserve causal_taskanchor)

if [[ -z "${AZURE_CONFIG_DIR:-}" ]]; then
  if AZURE_CONFIG_DIR=/home/v-yuejiali/.azure az account get-access-token \
      --subscription b6dc87f3-c479-49c8-8cb5-7896da3ff895 \
      --query expiresOn -o tsv >/dev/null 2>&1; then
    export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure
  else
    export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure-studentmodeling-feeds
  fi
fi

mkdir -p "$state/preserved_results"
cd "$project" || exit 1
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"
cleanup(){ rm -f "$state/controller.active.pid"; }
trap cleanup EXIT
sleep_unlocked(){ sleep "$1" 9>&-; }
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }
rm -f "$state/needs_attention"

phase(){
  STATUS="$1" "$python_bin" - <<'PY'
import json,os
rows=json.loads(os.environ['STATUS'])
states={str(row.get('status') or '').casefold() for row in rows}
active={'preparing','pre-queued','queued','running','resuming'}
if states and states <= {'pass'}: print('pass')
elif states & active: print('active')
elif 'paused' in states: print('paused')
elif states: print('terminal')
else: print('invalid')
PY
}

submit_jobs(){
  local next="$1"; shift
  local requested=("$@") selectors=() output rc current class job
  for job in "${requested[@]}"; do selectors+=(":$job"); done
  exec 8>"$project/results/feeds_submission_global.lock"; flock -x 8
  "$python_bin" scripts/validate_state8_matched_checkpoint2000_rescue_amlt.py \
    --config "$config" "${requested[@]/#/--job=}" >> "$state/controller.log" 2>&1 \
    || { flock -u 8; return 1; }
  amlt run "$config" "${selectors[@]}" --dump -t Feeds \
    > "$state/$next.preflight.yaml" 2> "$state/$next.preflight.stderr" \
    || { flock -u 8; return 1; }
  "$python_bin" scripts/validate_state8_matched_checkpoint2000_rescue_amlt.py \
    --config "$state/$next.preflight.yaml" "${requested[@]/#/--job=}" \
    >> "$state/controller.log" 2>&1 || { flock -u 8; return 1; }
  amlt --json-tables ti sing -t Feeds > "$state/$next.quota.json" \
    || { flock -u 8; return 1; }
  output="$(amlt run "$config" "${selectors[@]}" "$next" -t Feeds -y \
    --description "Automatic hash-bound recovery of interrupted State8 checkpoint-2000 causal rescue jobs" 2>&1)"
  rc=$?; printf '%s\n' "$output" > "$state/$next.submission.txt"
  if (( rc != 0 )); then
    current="$(amlt status --no-update-db "$next" 2>&1)"; rc=$?
    class="$("$python_bin" - "$rc" "$current" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]),sys.argv[2]))
PY
)" || { flock -u 8; return 1; }
    [[ "$class" == exists ]] || { flock -u 8; return 1; }
  fi
  flock -u 8
}

preserve_passed(){
  local status="$1" job
  mapfile -t observed < <(STATUS="$status" "$python_bin" - <<'PY'
import json,os
for row in json.loads(os.environ['STATUS']):
 if str(row.get('status') or '').casefold()=='pass':
  print(str(row['job_name']).lstrip(':'))
PY
  )
  for job in "${observed[@]}"; do
    [[ " ${jobs[*]} " == *" $job "* ]] || continue
    [[ ! -f "$state/passed.$job" ]] || continue
    amlt results download "$experiment" ":$job" \
      --include '*.json' --include '*.jsonl' --include '*.txt' \
      --output "$state/preserved_results" >> "$state/controller.log" 2>&1 \
      || return 1
    touch "$state/passed.$job"
    log "preserved passed rescue job=$job"
  done
}

wave=0
if [[ -s "$state/active_experiment.txt" ]]; then
  experiment="$(<"$state/active_experiment.txt")"
  [[ "$experiment" =~ recovery([0-9]+)-20260810$ ]] && wave="${BASH_REMATCH[1]}"
else
  printf '%s\n' "$experiment" > "$state/active_experiment.txt"
fi

while [[ ! -f "$state/remote_passed" ]]; do
  heartbeat
  status="$(timeout 90 amlt --json-tables status "$experiment" --hide-urls 2>/dev/null)" \
    || { sleep_unlocked 60; continue; }
  printf '%s\n' "$status" > "$state/latest_status.json"
  preserve_passed "$status" || fail "could not download a passed rescue job"
  missing=()
  for job in "${jobs[@]}"; do [[ -f "$state/passed.$job" ]] || missing+=("$job"); done
  if (( ${#missing[@]} == 0 )); then touch "$state/remote_passed"; break; fi
  current_phase="$(phase "$status")"
  if [[ "$current_phase" == active ]]; then sleep_unlocked 60; continue; fi
  if [[ "$current_phase" == paused ]]; then
    selectors=(); for job in "${missing[@]}"; do selectors+=(":$job"); done
    timeout 90 amlt resume "$experiment" "${selectors[@]}" \
      >> "$state/controller.log" 2>&1 || true
    sleep_unlocked 60; continue
  fi
  timeout 90 amlt show "$experiment" > "$state/$experiment.show.txt" 2>&1 || true
  timeout 180 amlt logs view -n 3000 "$experiment" \
    > "$state/$experiment.failure.log" 2>&1 || true
  (( wave < max_recovery_waves )) \
    || fail "rescue exhausted automatic recovery waves: ${missing[*]}"
  wave=$((wave+1)); next="${recovery_prefix}${wave}-20260810"
  until submit_jobs "$next" "${missing[@]}"; do
    heartbeat; log "transient rescue recovery submission failure wave=$wave; retrying"
    sleep_unlocked 120
  done
  experiment="$next"
  printf '%s\n' "$experiment" > "$state/active_experiment.txt"
  log "submitted rescue recovery wave=$wave experiment=$experiment jobs=${missing[*]}"
done

[[ -x "$judge_python" ]] || fail "TRAPI judge environment unavailable"
[[ -s "$baseline" ]] || fail "checkpoint-2000 baseline summary unavailable"
for index in "${!jobs[@]}"; do
  job="${jobs[$index]}"; mode="${modes[$index]}"; root="$state/$mode"
  mkdir -p "$root/t3_semantic" "$root/c4_semantic"
  mapfile -t job_roots < <(find "$state/preserved_results" -type d -name "$job")
  [[ ${#job_roots[@]} -eq 1 ]] || fail "artifact root cardinality drift job=$job"
  job_root="${job_roots[0]}"
  mapfile -t formal_pred < <(find "$job_root" -type f -path '*/eval/formal_t3_300/predictions.jsonl')
  mapfile -t formal_metrics < <(find "$job_root" -type f -path '*/eval/formal_t3_300/metrics.json')
  mapfile -t diag_pred < <(find "$job_root" -type f -path '*/eval/diagnosis_c4_50/judge_predictions.jsonl')
  mapfile -t diag_eval < <(find "$job_root" -type f -path '*/eval/diagnosis_c4_50/judge_eval.jsonl')
  mapfile -t diag_metrics < <(find "$job_root" -type f -path '*/eval/diagnosis_c4_50/metrics.json')
  mapfile -t answer < <(find "$job_root" -type f -path '*/eval/answer_c4_50/recurrent_causal_gate.json')
  [[ ${#formal_pred[@]} -eq 1 && ${#formal_metrics[@]} -eq 1 \
      && ${#diag_pred[@]} -eq 1 && ${#diag_eval[@]} -eq 1 \
      && ${#diag_metrics[@]} -eq 1 && ${#answer[@]} -eq 1 ]] \
    || fail "evaluation artifact cardinality drift job=$job"

  if [[ ! -f "$root/t3_semantic_complete" ]]; then
    attempt=0
    until PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
        --predictions-jsonl "${formal_pred[0]}" --eval-jsonl "$eval_jsonl" \
        --output-dir "$root/t3_semantic" --judge-backend trapi \
        --trapi-deployment gpt-4o_2024-11-20 --judge-workers 12 \
        --judge-max-retries 12 --judge-timeout-seconds 60 \
        --judge-cache "$root/t3_semantic/cache.jsonl" \
        --judge-audit-samples 100 --seed 42 >> "$root/t3_semantic/judge.log" 2>&1; do
      attempt=$((attempt+1)); (( attempt < 30 )) || fail "T3 judge exhausted job=$job"
      heartbeat; sleep_unlocked 120
    done
    touch "$root/t3_semantic_complete"
  fi
  if [[ ! -f "$root/c4_semantic_complete" ]]; then
    attempt=0
    until PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
        --predictions-jsonl "${diag_pred[0]}" --eval-jsonl "${diag_eval[0]}" \
        --output-dir "$root/c4_semantic" --judge-backend trapi \
        --trapi-deployment gpt-4o_2024-11-20 --judge-workers 12 \
        --judge-max-retries 12 --judge-timeout-seconds 60 \
        --judge-cache "$root/c4_semantic/cache.jsonl" \
        --judge-audit-samples 100 --seed 42 >> "$root/c4_semantic/judge.log" 2>&1; do
      attempt=$((attempt+1)); (( attempt < 30 )) || fail "diagnosis-C4 judge exhausted job=$job"
      heartbeat; sleep_unlocked 120
    done
    "$python_bin" scripts/summarize_state8_diagnosis_c4.py \
      --judge-redacted "$root/c4_semantic/predictions_redacted.jsonl" \
      --generation-metrics "${diag_metrics[0]}" \
      --output "$root/c4_semantic/summary.json" \
      >> "$root/c4_semantic/summarize.log" 2>&1 \
      || fail "diagnosis-C4 summary failed job=$job"
    touch "$root/c4_semantic_complete"
  fi

  "$python_bin" - "$mode" "$root/t3_semantic/metrics.json" \
      "$root/c4_semantic/summary.json" "${answer[0]}" "${formal_metrics[0]}" \
      "$root/summary.json" <<'PY'
import json,pathlib,sys
mode=sys.argv[1]
t3,diagnosis,answer,formal=map(lambda p:json.load(open(p)),sys.argv[2:6])
groups=t3['metrics_by_group']; support=sum(int(x['llm_judge_n']) for x in groups.values())
accuracy=sum(float(x['llm_judge_accuracy'])*int(x['llm_judge_n']) for x in groups.values())/support
a=answer['summary']; clean=a['clean']
out={'stage':'state8_matched_checkpoint2000_rescue_semantic_summary_v1','status':'complete',
 'mode':mode,'state_token_count':8,'raw_history_visible_to_decoder':False,
 'semantic_judge_authoritative':True,'exact_match_authoritative':False,
 'canonical_label_role':'auxiliary_ordinary_weight','t3_semantic_accuracy':accuracy,
 't3_support':support,'t3_metrics_by_group':groups,'formal_metrics':formal['metrics_by_group'],
 'answer_c4':{'clean_n':a['clean_n'],'swap':clean['swap']['flip_to_A_rate'],
  'random':clean['random']['flip_to_A_rate'],'same_rule':clean['samemal']['flip_to_A_rate'],
  'noop':clean['noop']['flip_to_A_rate'],'D_specific':a['D_specific'],
  'checks':a['checks'],'scientific_gate_pass':a['scientific_gate_pass']},
 'diagnosis_c4':diagnosis,'artifact_sha256':answer['artifact_sha256']}
path=pathlib.Path(sys.argv[6]); tmp=path.with_suffix('.json.partial')
tmp.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); tmp.replace(path)
PY
done

"$python_bin" - "$baseline" "$state/semantic_preserve/summary.json" \
    "$state/causal_taskanchor/summary.json" "$state/comparison.json" <<'PY'
import json,pathlib,sys
rows=[json.load(open(p)) for p in sys.argv[1:4]]
names=['checkpoint2000','semantic_preserve','causal_taskanchor']
table=[]
for name,row in zip(names,rows):
 a=row['answer_c4']; d=row['diagnosis_c4']
 answer_diagnostic_gate=(a['D_specific']>=0.15 and a['random']<=0.25
  and a['same_rule']<=0.10 and a['noop']<=0.05
  and all(bool(value) for key,value in a['checks'].items() if key!='candidate_support'))
 diagnosis_diagnostic_gate=(d['semantic_D_specific']>=0.15
  and d['random_matches_either_stable_target_rate']<=0.25
  and d['noop_recipient_retention']>=0.75
  and d['same_rule_recipient_retention']>=0.65
  and d['cells']['swap_to_donor']['semantic_equivalence_rate']>=0.50)
 table.append({'condition':name,'t3_semantic_accuracy':row['t3_semantic_accuracy'],
  'answer_swap':a['swap'],'answer_random':a['random'],'answer_same_rule':a['same_rule'],
  'answer_D_specific':a['D_specific'],'answer_gate':a['scientific_gate_pass'],
  'answer_diagnostic_gate':answer_diagnostic_gate,
  'diagnosis_swap_to_donor':d['cells']['swap_to_donor']['semantic_equivalence_rate'],
  'diagnosis_random_any':d['random_matches_either_stable_target_rate'],
  'diagnosis_same_rule_retention':d['same_rule_recipient_retention'],
  'diagnosis_D_specific':d['semantic_D_specific'],'diagnosis_gate':d['pass'],
  'diagnosis_diagnostic_gate':diagnosis_diagnostic_gate})
eligible=[x for x in table[1:] if x['answer_diagnostic_gate'] and x['diagnosis_diagnostic_gate']]
selected=max(eligible,key=lambda x:(x['t3_semantic_accuracy'],x['diagnosis_D_specific']))['condition'] if eligible else None
out={'stage':'state8_matched_checkpoint2000_rescue_comparison_v1','status':'complete',
 'semantic_judge_authoritative':True,'exact_match_authoritative':False,
 'diagnostic_only':True,'pair_count':50,
 'selection_rule':'50-pair support-adjusted answer and diagnosis causal screens, then T3 semantic accuracy, then diagnosis D_specific; requires 150-pair confirmation before promotion',
 'selected_for_followup':selected,'conditions':table}
path=pathlib.Path(sys.argv[4]); tmp=path.with_suffix('.json.partial')
tmp.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); tmp.replace(path)
PY

touch "$state/complete"; heartbeat
log "matched checkpoint-2000 rescue and authoritative semantic comparison complete"
