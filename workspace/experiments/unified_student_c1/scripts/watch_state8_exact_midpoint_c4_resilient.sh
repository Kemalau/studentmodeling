#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
config="$project/amlt/feeds_state8_exact_midpoint_c4_resilient_20260809.yaml"
state="$project/results/state8_exact_diagnosis_midpoint_eval_20260809/resilient_c4"
t3_state="$project/results/state8_exact_diagnosis_midpoint_eval_20260809/t3_semantic"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
base_experiment=state8-exact-midpoint-c4-resilient-20260809
recovery_prefix=state8-exact-midpoint-c4-resilient-recovery
recovery_suffix=-20260809
max_recovery_waves=16
answer_job=state8-exact-midpoint-step2000-answer-c4-resumable
diagnosis_jobs=(
  state8-exact-midpoint-step2000-diagnosis-c4-shard00
  state8-exact-midpoint-step2000-diagnosis-c4-shard10
  state8-exact-midpoint-step2000-diagnosis-c4-shard20
  state8-exact-midpoint-step2000-diagnosis-c4-shard30
  state8-exact-midpoint-step2000-diagnosis-c4-shard40
)
jobs=("$answer_job" "${diagnosis_jobs[@]}")

export AZURE_CONFIG_DIR="${AZURE_CONFIG_DIR:-/home/v-yuejiali/.azure}"
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
  local requested=("$@") selectors=() status phase rc
  for job in "${requested[@]}"; do selectors+=(":$job"); done
  exec 8>"$project/results/feeds_submission_global.lock"; flock -x 8
  "$python_bin" scripts/validate_state8_exact_midpoint_c4_resilient_amlt.py \
    --config "$config" >> "$state/controller.log" 2>&1 \
    || { flock -u 8; return 1; }
  amlt run "$config" "${selectors[@]}" --dump -t Feeds \
    > "$state/$experiment.preflight.yaml" \
    2> "$state/$experiment.preflight.stderr" \
    || { flock -u 8; return 1; }
  "$python_bin" scripts/validate_state8_exact_midpoint_c4_resilient_amlt.py \
    --config "$state/$experiment.preflight.yaml" \
    >> "$state/controller.log" 2>&1 \
    || { flock -u 8; return 1; }
  amlt --json-tables ti sing -t Feeds \
    > "$state/$experiment.quota.json" 2>> "$state/controller.log" \
    || { flock -u 8; return 1; }
  amlt run "$config" "${selectors[@]}" "$experiment" -t Feeds -y \
    --description "Automatic recovery of missing immutable step-2000 State8 C4 shards; remote answer progress retained" \
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
    log "preserved passed resilient C4 job=$job experiment=$experiment"
  done
}

experiment="$base_experiment"
wave=0
if [[ -s "$state/active_experiment.txt" ]]; then
  experiment="$(<"$state/active_experiment.txt")"
  [[ "$experiment" =~ recovery([0-9]+)-20260809 ]] && wave="${BASH_REMATCH[1]}"
else
  printf '%s\n' "$experiment" > "$state/active_experiment.txt"
  touch "$state/submitted.$experiment"
fi

while [[ ! -f "$state/remote_passed" ]]; do
  missing=()
  for job in "${jobs[@]}"; do [[ -f "$state/passed.$job" ]] || missing+=("$job"); done
  if (( ${#missing[@]} == 0 )); then touch "$state/remote_passed"; break; fi
  if [[ ! -f "$state/submitted.$experiment" ]]; then
    until submit_jobs "$experiment" "${missing[@]}"; do
      heartbeat; log "transient resilient C4 submission failure experiment=$experiment; retrying"
      sleep_unlocked 120
    done
    printf '%s\n' "$experiment" > "$state/active_experiment.txt"
    printf '%s\n' "${missing[@]}" > "$state/active_jobs.txt"
    touch "$state/submitted.$experiment"
    log "submitted resilient C4 experiment=$experiment jobs=${missing[*]}"
  fi
  status="$(timeout 90 amlt --json-tables status "$experiment" --hide-urls 2>/dev/null)" \
    || { sleep_unlocked 120; continue; }
  printf '%s\n' "$status" > "$state/latest_status.json"
  preserve_passed "$experiment" "$status" \
    || fail "could not preserve passed resilient C4 jobs"
  missing=()
  for job in "${jobs[@]}"; do [[ -f "$state/passed.$job" ]] || missing+=("$job"); done
  if (( ${#missing[@]} == 0 )); then touch "$state/remote_passed"; break; fi
  phase="$(status_class "$status")"
  [[ "$phase" == active ]] && { sleep_unlocked 120; continue; }
  timeout 90 amlt show "$experiment" > "$state/$experiment.show.txt" 2>&1 || true
  timeout 180 amlt logs view -n 4000 "$experiment" \
    > "$state/$experiment.failure.log" 2>&1 || true
  (( wave < max_recovery_waves )) || fail "resilient C4 exhausted recovery waves"
  wave=$((wave+1)); experiment="${recovery_prefix}${wave}${recovery_suffix}"
  log "recovering terminal resilient C4 jobs wave=$wave jobs=${missing[*]}"
done

mapfile -t answer_roots < <(find "$state/preserved_results" -type d -name "$answer_job")
[[ ${#answer_roots[@]} -eq 1 ]] || fail "answer-C4 artifact root cardinality drift"
mapfile -t answer_json < <(find "${answer_roots[0]}" -type f -path '*/eval/answer_c4_50/recurrent_causal_gate.json')
[[ ${#answer_json[@]} -eq 1 ]] || fail "answer-C4 result cardinality drift"

shard_args=()
for job in "${diagnosis_jobs[@]}"; do
  mapfile -t roots < <(find "$state/preserved_results" -type d -name "$job")
  [[ ${#roots[@]} -eq 1 ]] || fail "diagnosis shard root cardinality drift job=$job"
  mapfile -t metrics < <(find "${roots[0]}" -type f -path '*/eval/diagnosis_c4_shard_*/metrics.json')
  [[ ${#metrics[@]} -eq 1 ]] || fail "diagnosis shard metric cardinality drift job=$job"
  shard_args+=(--shard "$(dirname "${metrics[0]}")")
done

rm -rf "$state/merged"; mkdir -p "$state/merged"
PYTHONPATH="$project/scripts" "$python_bin" scripts/merge_state8_diagnosis_c4_shards.py \
  "${shard_args[@]}" --expected-pairs 50 --output-dir "$state/merged" \
  >> "$state/controller.log" 2>&1 || fail "resilient diagnosis-C4 merge failed"

[[ -x "$judge_python" ]] || fail "TRAPI judge environment unavailable"
if [[ ! -f "$state/c4_semantic_complete" ]]; then
  mkdir -p "$state/c4_semantic"
  attempt=0
  until PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
      --predictions-jsonl "$state/merged/judge_predictions.jsonl" \
      --eval-jsonl "$state/merged/judge_eval.jsonl" \
      --output-dir "$state/c4_semantic" --judge-backend trapi \
      --judge-workers 12 --judge-max-retries 12 --judge-timeout-seconds 60 \
      --judge-cache "$state/c4_semantic/cache.jsonl" --judge-audit-samples 100 \
      --seed 42 >> "$state/c4_semantic/judge.log" 2>&1; do
    attempt=$((attempt+1)); (( attempt < 30 )) || fail "resilient C4 semantic judge exhausted"
    heartbeat; sleep_unlocked 120
  done
  "$python_bin" scripts/summarize_state8_diagnosis_c4.py \
    --judge-redacted "$state/c4_semantic/predictions_redacted.jsonl" \
    --generation-metrics "$state/merged/metrics.json" \
    --output "$state/c4_semantic/summary.json" \
    >> "$state/controller.log" 2>&1 || fail "resilient C4 semantic summary failed"
  touch "$state/c4_semantic_complete"
fi

[[ -f "$t3_state/metrics.json" ]] || fail "authoritative midpoint T3 semantic metrics missing"
"$python_bin" - "${answer_json[0]}" "$state/c4_semantic/summary.json" \
  "$t3_state/metrics.json" "$state/summary.json" <<'PY'
import json,pathlib,sys
answer,diagnosis,t3=map(lambda p:json.load(open(p)),sys.argv[1:4])
groups=t3['metrics_by_group']; support=sum(int(r['llm_judge_n']) for r in groups.values())
api=sum(int(r['llm_judge_api_or_cache']) for r in groups.values())
shortcuts=sum(int(r['llm_judge_exact_shortcuts']) for r in groups.values())
assert t3['judge_backend']=='trapi' and t3['judge_errors']==0,t3
assert support==75 and api==75 and shortcuts==0,groups
t3_accuracy=sum(float(r['llm_judge_accuracy'])*int(r['llm_judge_n']) for r in groups.values())/support
s=answer['summary']; clean=s['clean']
out={
 'stage':'state8_exact_diagnosis_midpoint_step2000_resilient_screen_v1',
 'status':'complete','checkpoint_step':2000,'state_shape':[64,8,5120],
 'history_dropout_counts':{'0.0':1000,'0.25':1000,'0.5':0,'0.75':0,'1.0':0},
 'raw_history_visible_to_decoder':False,'persistent_history_kv_cache':False,
 'semantic_judge_authoritative':True,'exact_match_authoritative':False,
 'canonical_label_role':'auxiliary_ordinary_weight',
 't3_semantic_accuracy':t3_accuracy,'t3_support':support,
 't3_metrics_by_group':groups,
 'answer_c4':{'clean_n':s['clean_n'],'swap':clean['swap']['flip_to_A_rate'],
              'random':clean['random']['flip_to_A_rate'],
              'same_rule':clean['samemal']['flip_to_A_rate'],
              'noop':clean['noop']['flip_to_A_rate'],'D_specific':s['D_specific'],
              'D_damage':s['D_damage'],'checks':s['checks'],
              'scientific_gate_pass':s['scientific_gate_pass']},
 'diagnosis_c4':diagnosis,'artifact_sha256':answer['artifact_sha256'],
 'promotion_eligible':False,'diagnostic_role':'early_curriculum_health_screen',
}
assert out['artifact_sha256']==diagnosis['generation']['artifact_sha256'],(out,diagnosis)
path=pathlib.Path(sys.argv[4]); tmp=path.with_suffix('.json.partial')
tmp.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); tmp.replace(path)
print(json.dumps(out,sort_keys=True))
PY
touch "$state/complete"; heartbeat
log "resilient step-2000 answer/diagnosis C4 and semantic T3 joint screen complete"
