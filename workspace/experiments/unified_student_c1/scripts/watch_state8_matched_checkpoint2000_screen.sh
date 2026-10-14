#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
parent="$project/results/state8_diagnosis_first_matched_curriculum_20260810"
state="$project/results/state8_matched_checkpoint2000_screen_20260810"
template="$project/amlt/feeds_state8_matched_checkpoint2000_screen_20260810.yaml"
rendered="$project/amlt/feeds_state8_matched_checkpoint2000_screen_generated_20260810.yaml"
account=csnewsandfeeds4150361735
container=amulet
remote_root=studentmodeling/checkpoints/state8_qualified_teacher_curriculum_seed42_exact5000_rounds1250_kd1_v2_20260805_taxonomy_semantic_v1_diagnosis_first_matched_teacher_seed67_single_ga4/checkpoint-2000
receipt_blob="$remote_root/checkpoint_completion.json"
base=state8-matched-checkpoint2000-screen-20260810
recovery_prefix=state8-matched-checkpoint2000-screen-recovery
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
eval_jsonl="$project/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
max_waves=16
jobs=(
  state8-matched-checkpoint2000-t3
  state8-matched-checkpoint2000-answer-c4
  state8-matched-checkpoint2000-diagnosis-c4-shard00
  state8-matched-checkpoint2000-diagnosis-c4-shard10
  state8-matched-checkpoint2000-diagnosis-c4-shard20
  state8-matched-checkpoint2000-diagnosis-c4-shard30
  state8-matched-checkpoint2000-diagnosis-c4-shard40
)
export AZURE_CONFIG_DIR="${AZURE_CONFIG_DIR:-/home/v-yuejiali/.azure}"

mkdir -p "$state/preserved_results"
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

status_class(){
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

while [[ "$(az storage blob exists --auth-mode login --account-name "$account" \
    --container-name "$container" --name "$receipt_blob" \
    --query exists -o tsv 2>/dev/null)" != true ]]; do
  heartbeat
  [[ ! -f "$parent/needs_attention" ]] \
    || fail "parent curriculum needs attention before checkpoint-2000"
  sleep_unlocked 30
done

az storage blob download --auth-mode login --account-name "$account" \
  --container-name "$container" --name "$receipt_blob" \
  --file "$state/checkpoint_completion.json" --overwrite -o none \
  || fail "could not download checkpoint-2000 receipt"
"$python_bin" scripts/materialize_state8_matched_checkpoint2000_screen.py \
  --template "$template" --receipt "$state/checkpoint_completion.json" \
  --output "$rendered" > "$state/materialize.json" \
  || fail "could not hash-bind checkpoint-2000 config"
"$python_bin" scripts/validate_state8_matched_checkpoint2000_screen_amlt.py \
  --config "$rendered" >> "$state/controller.log" 2>&1 \
  || fail "rendered checkpoint-2000 config is invalid"

submit_jobs(){
  local experiment="$1"; shift
  local requested=("$@") selectors=() output rc current class
  for job in "${requested[@]}"; do selectors+=(":$job"); done
  exec 8>"$project/results/feeds_submission_global.lock"; flock -x 8
  amlt run "$rendered" "${selectors[@]}" --dump -t Feeds \
    > "$state/$experiment.preflight.yaml" 2> "$state/$experiment.preflight.stderr" \
    || { flock -u 8; return 1; }
  validation=("$python_bin" scripts/validate_state8_matched_checkpoint2000_screen_amlt.py \
    --config "$state/$experiment.preflight.yaml")
  for job in "${requested[@]}"; do validation+=(--job "$job"); done
  "${validation[@]}" >> "$state/controller.log" 2>&1 \
    || { flock -u 8; return 1; }
  amlt --json-tables ti sing -t Feeds > "$state/$experiment.quota.json" \
    || { flock -u 8; return 1; }
  output="$(amlt run "$rendered" "${selectors[@]}" "$experiment" -t Feeds -y \
    --description "Hash-bound State8 checkpoint-2000 state-only T3, answer C4, and five diagnosis-C4 shards" 2>&1)"; rc=$?
  printf '%s\n' "$output" > "$state/$experiment.submission.txt"
  if (( rc != 0 )); then
    current="$(amlt status --no-update-db "$experiment" 2>&1)"; rc=$?
    class="$("$python_bin" - "$rc" "$current" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]),sys.argv[2]))
PY
)" || { flock -u 8; return 1; }
    [[ "$class" == exists ]] || { flock -u 8; return 1; }
  fi
  flock -u 8
  return 0
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
    [[ ! -f "$state/passed.$job" ]] || continue
    amlt results download "$experiment" ":$job" \
      --include '*.json' --include '*.jsonl' --include '*.txt' \
      --output "$state/preserved_results" >> "$state/controller.log" 2>&1 \
      || return 1
    touch "$state/passed.$job"
    log "preserved checkpoint-2000 job=$job experiment=$experiment"
  done
}

experiment="$base"; wave=0
if [[ -s "$state/active_experiment.txt" ]]; then
  experiment="$(<"$state/active_experiment.txt")"
  [[ "$experiment" =~ recovery([0-9]+)-20260810$ ]] && wave="${BASH_REMATCH[1]}"
fi
while [[ ! -f "$state/remote_passed" ]]; do
  missing=(); for job in "${jobs[@]}"; do [[ -f "$state/passed.$job" ]] || missing+=("$job"); done
  if (( ${#missing[@]} == 0 )); then touch "$state/remote_passed"; break; fi
  if [[ ! -f "$state/submitted.$experiment" ]]; then
    until submit_jobs "$experiment" "${missing[@]}"; do
      heartbeat; log "transient checkpoint-2000 submission failure experiment=$experiment"; sleep_unlocked 120
    done
    printf '%s\n' "$experiment" > "$state/active_experiment.txt"
    touch "$state/submitted.$experiment"
    log "submitted checkpoint-2000 experiment=$experiment jobs=${missing[*]}"
  fi
  heartbeat
  status="$(timeout 90 amlt --json-tables status "$experiment" --hide-urls 2>/dev/null)" \
    || { sleep_unlocked 60; continue; }
  printf '%s\n' "$status" > "$state/latest_status.json"
  preserve_passed "$experiment" "$status" || fail "could not preserve passed checkpoint-2000 jobs"
  missing=(); for job in "${jobs[@]}"; do [[ -f "$state/passed.$job" ]] || missing+=("$job"); done
  if (( ${#missing[@]} == 0 )); then touch "$state/remote_passed"; break; fi
  phase="$(status_class "$status")"
  if [[ "$phase" == active ]]; then sleep_unlocked 60; continue; fi
  if [[ "$phase" == paused ]]; then
    selectors=(); for job in "${missing[@]}"; do selectors+=(":$job"); done
    timeout 90 amlt resume "$experiment" "${selectors[@]}" >> "$state/controller.log" 2>&1 || true
    sleep_unlocked 60; continue
  fi
  timeout 90 amlt show "$experiment" > "$state/$experiment.show.txt" 2>&1 || true
  timeout 180 amlt logs view -n 3000 "$experiment" > "$state/$experiment.failure.log" 2>&1 || true
  (( wave < max_waves )) || fail "checkpoint-2000 screen exhausted recovery waves"
  wave=$((wave+1)); experiment="${recovery_prefix}${wave}-20260810"
  log "recovering checkpoint-2000 failed jobs wave=$wave jobs=${missing[*]}"
done

mapfile -t t3_predictions < <(find "$state/preserved_results" -type f -path '*/state8-matched-checkpoint2000-t3/eval/formal_t3_75/predictions.jsonl')
mapfile -t t3_metrics < <(find "$state/preserved_results" -type f -path '*/state8-matched-checkpoint2000-t3/eval/formal_t3_75/metrics.json')
mapfile -t answer < <(find "$state/preserved_results" -type f -path '*/state8-matched-checkpoint2000-answer-c4/eval/answer_c4_50/recurrent_causal_gate.json')
[[ ${#t3_predictions[@]} -eq 1 && ${#t3_metrics[@]} -eq 1 && ${#answer[@]} -eq 1 ]] \
  || fail "checkpoint-2000 T3/answer artifact cardinality drift"
mapfile -t diagnosis_metrics < <(find "$state/preserved_results" -type f -path '*/state8-matched-checkpoint2000-diagnosis-c4-shard*/eval/diagnosis_c4_shard_*/metrics.json' | sort)
[[ ${#diagnosis_metrics[@]} -eq 5 ]] || fail "checkpoint-2000 diagnosis shard cardinality drift"
merge_args=(); for metric in "${diagnosis_metrics[@]}"; do merge_args+=(--shard "$(dirname "$metric")"); done
rm -rf "$state/merged"; mkdir -p "$state/merged"
PYTHONPATH="$project/scripts" "$python_bin" scripts/merge_state8_diagnosis_c4_shards.py \
  "${merge_args[@]}" --expected-pairs 50 --output-dir "$state/merged" \
  >> "$state/controller.log" 2>&1 || fail "checkpoint-2000 diagnosis merge failed"

[[ -x "$judge_python" ]] || fail "TRAPI judge environment unavailable"
mkdir -p "$state/t3_semantic" "$state/c4_semantic"
PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
  --predictions-jsonl "${t3_predictions[0]}" --eval-jsonl "$eval_jsonl" \
  --output-dir "$state/t3_semantic" --judge-backend trapi \
  --trapi-deployment gpt-4o_2024-11-20 --judge-workers 12 --judge-max-retries 12 \
  --judge-timeout-seconds 60 --judge-cache "$state/t3_semantic/cache.jsonl" \
  --judge-audit-samples 75 --seed 2000 >> "$state/t3_semantic/judge.log" 2>&1 \
  || fail "checkpoint-2000 T3 semantic judge failed"
PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
  --predictions-jsonl "$state/merged/judge_predictions.jsonl" \
  --eval-jsonl "$state/merged/judge_eval.jsonl" --output-dir "$state/c4_semantic" \
  --judge-backend trapi --trapi-deployment gpt-4o_2024-11-20 \
  --judge-workers 12 --judge-max-retries 12 --judge-timeout-seconds 60 \
  --judge-cache "$state/c4_semantic/cache.jsonl" --judge-audit-samples 100 \
  --seed 2000 >> "$state/c4_semantic/judge.log" 2>&1 \
  || fail "checkpoint-2000 C4 semantic judge failed"
"$python_bin" scripts/summarize_state8_diagnosis_c4.py \
  --judge-redacted "$state/c4_semantic/predictions_redacted.jsonl" \
  --generation-metrics "$state/merged/metrics.json" \
  --output "$state/c4_semantic/summary.json" >> "$state/controller.log" 2>&1 \
  || fail "checkpoint-2000 C4 semantic summary failed"

"$python_bin" - "${t3_metrics[0]}" "$state/t3_semantic/metrics.json" \
  "${answer[0]}" "$state/c4_semantic/summary.json" "$state/summary.json" <<'PY'
import json,pathlib,sys
formal,t3,answer,diagnosis=map(lambda p:json.load(open(p)),sys.argv[1:5])
groups=t3['metrics_by_group']; support=sum(int(row['llm_judge_n']) for row in groups.values())
assert support==75 and t3['judge_backend']=='trapi' and t3['judge_errors']==0,t3
accuracy=sum(float(row['llm_judge_accuracy'])*int(row['llm_judge_n']) for row in groups.values())/support
s=answer['summary']; clean=s['clean']
out={'stage':'state8_matched_checkpoint2000_screen_v1','status':'complete','diagnostic_only':True,
 'checkpoint_step':2000,'state_shape':[64,8,5120],
 'history_dropout_counts':{'0.0':1000,'0.25':1000,'0.5':0,'0.75':0,'1.0':0},
 'raw_history_visible_to_decoder':False,'persistent_history_kv_cache':False,
 'semantic_judge_authoritative':True,'exact_match_authoritative':False,
 'canonical_label_role':'auxiliary_ordinary_weight','t3_semantic_accuracy':accuracy,
 't3_support':support,'t3_metrics_by_group':groups,'formal_metrics':formal['metrics_by_group'],
 'answer_c4':{'clean_n':s['clean_n'],'swap':clean['swap']['flip_to_A_rate'],
  'random':clean['random']['flip_to_A_rate'],'same_rule':clean['samemal']['flip_to_A_rate'],
  'noop':clean['noop']['flip_to_A_rate'],'D_specific':s['D_specific'],
  'checks':s['checks'],'scientific_gate_pass':s['scientific_gate_pass']},
 'diagnosis_c4':diagnosis,'artifact_sha256':answer['artifact_sha256']}
path=pathlib.Path(sys.argv[5]); partial=path.with_suffix('.json.partial')
partial.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); partial.replace(path)
print(json.dumps(out,sort_keys=True))
PY
touch "$state/complete"; heartbeat
log "checkpoint-2000 semantic T3/answer-C4/diagnosis-C4 screen complete"
