#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
step="${STATE8_MATCHED_MILESTONE_STEP:?STATE8_MATCHED_MILESTONE_STEP is required}"
case "$step" in 3000|4000) ;; *) echo "Unsupported milestone step=$step" >&2; exit 2 ;; esac
parent="$project/results/state8_diagnosis_first_matched_curriculum_20260810"
state="$project/results/state8_matched_checkpoint${step}_screen_20260810"
template="$project/amlt/feeds_state8_matched_checkpoint${step}_screen_20260810.yaml"
rendered="$project/amlt/feeds_state8_matched_checkpoint${step}_screen_generated_20260810.yaml"
account=csnewsandfeeds4150361735; container=amulet
remote_base=studentmodeling/checkpoints/state8_qualified_teacher_curriculum_seed42_exact5000_rounds1250_kd1_v2_20260805_taxonomy_semantic_v1_diagnosis_first_matched_teacher_seed67_single_ga4
receipt_blob="$remote_base/checkpoint-$step/checkpoint_completion.json"
base="state8-matched-checkpoint${step}-screen-20260810"
recovery_prefix="state8-matched-checkpoint${step}-screen-recovery"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
eval_jsonl="$project/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
max_waves=12
jobs=("state8-matched-checkpoint${step}-t3" "state8-matched-checkpoint${step}-c4")
export AZURE_CONFIG_DIR="${AZURE_CONFIG_DIR:-/home/v-yuejiali/.azure}"

mkdir -p "$state/preserved_results"; cd "$project" || exit 1
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"
cleanup(){ rm -f "$state/controller.active.pid"; }; trap cleanup EXIT
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }
sleep_unlocked(){ sleep "$1" 9>&-; }
rm -f "$state/needs_attention"

while [[ "$(az storage blob exists --auth-mode login --account-name "$account" --container-name "$container" \
    --name "$receipt_blob" --query exists -o tsv 2>/dev/null)" != true ]]; do
  heartbeat; [[ ! -f "$parent/needs_attention" ]] || fail "parent curriculum needs attention"; sleep_unlocked 30
done
az storage blob download --auth-mode login --account-name "$account" --container-name "$container" \
  --name "$receipt_blob" --file "$state/checkpoint_completion.json" --overwrite -o none \
  || fail "could not download checkpoint-$step receipt"
"$python_bin" scripts/materialize_state8_matched_checkpoint2000_screen.py \
  --template "$template" --receipt "$state/checkpoint_completion.json" --output "$rendered" --step "$step" \
  > "$state/materialize.json" || fail "could not hash-bind checkpoint-$step config"
"$python_bin" scripts/validate_state8_matched_checkpoint_milestone_screen_amlt.py \
  --config "$rendered" --step "$step" >> "$state/controller.log" 2>&1 \
  || fail "rendered checkpoint-$step config is invalid"

submit_jobs(){
  local experiment="$1"; shift; local requested=("$@") selectors=() output rc current class
  for job in "${requested[@]}"; do selectors+=(":$job"); done
  exec 8>"$project/results/feeds_submission_global.lock"; flock -x 8
  amlt run "$rendered" "${selectors[@]}" --dump -t Feeds > "$state/$experiment.preflight.yaml" \
    2> "$state/$experiment.preflight.stderr" || { flock -u 8; return 1; }
  "$python_bin" scripts/validate_state8_matched_checkpoint_milestone_screen_amlt.py \
    --config "$state/$experiment.preflight.yaml" --step "$step" >> "$state/controller.log" 2>&1 \
    || { flock -u 8; return 1; }
  amlt --json-tables ti sing -t Feeds > "$state/$experiment.quota.json" || { flock -u 8; return 1; }
  output="$(amlt run "$rendered" "${selectors[@]}" "$experiment" -t Feeds -y \
    --description "Hash-bound State8 checkpoint-$step read-only T3 and compact answer/natural-diagnosis C4 trajectory screen" 2>&1)"; rc=$?
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
}

preserve_passed(){
  local experiment="$1" status="$2" job
  mapfile -t observed < <(STATUS="$status" "$python_bin" - <<'PY'
import json,os
for row in json.loads(os.environ['STATUS']):
 if str(row.get('status') or '').casefold()=='pass': print(str(row['job_name']).lstrip(':'))
PY
)
  for job in "${observed[@]}"; do
    [[ " ${jobs[*]} " == *" $job "* ]] || continue
    [[ ! -f "$state/passed.$job" ]] || continue
    amlt results download "$experiment" ":$job" --include '*.json' --include '*.jsonl' --include '*.txt' \
      --output "$state/preserved_results" >> "$state/controller.log" 2>&1 || return 1
    touch "$state/passed.$job"; log "preserved milestone job=$job experiment=$experiment"
  done
}

experiment="$base"; wave=0
while [[ ! -f "$state/remote_passed" ]]; do
  missing=(); for job in "${jobs[@]}"; do [[ -f "$state/passed.$job" ]] || missing+=("$job"); done
  if (( ${#missing[@]} == 0 )); then touch "$state/remote_passed"; break; fi
  if [[ ! -f "$state/submitted.$experiment" ]]; then
    until submit_jobs "$experiment" "${missing[@]}"; do heartbeat; sleep_unlocked 120; done
    touch "$state/submitted.$experiment"; log "submitted checkpoint-$step screen jobs=${missing[*]}"
  fi
  heartbeat
  status="$(timeout 90 amlt --json-tables status "$experiment" --hide-urls 2>/dev/null)" \
    || { sleep_unlocked 60; continue; }
  printf '%s\n' "$status" > "$state/latest_status.json"
  preserve_passed "$experiment" "$status" || fail "could not preserve passed jobs"
  missing=(); for job in "${jobs[@]}"; do [[ -f "$state/passed.$job" ]] || missing+=("$job"); done
  (( ${#missing[@]} > 0 )) || { touch "$state/remote_passed"; break; }
  phase="$(STATUS="$status" "$python_bin" - <<'PY'
import json,os
s={str(x.get('status') or '').casefold() for x in json.loads(os.environ['STATUS'])}
print('active' if s & {'preparing','pre-queued','queued','running','resuming'} else 'paused' if 'paused' in s else 'terminal')
PY
)"
  if [[ "$phase" == active ]]; then sleep_unlocked 60; continue; fi
  if [[ "$phase" == paused ]]; then selectors=(); for job in "${missing[@]}"; do selectors+=(":$job"); done; \
    timeout 90 amlt resume "$experiment" "${selectors[@]}" >> "$state/controller.log" 2>&1 || true; sleep_unlocked 60; continue; fi
  timeout 90 amlt show "$experiment" > "$state/$experiment.show.txt" 2>&1 || true
  timeout 180 amlt logs view -n 3000 "$experiment" > "$state/$experiment.failure.log" 2>&1 || true
  (( wave < max_waves )) || fail "milestone screen exhausted recovery waves"
  wave=$((wave+1)); experiment="${recovery_prefix}${wave}-20260810"
done

mapfile -t formal < <(find "$state/preserved_results" -type f -path "*/state8-matched-checkpoint${step}-t3/eval/formal_t3_75/predictions.jsonl")
mapfile -t answer < <(find "$state/preserved_results" -type f -path "*/state8-matched-checkpoint${step}-c4/eval/answer_c4_50/recurrent_causal_gate.json")
mapfile -t diag_pred < <(find "$state/preserved_results" -type f -path "*/state8-matched-checkpoint${step}-c4/eval/diagnosis_c4_20/judge_predictions.jsonl")
mapfile -t diag_eval < <(find "$state/preserved_results" -type f -path "*/state8-matched-checkpoint${step}-c4/eval/diagnosis_c4_20/judge_eval.jsonl")
mapfile -t diag_metrics < <(find "$state/preserved_results" -type f -path "*/state8-matched-checkpoint${step}-c4/eval/diagnosis_c4_20/metrics.json")
[[ ${#formal[@]} -eq 1 && ${#answer[@]} -eq 1 && ${#diag_pred[@]} -eq 1 && ${#diag_eval[@]} -eq 1 && ${#diag_metrics[@]} -eq 1 ]] \
  || fail "milestone artifact cardinality drift"
[[ -x "$judge_python" ]] || fail "TRAPI judge environment unavailable"
mkdir -p "$state/t3_semantic" "$state/c4_semantic"
PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
  --predictions-jsonl "${formal[0]}" --eval-jsonl "$eval_jsonl" --output-dir "$state/t3_semantic" \
  --judge-backend trapi --trapi-deployment gpt-4o_2024-11-20 --judge-workers 12 --judge-max-retries 12 \
  --judge-timeout-seconds 60 --judge-cache "$state/t3_semantic/cache.jsonl" --judge-audit-samples 75 --seed "$step" \
  >> "$state/t3_semantic/judge.log" 2>&1 || fail "T3 semantic judge failed"
PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
  --predictions-jsonl "${diag_pred[0]}" --eval-jsonl "${diag_eval[0]}" --output-dir "$state/c4_semantic" \
  --judge-backend trapi --trapi-deployment gpt-4o_2024-11-20 --judge-workers 12 --judge-max-retries 12 \
  --judge-timeout-seconds 60 --judge-cache "$state/c4_semantic/cache.jsonl" --judge-audit-samples 100 --seed "$step" \
  >> "$state/c4_semantic/judge.log" 2>&1 || fail "diagnosis C4 semantic judge failed"
"$python_bin" scripts/summarize_state8_diagnosis_c4.py --judge-redacted "$state/c4_semantic/predictions_redacted.jsonl" \
  --generation-metrics "${diag_metrics[0]}" --output "$state/c4_semantic/summary.json" \
  >> "$state/controller.log" 2>&1 || fail "diagnosis C4 summary failed"
"$python_bin" - "$step" "$state/t3_semantic/metrics.json" "${answer[0]}" "$state/c4_semantic/summary.json" "$state/summary.json" <<'PY'
import json,pathlib,sys
step=int(sys.argv[1]); t3,answer,diagnosis=map(lambda p:json.load(open(p)),sys.argv[2:5])
groups=t3['metrics_by_group']; n=sum(int(v['llm_judge_n']) for v in groups.values())
accuracy=sum(float(v['llm_judge_accuracy'])*int(v['llm_judge_n']) for v in groups.values())/n
s=answer['summary']; c=s['clean']
out={'stage':'state8_matched_milestone_semantic_screen_v1','status':'complete','diagnostic_only':True,
 'checkpoint_step':step,'t3_semantic_accuracy':accuracy,'t3_support':n,'t3_metrics_by_group':groups,
 'answer_c4':{'clean_n':s['clean_n'],'swap':c['swap']['flip_to_A_rate'],'random':c['random']['flip_to_A_rate'],
  'same_rule':c['samemal']['flip_to_A_rate'],'noop':c['noop']['flip_to_A_rate'],'D_specific':s['D_specific'],
  'scientific_gate_pass':s['scientific_gate_pass']},'diagnosis_c4':diagnosis,
 'semantic_judge_authoritative':True,'exact_match_authoritative':False,'final_inference_route':'strict_state_only'}
path=pathlib.Path(sys.argv[5]); path.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
PY
touch "$state/complete"; heartbeat; log "checkpoint-$step milestone semantic screen complete"
