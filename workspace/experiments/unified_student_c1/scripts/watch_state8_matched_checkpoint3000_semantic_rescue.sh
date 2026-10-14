#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
state="$project/results/state8_matched_checkpoint3000_semantic_rescue_20260810"
baseline="$project/results/state8_matched_checkpoint3000_screen_20260810/summary.json"
config="${STATE8_CP3000_CONFIG:-$project/amlt/feeds_state8_matched_checkpoint3000_semantic_rescue_20260810.yaml}"
base="${STATE8_CP3000_BASE_EXPERIMENT:-state8-matched-cp3000-semantic-rescue-20260810}"
recovery_prefix="${STATE8_CP3000_RECOVERY_PREFIX:-state8-matched-cp3000-semantic-rescue-recovery}"
validator="${STATE8_CP3000_VALIDATOR:-scripts/validate_state8_matched_checkpoint3000_semantic_rescue_amlt.py}"
job=state8-matched-cp3000-rescue-semantic-preserve
eval_jsonl="$project/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
max_waves=12
export AZURE_CONFIG_DIR="${AZURE_CONFIG_DIR:-/home/v-yuejiali/.azure}"

mkdir -p "$state"; cd "$project" || exit 1
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"
cleanup(){ rm -f "$state/controller.active.pid"; }
trap cleanup EXIT
rm -f "$state/needs_attention"
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }
sleep_unlocked(){ sleep "$1" 9>&-; }

submit_wave(){
  local experiment="$1" output rc class current
  exec 8>"$project/results/feeds_submission_global.lock"; flock -x 8
  amlt run "$config" ":$job" --dump -t Feeds \
    > "$state/$experiment.preflight.yaml" 2> "$state/$experiment.preflight.stderr" \
    || { flock -u 8; return 1; }
  "$python_bin" "$validator" \
    --config "$state/$experiment.preflight.yaml" >> "$state/controller.log" 2>&1 \
    || { flock -u 8; return 1; }
  current="$(amlt status --no-update-db "$experiment" 2>&1)"; rc=$?
  class="$("$python_bin" - "$rc" "$current" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]),sys.argv[2]))
PY
)" || { flock -u 8; return 1; }
  if [[ "$class" == missing ]]; then
    amlt --json-tables ti sing -t Feeds > "$state/$experiment.quota.json" \
      2>> "$state/controller.log" || { flock -u 8; return 1; }
    output="$(amlt run "$config" ":$job" "$experiment" -t Feeds -y \
      --description "Checkpoint-3000 semantic-preserve causal rescue recovery; hash-bound strict-state-only State8" 2>&1)"; rc=$?
    printf '[%s] rc=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$output" \
      > "$state/$experiment.submission.txt"
    (( rc == 0 )) || { flock -u 8; return 1; }
  elif [[ "$class" != exists ]]; then
    flock -u 8; return 1
  fi
  flock -u 8
  printf '%s\n' "$experiment" > "$state/active_experiment.txt"
}

wave=0
if [[ -s "$state/active_experiment.txt" ]]; then
  experiment="$(<"$state/active_experiment.txt")"
  [[ "$experiment" =~ recovery([0-9]+)-20260810$ ]] && wave="${BASH_REMATCH[1]}"
else
  experiment="$base"; printf '%s\n' "$experiment" > "$state/active_experiment.txt"
fi

while [[ ! -f "$state/remote_passed" ]]; do
  heartbeat
  status="$(timeout 90 amlt --json-tables status "$experiment" --hide-urls --force-status-update 2>/dev/null)" \
    || { sleep_unlocked 60; continue; }
  [[ -n "$status" && "$status" != '[]' ]] || { sleep_unlocked 30; continue; }
  printf '%s\n' "$status" > "$state/latest_status.json"
  phase="$(STATUS="$status" "$python_bin" - <<'PY'
import json,os
rows=json.loads(os.environ['STATUS']); assert len(rows)==1,rows
print(str(rows[0].get('status') or '').casefold())
PY
)" || { sleep_unlocked 30; continue; }
  case "$phase" in
    pass) touch "$state/remote_passed" ;;
    preparing|pre-queued|queued|running|resuming) sleep_unlocked 60 ;;
    paused)
      timeout 90 amlt resume "$experiment" ":$job" >> "$state/controller.log" 2>&1 || true
      sleep_unlocked 60 ;;
    *)
      timeout 90 amlt show "$experiment" > "$state/$experiment.show.txt" 2>&1 || true
      timeout 180 amlt logs view -n 3000 "$experiment" ":$job" \
        > "$state/$experiment.failure.log" 2>&1 || true
      (( wave < max_waves )) || fail "cp3000 rescue exhausted recovery waves"
      wave=$((wave+1)); experiment="${recovery_prefix}${wave}-20260810"
      until submit_wave "$experiment"; do heartbeat; sleep_unlocked 120; done
      log "submitted cp3000 rescue recovery wave=$wave experiment=$experiment" ;;
  esac
done

rm -rf "$state/download"; mkdir -p "$state/download"
amlt results download "$experiment" --status pass \
  --include '*.json' --include '*.jsonl' --include '*.txt' \
  --output "$state/download" >> "$state/controller.log" 2>&1 \
  || fail "cp3000 rescue result download failed"
# The finalizer reuses the full State8 evaluator names rather than the earlier
# rescue-screen aliases.  Bind the watcher to the actual immutable output
# directories so a passed GPU job cannot be mistaken for missing artifacts.
mapfile -t formal_pred < <(find "$state/download" -type f -path '*/eval/formal_t3_300/predictions.jsonl')
mapfile -t formal_metrics < <(find "$state/download" -type f -path '*/eval/formal_t3_300/metrics.json')
mapfile -t diag_pred < <(find "$state/download" -type f -path '*/eval/diagnosis_c4_50/judge_predictions.jsonl')
mapfile -t diag_eval < <(find "$state/download" -type f -path '*/eval/diagnosis_c4_50/judge_eval.jsonl')
mapfile -t diag_metrics < <(find "$state/download" -type f -path '*/eval/diagnosis_c4_50/metrics.json')
mapfile -t answer < <(find "$state/download" -type f -path '*/eval/answer_c4_50/recurrent_causal_gate.json')
mapfile -t rescue_receipt < <(find "$state/download" -type f -path '*/metrics/rescue_receipt.json')
[[ ${#formal_pred[@]} -eq 1 && ${#formal_metrics[@]} -eq 1 && ${#diag_pred[@]} -eq 1 \
   && ${#diag_eval[@]} -eq 1 && ${#diag_metrics[@]} -eq 1 && ${#answer[@]} -eq 1 \
   && ${#rescue_receipt[@]} -eq 1 ]] \
  || fail "cp3000 rescue artifact cardinality drift"
[[ -x "$judge_python" ]] || fail "TRAPI judge environment unavailable"
mkdir -p "$state/t3_semantic" "$state/c4_semantic"
until PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
    --predictions-jsonl "${formal_pred[0]}" --eval-jsonl "$eval_jsonl" \
    --output-dir "$state/t3_semantic" --judge-backend trapi \
    --trapi-deployment gpt-4o_2024-11-20 --judge-workers 12 \
    --judge-max-retries 12 --judge-timeout-seconds 60 \
    --judge-cache "$state/t3_semantic/cache.jsonl" --judge-audit-samples 75 --seed 3066 \
    >> "$state/t3_semantic/judge.log" 2>&1; do heartbeat; sleep_unlocked 120; done
until PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
    --predictions-jsonl "${diag_pred[0]}" --eval-jsonl "${diag_eval[0]}" \
    --output-dir "$state/c4_semantic" --judge-backend trapi \
    --trapi-deployment gpt-4o_2024-11-20 --judge-workers 12 \
    --judge-max-retries 12 --judge-timeout-seconds 60 \
    --judge-cache "$state/c4_semantic/cache.jsonl" --judge-audit-samples 100 --seed 42 \
    >> "$state/c4_semantic/judge.log" 2>&1; do heartbeat; sleep_unlocked 120; done
"$python_bin" scripts/summarize_state8_diagnosis_c4.py \
  --judge-redacted "$state/c4_semantic/predictions_redacted.jsonl" \
  --generation-metrics "${diag_metrics[0]}" --output "$state/c4_semantic/summary.json" \
  >> "$state/c4_semantic/summarize.log" 2>&1 || fail "cp3000 diagnosis-C4 summary failed"

"$python_bin" - "$baseline" "$state/t3_semantic/metrics.json" \
    "$state/c4_semantic/summary.json" "${answer[0]}" "${formal_metrics[0]}" \
    "$state/summary.json" <<'PY'
import json,pathlib,sys
baseline,t3,diagnosis,answer,formal=map(lambda p:json.load(open(p)),sys.argv[1:6])
groups=t3['metrics_by_group']; support=sum(int(x['llm_judge_n']) for x in groups.values())
accuracy=sum(float(x['llm_judge_accuracy'])*int(x['llm_judge_n']) for x in groups.values())/support
s=answer['summary']; clean=s['clean']
out={'stage':'state8_matched_checkpoint3000_semantic_rescue_summary_v1','status':'complete',
 'source_checkpoint_step':3000,'diagnostic_only':True,'state_token_count':8,
 'raw_history_visible_to_decoder':False,'semantic_judge_authoritative':True,
 'exact_match_authoritative':False,'t3_semantic_accuracy':accuracy,'t3_support':support,
 't3_metrics_by_group':groups,'formal_metrics':formal['metrics_by_group'],
 'answer_c4':{'clean_n':s['clean_n'],'swap':clean['swap']['flip_to_A_rate'],
  'random':clean['random']['flip_to_A_rate'],'same_rule':clean['samemal']['flip_to_A_rate'],
  'noop':clean['noop']['flip_to_A_rate'],'D_specific':s['D_specific'],
  'scientific_gate_pass':s['scientific_gate_pass']},'diagnosis_c4':diagnosis,
 'artifact_sha256':answer['artifact_sha256'],
 'delta_vs_checkpoint3000':{'T3_semantic':accuracy-baseline['t3_semantic_accuracy'],
  'answer_D_specific':s['D_specific']-baseline['answer_c4']['D_specific'],
  'diagnosis_D_specific':diagnosis['semantic_D_specific']-baseline['diagnosis_c4']['semantic_D_specific']}}
path=pathlib.Path(sys.argv[6]); tmp=path.with_suffix('.json.partial')
tmp.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); tmp.replace(path)
PY
set +e
"$python_bin" scripts/select_state8_matched_cp3000_semantic_rescue.py \
  --summary "$state/summary.json" --rescue-receipt "${rescue_receipt[0]}" \
  --output "$state/selection.json" > "$state/selection.log" 2>&1
selection_rc=$?
set -e
if (( selection_rc == 0 )); then
  az storage blob upload --auth-mode login \
    --account-name csnewsandfeeds4150361735 --container-name amulet \
    --name studentmodeling/gates/state8_matched_cp3000_semantic_v1/selection.json \
    --file "$state/selection.json" --overwrite \
    > "$state/promotion_upload.txt" 2>&1 \
    || fail "cp3000 semantic promotion upload failed"
  touch "$state/scientific_gate_pass"
elif (( selection_rc == 2 )); then
  touch "$state/scientific_gate_failed"
else
  fail "cp3000 semantic promotion selector crashed rc=$selection_rc"
fi
touch "$state/complete"; heartbeat
log "checkpoint-3000 semantic rescue and authoritative semantic comparison complete"
