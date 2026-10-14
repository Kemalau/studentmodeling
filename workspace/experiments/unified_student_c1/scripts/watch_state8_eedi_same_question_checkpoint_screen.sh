#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
source "$project/scripts/use_studentmodeling_feeds_auth.sh"
: "${SCREEN_ARM:?SCREEN_ARM must be updater or full}" "${CHECKPOINT_STEP:?CHECKPOINT_STEP must be 20 or 100}"
[[ "$SCREEN_ARM" == updater || "$SCREEN_ARM" == full ]] || exit 2
[[ "$CHECKPOINT_STEP" == 20 || "$CHECKPOINT_STEP" == 100 ]] || exit 2
arm="$SCREEN_ARM"; step="$CHECKPOINT_STEP"
if [[ "$arm" == updater ]]; then
  seed=3072; tag=source_anchor_updater_mse1_eedi_same_question_v1
else
  seed=3071; tag=source_anchor_full_mse1_eedi_same_question_v1
fi
state="$project/results/state8_eedi_same_question_${arm}_checkpoint${step}_screen_20260811"
template="$project/amlt/feeds_state8_eedi_same_question_checkpoint_screen_20260811.yaml"
base="state8-eedi-same-question-${arm}-cp${step}-screen-20260811"
experiment="$base"; recovery=0; max_recovery=3
job="state8-eedi-same-question-${arm}-cp${step}-screen"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
eval_jsonl="$project/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
remote_root="studentmodeling/checkpoints/state8_stateonly_repair_seed${seed}_taskanchor_${tag}/checkpoint-${step}"
blob="$remote_root/checkpoint_completion.json"
snapshot_prefix="studentmodeling/checkpoint_snapshots/state8_eedi_same_question_20260811/${arm}/checkpoint-${step}"
mkdir -p "$state"; cd "$project" || exit 1
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"; trap 'rm -f "$state/controller.active.pid"' EXIT
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
sleep_unlocked(){ sleep "$1" 9>&-; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }

receipt="$state/checkpoint_completion.json"
while [[ ! -f "$state/checkpoint_ready" ]]; do
  heartbeat
  exists="$(az storage blob exists --account-name csnewsandfeeds4150361735 \
    --container-name amulet --name "$blob" --auth-mode login --query exists -o tsv 2>/dev/null)" \
    || { sleep_unlocked 60; continue; }
  [[ "$exists" == true ]] || { sleep_unlocked 60; continue; }
  az storage blob download --account-name csnewsandfeeds4150361735 --container-name amulet \
    --name "$blob" --file "$receipt" --auth-mode login --no-progress -o none \
    || { rm -f "$receipt"; sleep_unlocked 60; continue; }
  "$python_bin" - "$receipt" "$step" <<'PY' || { rm -f "$receipt"; sleep_unlocked 30; continue; }
import json,sys
x=json.load(open(sys.argv[1])); step=int(sys.argv[2])
assert x.get('status')=='complete' and int(x.get('step',-1))==step,x
assert set(x.get('sha256') or {})=={'adapter_model','recurrent_state'},x
PY
  touch "$state/checkpoint_ready"; log "checkpoint-$step receipt is complete"
done

while [[ ! -f "$state/snapshot_complete" ]]; do
  heartbeat
  if "$python_bin" scripts/snapshot_state8_checkpoint_blob.py \
    --source-prefix "$remote_root" --destination-prefix "$snapshot_prefix" \
    --receipt "$receipt" --output "$state/snapshot.json" \
    >> "$state/controller.log" 2>&1; then
    touch "$state/snapshot_complete"
    log "checkpoint-$step immutable snapshot complete"
  else
    # Blob authentication, transient Azure failures, and local write pressure
    # do not invalidate an already receipt-complete remote checkpoint.  Keep
    # the watcher resumable and retry instead of marking the model experiment
    # failed; the receipt-last snapshot remains fail closed.
    rm -f "$state/snapshot.json.partial"
    log "checkpoint snapshot unavailable; preserving checkpoint and retrying"
    sleep_unlocked 60
  fi
done

config="$state/bound.yaml"
"$python_bin" scripts/materialize_state8_eedi_same_question_checkpoint_screen.py \
  --template "$template" --receipt "$receipt" --arm "$arm" --step "$step" --output "$config" \
  >> "$state/controller.log" 2>&1 || fail "config materialization failed"
"$python_bin" scripts/validate_state8_eedi_same_question_checkpoint_screen_amlt.py \
  --config "$config" --expected-arm "$arm" --expected-step "$step" >> "$state/controller.log" 2>&1 \
  || fail "bound config validation failed"

submit_experiment(){
  local name="$1" output rc current class
  exec 8>"$project/results/feeds_submission_global.lock"; flock -x 8
  amlt run "$config" --dump -t Feeds > "$state/$name.preflight.yaml" \
    2> "$state/$name.preflight.stderr" || { flock -u 8; return 1; }
  "$python_bin" scripts/validate_state8_eedi_same_question_checkpoint_screen_amlt.py \
    --config "$state/$name.preflight.yaml" --expected-arm "$arm" --expected-step "$step" \
    >> "$state/controller.log" 2>&1 || { flock -u 8; return 1; }
  current="$(amlt status --no-update-db "$name" 2>&1)"; rc=$?
  class="$("$python_bin" - "$rc" "$current" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]),sys.argv[2]))
PY
)" || { flock -u 8; return 1; }
  if [[ "$class" == missing ]]; then
    amlt --json-tables ti sing -t Feeds > "$state/$name.quota.json" 2>> "$state/controller.log" \
      || { flock -u 8; return 1; }
    output="$(amlt run "$config" "$name" -t Feeds -y \
      --description "Frozen $arm same-question State8 checkpoint-$step semantic T3 plus MalRule/Eedi C4 screen" 2>&1)"; rc=$?
    printf '[%s] rc=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$output" \
      > "$state/$name.submission.txt"
    (( rc == 0 )) || { flock -u 8; return 1; }
  elif [[ "$class" != exists ]]; then flock -u 8; return 1; fi
  flock -u 8; printf '%s\n' "$name" > "$state/active_experiment.txt"
}

if [[ -s "$state/active_experiment.txt" ]]; then experiment="$(<"$state/active_experiment.txt")"
else until submit_experiment "$experiment"; do heartbeat; sleep_unlocked 120; done; fi

while [[ ! -f "$state/remote_passed" ]]; do
  heartbeat
  status="$(timeout 90 amlt --json-tables status "$experiment" --hide-urls --force-status-update 2>/dev/null)" \
    || { sleep_unlocked 60; continue; }
  [[ -n "$status" && "$status" != '[]' ]] || { sleep_unlocked 30; continue; }
  printf '%s\n' "$status" > "$state/latest_status.json"
  phase="$(STATUS="$status" "$python_bin" - <<'PY'
import json,os
s={str(x.get('status') or '').casefold() for x in json.loads(os.environ['STATUS'])}
if s=={'pass'}: print('pass')
elif s&{'preparing','pre-queued','queued','running','resuming'}: print('active')
elif 'paused' in s: print('paused')
elif not s or s<={'','unknown'}: print('unknown')
else: print('terminal')
PY
)"
  case "$phase" in
    pass) touch "$state/remote_passed" ;;
    active) sleep_unlocked 60 ;;
    paused) timeout 90 amlt resume "$experiment" ":$job" >> "$state/controller.log" 2>&1 || true; sleep_unlocked 60 ;;
    unknown)
      log "AMLT returned unknown status; preserving remote screen and retrying"
      sleep_unlocked 60 ;;
    *)
      timeout 180 amlt logs view -n 3000 "$experiment" ":$job" > "$state/$experiment.failure.log" 2>&1 || true
      (( recovery < max_recovery )) || fail "checkpoint-$step screen recovery exhausted"
      recovery=$((recovery+1)); experiment="${base}-recovery${recovery}"
      until submit_experiment "$experiment"; do heartbeat; sleep_unlocked 120; done
      log "submitted recovery=$recovery" ;;
  esac
done

rm -rf "$state/download"; mkdir -p "$state/download"
amlt results download "$experiment" ":$job" --include '*.json' --include '*.jsonl' \
  --include '*.txt' --output "$state/download" >> "$state/controller.log" 2>&1 \
  || fail "result download failed"
mapfile -t roots < <(find "$state/download" -type d -name "$job")
[[ ${#roots[@]} -eq 1 ]] || fail "artifact root cardinality drift"
root="${roots[0]}"; out="$state/semantic"; mkdir -p "$out/t3" "$out/malrule_c4" "$out/eedi_c4"
mapfile -t formal < <(find "$root" -type f -path '*/eval/formal_t3_75/predictions.jsonl')
mapfile -t malrule_pred < <(find "$root" -type f -path '*/eval/malrule_diagnosis_c4_20/judge_predictions.jsonl')
mapfile -t malrule_eval < <(find "$root" -type f -path '*/eval/malrule_diagnosis_c4_20/judge_eval.jsonl')
mapfile -t malrule_metrics < <(find "$root" -type f -path '*/eval/malrule_diagnosis_c4_20/metrics.json')
mapfile -t eedi_pred < <(find "$root" -type f -path '*/eval/eedi_same_question_c4_25/judge_predictions.jsonl')
mapfile -t eedi_eval < <(find "$root" -type f -path '*/eval/eedi_same_question_c4_25/judge_eval.jsonl')
mapfile -t eedi_metrics < <(find "$root" -type f -path '*/eval/eedi_same_question_c4_25/metrics.json')
[[ ${#formal[@]} -eq 1 && ${#malrule_pred[@]} -eq 1 && ${#malrule_eval[@]} -eq 1 \
   && ${#malrule_metrics[@]} -eq 1 && ${#eedi_pred[@]} -eq 1 && ${#eedi_eval[@]} -eq 1 \
   && ${#eedi_metrics[@]} -eq 1 ]] || fail "semantic artifact cardinality drift"
judge(){
  local pred="$1" eval="$2" dest="$3" audit="$4"
  until PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
    --predictions-jsonl "$pred" --eval-jsonl "$eval" --output-dir "$dest" \
    --judge-backend trapi --trapi-deployment gpt-4o_2024-11-20 --judge-workers 12 \
    --judge-max-retries 12 --judge-timeout-seconds 60 --judge-cache "$dest/cache.jsonl" \
    --judge-audit-samples "$audit" --seed 42 >> "$dest/judge.log" 2>&1; do heartbeat; sleep_unlocked 120; done
}
judge "${formal[0]}" "$eval_jsonl" "$out/t3" 75
judge "${malrule_pred[0]}" "${malrule_eval[0]}" "$out/malrule_c4" 100
judge "${eedi_pred[0]}" "${eedi_eval[0]}" "$out/eedi_c4" 100
"$python_bin" scripts/summarize_state8_diagnosis_c4.py \
  --judge-redacted "$out/malrule_c4/predictions_redacted.jsonl" \
  --generation-metrics "${malrule_metrics[0]}" --output "$out/malrule_c4/summary.json"
"$python_bin" scripts/summarize_state8_diagnosis_c4.py \
  --judge-redacted "$out/eedi_c4/predictions_redacted.jsonl" \
  --generation-metrics "${eedi_metrics[0]}" --output "$out/eedi_c4/summary.json"
"$python_bin" - "$out/t3/metrics.json" "$out/malrule_c4/summary.json" \
  "$out/eedi_c4/summary.json" "$state/summary.json" "$arm" "$step" <<'PY'
import json,pathlib,sys
t3,malrule,eedi=(json.load(open(p)) for p in sys.argv[1:4])
groups=t3['metrics_by_group']
def acc(fragment):
 rows=[v for k,v in groups.items() if fragment in k]; assert len(rows)==1,(fragment,groups.keys())
 return rows[0]['llm_judge_accuracy']
out={'stage':'state8_eedi_same_question_checkpoint_semantic_summary_v1','status':'complete',
 'arm':sys.argv[5],'checkpoint_step':int(sys.argv[6]),'diagnostic_only':True,
 'semantic_judge_authoritative':True,'exact_match_authoritative':False,
 'T3_eedi_semantic':acc('eedi_mining/open_taxonomy_generation'),
 'T3_synthetic_cross_semantic':acc('malrule/synthetic_cross_template'),
 'T3_synthetic_same_semantic':acc('malrule/synthetic_same_template'),
 'malrule_diagnosis_c4':malrule,'eedi_same_question_diagnosis_c4':eedi}
pathlib.Path(sys.argv[4]).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
PY
touch "$state/complete"; heartbeat; log "$arm checkpoint-$step dual-C4 semantic screen complete"
