#!/usr/bin/env bash
set -euo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
source "$project/scripts/use_studentmodeling_feeds_auth.sh"
state="${STATE_DIR:-$project/results/state8_eedi_same_question_causal_repair_20260811}"
experiment="${EXPERIMENT_NAME:-state8-eedi-same-question-causal-repair-20260811}"
base_experiment="${BASE_EXPERIMENT:-$experiment}"
config="${CONFIG_PATH:-$project/amlt/feeds_state8_eedi_same_question_causal_repair_20260811.yaml}"
wave="${INITIAL_RECOVERY_WAVE:-0}"
max_recovery_waves="${MAX_RECOVERY_WAVES:-8}"
job="${JOB_NAME:-state8-eedi-same-question-updater-mse1-seed3072}"
auto_recovery="${AUTO_RECOVERY:-1}"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
eval_jsonl="$project/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"

mkdir -p "$state"
cd "$project"
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"
cleanup(){ rm -f "$state/controller.active.pid"; }
trap cleanup EXIT
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
sleep_unlocked(){ sleep "$1" 9>&-; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }

submit_recovery(){
  local name="$1" current class output rc audit quota
  exec 8>"$project/results/feeds_submission_global.lock"; flock -x 8
  amlt run "$config" --dump -t Feeds > "$state/$name.preflight.yaml" \
    2> "$state/$name.preflight.stderr" || { flock -u 8; return 1; }
  "$python_bin" scripts/validate_state8_current_query_causal_repair_amlt.py \
    --config "$state/$name.preflight.yaml" >> "$state/controller.log" 2>&1 \
    || { flock -u 8; return 1; }
  current="$(amlt status --no-update-db "$name" 2>&1)"; rc=$?
  class="$("$python_bin" - "$rc" "$current" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]),sys.argv[2]))
PY
)" || { flock -u 8; return 1; }
  if [[ "$class" == missing ]]; then
    audit="$state/$name.active_a100.json"
    "$python_bin" scripts/count_active_a100_requests.py --user-limit 32 --recent 500 \
      --output "$audit" > /dev/null 2>> "$state/controller.log" \
      || { flock -u 8; return 1; }
    "$python_bin" - "$audit" <<'PY' || { flock -u 8; return 1; }
import json,sys
x=json.load(open(sys.argv[1])); assert x['auditable'] is True,x
assert int(x['available_user_slots'])>=1,x
PY
    quota="$state/$name.quota.json"
    amlt --json-tables ti sing -t Feeds > "$quota" 2>> "$state/controller.log" \
      || { flock -u 8; return 1; }
    "$python_bin" - "$quota" <<'PY' || { flock -u 8; return 1; }
import sys
from scripts.feeds_quota_utils import parse_json_table,select_a100_80gb
row=select_a100_80gb(parse_json_table(open(sys.argv[1]).read()))
assert row['series']=='NDAMv4' and row['accelerator']['type']=='A100',row
assert int(row['accelerator']['memory_gb'])==80,row
quota=row['quotas_(used/limit)']; assert quota['standard']['limit']-quota['standard']['used']>=1,quota
assert int(quota['user_max']['limit'])>=1,quota
PY
    output="$(amlt run "$config" "$name" -t Feeds -y \
      --description "Same-question State8 exact-resume recovery wave=$wave; immutable recipe and output root preserved" 2>&1)"; rc=$?
    printf '[%s] rc=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$output" \
      > "$state/$name.submission.txt"
    (( rc == 0 )) || { flock -u 8; return 1; }
  elif [[ "$class" != exists ]]; then
    flock -u 8; return 1
  fi
  flock -u 8
  printf '%s\n' "$name" > "$state/active_experiment.txt"
  return 0
}

if [[ -s "$state/active_experiment.txt" ]]; then
  experiment="$(<"$state/active_experiment.txt")"
  [[ "$experiment" =~ recovery([0-9]+) ]] && wave="${BASH_REMATCH[1]}"
fi

while [[ ! -f "$state/remote_passed" ]]; do
  heartbeat
  status="$(timeout 90 amlt --json-tables status "$experiment" --hide-urls --force-status-update 2>/dev/null)" \
    || { sleep_unlocked 60; continue; }
  [[ -n "$status" && "$status" != '[]' ]] || { sleep_unlocked 30; continue; }
  printf '%s\n' "$status" > "$state/latest_status.json"
  phase="$(STATUS="$status" "$python_bin" - <<'PY'
import json,os
rows=json.loads(os.environ['STATUS']); states={str(x.get('status') or '').casefold() for x in rows}
if len(rows)==1 and states=={'pass'}: print('pass')
elif states & {'preparing','pre-queued','queued','running','resuming'}: print('active')
elif 'paused' in states: print('paused')
elif not states or states <= {'', 'unknown'}: print('unknown')
else: print('terminal')
PY
)" || fail "status classification failed"
  case "$phase" in
    pass) touch "$state/remote_passed" ;;
    active) sleep_unlocked 60 ;;
    paused)
      timeout 90 amlt resume "$experiment" ":$job" >> "$state/controller.log" 2>&1 || true
      sleep_unlocked 60 ;;
    unknown)
      log "AMLT returned unknown status; preserving remote job and retrying"
      sleep_unlocked 60 ;;
    *)
      timeout 90 amlt show "$experiment" > "$state/show.txt" 2>&1 || true
      timeout 180 amlt logs view -n 3000 "$experiment" ":$job" \
        > "$state/$experiment.failure.log" 2>&1 || true
      [[ "$auto_recovery" == 1 ]] \
        || fail "remote job became terminal; automatic recovery disabled"
      (( wave < max_recovery_waves )) \
        || fail "same-question repair exhausted recovery waves"
      wave=$((wave+1)); experiment="${base_experiment}-recovery${wave}"
      until submit_recovery "$experiment"; do
        heartbeat; log "waiting for auditable A100 slot for recovery wave=$wave"; sleep_unlocked 120
      done
      log "submitted same-question exact-resume recovery wave=$wave" ;;
  esac
done

while [[ ! -f "$state/results_downloaded" ]]; do
  heartbeat
  rm -rf "$state/download"; mkdir -p "$state/download"
  if amlt results download "$experiment" ":$job" \
      --include '*.json' --include '*.jsonl' --include '*.txt' \
      --output "$state/download" >> "$state/controller.log" 2>&1; then
    touch "$state/results_downloaded"
    log "passed same-question artifacts downloaded"
  else
    # Authentication and result-service failures do not invalidate a Passed
    # remote job. Keep the receipt and retry without resubmitting training.
    log "result download unavailable; preserving Passed job and retrying"
    sleep_unlocked 120
  fi
done
mapfile -t roots < <(find "$state/download" -type d -name "$job")
[[ ${#roots[@]} -eq 1 ]] || fail "artifact root cardinality drift"
root="${roots[0]}"; out="$state/semantic"; mkdir -p "$out/t3" "$out/malrule_c4" "$out/eedi_c4"
mapfile -t formal_pred < <(find "$root" -type f -path '*/eval/formal_t3_300/predictions.jsonl')
mapfile -t formal_metrics < <(find "$root" -type f -path '*/eval/formal_t3_300/metrics.json')
mapfile -t malrule_pred < <(find "$root" -type f -path '*/eval/diagnosis_c4_50/judge_predictions.jsonl')
mapfile -t malrule_eval < <(find "$root" -type f -path '*/eval/diagnosis_c4_50/judge_eval.jsonl')
mapfile -t malrule_metrics < <(find "$root" -type f -path '*/eval/diagnosis_c4_50/metrics.json')
mapfile -t eedi_pred < <(find "$root" -type f -path '*/eval/eedi_same_question_c4_25/judge_predictions.jsonl')
mapfile -t eedi_eval < <(find "$root" -type f -path '*/eval/eedi_same_question_c4_25/judge_eval.jsonl')
mapfile -t eedi_metrics < <(find "$root" -type f -path '*/eval/eedi_same_question_c4_25/metrics.json')
[[ ${#formal_pred[@]} -eq 1 && ${#formal_metrics[@]} -eq 1 \
   && ${#malrule_pred[@]} -eq 1 && ${#malrule_eval[@]} -eq 1 && ${#malrule_metrics[@]} -eq 1 \
   && ${#eedi_pred[@]} -eq 1 && ${#eedi_eval[@]} -eq 1 && ${#eedi_metrics[@]} -eq 1 ]] \
  || fail "evaluation artifact cardinality drift"

judge(){
  local predictions="$1" evaluation="$2" output="$3" audit="$4"
  until PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
    --predictions-jsonl "$predictions" --eval-jsonl "$evaluation" \
    --output-dir "$output" --judge-backend trapi \
    --trapi-deployment gpt-4o_2024-11-20 --judge-workers 12 \
    --judge-max-retries 12 --judge-timeout-seconds 60 \
    --judge-cache "$output/cache.jsonl" --judge-audit-samples "$audit" --seed 42 \
    >> "$output/judge.log" 2>&1; do heartbeat; sleep_unlocked 120; done
}
judge "${formal_pred[0]}" "$eval_jsonl" "$out/t3" 75
judge "${malrule_pred[0]}" "${malrule_eval[0]}" "$out/malrule_c4" 100
judge "${eedi_pred[0]}" "${eedi_eval[0]}" "$out/eedi_c4" 100
"$python_bin" scripts/summarize_state8_diagnosis_c4.py \
  --judge-redacted "$out/malrule_c4/predictions_redacted.jsonl" \
  --generation-metrics "${malrule_metrics[0]}" --output "$out/malrule_c4/summary.json"
"$python_bin" scripts/summarize_state8_diagnosis_c4.py \
  --judge-redacted "$out/eedi_c4/predictions_redacted.jsonl" \
  --generation-metrics "${eedi_metrics[0]}" --output "$out/eedi_c4/summary.json"
"$python_bin" - "${formal_metrics[0]}" "$out/t3/metrics.json" \
  "$out/malrule_c4/summary.json" "$out/eedi_c4/summary.json" "$state/summary.json" <<'PY'
import json,pathlib,sys
formal,t3,malrule,eedi=(json.load(open(path)) for path in sys.argv[1:5])
out={'stage':'state8_eedi_same_question_causal_repair_semantic_summary_v1',
     'status':'complete','semantic_judge_authoritative':True,
     'exact_match_authoritative':False,'formal_metrics':formal,
     't3_semantic':t3,'malrule_diagnosis_c4':malrule,
     'eedi_same_question_diagnosis_c4':eedi}
pathlib.Path(sys.argv[5]).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
PY
touch "$state/complete"; heartbeat
log "same-question repair semantic evaluation complete"
