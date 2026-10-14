#!/usr/bin/env bash
set -euo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
source "$project/scripts/use_studentmodeling_feeds_auth.sh"
state="${STATE_DIR:-$project/results/state8_diagnosis_first_eedi_curriculum_20260812}"
config="${CONFIG_PATH:-$project/amlt/feeds_state8_diagnosis_first_eedi_curriculum_20260812.yaml}"
base="${EXPERIMENT_BASE:-state8-diagnosis-first-eedi-curriculum}"
experiment="${EXPERIMENT_NAME:-${base}-recovery1-20260812}"
wave="${INITIAL_RECOVERY_WAVE:-1}"
max_recovery_waves=8
job="${JOB_NAME:-state8-diagnosis-first-eedi-curriculum-seed83}"
validator="${VALIDATOR:-scripts/validate_state8_diagnosis_first_eedi_curriculum_amlt.py}"
skip_postprocess="${SKIP_POSTPROCESS:-0}"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
formal_eval="$project/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"

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
  local name="$1" output rc current class audit quota
  exec 8>"$project/results/feeds_submission_global.lock"; flock -x 8
  amlt run "$config" --dump -t Feeds > "$state/$name.preflight.yaml" \
    2> "$state/$name.preflight.stderr" || { flock -u 8; return 1; }
  PYTHONPATH="$project/scripts" "$python_bin" \
    "$validator" \
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
    output="$(amlt run "$config" "$name" -t Feeds -y --description \
      "Exact-resume State8 Eedi curriculum recovery wave=$wave; immutable seed83 recipe and remote run directory" 2>&1)"; rc=$?
    printf '[%s] rc=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$output" \
      > "$state/$name.submission.txt"
    (( rc == 0 )) || { flock -u 8; return 1; }
  elif [[ "$class" != exists ]]; then
    flock -u 8; return 1
  fi
  flock -u 8
  printf '%s\n' "$name" > "$state/active_experiment.txt"
  if [[ -d "$state/milestones" ]]; then
    printf '%s\n' "$name" > "$state/milestones/active_experiment.txt"
  fi
  return 0
}

if [[ -s "$state/active_experiment.txt" ]]; then
  experiment="$(<"$state/active_experiment.txt")"
fi
if [[ "$experiment" =~ -recovery([0-9]+)-20260812$ ]]; then
  wave="${BASH_REMATCH[1]}"
fi
printf '%s\n' "$experiment" > "$state/active_experiment.txt"
mkdir -p "$state/milestones"
printf '%s\n' "$experiment" > "$state/milestones/active_experiment.txt"

while [[ ! -f "$state/remote_passed" ]]; do
  heartbeat
  status="$(timeout 90 amlt --json-tables status "$experiment" --hide-urls --force-status-update 2>/dev/null)" \
    || { sleep_unlocked 60; continue; }
  [[ -n "$status" && "$status" != '[]' ]] || { sleep_unlocked 60; continue; }
  printf '%s\n' "$status" > "$state/latest_status.json"
  phase="$(STATUS="$status" "$python_bin" - <<'PY'
import json,os
rows=json.loads(os.environ['STATUS']); states={str(row.get('status') or '').casefold() for row in rows}
if len(rows)==1 and states=={'pass'}: print('pass')
elif states & {'preparing','pre-queued','queued','running','resuming'}: print('active')
elif 'paused' in states: print('paused')
else: print('terminal')
PY
)"
  case "$phase" in
    pass) touch "$state/remote_passed" ;;
    active) sleep_unlocked 60 ;;
    paused)
      timeout 90 amlt resume "$experiment" ":$job" >> "$state/controller.log" 2>&1 || true
      sleep_unlocked 60
      ;;
    *)
      timeout 90 amlt show "$experiment" > "$state/show.txt" 2>&1 || true
      timeout 180 amlt logs view -n 3000 "$experiment" ":$job" \
        > "$state/$experiment.failure.log" 2>&1 || true
      (( wave < max_recovery_waves )) \
        || fail "remote curriculum exhausted exact-resume recovery waves"
      wave=$((wave + 1))
      experiment="${base}-recovery${wave}-20260812"
      until submit_recovery "$experiment"; do
        heartbeat
        log "waiting for auditable A100 capacity for recovery wave=$wave"
        sleep_unlocked 120
      done
      log "submitted exact-resume recovery wave=$wave experiment=$experiment"
      ;;
  esac
done

if [[ "$skip_postprocess" == 1 ]]; then
  touch "$state/training_complete"
  heartbeat
  log "remote curriculum Passed; independent bound evaluation required"
  exit 0
fi

while [[ ! -f "$state/results_downloaded" ]]; do
  heartbeat
  rm -rf "$state/download"; mkdir -p "$state/download"
  if amlt results download "$experiment" ":$job" \
      --include '*.json' --include '*.jsonl' --include '*.txt' \
      --output "$state/download" >> "$state/controller.log" 2>&1; then
    touch "$state/results_downloaded"
  else
    log "Passed job result download unavailable; retrying without resubmission"
    sleep_unlocked 120
  fi
done

mapfile -t roots < <(find "$state/download" -type d -name "$job")
[[ ${#roots[@]} -eq 1 ]] || fail "artifact root cardinality drift"
root="${roots[0]}"
out="$state/semantic"
mkdir -p "$out/t3" "$out/eedi_c4"
mapfile -t formal_predictions < <(find "$root" -type f -path '*/eval/formal_screen/predictions.jsonl')
mapfile -t formal_metrics < <(find "$root" -type f -path '*/eval/formal_screen/metrics.json')
mapfile -t c4_predictions < <(find "$root" -type f -path '*/eval/eedi_same_question_c4_25/judge_predictions.jsonl')
mapfile -t c4_eval < <(find "$root" -type f -path '*/eval/eedi_same_question_c4_25/judge_eval.jsonl')
mapfile -t c4_metrics < <(find "$root" -type f -path '*/eval/eedi_same_question_c4_25/metrics.json')
[[ ${#formal_predictions[@]} -eq 1 && ${#formal_metrics[@]} -eq 1 \
   && ${#c4_predictions[@]} -eq 1 && ${#c4_eval[@]} -eq 1 \
   && ${#c4_metrics[@]} -eq 1 ]] || fail "semantic artifact cardinality drift"

judge(){
  local predictions="$1" evaluation="$2" output="$3" audit="$4"
  until PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
    --predictions-jsonl "$predictions" --eval-jsonl "$evaluation" \
    --output-dir "$output" --judge-backend trapi \
    --trapi-deployment gpt-4o_2024-11-20 --judge-workers 12 \
    --judge-max-retries 12 --judge-timeout-seconds 60 \
    --judge-cache "$output/cache.jsonl" --judge-audit-samples "$audit" --seed 83 \
    >> "$output/judge.log" 2>&1; do
    heartbeat; log "TRAPI judge unavailable; retrying cached panel"; sleep_unlocked 120
  done
}
judge "${formal_predictions[0]}" "$formal_eval" "$out/t3" 75
judge "${c4_predictions[0]}" "${c4_eval[0]}" "$out/eedi_c4" 100
"$python_bin" scripts/summarize_state8_diagnosis_c4.py \
  --judge-redacted "$out/eedi_c4/predictions_redacted.jsonl" \
  --generation-metrics "${c4_metrics[0]}" --output "$out/eedi_c4/summary.json"

mapfile -t curriculum_receipts < <(find "$root" -type f -path '*/metrics/eedi_curriculum.json')
mapfile -t merge_receipts < <(find "$root" -type f -path '*/metrics/eedi_causal_merge.json')
[[ ${#curriculum_receipts[@]} -eq 1 && ${#merge_receipts[@]} -eq 1 ]] \
  || fail "curriculum receipt cardinality drift"
"$python_bin" - "${formal_metrics[0]}" "$out/t3/metrics.json" \
  "$out/eedi_c4/summary.json" "${curriculum_receipts[0]}" \
  "${merge_receipts[0]}" "$state/summary.json" <<'PY'
import json,pathlib,sys
formal,t3,c4,curriculum,merge=(json.load(open(path)) for path in sys.argv[1:6])
summary={
 "stage":"state8_diagnosis_first_eedi_curriculum_semantic_summary_v1",
 "status":"complete",
 "semantic_judge_authoritative":True,
 "exact_match_authoritative":False,
 "formal_metrics":formal,
 "t3_semantic":t3,
 "eedi_same_question_diagnosis_c4":c4,
 "curriculum":curriculum,
 "causal_merge":merge,
}
pathlib.Path(sys.argv[6]).write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
PY
touch "$state/complete"; heartbeat
log "curriculum and semantic evaluation complete"
