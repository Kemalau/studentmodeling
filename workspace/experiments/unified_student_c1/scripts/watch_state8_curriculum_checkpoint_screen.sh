#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
source "$project/scripts/use_studentmodeling_feeds_auth.sh"
: "${CURRICULUM_ARM:?CURRICULUM_ARM must be generic or current_query}" \
  "${CHECKPOINT_STEP:?CHECKPOINT_STEP must be 2000, 3000, 4000, or 5000}"
case "$CURRICULUM_ARM" in generic|current_query) ;; *) exit 2 ;; esac
case "$CHECKPOINT_STEP" in 2000|3000|4000|5000) ;; *) exit 2 ;; esac
arm="$CURRICULUM_ARM"; step="$CHECKPOINT_STEP"
if [[ "$arm" == generic ]]; then
  curriculum_tag=state8_diagnosis_first_eedi_curriculum_20260812
else
  curriculum_tag=state8_diagnosis_first_eedi_current_query_curriculum_20260812
fi
milestones="$project/results/$curriculum_tag/milestones"
state="$project/results/state8_curriculum_screens_20260812/${arm}/checkpoint-${step}"
template="$project/amlt/feeds_state8_curriculum_checkpoint_screen_20260812.yaml"
base="state8-curriculum-${arm//_/-}-cp${step}-screen-20260812"
experiment="$base"; recovery=0; max_recovery=8
job="state8-curriculum-${arm//_/-}-cp${step}-screen"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
eval_jsonl="$project/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"

mkdir -p "$state"; cd "$project" || exit 1
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"
trap 'rm -f "$state/controller.active.pid"' EXIT
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
sleep_unlocked(){ sleep "$1" 9>&-; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }

while [[ ! -f "$milestones/checkpoint-${step}.snapshot_complete" ]]; do
  heartbeat
  [[ ! -f "$milestones/needs_attention" ]] || fail "curriculum milestone watcher needs attention"
  sleep_unlocked 30
done
receipt="$milestones/checkpoint-${step}.completion.json"
[[ -s "$receipt" ]] || fail "checkpoint receipt missing after snapshot completion"
config="$state/bound.yaml"
"$python_bin" scripts/materialize_state8_curriculum_checkpoint_screen.py \
  --template "$template" --receipt "$receipt" --arm "$arm" --step "$step" \
  --output "$config" >> "$state/controller.log" 2>&1 \
  || fail "checkpoint screen config materialization failed"
"$python_bin" scripts/validate_state8_curriculum_checkpoint_screen_amlt.py \
  --config "$config" --expected-arm "$arm" --expected-step "$step" \
  >> "$state/controller.log" 2>&1 || fail "bound checkpoint screen config is invalid"

submit_experiment(){
  local name="$1" current class output rc audit available auditable quota
  exec 8>"$project/results/feeds_submission_global.lock"; flock -x 8
  audit="$state/$name.active_a100.json"
  "$python_bin" scripts/count_active_a100_requests.py --user-limit 32 --recent 500 \
    --output "$audit" >/dev/null 2>> "$state/controller.log" \
    || { flock -u 8; return 1; }
  read -r available auditable < <("$python_bin" - "$audit" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); print(int(x['available_user_slots']),int(bool(x['auditable'])))
PY
  )
  [[ "$auditable" == 1 && "$available" -ge 1 ]] || { flock -u 8; return 2; }
  quota="$state/$name.quota.json"
  amlt --json-tables ti sing -t Feeds > "$quota" 2>> "$state/controller.log" \
    || { flock -u 8; return 1; }
  "$python_bin" - "$quota" <<'PY' || { flock -u 8; return 2; }
import json,sys
from scripts.feeds_quota_utils import parse_json_table,select_a100_80gb
row=select_a100_80gb(parse_json_table(open(sys.argv[1]).read()))
q=row['quotas_(used/limit)']; assert q['standard']['limit']-q['standard']['used']>=1,q
assert int(q['user_max']['limit'])>=1,q
PY
  amlt run "$config" --dump -t Feeds > "$state/$name.preflight.yaml" \
    2> "$state/$name.preflight.stderr" || { flock -u 8; return 1; }
  "$python_bin" scripts/validate_state8_curriculum_checkpoint_screen_amlt.py \
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
    output="$(amlt run "$config" "$name" -t Feeds -y --description \
      "Hash-bound read-only State8 $arm curriculum checkpoint-$step unified-task, answer-C4, MalRule diagnosis-C4, and identical-query Eedi C4 screen" 2>&1)"; rc=$?
    printf '[%s] rc=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$output" \
      > "$state/$name.submission.txt"
    (( rc == 0 )) || { flock -u 8; return 1; }
  elif [[ "$class" != exists ]]; then
    flock -u 8; return 1
  fi
  flock -u 8
  printf '%s\n' "$name" > "$state/active_experiment.txt"
}

if [[ -s "$state/active_experiment.txt" ]]; then
  experiment="$(<"$state/active_experiment.txt")"
  [[ "$experiment" =~ -recovery([0-9]+)$ ]] && recovery="${BASH_REMATCH[1]}"
else
  until submit_experiment "$experiment"; do
    heartbeat; log "waiting for an auditable A100 slot for arm=$arm step=$step"; sleep_unlocked 120
  done
  log "submitted checkpoint screen experiment=$experiment"
fi

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
    paused)
      timeout 90 amlt resume "$experiment" ":$job" >> "$state/controller.log" 2>&1 || true
      sleep_unlocked 60 ;;
    unknown) sleep_unlocked 60 ;;
    *)
      timeout 90 amlt show "$experiment" > "$state/$experiment.show.txt" 2>&1 || true
      timeout 180 amlt logs view -n 3000 "$experiment" ":$job" \
        > "$state/$experiment.failure.log" 2>&1 || true
      (( recovery < max_recovery )) || fail "checkpoint screen exhausted recovery waves"
      recovery=$((recovery+1)); experiment="${base}-recovery${recovery}"
      until submit_experiment "$experiment"; do heartbeat; sleep_unlocked 120; done
      log "submitted screen recovery=$recovery experiment=$experiment" ;;
  esac
done

rm -rf "$state/download"; mkdir -p "$state/download"
amlt results download "$experiment" ":$job" --include '*.json' --include '*.jsonl' \
  --include '*.txt' --output "$state/download" >> "$state/controller.log" 2>&1 \
  || fail "screen result download failed"
mapfile -t roots < <(find "$state/download" -type d -name "$job")
[[ ${#roots[@]} -eq 1 ]] || fail "artifact root cardinality drift"
root="${roots[0]}"; semantic="$state/semantic"
mkdir -p "$semantic/t3" "$semantic/malrule_c4" "$semantic/eedi_c4"
mapfile -t formal_metrics < <(find "$root" -type f -path '*/eval/formal_all_tasks/metrics.json')
mapfile -t formal_predictions < <(find "$root" -type f -path '*/eval/formal_all_tasks/predictions.jsonl')
mapfile -t answer < <(find "$root" -type f -path '*/eval/answer_c4_50/recurrent_causal_gate.json')
mapfile -t malrule_pred < <(find "$root" -type f -path '*/eval/malrule_diagnosis_c4_20/judge_predictions.jsonl')
mapfile -t malrule_eval < <(find "$root" -type f -path '*/eval/malrule_diagnosis_c4_20/judge_eval.jsonl')
mapfile -t malrule_metrics < <(find "$root" -type f -path '*/eval/malrule_diagnosis_c4_20/metrics.json')
mapfile -t eedi_pred < <(find "$root" -type f -path '*/eval/eedi_same_question_c4_25/judge_predictions.jsonl')
mapfile -t eedi_eval < <(find "$root" -type f -path '*/eval/eedi_same_question_c4_25/judge_eval.jsonl')
mapfile -t eedi_metrics < <(find "$root" -type f -path '*/eval/eedi_same_question_c4_25/metrics.json')
[[ ${#formal_metrics[@]} -eq 1 && ${#formal_predictions[@]} -eq 1 && ${#answer[@]} -eq 1 \
   && ${#malrule_pred[@]} -eq 1 && ${#malrule_eval[@]} -eq 1 && ${#malrule_metrics[@]} -eq 1 \
   && ${#eedi_pred[@]} -eq 1 && ${#eedi_eval[@]} -eq 1 && ${#eedi_metrics[@]} -eq 1 ]] \
  || fail "semantic artifact cardinality drift"
judge(){
  local predictions="$1" evaluation="$2" destination="$3" audit="$4"
  until PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
    --predictions-jsonl "$predictions" --eval-jsonl "$evaluation" --output-dir "$destination" \
    --judge-backend trapi --trapi-deployment gpt-4o_2024-11-20 --judge-workers 12 \
    --judge-max-retries 12 --judge-timeout-seconds 60 --judge-cache "$destination/cache.jsonl" \
    --judge-audit-samples "$audit" --seed "$step" >> "$destination/judge.log" 2>&1; do
      heartbeat; sleep_unlocked 120
  done
}
judge "${formal_predictions[0]}" "$eval_jsonl" "$semantic/t3" 75
judge "${malrule_pred[0]}" "${malrule_eval[0]}" "$semantic/malrule_c4" 40
judge "${eedi_pred[0]}" "${eedi_eval[0]}" "$semantic/eedi_c4" 50
"$python_bin" scripts/summarize_state8_diagnosis_c4.py \
  --judge-redacted "$semantic/malrule_c4/predictions_redacted.jsonl" \
  --generation-metrics "${malrule_metrics[0]}" --output "$semantic/malrule_c4/summary.json" \
  >> "$semantic/malrule_c4/summarize.log" 2>&1 || fail "MalRule diagnosis-C4 summary failed"
"$python_bin" scripts/summarize_state8_diagnosis_c4.py \
  --judge-redacted "$semantic/eedi_c4/predictions_redacted.jsonl" \
  --generation-metrics "${eedi_metrics[0]}" --output "$semantic/eedi_c4/summary.json" \
  >> "$semantic/eedi_c4/summarize.log" 2>&1 || fail "Eedi diagnosis-C4 summary failed"
"$python_bin" scripts/summarize_state8_curriculum_checkpoint_screen.py \
  --arm "$arm" --step "$step" --formal "${formal_metrics[0]}" \
  --t3 "$semantic/t3/metrics.json" --answer "${answer[0]}" \
  --malrule "$semantic/malrule_c4/summary.json" --eedi "$semantic/eedi_c4/summary.json" \
  --output "$state/summary.json" >> "$state/controller.log" 2>&1 \
  || fail "unified checkpoint summary failed"
touch "$state/complete"; heartbeat
log "curriculum checkpoint unified semantic screen complete arm=$arm step=$step"
