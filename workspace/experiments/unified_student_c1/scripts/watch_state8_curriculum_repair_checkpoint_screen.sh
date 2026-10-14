#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
source "$project/scripts/use_studentmodeling_feeds_auth.sh"
: "${REPAIR_SEED:?}" "${REPAIR_BRANCH:?}" "${CHECKPOINT_STEP:?}"
case "$REPAIR_SEED:$REPAIR_BRANCH" in
  5201:semantic_binding|5202:specificity_triplet|5203:balanced_continuation|\
  6201:semantic_binding|6202:specificity_triplet|6203:balanced_continuation) ;;
  8201:causal_teacher_low|8202:causal_teacher_high|\
  8301:early_label_low|8302:early_label_high) ;;
  *) exit 2 ;;
esac
case "$CHECKPOINT_STEP" in 20|100|240|500) ;; *) exit 2 ;; esac

slug="seed${REPAIR_SEED}_${REPAIR_BRANCH}"
state="$project/results/state8_curriculum_repair_screens_20260812/$slug/checkpoint-${CHECKPOINT_STEP}"
milestones="$project/results/state8_curriculum_repair_milestones_20260812"
receipt="$milestones/seed${REPAIR_SEED}.${REPAIR_BRANCH}.checkpoint-${CHECKPOINT_STEP}.completion.json"
marker="$milestones/seed${REPAIR_SEED}.${REPAIR_BRANCH}.checkpoint-${CHECKPOINT_STEP}.snapshot_complete"
template="$project/amlt/feeds_state8_curriculum_checkpoint_screen_20260812.yaml"
config="$state/bound.yaml"
base="state8-repair-${REPAIR_BRANCH//_/-}-seed${REPAIR_SEED}-cp${CHECKPOINT_STEP}-screen-20260812"
experiment="$base"; recovery=0; max_recovery=8
job="state8-repair-${REPAIR_BRANCH//_/-}-seed${REPAIR_SEED}-cp${CHECKPOINT_STEP}-screen"
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

while [[ ! -f "$marker" ]]; do
  heartbeat
  [[ ! -f "$milestones/needs_attention" ]] || fail "repair milestone watcher needs attention"
  sleep_unlocked 10
done
[[ -s "$receipt" ]] || fail "repair checkpoint receipt missing"
"$python_bin" scripts/materialize_state8_curriculum_repair_checkpoint_screen.py \
  --template "$template" --receipt "$receipt" --seed "$REPAIR_SEED" \
  --branch "$REPAIR_BRANCH" --step "$CHECKPOINT_STEP" --output "$config" \
  >> "$state/controller.log" 2>&1 || fail "screen config materialization failed"
"$python_bin" scripts/validate_state8_curriculum_repair_checkpoint_screen_amlt.py \
  --config "$config" --expected-seed "$REPAIR_SEED" \
  --expected-branch "$REPAIR_BRANCH" --expected-step "$CHECKPOINT_STEP" \
  >> "$state/controller.log" 2>&1 || fail "bound screen config is invalid"

submit_experiment(){
  local name="$1" current class output rc quota audit available auditable
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
import sys
from scripts.feeds_quota_utils import parse_json_table,select_a100_80gb
r=select_a100_80gb(parse_json_table(open(sys.argv[1]).read()))
q=r['quotas_(used/limit)']; assert q['standard']['limit']-q['standard']['used']>=1,q
assert int(q['user_max']['limit'])>=1,q
PY
  amlt run "$config" --dump -t Feeds > "$state/$name.preflight.yaml" \
    2> "$state/$name.preflight.stderr" || { flock -u 8; return 1; }
  "$python_bin" scripts/validate_state8_curriculum_repair_checkpoint_screen_amlt.py \
    --config "$state/$name.preflight.yaml" --expected-seed "$REPAIR_SEED" \
    --expected-branch "$REPAIR_BRANCH" --expected-step "$CHECKPOINT_STEP" \
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
      "Hash-bound read-only State8 repair checkpoint screen seed=$REPAIR_SEED branch=$REPAIR_BRANCH step=$CHECKPOINT_STEP" 2>&1)"; rc=$?
    printf '[%s] rc=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$output" \
      > "$state/$name.submission.txt"
    (( rc == 0 )) || { flock -u 8; return 1; }
  elif [[ "$class" != exists ]]; then flock -u 8; return 1; fi
  flock -u 8
  printf '%s\n' "$name" > "$state/active_experiment.txt"
}

if [[ -s "$state/active_experiment.txt" ]]; then
  experiment="$(<"$state/active_experiment.txt")"
  [[ "$experiment" =~ -recovery([0-9]+)$ ]] && recovery="${BASH_REMATCH[1]}"
else
  until submit_experiment "$experiment"; do heartbeat; sleep_unlocked 60; done
fi

while [[ ! -f "$state/remote_passed" ]]; do
  heartbeat
  status="$(timeout 90 amlt --json-tables status "$experiment" --hide-urls \
    --force-status-update 2>/dev/null)" || { sleep_unlocked 30; continue; }
  [[ -n "$status" && "$status" != '[]' ]] || { sleep_unlocked 30; continue; }
  printf '%s\n' "$status" > "$state/latest_status.json"
phase="$(STATUS="$status" "$python_bin" - <<'PY'
import json,os
s={str(x.get('status') or '').casefold() for x in json.loads(os.environ['STATUS'])}
if s=={'pass'}: print('pass')
elif s&{'preparing','pre-queued','queued','running','resuming'}: print('active')
elif 'paused' in s: print('paused')
elif s&{'failed','killed','cancelled','canceled','stopped'}: print('terminal')
else: print('unknown')
PY
)"
  case "$phase" in
    pass) touch "$state/remote_passed" ;;
    active) sleep_unlocked 30 ;;
    paused) timeout 90 amlt resume "$experiment" ":$job" >> "$state/controller.log" 2>&1 || true; sleep_unlocked 30 ;;
    unknown) sleep_unlocked 30 ;;
    *)
      timeout 180 amlt logs view -n 3000 "$experiment" ":$job" \
        > "$state/$experiment.failure.log" 2>&1 || true
      (( recovery < max_recovery )) || fail "screen exhausted recovery waves"
      recovery=$((recovery+1)); experiment="${base}-recovery${recovery}"
      until submit_experiment "$experiment"; do heartbeat; sleep_unlocked 60; done ;;
  esac
done

# A remote `pass` is authoritative for the GPU job, but it does not guarantee
# that the local Azure token will still be alive during result collection.
# Authentication expiry is therefore a wait condition, not a scientific or
# remote-job failure.  Once authentication is healthy, tolerate a small number
# of genuine transport failures before asking for attention.
download_failures=0
while :; do
  heartbeat
  source "$project/scripts/use_studentmodeling_feeds_auth.sh"
  if [[ "${STUDENTMODELING_FEEDS_AUTH_HEALTH:-unhealthy}" == unhealthy ]]; then
    log "result download waiting for Feeds authentication"
    sleep_unlocked 30
    continue
  fi
  rm -rf "$state/download"; mkdir -p "$state/download"
  if timeout --foreground 600 amlt results download "$experiment" ":$job" \
      --include '*.json' --include '*.jsonl' --include '*.txt' \
      --output "$state/download" >> "$state/controller.log" 2>&1; then
    break
  fi
  download_failures=$((download_failures+1))
  (( download_failures < 8 )) || fail "screen result download failed after authenticated retries"
  log "transient screen result download failure attempt=$download_failures; retrying"
  sleep_unlocked 30
done
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
  bash scripts/wait_for_trapi_auth.sh
  until PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
    --predictions-jsonl "$predictions" --eval-jsonl "$evaluation" \
    --output-dir "$destination" --judge-backend trapi \
    --trapi-deployment gpt-4o_2024-11-20 --judge-workers 12 \
    --judge-max-retries 12 --judge-timeout-seconds 60 \
    --judge-cache "$destination/cache.jsonl" --judge-audit-samples "$audit" \
    --seed "$CHECKPOINT_STEP" >> "$destination/judge.log" 2>&1; do
      heartbeat; sleep_unlocked 60
  done
}
judge "${formal_predictions[0]}" "$eval_jsonl" "$semantic/t3" 75
judge "${malrule_pred[0]}" "${malrule_eval[0]}" "$semantic/malrule_c4" 40
judge "${eedi_pred[0]}" "${eedi_eval[0]}" "$semantic/eedi_c4" 50
"$python_bin" scripts/summarize_state8_diagnosis_c4.py \
  --judge-redacted "$semantic/malrule_c4/predictions_redacted.jsonl" \
  --generation-metrics "${malrule_metrics[0]}" --output "$semantic/malrule_c4/summary.json" \
  || fail "MalRule diagnosis-C4 summary failed"
"$python_bin" scripts/summarize_state8_diagnosis_c4.py \
  --judge-redacted "$semantic/eedi_c4/predictions_redacted.jsonl" \
  --generation-metrics "${eedi_metrics[0]}" --output "$semantic/eedi_c4/summary.json" \
  || fail "Eedi diagnosis-C4 summary failed"
"$python_bin" scripts/summarize_state8_curriculum_checkpoint_screen.py \
  --arm "repair_${REPAIR_BRANCH}_seed${REPAIR_SEED}" --step "$CHECKPOINT_STEP" \
  --formal "${formal_metrics[0]}" --t3 "$semantic/t3/metrics.json" \
  --answer "${answer[0]}" --malrule "$semantic/malrule_c4/summary.json" \
  --eedi "$semantic/eedi_c4/summary.json" --output "$state/summary.json" \
  >> "$state/controller.log" 2>&1 || fail "unified repair summary failed"
touch "$state/complete"; heartbeat
log "repair checkpoint unified semantic screen complete seed=$REPAIR_SEED branch=$REPAIR_BRANCH step=$CHECKPOINT_STEP"
