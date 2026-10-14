#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
source "$project/scripts/use_studentmodeling_feeds_auth.sh"
updater="$project/results/state8_eedi_same_question_causal_repair_20260811"
full="$project/results/state8_eedi_same_question_full_mse1_20260811"
early="$project/results/state8_eedi_same_question_early_selection_20260811"
repair="$project/results/state8_eedi_same_question_repair_branches_20260811"
state="${STATE8_PROMOTION_STATE_DIR:-$project/results/state8_eedi_same_question_promotion_20260811}"
template="${STATE8_PROMOTION_TEMPLATE:-$project/amlt/feeds_state8_eedi_same_question_answer_c4_20260811.yaml}"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
base="${STATE8_PROMOTION_BASE_EXPERIMENT:-state8-eedi-same-question-answer-c4-20260811}"
experiment="$base"; wave=0; max_waves=12
job=state8-eedi-same-question-answer-c4-selected

mkdir -p "$state"; cd "$project" || exit 1
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"; trap 'rm -f "$state/controller.active.pid"' EXIT
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
sleep_unlocked(){ sleep "$1" 9>&-; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }

selection="$state/selection.json"
if [[ -n "${STATE8_PROMOTION_SELECTION_FILE:-}" && ! -f "$selection" ]]; then
  while [[ ! -s "$STATE8_PROMOTION_SELECTION_FILE" ]]; do heartbeat; sleep_unlocked 60; done
  cp "$STATE8_PROMOTION_SELECTION_FILE" "$selection"
  "$python_bin" - "$selection" <<'PY' || fail "external selection is not qualified"
import json,sys
x=json.load(open(sys.argv[1])); assert x['pass'] is True and x.get('selected'),x
PY
  touch "$state/diagnostic_gate_pass"
fi
while [[ ! -f "$selection" ]]; do
  heartbeat
  candidate_args=()
  [[ -s "$updater/summary.json" && -f "$updater/complete" ]] \
    && candidate_args+=(--candidate updater "$updater/summary.json")
  [[ -s "$full/summary.json" && -f "$full/complete" ]] \
    && candidate_args+=(--candidate full "$full/summary.json")
  while IFS= read -r summary; do
    slug="$(basename "$(dirname "$summary")")"
    candidate_args+=(--candidate "$slug" "$summary")
  done < <(find "$repair" -mindepth 2 -maxdepth 2 -type f -name summary.json \
    -path '*/repair_*/*' 2>/dev/null | sort)

  rc=2
  if (( ${#candidate_args[@]} > 0 )); then
    attempt="$state/selection.attempt.json"
    rm -f "$attempt"
    set +e
    "$python_bin" scripts/select_state8_eedi_same_question_candidate.py \
      "${candidate_args[@]}" --output "$attempt" \
      > "$state/selection.log" 2>&1
    rc=$?
    set -e
    if (( rc == 0 )); then
      mv "$attempt" "$selection"
      touch "$state/diagnostic_gate_pass"
      break
    fi
    (( rc == 2 )) || fail "same-question selector crashed rc=$rc"
    [[ -f "$attempt" ]] && mv "$attempt" "$state/selection.latest_failed.json"
  fi

  exhausted=0
  if [[ -f "$repair/complete" ]]; then exhausted=1
  elif [[ -f "$updater/complete" && -f "$full/complete" \
          && -f "$early/complete" && -f "$repair/not_required" ]]; then exhausted=1
  fi
  if (( exhausted )); then
    touch "$state/diagnostic_gate_failed" "$state/complete"; heartbeat
    log "all completed same-question candidates failed the joint diagnostic/unified gate"
    exit 0
  fi
  sleep_unlocked 60
done

if [[ ! -f "$state/diagnostic_gate_pass" ]]; then
  set +e
  "$python_bin" scripts/select_state8_eedi_same_question_candidate.py \
    --candidate updater "$updater/summary.json" \
    --candidate full "$full/summary.json" --output "$selection" \
    > "$state/selection.log" 2>&1
  rc=$?
  set -e
  if (( rc == 2 )); then
    touch "$state/diagnostic_gate_failed" "$state/complete"; heartbeat
    log "same-question candidates failed the joint diagnostic/unified gate"
    exit 0
  elif (( rc != 0 )); then fail "same-question selector crashed rc=$rc"; fi
  touch "$state/diagnostic_gate_pass"
fi

config="$state/answer_c4.generated.yaml"
"$python_bin" scripts/materialize_state8_eedi_same_question_answer_c4.py \
  --template "$template" --selection "$selection" --output "$config" \
  >> "$state/controller.log" 2>&1 || fail "answer-C4 materialization failed"
"$python_bin" scripts/validate_state8_eedi_same_question_answer_c4_amlt.py \
  --config "$config" >> "$state/controller.log" 2>&1 || fail "answer-C4 config validation failed"

submit(){
  local name="$1" current class output rc
  exec 8>"$project/results/feeds_submission_global.lock"; flock -x 8
  amlt run "$config" --dump -t Feeds > "$state/$name.preflight.yaml" \
    2> "$state/$name.preflight.stderr" || { flock -u 8; return 1; }
  "$python_bin" scripts/validate_state8_eedi_same_question_answer_c4_amlt.py \
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
    amlt --json-tables ti sing -t Feeds > "$state/$name.quota.json" \
      2>> "$state/controller.log" || { flock -u 8; return 1; }
    output="$(amlt run "$config" "$name" -t Feeds -y \
      --description "Frozen 150-pair answer-C4 for selected identical-query Eedi State8 candidate" 2>&1)"; rc=$?
    printf '[%s] rc=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$output" \
      > "$state/$name.submission.txt"
    (( rc == 0 )) || { flock -u 8; return 1; }
  elif [[ "$class" != exists ]]; then flock -u 8; return 1; fi
  flock -u 8; printf '%s\n' "$name" > "$state/active_experiment.txt"
}

if [[ -s "$state/active_experiment.txt" ]]; then experiment="$(<"$state/active_experiment.txt")"
else until submit "$experiment"; do heartbeat; sleep_unlocked 120; done; fi

while [[ ! -f "$state/remote_passed" ]]; do
  heartbeat
  status="$(timeout 90 amlt --json-tables status "$experiment" --hide-urls --force-status-update 2>/dev/null)" \
    || { sleep_unlocked 60; continue; }
  [[ -n "$status" && "$status" != '[]' ]] || { sleep_unlocked 30; continue; }
  printf '%s\n' "$status" > "$state/latest_status.json"
  phase="$(STATUS="$status" "$python_bin" - <<'PY'
import json,os
states={str(x.get('status') or '').casefold() for x in json.loads(os.environ['STATUS'])}
if states=={'pass'}: print('pass')
elif states&{'preparing','pre-queued','queued','running','resuming'}: print('active')
elif 'paused' in states: print('paused')
elif not states or states<={'','unknown'}: print('unknown')
else: print('terminal')
PY
)"
  case "$phase" in
    pass) touch "$state/remote_passed" ;;
    active|unknown) sleep_unlocked 60 ;;
    paused) timeout 90 amlt resume "$experiment" ":$job" >> "$state/controller.log" 2>&1 || true; sleep_unlocked 60 ;;
    *)
      (( wave < max_waves )) || fail "answer-C4 recovery exhausted"
      wave=$((wave+1)); experiment="${base}-recovery${wave}"
      until submit "$experiment"; do heartbeat; sleep_unlocked 120; done ;;
  esac
done

rm -rf "$state/download"; mkdir -p "$state/download"
amlt results download "$experiment" ":$job" --include '*.json' --include '*.txt' \
  --output "$state/download" >> "$state/controller.log" 2>&1 || fail "answer-C4 download failed"
mapfile -t answer < <(find "$state/download" -type f -path '*/eval/answer_c4_seed42_n150/recurrent_causal_gate.json')
[[ ${#answer[@]} -eq 1 ]] || fail "answer-C4 artifact cardinality drift"
set +e
"$python_bin" scripts/finalize_state8_eedi_same_question_promotion.py \
  --selection "$selection" --answer-c4 "${answer[0]}" --output "$state/promotion.json" \
  >> "$state/controller.log" 2>&1
rc=$?
set -e
if (( rc == 0 )); then
  gate_blob="${STATE8_PROMOTION_GATE_BLOB:-studentmodeling/gates/state8_eedi_same_question_v1/promotion.json}"
  az storage blob upload --auth-mode login \
    --account-name csnewsandfeeds4150361735 --container-name amulet \
    --name "$gate_blob" \
    --file "$state/promotion.json" --overwrite --only-show-errors \
    > "$state/promotion_upload.txt" 2>&1 \
    || fail "same-question promotion gate upload failed"
  touch "$state/scientific_gate_pass"
elif (( rc == 2 )); then touch "$state/scientific_gate_failed"
else fail "same-question promotion finalizer crashed rc=$rc"; fi
touch "$state/complete"; heartbeat
log "same-question joint promotion reached terminal state rc=$rc"
