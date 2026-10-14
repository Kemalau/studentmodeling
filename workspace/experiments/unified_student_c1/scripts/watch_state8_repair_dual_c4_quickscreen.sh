#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
source "$project/scripts/use_studentmodeling_feeds_auth.sh"
state="$project/results/state8_repair_dual_c4_quickscreen_20260813"
template="$project/amlt/feeds_state8_repair_dual_c4_quickscreen_20260813.yaml"
receipts="$project/results/state8_curriculum_repair_milestones_20260812"
config="$state/bound.yaml"
base=state8-ct-cp20-dual-c4-quickscreen-20260813
experiment="$base"; recovery=0; max_recovery=4
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"

mkdir -p "$state"; cd "$project" || exit 1
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"
trap 'rm -f "$state/controller.active.pid"' EXIT
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
sleep_unlocked(){ sleep "$1" 9>&-; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }

for spec in '8201 causal_teacher_low' '8202 causal_teacher_high'; do
  set -- $spec
  receipt="$receipts/seed${1}.${2}.checkpoint-20.completion.json"
  marker="$receipts/seed${1}.${2}.checkpoint-20.snapshot_complete"
  [[ -s "$receipt" && -f "$marker" ]] || fail "cp20 snapshot missing seed=$1 branch=$2"
done
"$python_bin" scripts/materialize_state8_repair_dual_c4_quickscreen.py \
  --template "$template" --receipt-dir "$receipts" --output "$config" \
  >> "$state/controller.log" 2>&1 || fail "quickscreen materialization failed"
"$python_bin" scripts/validate_state8_repair_dual_c4_quickscreen_amlt.py \
  --config "$config" >> "$state/controller.log" 2>&1 || fail "bound quickscreen is invalid"

submit_experiment(){
  local name="$1" audit quota available auditable output rc
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
  [[ "$auditable" == 1 && "$available" -ge 2 ]] || { flock -u 8; return 2; }
  quota="$state/$name.quota.json"
  amlt --json-tables ti sing -t Feeds > "$quota" 2>> "$state/controller.log" \
    || { flock -u 8; return 1; }
  "$python_bin" - "$quota" <<'PY' || { flock -u 8; return 2; }
import sys
from scripts.feeds_quota_utils import parse_json_table,select_a100_80gb
r=select_a100_80gb(parse_json_table(open(sys.argv[1]).read()))
q=r['quotas_(used/limit)']; assert q['standard']['limit']-q['standard']['used']>=2,q
assert int(q['user_max']['limit'])>=2,q
PY
  amlt run "$config" --dump -t Feeds > "$state/$name.preflight.yaml" \
    2> "$state/$name.preflight.stderr" || { flock -u 8; return 1; }
  "$python_bin" scripts/validate_state8_repair_dual_c4_quickscreen_amlt.py \
    --config "$state/$name.preflight.yaml" >> "$state/controller.log" 2>&1 \
    || { flock -u 8; return 1; }
  output="$(amlt run "$config" "$name" -t Feeds -y --description \
    "Hash-bound read-only dual diagnosis-C4 quickscreen for causal-teacher cp20 checkpoints" 2>&1)"; rc=$?
  printf '[%s] rc=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$output" \
    > "$state/$name.submission.txt"
  flock -u 8
  (( rc == 0 )) || return 1
  printf '%s\n' "$name" > "$state/active_experiment.txt"
}

if [[ -s "$state/active_experiment.txt" ]]; then
  experiment="$(<"$state/active_experiment.txt")"
  [[ "$experiment" =~ -recovery([0-9]+)$ ]] && recovery="${BASH_REMATCH[1]}"
else
  until submit_experiment "$experiment"; do heartbeat; sleep_unlocked 60; done
fi

jobs=(state8-ct-causal-teacher-low-seed8201-cp20-dual-c4 state8-ct-causal-teacher-high-seed8202-cp20-dual-c4)
while [[ ! -f "$state/remote_passed" ]]; do
  heartbeat
  status="$(timeout 90 amlt --json-tables status "$experiment" --hide-urls --force-status-update 2>/dev/null)" \
    || { sleep_unlocked 30; continue; }
  [[ -n "$status" && "$status" != '[]' ]] || { sleep_unlocked 30; continue; }
  printf '%s\n' "$status" > "$state/latest_status.json"
  phase="$(STATUS="$status" "$python_bin" - <<'PY'
import json,os
s={str(x.get('status') or '').casefold() for x in json.loads(os.environ['STATUS'])}
if s=={'pass'}: print('pass')
elif s&{'preparing','pre-queued','queued','running','resuming'}: print('active')
elif 'paused' in s: print('paused')
else: print('terminal')
PY
)"
  case "$phase" in
    pass) touch "$state/remote_passed" ;;
    active) sleep_unlocked 30 ;;
    paused) for job in "${jobs[@]}"; do timeout 90 amlt resume "$experiment" ":$job" >> "$state/controller.log" 2>&1 || true; done; sleep_unlocked 30 ;;
    *)
      for job in "${jobs[@]}"; do timeout 180 amlt logs view -n 3000 "$experiment" ":$job" > "$state/$experiment.$job.failure.log" 2>&1 || true; done
      (( recovery < max_recovery )) || fail "quickscreen exhausted recovery waves"
      recovery=$((recovery+1)); experiment="${base}-recovery${recovery}"
      until submit_experiment "$experiment"; do heartbeat; sleep_unlocked 60; done ;;
  esac
done

rm -rf "$state/download"; mkdir -p "$state/download"
amlt results download "$experiment" --status pass --include '*.json' --include '*.jsonl' \
  --include '*.txt' --output "$state/download" >> "$state/controller.log" 2>&1 \
  || fail "quickscreen result download failed"

judge(){
  local predictions="$1" evaluation="$2" destination="$3" audit="$4" seed="$5"
  mkdir -p "$destination"
  until PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
    --predictions-jsonl "$predictions" --eval-jsonl "$evaluation" \
    --output-dir "$destination" --judge-backend trapi \
    --trapi-deployment gpt-4o_2024-11-20 --judge-workers 12 \
    --judge-max-retries 12 --judge-timeout-seconds 60 \
    --judge-cache "$destination/cache.jsonl" --judge-audit-samples "$audit" \
    --seed "$seed" >> "$destination/judge.log" 2>&1; do
      heartbeat; sleep_unlocked 60
  done
}

for spec in '8201 causal_teacher_low' '8202 causal_teacher_high'; do
  set -- $spec; seed=$1; branch=$2
  job="state8-ct-${branch//_/-}-seed${seed}-cp20-dual-c4"
  mapfile -t roots < <(find "$state/download" -type d -name "$job")
  [[ ${#roots[@]} -eq 1 ]] || fail "quickscreen artifact root cardinality drift job=$job"
  root="${roots[0]}"; semantic="$state/semantic/seed${seed}_${branch}"
  for domain in malrule eedi; do
    if [[ "$domain" == malrule ]]; then generated="$root/eval/malrule_diagnosis_c4_20"; else generated="$root/eval/eedi_same_question_c4_25"; fi
    judge "$generated/judge_predictions.jsonl" "$generated/judge_eval.jsonl" \
      "$semantic/${domain}_gold" 50 "$seed"
    judge "$root/eval/${domain}_pairwise/judge_predictions.jsonl" \
      "$root/eval/${domain}_pairwise/judge_eval.jsonl" \
      "$semantic/${domain}_pairwise" 40 "$((seed+1))"
    "$python_bin" scripts/summarize_state8_diagnosis_c4.py \
      --judge-redacted "$semantic/${domain}_gold/predictions_redacted.jsonl" \
      --generation-metrics "$generated/metrics.json" \
      --output "$semantic/${domain}_gold_summary.json" \
      >> "$state/controller.log" 2>&1 || fail "$domain gold summary failed"
    "$python_bin" scripts/summarize_state8_diagnosis_c4_pairwise.py \
      --judge-redacted "$semantic/${domain}_pairwise/predictions_redacted.jsonl" \
      --panel-receipt "$root/eval/${domain}_pairwise/panel_receipt.json" \
      --gold-summary "$semantic/${domain}_gold_summary.json" \
      --output "$semantic/${domain}_pairwise_summary.json" \
      >> "$state/controller.log" 2>&1 || fail "$domain pairwise summary failed"
  done
  "$python_bin" scripts/summarize_state8_repair_dual_c4_quickscreen.py \
    --receipt "$root/metrics/quickscreen_receipt.json" \
    --malrule-gold "$semantic/malrule_gold_summary.json" \
    --eedi-gold "$semantic/eedi_gold_summary.json" \
    --malrule-pairwise "$semantic/malrule_pairwise_summary.json" \
    --eedi-pairwise "$semantic/eedi_pairwise_summary.json" \
    --output "$semantic/summary.json" >> "$state/controller.log" 2>&1 \
    || fail "joint quickscreen summary failed seed=$seed"
done
touch "$state/complete"; heartbeat
log "causal-teacher cp20 dual-C4 semantic quickscreen complete"
