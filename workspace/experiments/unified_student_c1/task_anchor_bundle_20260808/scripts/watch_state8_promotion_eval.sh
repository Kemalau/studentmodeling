#!/usr/bin/env bash
set -uo pipefail

project_dir=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
config="$project_dir/amlt/feeds_state8_promotion_eval_20260805.yaml"
promotion_variant="${PROMOTION_VARIANT:-t3_heavy}"
case "$promotion_variant" in
  t3_heavy)
    train_experiment=state8-qualified-teacher-curriculum-20260805
    base_experiment=state8-qualified-promotion-eval-20260805
    recovery_prefix=state8-qualified-promotion-eval-recovery
    state_dir="$project_dir/results/state8_qualified_promotion_eval_20260805"
    train_dir="$project_dir/results/state8_qualified_teacher_pipeline_20260805"
    train_active_dir="$train_dir/primary_curriculum"
    train_artifact_dir="$train_dir/stage_b_download"
    promotion_stage=state8_qualified_promotion_gate_v1
    job_prefix=state8-promotion
    ;;
  balanced)
    train_experiment=state8-qualified-teacher-balanced-curriculum-20260805
    base_experiment=state8-qualified-balanced-promotion-eval-20260805
    recovery_prefix=state8-qualified-balanced-promotion-eval-recovery
    state_dir="$project_dir/results/state8_qualified_balanced_promotion_eval_20260805"
    train_dir="$project_dir/results/state8_qualified_balanced_teacher_pipeline_20260805"
    train_active_dir="$train_dir"
    train_artifact_dir="$train_dir/download"
    promotion_stage=state8_qualified_balanced_promotion_gate_v1
    job_prefix=state8-balanced-promotion
    ;;
  highkd)
    train_experiment=state8-qualified-teacher-highkd-curriculum-20260805
    base_experiment=state8-qualified-highkd-promotion-eval-20260805
    recovery_prefix=state8-qualified-highkd-promotion-eval-recovery
    state_dir="$project_dir/results/state8_qualified_highkd_promotion_eval_20260805"
    train_dir="$project_dir/results/state8_qualified_highkd_teacher_pipeline_20260805"
    train_active_dir="$train_dir"
    train_artifact_dir="$train_dir/download"
    promotion_stage=state8_qualified_highkd_promotion_gate_v1
    job_prefix=state8-highkd-promotion
    ;;
  *) echo "Unknown PROMOTION_VARIANT=$promotion_variant" >&2; exit 2 ;;
esac
eval_jsonl="$project_dir/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project_dir/scripts:/home/v-yuejiali/fkft/src"
max_recovery_waves="${MAX_RECOVERY_WAVES:-24}"
all_jobs=(
  "$job_prefix-t3-shard0"
  "$job_prefix-t3-shard1"
  "$job_prefix-t3-shard2"
  "$job_prefix-t3-shard3"
  "$job_prefix-c4-seed13"
  "$job_prefix-c4-seed42"
  "$job_prefix-c4-seed97"
)

mkdir -p "$state_dir/preserved_results" "$state_dir/t3_semantic/full1494"
cd "$project_dir" || exit 1
exec 9>"$state_dir/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state_dir/controller.active.pid"
heartbeat_pid=""
cleanup(){
  if [[ -n "$heartbeat_pid" ]]; then
    kill "$heartbeat_pid" 2>/dev/null || true
    wait "$heartbeat_pid" 2>/dev/null || true
  fi
  rm -f "$state_dir/controller.active.pid"
}
trap cleanup EXIT
rm -f "$state_dir/needs_attention"

heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state_dir/needs_attention"; exit 1; }

wait_training(){
  while true; do
    heartbeat
    if [[ -f "$train_dir/needs_attention" ]]; then
      fail "qualified state8 training/evaluation monitor needs attention"
    fi
    if [[ -f "$train_dir/complete" ]]; then
      return 0
    fi
    # The upstream controller owns failed-job recovery and may switch from the
    # base experiment to a recovery experiment.  Treat its fail-closed local
    # receipts as authoritative; the base job becoming terminal is diagnostic,
    # not grounds to kill this downstream watcher.
    active="$(cat "$train_active_dir/active_experiment.txt" 2>/dev/null || true)"
    if [[ -n "$active" ]]; then
      status="$(timeout 90 amlt status "$active" 2>&1)" || { sleep 120; continue; }
      printf '[%s] experiment=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        "$active" "$status" >> "$state_dir/training_status.log"
    fi
    sleep 120
  done
}

wait_quota(){
  local required="$1" quota
  while true; do
    heartbeat
    quota="$(timeout 90 amlt --json-tables ti sing -t Feeds 2>&1)" || { sleep 30; continue; }
    if "$python_bin" - "$quota" "$required" "$state_dir/quota.json" <<'PY'
import json,sys
from scripts.feeds_quota_utils import parse_json_table,select_a100_80gb
row=select_a100_80gb(parse_json_table(sys.argv[1])); n=int(sys.argv[2])
q=row['quotas_(used/limit)']; standard_limit=int(q['standard']['limit'])
available=standard_limit-int(q['standard']['used'])
user_limit=int(q['user_max']['limit']); user_available=user_limit-int(q['user_max'].get('used') or 0)
open(sys.argv[3],'w').write(json.dumps({'selected_row':row,'required':n,'available':available,
 'user_available':user_available,'scheduler_queue_allowed':True},indent=2,sort_keys=True)+'\n')
assert standard_limit>=n and user_limit>=n
PY
    then return 0; fi
    sleep 30
  done
}

preserve_passed(){
  local experiment="$1" status
  status="$(timeout 90 amlt status "$experiment" 2>&1)" || return 1
  grep -Eqi '(^|[[:space:]])pass([[:space:]]|$)' <<< "$status" || return 0
  amlt results download "$experiment" --status pass \
    --include '*.json' --include '*.jsonl' --include '*.txt' \
    --output "$state_dir/preserved_results" >> "$state_dir/controller.log" 2>&1
}

submit_jobs(){
  local experiment="$1"; shift
  local names=("$@") selectors=() output current class rc
  for name in "${names[@]}"; do selectors+=(":$name"); done
  exec 8>"$project_dir/results/feeds_submission_global.lock"; flock 8
  amlt run "$config" "${selectors[@]}" --dump -t Feeds \
    > "$state_dir/${experiment}.preflight.dump.yaml" \
    2> "$state_dir/${experiment}.preflight.stderr" \
    || { flock -u 8; return 1; }
  "$python_bin" scripts/validate_state8_promotion_eval_amlt.py \
    --config "$state_dir/${experiment}.preflight.dump.yaml" --jobs "${names[@]}" \
    >> "$state_dir/controller.log" 2>&1 || { flock -u 8; return 1; }
  current="$(amlt status --no-update-db "$experiment" 2>&1)"; rc=$?
  class="$("$python_bin" - "$rc" "$current" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]),sys.argv[2]))
PY
)" || { flock -u 8; return 1; }
  if [[ "$class" == missing ]]; then
    wait_quota "${#names[@]}"
    output="$(amlt run "$config" "${selectors[@]}" "$experiment" -t Feeds -y \
      --description "Immutable qualified-teacher state8 checkpoint variant=$promotion_variant; four disjoint full-T3 shards and three pair-seed C4 gates; failed siblings recover independently" 2>&1)"; rc=$?
    printf '[%s] submit rc=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$output" \
      > "$state_dir/${experiment}.submission.txt"
    (( rc == 0 )) || { flock -u 8; return 1; }
  elif [[ "$class" == exists ]]; then
    printf 'status=preexisting_not_resubmitted\n%s\n' "$current" \
      > "$state_dir/${experiment}.submission.txt"
  else
    flock -u 8; return 1
  fi
  flock -u 8
  printf '%s\n' "$experiment" > "$state_dir/active_experiment.txt"
  printf '%s\n' "${names[@]}" > "$state_dir/active_jobs.txt"
  log "submitted promotion evaluation experiment=$experiment jobs=${#names[@]}"
}

wait_training
experiment="$base_experiment"
wave=0
if [[ -s "$state_dir/active_experiment.txt" ]]; then
  experiment="$(<"$state_dir/active_experiment.txt")"
  [[ "$experiment" =~ recovery([0-9]+)-20260805 ]] && wave="${BASH_REMATCH[1]}"
else
  submit_jobs "$experiment" "${all_jobs[@]}" || fail "initial promotion-eval submission failed"
fi

while true; do
  heartbeat
  status="$(timeout 90 amlt status "$experiment" 2>&1)" || { sleep 120; continue; }
  printf '[%s]\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" >> "$state_dir/status.log"
  if "$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import has_forward_progress_job
raise SystemExit(0 if has_forward_progress_job(sys.argv[1]) else 1)
PY
  then
    sleep 120; continue
  fi
  expected_jobs="$(wc -l < "$state_dir/active_jobs.txt")"
  if "$python_bin" - "$status" "$expected_jobs" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1],int(sys.argv[2])) else 1)
PY
  then
    preserve_passed "$experiment" || fail "could not preserve passed promotion results"
    break
  fi
  mapfile -t paused < <("$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import paused_job_names
values=paused_job_names(sys.argv[1])
sys.stdout.write(''.join(value+'\n' for value in values))
PY
  )
  if (( ${#paused[@]} > 0 )); then
    log "requesting resume for paused promotion jobs: ${paused[*]}"
    selectors=(); for job in "${paused[@]}"; do selectors+=(":$job"); done
    timeout 90 amlt resume "$experiment" "${selectors[@]}" \
      >> "$state_dir/controller.log" 2>&1 || true
    sleep 120; continue
  fi
  preserve_passed "$experiment" || fail "could not preserve passed promotion siblings"
  mapfile -t failed < <("$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import terminal_nonpass_job_names
values=terminal_nonpass_job_names(sys.argv[1])
sys.stdout.write(''.join(value+'\n' for value in values))
PY
  )
  (( ${#failed[@]} > 0 )) || fail "terminal promotion status has no failed jobs"
  for job in "${failed[@]}"; do
    timeout 180 amlt logs view -n 1000 "$experiment" ":$job" \
      > "$state_dir/${experiment}.${job}.failure.log" 2>&1 || true
  done
  (( wave < max_recovery_waves )) || fail "promotion evaluation exhausted recovery waves"
  wave=$((wave+1)); experiment="${recovery_prefix}${wave}-20260805"
  submit_jobs "$experiment" "${failed[@]}" || fail "promotion failed-sibling recovery failed wave=$wave"
done

rm -rf "$state_dir/download"; mkdir -p "$state_dir/download"
cp -a "$state_dir/preserved_results/." "$state_dir/download/"
readarray -t predictions < <(find "$state_dir/download" -type f -path '*/eval/t3/predictions.jsonl' | sort)
readarray -t t3_metrics < <(find "$state_dir/download" -type f -path '*/eval/t3/metrics.json' | sort)
readarray -t c4_results < <(find "$state_dir/download" -type f -path '*/eval/c4/recurrent_causal_gate.json' | sort)
[[ ${#predictions[@]} -eq 4 && ${#t3_metrics[@]} -eq 4 && ${#c4_results[@]} -eq 3 ]] \
  || fail "promotion result cardinality drift predictions=${#predictions[@]} metrics=${#t3_metrics[@]} c4=${#c4_results[@]}"
adapter_sha="$("$python_bin" - "${t3_metrics[0]}" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['artifact_sha256']['adapter_model'])
PY
)"
state_sha="$("$python_bin" - "${t3_metrics[0]}" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['artifact_sha256']['recurrent_state'])
PY
)"
"$python_bin" scripts/merge_state8_promotion_eval.py \
  --predictions "${predictions[@]}" --t3-metrics "${t3_metrics[@]}" \
  --eval-jsonl "$eval_jsonl" --c4 "${c4_results[@]}" \
  --adapter-sha256 "$adapter_sha" --recurrent-state-sha256 "$state_sha" \
  --output-dir "$state_dir/merged" >> "$state_dir/controller.log" 2>&1 \
  || fail "promotion merge audit failed"

exec 11>"$state_dir/t3_semantic/judge.lock"; flock -x 11
(while true; do heartbeat; sleep 60; done) & heartbeat_pid=$!
attempt=0
while [[ ! -f "$state_dir/t3_semantic/complete" ]]; do
  heartbeat; attempt=$((attempt+1))
  if PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
      --predictions-jsonl "$state_dir/merged/t3_predictions.jsonl" \
      --eval-jsonl "$eval_jsonl" --output-dir "$state_dir/t3_semantic/full1494" \
      --judge-backend trapi --judge-workers 12 --judge-max-retries 12 \
      --judge-timeout-seconds 60 --judge-cache "$state_dir/t3_semantic/cache.jsonl" \
      --judge-audit-samples 100 --seed 42 >> "$state_dir/t3_semantic/full.log" 2>&1; then
    if "$python_bin" scripts/validate_t3_semantic_result.py \
        --metrics "$state_dir/t3_semantic/full1494/metrics.json" \
        --output "$state_dir/t3_semantic/t3_semantic_gate.json" \
        >> "$state_dir/t3_semantic/controller.log" 2>&1; then
      touch "$state_dir/t3_semantic/scientific_gate_pass"
    else
      rc=$?; (( rc == 2 )) || fail "semantic T3 validator failed rc=$rc"
      touch "$state_dir/t3_semantic/scientific_gate_failed"
    fi
    touch "$state_dir/t3_semantic/complete"
    break
  fi
  (( attempt < 30 )) || fail "semantic judge exhausted"
  sleep 120
done
kill "$heartbeat_pid" 2>/dev/null || true; wait "$heartbeat_pid" 2>/dev/null || true; heartbeat_pid=""
flock -u 11

# Stage-B produces the frozen 25-per-group T1/T2/T3/T5/T6 screen on the exact
# immutable qualified checkpoint. Reuse that checkpoint-bound evidence here.
while [[ ! -f "$train_dir/complete" ]]; do
  heartbeat
  [[ ! -f "$train_dir/needs_attention" ]] || fail "state8 small-screen monitor needs attention"
  sleep 60
done
readarray -t small_metrics < <(
  find "$train_artifact_dir" -type f -path '*/eval/formal_screen/metrics.json' | sort
)
[[ ${#small_metrics[@]} -eq 1 ]] || fail "expected one state8 formal small-screen metric"
"$python_bin" scripts/summarize_layerwise_checkpoint250_formal_screen.py \
  --root "$(dirname "${small_metrics[0]}")" --checkpoint-round 1250 \
  --output "$state_dir/formal_screen_gate.json" >> "$state_dir/controller.log" 2>&1
formal_rc=$?
(( formal_rc == 0 || formal_rc == 2 )) \
  || fail "formal small-screen validator failed operationally rc=$formal_rc"

"$python_bin" - "$state_dir/merged/c4_multiseed.json" \
  "$state_dir/t3_semantic/t3_semantic_gate.json" "$state_dir/formal_screen_gate.json" \
  "$state_dir/merged/t3_metrics.json" "$state_dir/promotion_gate.json" \
  "$promotion_stage" "$promotion_variant" <<'PY'
import json,pathlib,sys
c4=json.load(open(sys.argv[1])); t3=json.load(open(sys.argv[2]))
formal=json.load(open(sys.argv[3])); merged=json.load(open(sys.argv[4]))
checks={'C4_multiseed':c4['replication_gate_pass'],'T3_semantic':t3['pass'],
        'T1_T2_T5_T6_small_screen':formal['diagnostic_gate_pass']}
artifacts=merged['artifact_sha256']
assert set(artifacts)=={'adapter_model','recurrent_state'} and all(len(x)==64 for x in artifacts.values()),artifacts
out={'stage':sys.argv[6],'promotion_variant':sys.argv[7],'checks':checks,'pass':all(checks.values()),
     'artifact_sha256':artifacts,'C4':c4,'T3_semantic':t3,'formal_screen':formal}
path=pathlib.Path(sys.argv[5]); path.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
raise SystemExit(0 if out['pass'] else 2)
PY
rc=$?
if (( rc == 0 )); then touch "$state_dir/scientific_gate_pass"
elif (( rc == 2 )); then touch "$state_dir/scientific_gate_failed"
else fail "joint promotion gate failed operationally rc=$rc"; fi
touch "$state_dir/complete"
heartbeat
log "qualified state8 promotion evaluation completed"
