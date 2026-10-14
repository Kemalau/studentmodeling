#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
source "$project/scripts/use_studentmodeling_feeds_auth.sh"
: "${TASKONLY_PUBLIC_TASK:?}"

case "$TASKONLY_PUBLIC_TASK" in
  T1)
    task=predict_correctness; train_rows=110525
    job=qwen3-32b-taskonly-t1-formal-full-seed42
    adapter_tag=taskonly_qwen_t1_correctness_oneepoch_seed42_v1 ;;
  T2)
    task=predict_response; train_rows=55569
    job=qwen3-32b-taskonly-t2-formal-full-seed42
    adapter_tag=taskonly_qwen_t2_response_oneepoch_seed42_v1 ;;
  T3)
    task=diagnose_misconception; train_rows=6776
    job=qwen3-32b-taskonly-t3-formal-full-seed42
    adapter_tag=taskonly_qwen_t3_full_oneepoch_seed42_v1 ;;
  T4)
    task=predict_prospective_error; train_rows=78600
    job=qwen3-32b-taskonly-t4-formal-full-seed42
    adapter_tag=taskonly_qwen_t4_prospective_oneepoch_seed42_v1 ;;
  T5)
    task=predict_error_persistence; train_rows=8375
    job=qwen3-32b-taskonly-t5-formal-full-seed42
    adapter_tag=taskonly_qwen_t5_persistence_oneepoch_seed42_v1 ;;
  *) echo "Unknown TASKONLY_PUBLIC_TASK=$TASKONLY_PUBLIC_TASK" >&2; exit 2 ;;
esac

state="$project/results/taskonly_qwen_formal_eval_20260813/$TASKONLY_PUBLIC_TASK"
config="$project/amlt/feeds_taskonly_qwen_formal_eval_20260813.yaml"
validator="$project/scripts/validate_taskonly_qwen_formal_eval_amlt.py"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
account=csnewsandfeeds4150361735
container=amulet
completion_blob="studentmodeling/checkpoints/$adapter_tag/completion.json"
base_experiment="taskonly-qwen-${TASKONLY_PUBLIC_TASK,,}-formal-full-20260813"
max_recoveries=3

mkdir -p "$state"; cd "$project" || exit 1
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"
trap 'rm -f "$state/controller.active.pid"' EXIT
rm -f "$state/needs_attention"

heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }

while true; do
  heartbeat
  source "$project/scripts/use_studentmodeling_feeds_auth.sh"
  exists="$(timeout 60 az storage blob exists --auth-mode login \
    --account-name "$account" --container-name "$container" \
    --name "$completion_blob" --query exists -o tsv 2>/dev/null)" || {
      sleep 60; continue;
    }
  [[ "$exists" == true ]] && break
  printf '%s\twaiting_for_adapter\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$state/latest_status.tsv"
  sleep 60
done

while true; do
  heartbeat
  source "$project/scripts/use_studentmodeling_feeds_auth.sh"
  [[ "${STUDENTMODELING_FEEDS_AUTH_HEALTH:-unhealthy}" != unhealthy ]] \
    || { sleep 60; continue; }
  az storage blob download --auth-mode login --account-name "$account" \
    --container-name "$container" --name "$completion_blob" \
    --file "$state/train_completion.remote.json" --overwrite \
    > "$state/train_completion.download.log" 2>&1 && break
  log "transient adapter completion receipt download failure; retrying"
  sleep 60
done
"$python_bin" - "$state/train_completion.remote.json" "$task" "$train_rows" <<'PY' \
  || fail "adapter completion receipt invalid"
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
assert x['status']=='complete',x
assert x['task']==sys.argv[2],x
assert int(x['train_rows'])==int(sys.argv[3]),x
assert int(x['epochs'])==1 and len(x['adapter_sha256'])==64,x
PY
touch "$state/adapter_ready"; log "adapter ready task=$TASKONLY_PUBLIC_TASK tag=$adapter_tag"

wait_capacity(){
  while true; do
    heartbeat
    source "$project/scripts/use_studentmodeling_feeds_auth.sh"
    timeout 90 amlt --json-tables ti sing -t Feeds > "$state/quota.json" 2>> "$state/controller.log" \
      || { sleep 60; continue; }
    "$python_bin" - "$state/quota.json" <<'PY' || { sleep 60; continue; }
import json,sys
from scripts.feeds_quota_utils import parse_json_table,select_a100_80gb
row=select_a100_80gb(parse_json_table(open(sys.argv[1]).read()))
assert row['series']=='NDAMv4',row
assert row['accelerator']=={'type':'A100','memory_gb':80},row
q=row['quotas_(used/limit)']; assert int(q['standard']['limit'])-int(q['standard']['used'])>=1,q
assert int(q['user_max']['limit'])>=1,q
PY
    "$python_bin" scripts/count_active_a100_requests.py \
      --user-limit 32 --recent 500 --output "$state/active_a100.json" \
      > /dev/null 2>> "$state/controller.log" || { sleep 60; continue; }
    available="$($python_bin - "$state/active_a100.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); print(int(x['available_user_slots']) if x['auditable'] else 0)
PY
)"
    (( available >= 1 )) && return 0
    sleep 60
  done
}

submit(){
  local experiment="$1" output rc current class
  exec 8>"$project/results/feeds_submission_global.lock"; flock -x 8
  amlt run "$config" ":$job" --dump -t Feeds \
    > "$state/$experiment.preflight.yaml" 2> "$state/$experiment.preflight.stderr" \
    || { flock -u 8; fail "AMLT preflight dump failed"; }
  "$python_bin" "$validator" --config "$state/$experiment.preflight.yaml" --jobs "$job" \
    >> "$state/controller.log" 2>&1 \
    || { flock -u 8; fail "AMLT preflight validation failed"; }
  current="$(amlt status --no-update-db "$experiment" 2>&1)"; rc=$?
  class="$($python_bin - "$rc" "$current" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]),sys.argv[2]))
PY
)" || { flock -u 8; fail "could not classify preexisting experiment"; }
  if [[ "$class" == missing ]]; then
    wait_capacity
    output="$(amlt run "$config" ":$job" "$experiment" -t Feeds -y \
      --description "Full frozen formal evaluation for one-epoch same-backbone task-only Qwen3-32B $TASKONLY_PUBLIC_TASK; complete predictions retained and adapter hash bound" 2>&1)"; rc=$?
    printf '[%s] rc=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$output" \
      > "$state/$experiment.submission.txt"
    if (( rc != 0 )); then
      current="$(amlt status --no-update-db "$experiment" 2>&1)"; rc=$?
      class="$($python_bin - "$rc" "$current" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]),sys.argv[2]))
PY
)"
      [[ "$class" == exists ]] \
        || { flock -u 8; fail "submission returned nonzero and acceptance is unproven"; }
    fi
  elif [[ "$class" != exists ]]; then
    flock -u 8; fail "ambiguous experiment pre-state class=$class"
  fi
  flock -u 8
  printf '%s\n' "$experiment" > "$state/active_experiment.txt"
  log "formal evaluation accepted experiment=$experiment job=$job"
}

wave=0
if [[ -s "$state/active_experiment.txt" ]]; then
  experiment="$(<"$state/active_experiment.txt")"
  [[ "$experiment" =~ recovery([0-9]+)$ ]] && wave="${BASH_REMATCH[1]}"
else
  experiment="$base_experiment"; submit "$experiment"
fi

while true; do
  heartbeat
  source "$project/scripts/use_studentmodeling_feeds_auth.sh"
  status="$(timeout 90 amlt --json-tables status "$experiment" --hide-urls --force-status-update 2>/dev/null)" \
    || { sleep 60; continue; }
  [[ -n "$status" && "$status" != '[]' ]] || { sleep 30; continue; }
  printf '%s\n' "$status" > "$state/latest_status.json"
  job_status="$(STATUS="$status" JOB="$job" "$python_bin" - <<'PY'
import json,os
rows=json.loads(os.environ['STATUS']); name=os.environ['JOB']
matches=[x for x in rows if str(x.get('job_name') or '').lstrip(':')==name]
print(str(matches[0].get('status') or '').casefold() if len(matches)==1 else 'unknown')
PY
)"
  case "$job_status" in
    pass) break ;;
    paused)
      timeout 90 amlt resume "$experiment" ":$job" >> "$state/controller.log" 2>&1 || true ;;
    preparing|pre-queued|queued|running|resuming) ;;
    failed|killed|cancelled|canceled|stopped)
      timeout 180 amlt logs view -n 2000 "$experiment" ":$job" \
        > "$state/$experiment.failure.log" 2>&1 || true
      (( wave < max_recoveries )) || fail "formal evaluation exhausted recoveries"
      wave=$((wave+1)); experiment="$base_experiment-recovery$wave"; submit "$experiment" ;;
    *) ;;
  esac
  sleep 60
done

download_failures=0
while true; do
  heartbeat
  source "$project/scripts/use_studentmodeling_feeds_auth.sh"
  [[ "${STUDENTMODELING_FEEDS_AUTH_HEALTH:-unhealthy}" != unhealthy ]] \
    || { sleep 60; continue; }
  rm -rf "$state/download"; mkdir -p "$state/download"
  if timeout --foreground 600 amlt results download "$experiment" --status pass \
      --include '*.json' --include '*.jsonl' --include '*.txt' \
      --output "$state/download" </dev/null \
      >> "$state/controller.log" 2>&1; then
    break
  fi
  download_failures=$((download_failures+1))
  (( download_failures < 8 )) \
    || fail "formal result download failed after authenticated retries"
  log "transient formal result download failure attempt=$download_failures; retrying"
  sleep 60
done
mapfile -t predictions < <(find "$state/download" -type f -path "*/eval/$TASKONLY_PUBLIC_TASK/predictions.jsonl")
mapfile -t metrics < <(find "$state/download" -type f -path "*/eval/$TASKONLY_PUBLIC_TASK/metrics.json")
mapfile -t completions < <(find "$state/download" -type f -path "*/eval/$TASKONLY_PUBLIC_TASK/completion.json")
[[ ${#predictions[@]} -eq 1 && ${#metrics[@]} -eq 1 && ${#completions[@]} -eq 1 ]] \
  || fail "download cardinality drift predictions=${#predictions[@]} metrics=${#metrics[@]} completions=${#completions[@]}"
"$python_bin" - "${completions[0]}" "$TASKONLY_PUBLIC_TASK" <<'PY' \
  || fail "formal completion receipt invalid"
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
assert x['status']=='complete' and x['stage']=='taskonly_qwen_formal_eval_v1',x
assert x['public_task']==sys.argv[2],x
PY

if [[ "$TASKONLY_PUBLIC_TASK" == T3 ]]; then
  [[ -x "$judge_python" ]] || fail "TRAPI judge environment unavailable"
  mkdir -p "$state/t3_semantic"
  bash scripts/wait_for_trapi_auth.sh
  attempt=0
  while [[ ! -s "$state/t3_semantic/metrics.json" ]]; do
    heartbeat; attempt=$((attempt+1))
    PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
      --predictions-jsonl "${predictions[0]}" \
      --eval-jsonl data/processed/formal_evaluation_v1/test.jsonl \
      --output-dir "$state/t3_semantic" --judge-backend trapi \
      --trapi-deployment gpt-4o_2024-11-20 --judge-workers 12 \
      --judge-max-retries 12 --judge-timeout-seconds 60 \
      --judge-cache "$state/t3_semantic/cache.jsonl" \
      --judge-audit-samples 100 --seed 42 \
      >> "$state/t3_semantic/judge.log" 2>&1 && break
    (( attempt < 30 )) || fail "T3 TRAPI semantic judge exhausted retries"
    sleep 120
  done
  "$python_bin" - "$state/t3_semantic/metrics.json" <<'PY' \
    || fail "T3 semantic metrics invalid"
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
assert x['judge_backend']=='trapi',x
assert x['generation_rows']==1494 and x['judged_rows']==1494 and x['judge_errors']==0,x
PY
fi

touch "$state/complete"; heartbeat
log "formal task-only baseline complete task=$TASKONLY_PUBLIC_TASK experiment=$experiment"
