#!/usr/bin/env bash
set -euo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
source "$project/scripts/use_studentmodeling_feeds_auth.sh"
: "${TASKONLY_PUBLIC_TASK:?TASKONLY_PUBLIC_TASK must be T1, T2, or T4}"

case "$TASKONLY_PUBLIC_TASK" in
  T1)
    old_job=qwen3-32b-taskonly-t1-correctness-oneepoch-seed42
    recovery_job=qwen3-32b-taskonly-t1-resumable-seed42
    adapter_tag=taskonly_qwen_t1_correctness_oneepoch_seed42_v1
    expected_task=predict_correctness; expected_rows=110525; expected_steps=27632 ;;
  T2)
    old_job=qwen3-32b-taskonly-t2-response-oneepoch-seed42
    recovery_job=qwen3-32b-taskonly-t2-resumable-seed42
    adapter_tag=taskonly_qwen_t2_response_oneepoch_seed42_v1
    expected_task=predict_response; expected_rows=55569; expected_steps=13893 ;;
  T4)
    old_job=qwen3-32b-taskonly-t4-prospective-oneepoch-seed42
    recovery_job=qwen3-32b-taskonly-t4-resumable-seed42
    adapter_tag=taskonly_qwen_t4_prospective_oneepoch_seed42_v1
    expected_task=predict_prospective_error; expected_rows=78600; expected_steps=19650 ;;
  *) exit 2 ;;
esac

old_experiment=taskonly-qwen-t1-t2-t4-oneepoch-20260813
config="$project/amlt/feeds_taskonly_qwen_resumable_recovery_20260814.yaml"
validator="$project/scripts/validate_taskonly_qwen_resumable_amlt.py"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
state="$project/results/taskonly_qwen_resumable_recovery_20260814/$TASKONLY_PUBLIC_TASK"
completion_blob="studentmodeling/checkpoints/$adapter_tag/completion.json"
account=csnewsandfeeds4150361735
container=amulet
poll_seconds="${TASKONLY_RESUMABLE_POLL_SECONDS:-60}"
max_recovery_waves="${TASKONLY_RESUMABLE_MAX_WAVES:-500}"

# Keep the cache selected by use_studentmodeling_feeds_auth.sh.  In particular,
# do not overwrite a healthy browser-authenticated default cache with an
# expired dedicated cache after the helper has already selected the former.
mkdir -p "$state"
cd "$project"
exec 9>"$state/controller.lock"
flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"
trap 'rm -f "$state/controller.active.pid"' EXIT

heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }

completion_exists(){
  source "$project/scripts/use_studentmodeling_feeds_auth.sh"
  [[ "${STUDENTMODELING_FEEDS_AUTH_HEALTH:-unhealthy}" != unhealthy ]] || return 1
  [[ "$(timeout 90 az storage blob exists --auth-mode login \
      --account-name "$account" --container-name "$container" \
      --name "$completion_blob" --query exists -o tsv 2>/dev/null)" == true ]]
}

validate_completion(){
  source "$project/scripts/use_studentmodeling_feeds_auth.sh"
  timeout 180 az storage blob download --auth-mode login \
    --account-name "$account" --container-name "$container" \
    --name "$completion_blob" --file "$state/completion.remote.json" \
    --overwrite > "$state/completion.download.log" 2>&1 || return 1
  "$python_bin" - "$state/completion.remote.json" "$expected_task" \
      "$expected_rows" "$expected_steps" <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
assert x['status']=='complete' and x['task']==sys.argv[2],x
assert int(x['train_rows'])==int(sys.argv[3]) and int(x['epochs'])==1,x
assert int(x['global_batch'])==4 and len(x['adapter_sha256'])==64,x
if x.get('resumable') is True:
    assert int(x['optimizer_steps'])==int(sys.argv[4]),x
PY
}

job_status(){
  local experiment="$1" job="$2" output
  output="$(timeout 90 amlt --json-tables status "$experiment" ":$job" \
    --no-calculate-result-size --hide-urls --force-status-update 2>/dev/null)" \
    || return 1
  printf '%s\n' "$output" > "$state/status.json"
  STATUS="$output" JOB="$job" "$python_bin" - <<'PY'
import json,os
rows=json.loads(os.environ['STATUS']); job=os.environ['JOB']
matches=[r for r in rows if str(r.get('job_name') or '').lstrip(':')==job]
print(str(matches[0].get('status') or '').casefold() if len(matches)==1 else 'unknown')
PY
}

wait_capacity(){
  while true; do
    heartbeat
    completion_exists && return 2
    source "$project/scripts/use_studentmodeling_feeds_auth.sh"
    [[ "${STUDENTMODELING_FEEDS_AUTH_HEALTH:-unhealthy}" != unhealthy ]] \
      || { sleep "$poll_seconds"; continue; }
    if ! timeout 90 amlt --json-tables ti sing -t Feeds -a \
        > "$state/quota.json" 2>> "$state/controller.log"; then
      sleep "$poll_seconds"; continue
    fi
    if ! "$python_bin" - "$state/quota.json" <<'PY'
import json,sys
rows=json.load(open(sys.argv[1],encoding='utf-8'))
m=[r for r in rows if r.get('target_name')=='Feeds' and r.get('series')=='NDAMv4'
   and (r.get('accelerator') or {}).get('type')=='A100'
   and int((r.get('accelerator') or {}).get('memory_gb',0))==80]
assert len(m)==1,m
q=m[0]['quotas_(used/limit)']
assert int(q['standard']['limit'])-int(q['standard']['used'])>=1,q
assert int(q['user_max']['limit'])==32,q
PY
    then sleep "$poll_seconds"; continue; fi
    if ! timeout 240 "$python_bin" scripts/count_active_a100_requests.py \
        --user-limit 32 --recent 500 --output "$state/active_a100.json" \
        > /dev/null 2>> "$state/controller.log"; then
      sleep "$poll_seconds"; continue
    fi
    available="$($python_bin - "$state/active_a100.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
print(int(x['available_user_slots']) if x.get('auditable') else 0)
PY
)"
    if (( available >= 1 )); then return 0; fi
    printf '%s\twaiting_for_user_a100_slot\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      > "$state/latest_status.tsv"
    sleep "$poll_seconds"
  done
}

submit_recovery(){
  local experiment="$1" dump output rc
  completion_exists && return 2
  wait_capacity || return $?
  exec 8>"$project/results/feeds_submission_global.lock"
  flock -x 8
  completion_exists && { flock -u 8; return 2; }
  dump="$state/$experiment.preflight.yaml"
  amlt run "$config" ":$recovery_job" --dump -t Feeds > "$dump" \
    2> "$state/$experiment.preflight.stderr" \
    || { flock -u 8; return 1; }
  "$python_bin" "$validator" --config "$dump" --jobs "$recovery_job" \
    >> "$state/controller.log" 2>&1 \
    || { flock -u 8; return 1; }
  if amlt status --no-update-db "$experiment" > "$state/$experiment.preexisting.txt" 2>&1; then
    flock -u 8; return 1
  fi
  output="$(amlt run "$config" ":$recovery_job" "$experiment" -t Feeds -y \
    --description "Resumable one-epoch same-backbone Qwen3-32B $TASKONLY_PUBLIC_TASK recovery; exact target exposure preserved; full Trainer checkpoint every 500 steps" 2>&1)"; rc=$?
  printf '[%s] rc=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$output" \
    > "$state/$experiment.submission.txt"
  flock -u 8
  (( rc == 0 )) || return 1
  printf '%s\n' "$experiment" > "$state/active_experiment.txt"
  log "submitted resumable recovery experiment=$experiment job=$recovery_job"
}

rm -f "$state/needs_attention"
if completion_exists; then
  validate_completion || fail "preexisting completion receipt invalid"
  touch "$state/complete"; log "completion already exists; no recovery needed"; exit 0
fi

original_paused_polls=0
if [[ ! -s "$state/active_experiment.txt" ]]; then
  while true; do
    heartbeat
    completion_exists && {
      validate_completion || fail "original completion receipt invalid"
      touch "$state/complete"; log "original job completed; recovery not submitted"; exit 0
    }
    status="$(job_status "$old_experiment" "$old_job")" \
      || { sleep "$poll_seconds"; continue; }
    printf '%s\toriginal_%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" \
      > "$state/latest_status.tsv"
    case "$status" in
      preparing|pre-queued|queued|running|resuming)
        original_paused_polls=0
        sleep "$poll_seconds" ;;
      paused)
        original_paused_polls=$((original_paused_polls+1))
        # A short scheduler pause may resume by itself.  A sustained pause does
        # not: hand it to the checkpoint-aware recovery path after ten polls.
        if (( original_paused_polls >= 10 )); then
          log "original job remained paused for $original_paused_polls polls; starting resumable recovery"
          break
        fi
        sleep "$poll_seconds" ;;
      pass)
        sleep "$poll_seconds" ;;
      failed|killed|cancelled|canceled|stopped)
        break ;;
      *) sleep "$poll_seconds" ;;
    esac
  done
fi

wave=1
if [[ -s "$state/active_experiment.txt" ]]; then
  experiment="$(<"$state/active_experiment.txt")"
  [[ "$experiment" =~ recovery([0-9]+)-20260814$ ]] \
    && wave="${BASH_REMATCH[1]}"
else
  experiment="taskonly-qwen-${TASKONLY_PUBLIC_TASK,,}-resumable-recovery${wave}-20260814"
  rc=0; submit_recovery "$experiment" || rc=$?
  if (( rc == 2 )); then
    validate_completion || fail "completion appeared during submit gate but is invalid"
    touch "$state/complete"; exit 0
  fi
  (( rc == 0 )) || fail "resumable submission failed"
fi

while true; do
  heartbeat
  completion_exists && {
    validate_completion || fail "recovery completion receipt invalid"
    touch "$state/complete"; log "resumable baseline complete experiment=$experiment"; exit 0
  }
  status="$(job_status "$experiment" "$recovery_job")" \
    || { sleep "$poll_seconds"; continue; }
  printf '%s\trecovery_%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" \
    > "$state/latest_status.tsv"
  case "$status" in
    preparing|pre-queued|queued|running|resuming|pass)
      recovery_paused_polls=0
      sleep "$poll_seconds" ;;
    paused)
      recovery_paused_polls=${recovery_paused_polls:-0}
      recovery_paused_polls=$((recovery_paused_polls+1))
      # Scheduler pauses can be transient, so first ask AMLT to resume the
      # same job.  A job that has exhausted max_run_duration remains paused;
      # after three confirmed polls, roll forward from the latest validated
      # shared checkpoint instead of waiting forever.
      timeout 90 amlt resume "$experiment" ":$recovery_job" \
        >> "$state/controller.log" 2>&1 || true
      if (( recovery_paused_polls < 3 )); then
        sleep "$poll_seconds"; continue
      fi
      (( wave < max_recovery_waves )) || fail "resumable recovery waves exhausted"
      wave=$((wave+1))
      experiment="taskonly-qwen-${TASKONLY_PUBLIC_TASK,,}-resumable-recovery${wave}-20260814"
      rc=0; submit_recovery "$experiment" || rc=$?
      if (( rc == 2 )); then recovery_paused_polls=0; continue; fi
      (( rc == 0 )) || fail "resumable recovery submission failed wave=$wave"
      recovery_paused_polls=0 ;;
    failed|killed|cancelled|canceled|stopped)
      timeout 180 amlt logs view -n 3000 "$experiment" ":$recovery_job" \
        > "$state/$experiment.failure.log" 2>&1 || true
      (( wave < max_recovery_waves )) || fail "resumable recovery waves exhausted"
      wave=$((wave+1))
      experiment="taskonly-qwen-${TASKONLY_PUBLIC_TASK,,}-resumable-recovery${wave}-20260814"
      rc=0; submit_recovery "$experiment" || rc=$?
      if (( rc == 2 )); then continue; fi
      (( rc == 0 )) || fail "resumable recovery submission failed wave=$wave" ;;
    *) sleep "$poll_seconds" ;;
  esac
done
