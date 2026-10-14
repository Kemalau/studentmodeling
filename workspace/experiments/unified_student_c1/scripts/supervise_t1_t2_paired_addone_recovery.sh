#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
state="$project/results/t1_t2_paired_addone_recovery_20260901"
config="$project/amlt/feeds_t1_t2_paired_addone_seed42_20260826.yaml"
job=qwen3-32b-t1-t2-paired-addone-recovery-seed42
target="${T1_T2_RECOVERY_TARGET:-Feeds}"
account=csnewsandfeeds4150361735
container=amulet
completion_blob=studentmodeling/checkpoints/qwen3_32b_t1_t2_paired_addone_oneepoch_seed42_v1/completion.json
poll_seconds="${T1_T2_RECOVERY_POLL_SECONDS:-60}"
mkdir -p "$state"
cd "$project" || exit 1
exec 9>"$state/supervisor.lock"
flock -n 9 || exit 0
echo $$ > "$state/supervisor.pid"
trap 'rm -f "$state/supervisor.pid"' EXIT

heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$state/supervisor.log"; }

completion_exists(){
  [[ "$(timeout 90 az storage blob exists --auth-mode login --account-name "$account" \
    --container-name "$container" --name "$completion_blob" --query exists -o tsv 2>/dev/null)" == true ]]
}

job_status(){
  local experiment="$1" output
  output="$(timeout 90 amlt --json-tables status "$experiment" ":$job" \
    --no-calculate-result-size --force-status-update 2>/dev/null)" || return 1
  printf '%s\n' "$output" > "$state/latest_status.json"
  STATUS="$output" python - <<'PY'
import json,os
x=json.loads(os.environ['STATUS'])
print(str(x[0].get('status') or '').casefold() if len(x)==1 else 'unknown')
PY
}

capacity_available(){
  local quota_tmp active_tmp
  quota_tmp="$(mktemp "$state/quota.json.partial.XXXXXX")"
  active_tmp="$(mktemp "$state/active_a100.json.partial.XXXXXX")"
  timeout 90 amlt --json-tables ti sing -t "$target" -a > "$quota_tmp" 2>/dev/null \
    || { rm -f "$quota_tmp" "$active_tmp"; return 1; }
  user_cap="$(python - "$quota_tmp" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
m=[r for r in x if r.get('target_name') in ('Feeds','Feeds-LLM') and r.get('series')=='NDAMv4'
   and (r.get('accelerator') or {}).get('type')=='A100'
   and int((r.get('accelerator') or {}).get('memory_gb',0))==80]
assert len(m)==1,m
q=m[0]['quotas_(used/limit)']
assert int(q['standard']['used']) < int(q['standard']['limit']),q
cap=q.get('user_max')
print('' if cap is None else int(cap['limit']))
PY
  )" || { rm -f "$quota_tmp" "$active_tmp"; return 1; }
  if [[ -n "$user_cap" ]]; then
    /home/v-yuejiali/.venvs/studentmodeling/bin/python scripts/count_active_a100_requests.py \
      --user-limit "$user_cap" --recent 500 --output "$active_tmp" >/dev/null 2>&1 \
      || { rm -f "$quota_tmp" "$active_tmp"; return 1; }
    python - "$active_tmp" <<'PY' || { rm -f "$quota_tmp" "$active_tmp"; return 1; }
import json,sys
x=json.load(open(sys.argv[1]))
assert x.get('auditable') and int(x['available_user_slots']) >= 1,x
PY
  else
    # Feeds-LLM currently exposes no per-user cap.  Do not apply Feeds' 32-GPU
    # cap across unrelated targets; record that this decision used target-pool
    # headroom and that immediate scheduling is still not guaranteed.
    python - "$quota_tmp" "$active_tmp" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
r=[v for v in x if v.get('target_name')=='Feeds-LLM' and v.get('series')=='NDAMv4'][0]
q=r['quotas_(used/limit)']['standard']
json.dump({'stage':'target_capacity_audit_v1','target':'Feeds-LLM','sku':'80G1-A100',
           'sla':'Standard','used':int(q['used']),'limit':int(q['limit']),
           'available_pool_slots':int(q['limit'])-int(q['used']),
           'user_max':None,'auditable':True,
           'note':'pool headroom is not a scheduling guarantee'},
          open(sys.argv[2],'w'),indent=2,sort_keys=True)
PY
  fi
  mv "$quota_tmp" "$state/quota.json"
  mv "$active_tmp" "$state/active_a100.json"
}

submit_wave(){
  local wave="$1" experiment="t1-t2-paired-addone-recovery${wave}-20260901"
  exec 8>"$project/results/feeds_submission_global.lock"
  flock -x 8
  if ! capacity_available; then flock -u 8; return 2; fi
  if amlt status --no-update-db "$experiment" >/dev/null 2>&1; then flock -u 8; return 1; fi
  if ! amlt run -y "$config" ":$job" "$experiment" -t "$target" \
    --description "T1 plus interaction-matched T2 add-one recovery wave $wave; resume latest complete Trainer checkpoint and run formal T1 evaluation" \
    > "$state/$experiment.submission.log" 2>&1; then flock -u 8; return 1; fi
  flock -u 8
  printf '%s\n' "$experiment" > "$state/active_experiment.txt"
  log "submitted $experiment"
}

wave=1
[[ -s "$state/wave.txt" ]] && wave="$(<"$state/wave.txt")"
while true; do
  heartbeat
  source scripts/use_studentmodeling_feeds_auth.sh
  if [[ "${STUDENTMODELING_FEEDS_AUTH_HEALTH:-unhealthy}" == unhealthy ]]; then sleep "$poll_seconds"; continue; fi
  if [[ -s "$state/active_experiment.txt" ]]; then
    experiment="$(<"$state/active_experiment.txt")"
    status="$(job_status "$experiment")" || { sleep "$poll_seconds"; continue; }
    printf '%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$experiment" "$status" > "$state/latest_status.tsv"
    case "$status" in
      preparing|pre-queued|queued|running|resuming) sleep "$poll_seconds"; continue ;;
      pass)
        rm -rf "$state/download"; mkdir -p "$state/download"
        timeout --foreground 900 amlt results download "$experiment" ":$job" \
          --include '*.json' --include '*.jsonl' --output "$state/download" </dev/null \
          > "$state/download.log" 2>&1 || { touch "$state/needs_attention"; exit 1; }
        python scripts/finalize_t1_t2_paired_addone.py --download "$state/download" --state "$state"
        touch "$state/complete"; log "complete $experiment"; exit 0 ;;
      paused)
        timeout 90 amlt cancel "$experiment" ":$job" -y >> "$state/supervisor.log" 2>&1 || true ;;
      failed|killed|cancelled|canceled|stopped)
        timeout 180 amlt logs view -n 4000 "$experiment" ":$job" > "$state/$experiment.failure.log" 2>&1 || true ;;
      *) sleep "$poll_seconds"; continue ;;
    esac
    wave=$((wave+1)); printf '%s\n' "$wave" > "$state/wave.txt"; rm -f "$state/active_experiment.txt"
  fi
  if capacity_available; then
    submit_wave "$wave" || true
  else
    printf '%s\twaiting_capacity\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$state/latest_status.tsv"
  fi
  sleep "$poll_seconds"
done
