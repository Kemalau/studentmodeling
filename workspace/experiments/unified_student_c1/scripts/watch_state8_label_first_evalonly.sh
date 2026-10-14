#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
state="$project/results/state8_label_first_evalonly_20260815"
config="$project/amlt/feeds_state8_label_first_evalonly_20260815.yaml"
experiment=state8-label-first-evalonly-20260815
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
mkdir -p "$state"
cd "$project" || exit 1
exec 9>"$state/watcher.lock"; flock -n 9 || exit 0
echo $$ > "$state/watcher.active.pid"
trap 'rm -f "$state/watcher.active.pid"' EXIT

heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/watcher.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/watcher.log"; }
sleep_unlocked(){ sleep "$1" 9>&-; }

while true; do
  heartbeat
  source scripts/use_studentmodeling_feeds_auth.sh
  if [[ "${STUDENTMODELING_FEEDS_AUTH_HEALTH:-unhealthy}" == unhealthy ]]; then
    log "auth unhealthy; retrying"
    sleep_unlocked 60
    continue
  fi

  status="$(timeout 90 amlt --json-tables status "$experiment" \
    --no-calculate-result-size 2>/dev/null || true)"
  if [[ -n "$status" && "$status" != '[]' ]]; then
    printf '%s\n' "$status" > "$state/latest_status.json"
    terminal="$($python_bin - "$state/latest_status.json" <<'PY'
import json,sys
rows=json.load(open(sys.argv[1]))
states=[str(r.get('status') or '').casefold() for r in rows]
if len(rows)!=2: print('active')
elif all(s in {'pass','passed'} for s in states): print('passed')
elif any(s in {'failed','killed','canceled','cancelled'} for s in states): print('failed')
else: print('active')
PY
)"
    case "$terminal" in
      passed)
        rm -rf "$state/download"
        if timeout --foreground 1200 amlt results download "$experiment" \
            --output "$state/download" >> "$state/watcher.log" 2>&1; then
          touch "$state/remote_passed"
          log "evaluation passed and results downloaded"
          exit 0
        fi
        log "result download retry"
        ;;
      failed)
        timeout 180 amlt logs view -n 1000 "$experiment" \
          > "$state/terminal.log" 2>&1 || true
        touch "$state/remote_failed"
        log "evaluation reached terminal failure"
        exit 1
        ;;
      *) ;;
    esac
    sleep_unlocked 30
    continue
  fi

  exec 8>"$project/results/feeds_submission_global.lock"; flock -x 8
  audit="$state/active_a100.json"
  if ! "$python_bin" scripts/count_active_a100_requests.py --user-limit 32 \
      --recent 500 --output "$audit" >/dev/null 2>> "$state/watcher.log"; then
    flock -u 8; sleep_unlocked 60; continue
  fi
  read -r available auditable < <("$python_bin" - "$audit" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); print(x['available_user_slots'],int(bool(x['auditable'])))
PY
  )
  if [[ "$auditable" != 1 || "$available" -lt 2 ]]; then
    flock -u 8
    log "waiting for two user A100 slots; available=$available"
    sleep_unlocked 60
    continue
  fi
  quota="$state/quota.json"
  if ! amlt --json-tables ti sing -t Feeds -a > "$quota" 2>> "$state/watcher.log"; then
    flock -u 8; sleep_unlocked 60; continue
  fi
  if ! "$python_bin" - "$quota" <<'PY'
import json,sys
rows=json.load(open(sys.argv[1]))
row=next(r for r in rows if r.get('series')=='NDAMv4' and (r.get('accelerator') or {}).get('type')=='A100' and (r.get('accelerator') or {}).get('memory_gb')==80)
q=row['quotas_(used/limit)']['standard']; assert int(q['limit'])-int(q['used'])>=2,row
PY
  then
    flock -u 8; log "waiting for two Feeds Standard A100 slots"; sleep_unlocked 60; continue
  fi
  if ! amlt run "$config" --dump -t Feeds > "$state/preflight.yaml" 2> "$state/preflight.stderr"; then
    flock -u 8; sleep_unlocked 60; continue
  fi
  output="$(amlt run "$config" "$experiment" -t Feeds -y --description \
    'Evaluation-only recovery of frozen completed 500-step State8 label-first checkpoint: query-free label readout and natural-language C4; no weight changes' 2>&1)"
  rc=$?
  printf '%s\n' "$output" > "$state/submission.log"
  flock -u 8
  if (( rc != 0 )); then
    log "submission rc=$rc; checking for ambiguous acceptance"
    sleep_unlocked 30
    continue
  fi
  log "submitted two frozen evaluation jobs"
  sleep_unlocked 30
done
