#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
root="$project_dir/results/recurrent_unified_supervisor_20260731"
state_dir="$root/reauth"
subscription="b6dc87f3-c479-49c8-8cb5-7896da3ff895"
source "$project_dir/scripts/use_studentmodeling_feeds_auth.sh"
retry_seconds="${REAUTH_RETRY_SECONDS:-15}"
ready_poll_seconds="${REAUTH_READY_POLL_SECONDS:-300}"

mkdir -p "$state_dir"
cd "$project_dir"
exec 9>"$state_dir/controller.lock"
flock -n 9 || exit 0
echo $$ > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT

heartbeat() {
  date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"
}

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" \
    | tee -a "$state_dir/controller.log"
}

while true; do
  heartbeat
  if az account get-access-token --subscription "$subscription" \
      --query expiresOn -o tsv >/dev/null 2>&1; then
    rm -f "$root/auth_required"
    [[ -f "$root/auth_ready" ]] \
      || date -u +%Y-%m-%dT%H:%M:%SZ > "$root/auth_ready"
    sleep "$ready_poll_seconds"
    continue
  fi

  rm -f "$root/auth_ready"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$root/auth_required"
  log "Feeds monitoring authentication is required; browser login must be completed manually"
  # Never start or fall back to device-code authentication. A human may run
  # the normal browser-based `az login` flow in this dedicated config, after
  # which this watcher will detect the token and resume automatically.
  sleep "$retry_seconds"
done
