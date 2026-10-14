#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
root="$project_dir/results/recurrent_unified_supervisor_20260731"
state_dir="$root/reauth"
if [[ -n "${AZURE_CONFIG_DIR:-}" ]]; then
  export AZURE_CONFIG_DIR
elif az account get-access-token \
    --subscription b6dc87f3-c479-49c8-8cb5-7896da3ff895 \
    --query expiresOn -o tsv >/dev/null 2>&1; then
  unset AZURE_CONFIG_DIR
else
  export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure-studentmodeling-feeds
fi
subscription="b6dc87f3-c479-49c8-8cb5-7896da3ff895"
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
  log "Feeds monitoring authentication is required; starting a bounded device-code attempt"
  # Do not wrap the helper in a long-lived `tee` pipeline: if a device-code
  # attempt is terminated, a surviving pipe process can keep this owner stuck
  # forever.  The tmux controller log still records status; detailed Azure
  # output is append-only in auth.log.
  if bash scripts/reauth_layerwise_unified_pipeline.sh \
      >> "$root/auth.log" 2>&1; then
    log "Feeds monitoring authentication and fixed-environment checks passed"
  else
    rc=$?
    log "device-code attempt ended rc=$rc; a fresh code will be requested"
    sleep "$retry_seconds"
  fi
done
