#!/usr/bin/env bash
set -euo pipefail

cd /home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
: "${WATCH_EXPERIMENT:?}" "${WATCH_JOB_ID:?}" "${WATCH_STATE_DIR:?}"
mkdir -p "$WATCH_STATE_DIR"
exec 9>"$WATCH_STATE_DIR/watcher.lock"
flock -n 9 || exit 0
echo $$ > "$WATCH_STATE_DIR/watcher.active.pid"
trap 'rm -f "$WATCH_STATE_DIR/watcher.active.pid"' EXIT

while true; do
  source scripts/use_studentmodeling_feeds_auth.sh
  if ! status="$(timeout 60 az ml job show \
      --name "$WATCH_JOB_ID" \
      -g rg-cs-newsandfeeds-singularity \
      -w CS-NewsAndFeeds-Singularity \
      --query status -o tsv)"; then
    tmp="$WATCH_STATE_DIR/latest_status.tsv.partial.${BASHPID}"
    printf '%s\tAUTH_OR_STATUS_RETRY\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$tmp"
    mv "$tmp" "$WATCH_STATE_DIR/latest_status.tsv"
    sleep 30
    continue
  fi
  tmp="$WATCH_STATE_DIR/latest_status.tsv.partial.${BASHPID}"
  printf '%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" > "$tmp"
  mv "$tmp" "$WATCH_STATE_DIR/latest_status.tsv"

  case "$status" in
    Completed)
      while true; do
        source scripts/use_studentmodeling_feeds_auth.sh
        if [[ "${STUDENTMODELING_FEEDS_AUTH_HEALTH:-unhealthy}" == unhealthy ]]; then
          sleep 30
          continue
        fi
        rm -rf "$WATCH_STATE_DIR/download"
        if timeout --foreground 600 amlt results download "$WATCH_EXPERIMENT" --output "$WATCH_STATE_DIR/download"; then
          touch "$WATCH_STATE_DIR/remote_passed"
          exit 0
        fi
        sleep 30
      done
      ;;
    Failed|Canceled)
      timeout 180 amlt logs view -n 400 "$WATCH_EXPERIMENT" > "$WATCH_STATE_DIR/terminal.log" 2>&1 || true
      touch "$WATCH_STATE_DIR/remote_failed"
      exit 1
      ;;
  esac
  sleep 30
done
