#!/usr/bin/env bash
set -euo pipefail

cd /home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1

experiment="${STATE8_INTERLEAVED_EXPERIMENT:-state8-interleaved-retention-recovery1-20260813}"
job_id="${STATE8_INTERLEAVED_JOB_ID:-blue_nut_db3kvbymhg}"
state_dir="${STATE8_INTERLEAVED_STATE_DIR:-results/state8_interleaved_retention_recovery1_20260813}"
mkdir -p "$state_dir"

while true; do
  # Tokens in either Azure CLI cache can expire during multi-hour jobs.  Pick
  # the healthy student-modeling cache on every poll instead of pinning the
  # watcher to whichever cache happened to work when it started.
  source scripts/use_studentmodeling_feeds_auth.sh
  if ! status="$(timeout 60 az ml job show \
    --name "$job_id" \
    -g rg-cs-newsandfeeds-singularity \
    -w CS-NewsAndFeeds-Singularity \
    --query status -o tsv)"; then
    printf '%s\tAUTH_OR_STATUS_RETRY\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      > "$state_dir/latest_status.tsv.partial"
    mv "$state_dir/latest_status.tsv.partial" "$state_dir/latest_status.tsv"
    sleep 30
    continue
  fi
  printf '%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" \
    > "$state_dir/latest_status.tsv.partial"
  mv "$state_dir/latest_status.tsv.partial" "$state_dir/latest_status.tsv"

  case "$status" in
    Completed)
      timeout 600 amlt results download "$experiment" --output "$state_dir/download"
      touch "$state_dir/remote_passed"
      exit 0
      ;;
    Failed|Canceled)
      timeout 180 amlt logs view -n 400 "$experiment" > "$state_dir/terminal.log" 2>&1 || true
      touch "$state_dir/remote_failed"
      exit 1
      ;;
  esac
  sleep 30
done
