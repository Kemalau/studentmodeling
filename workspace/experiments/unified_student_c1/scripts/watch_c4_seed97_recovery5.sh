#!/usr/bin/env bash
set -euo pipefail

export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure-studentmodeling-feeds-v2

job_id=honest_helmet_g7syhtqnl4
resource_group=rg-cs-newsandfeeds-singularity
workspace=CS-NewsAndFeeds-Singularity
experiment=c4-cf1-visionpad-seed97-rerun-recovery5-20260824
job_ref=:c4-cf1-visionpad-trainseed97-rerun-recovery5
out_dir=results/c4_seed97_recovery5_monitor_20260824
mkdir -p "$out_dir"

last=""
while true; do
  now=$(date -u +%FT%TZ)
  if raw=$(az ml job show --name "$job_id" -g "$resource_group" -w "$workspace" \
      --query '{status:status,error:error}' -o json 2>&1); then
    status=$(python -c 'import json,sys; print(json.load(sys.stdin).get("status", ""))' <<<"$raw")
  else
    status=query_error
  fi
  if [[ "$status" != "$last" ]]; then
    printf '%s %s\n%s\n' "$now" "$status" "$raw" | tee -a "$out_dir/status.log"
    last=$status
  fi
  case "$status" in
    Completed|Failed|Canceled|NotResponding)
      break
      ;;
  esac
  sleep 120
done

if [[ "$status" == Completed ]]; then
  amlt status "$experiment" >"$out_dir/final_status.txt" 2>&1 || true
  amlt logs view -n 400 "$experiment" "$job_ref" >"$out_dir/final_log.txt" 2>&1 || true
fi
