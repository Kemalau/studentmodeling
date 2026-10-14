#!/usr/bin/env bash
set -euo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
lock=/tmp/studentmodeling-trapi-auth-probe.lock
while true; do
  source "$project/scripts/use_studentmodeling_feeds_auth.sh"
  exec 218>"$lock"
  if flock -w 40 218; then
    if timeout 30 az account get-access-token --resource api://trapi \
        --query expiresOn -o tsv >/dev/null 2>&1; then
      flock -u 218
      exit 0
    fi
    flock -u 218
  fi
  sleep 60
done
