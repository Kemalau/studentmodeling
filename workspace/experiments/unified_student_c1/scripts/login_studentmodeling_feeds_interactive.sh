#!/usr/bin/env bash
set -euo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
source "$project/scripts/use_studentmodeling_feeds_auth.sh"
state="$project/results/recurrent_unified_supervisor_20260731"
tenant=72f988bf-86f1-41af-91ab-2d7cd011db47
subscription=b6dc87f3-c479-49c8-8cb5-7896da3ff895

mkdir -p "$state"
exec 7>"$state/device_login.lock"
flock -x 7
az login --tenant "$tenant" --use-device-code --output none
az account set --subscription "$subscription"
az account get-access-token --subscription "$subscription" \
  --query expiresOn -o tsv >/dev/null
flock -u 7

exec bash "$project/scripts/reauth_layerwise_unified_pipeline.sh"
