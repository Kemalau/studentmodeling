#!/usr/bin/env bash
set -euo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
state_dir="$project_dir/results/recurrent_unified_supervisor_20260731"
subscription="b6dc87f3-c479-49c8-8cb5-7896da3ff895"
source "$project_dir/scripts/use_studentmodeling_feeds_auth.sh"
tenant="72f988bf-86f1-41af-91ab-2d7cd011db47"
account="csnewsandfeeds4150361735"
container="amulet"
probe_blob="studentmodeling/checkpoints/layerwise_unified_world_model_seed42_full259845_v1_20260731/gradient_audit.json"

mkdir -p "$state_dir"
if [[ -n "${AZURE_CONFIG_DIR:-}" ]]; then
  mkdir -p "$AZURE_CONFIG_DIR"
  chmod 700 "$AZURE_CONFIG_DIR"
fi
rm -f "$state_dir/auth_ready"
date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/auth_required"

if ! az account get-access-token --subscription "$subscription" \
    --query expiresOn -o tsv >/dev/null 2>&1; then
  printf 'Authentication required. Use browser-based az login; device-code login is prohibited.\n' >&2
  exit 75
fi
az account set --subscription "$subscription"
python - "$tenant" "$subscription" <<'PY'
import json, subprocess, sys
value = json.loads(subprocess.check_output([
    'az', 'account', 'show', '--output', 'json',
], text=True))
assert value.get('tenantId') == sys.argv[1], value
assert value.get('id') == sys.argv[2] and value.get('state') == 'Enabled', value
assert (value.get('user') or {}).get('name') == 'v-yuejiali@microsoft.com', value
PY
az account get-access-token --subscription "$subscription" \
  --query expiresOn -o tsv >/dev/null
test "$(az storage blob exists --auth-mode login --account-name "$account" \
  --container-name "$container" --name "$probe_blob" --query exists -o tsv)" = true
amlt --json-tables ti sing -t Feeds > "$state_dir/quota_after_reauth.json"
python - "$state_dir/quota_after_reauth.json" <<'PY'
import sys
from scripts.feeds_quota_utils import parse_json_table, select_a100_80gb
row = select_a100_80gb(parse_json_table(open(sys.argv[1], encoding='utf-8').read()))
assert row['series'] == 'NDAMv4', row
assert row['accelerator']['type'] == 'A100' and row['accelerator']['memory_gb'] == 80, row
PY

# The active unified-model objective uses the same dedicated credential cache.
# Restart only its local idempotent controllers so they all observe the fresh
# token; remote jobs are neither cancelled nor resubmitted here.
bash scripts/restart_state8_eedi_same_question_controllers.sh
bash scripts/restart_current_state8_and_baseline_controllers_after_auth.sh

# Every restarted controller reads the dedicated AZURE_CONFIG_DIR itself.  This
# changes only local monitoring/auth state; the live Feeds training job remains
# untouched and is never cancelled or resubmitted here.
for session in \
  layerwise_unified_final_fit \
  layerwise_unified_final_eval \
  layerwise_unified_training_progress \
  layerwise_finalfit_checkpoint1000_c4 \
  layerwise_finalfit_checkpoint1000_formal_screen; do
  tmux kill-session -t "$session" 2>/dev/null || true
done
rm -f \
  "$project_dir/results/layerwise_unified_final_pipeline_20260731/final_fit/controller.active.pid" \
  "$project_dir/results/layerwise_unified_final_pipeline_20260731/final_eval/controller.active.pid" \
  "$project_dir/results/layerwise_unified_final_pipeline_20260731/progress/controller.active.pid" \
  "$project_dir/results/layerwise_unified_final_pipeline_20260731/checkpoint1000_c4/controller.active.pid" \
  "$project_dir/results/layerwise_unified_final_pipeline_20260731/checkpoint1000_formal_screen/controller.active.pid"

rm -f "$state_dir/auth_required"
date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/auth_ready"
printf '[%s] dedicated Feeds authentication verified; local controllers will restart\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$state_dir/auth.log"
