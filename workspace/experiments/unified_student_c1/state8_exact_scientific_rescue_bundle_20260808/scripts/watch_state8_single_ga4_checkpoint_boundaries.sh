#!/usr/bin/env bash
set -euo pipefail

# The already-submitted single-A100 attempt was packaged with a 250-micro-step
# save cadence although GA=4 checkpoints do not serialize in-flight gradients.
# Preserve non-boundary receipts under a diagnostic name and remove only the
# selector-visible receipt.  Model/state files are never changed.
project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
state="$project/results/state8_strong1_widen8_bridge_20260808/single_fallback_monitor/ga4_boundary_guard"
account=csnewsandfeeds4150361735
container=amulet
experiment=state8-strong1-widen8-single-fallback-recovery3-20260808
remote_root=studentmodeling/checkpoints/state8_qualified_teacher_curriculum_seed42_exact5000_rounds1250_kd1_v2_20260805_taxonomy_semantic_v1_strong1_widen8_single_lr5e7_seed60_single_ga4
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python

if [[ -z "${AZURE_CONFIG_DIR:-}" ]]; then
  if AZURE_CONFIG_DIR=/home/v-yuejiali/.azure az account get-access-token \
      --subscription b6dc87f3-c479-49c8-8cb5-7896da3ff895 \
      --query expiresOn -o tsv >/dev/null 2>&1; then
    export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure
  else
    export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure-studentmodeling-feeds
  fi
fi

mkdir -p "$state"
cd "$project"
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"
trap 'rm -f "$state/controller.active.pid"' EXIT
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }

phase(){
  local raw
  raw="$(timeout 90 amlt --json-tables status "$experiment" --hide-urls 2>/dev/null)" || return 1
  STATUS="$raw" "$python_bin" - <<'PY'
import json,os
rows=json.loads(os.environ['STATUS'])
print(str(rows[0].get('status') or '').casefold() if len(rows)==1 else 'unknown')
PY
}

quarantine_step(){
  local step="$1" root="$remote_root/checkpoint-$1"
  local visible="$root/checkpoint_completion.json"
  local preserved="$root/checkpoint_completion.nonresumable_ga4.json"
  local source="$state/checkpoint-$step.json" verify="$state/checkpoint-$step.verify.json"
  local visible_exists preserved_exists
  visible_exists="$(az storage blob exists --auth-mode login --account-name "$account" \
    --container-name "$container" --name "$visible" --query exists -o tsv 2>/dev/null || true)"
  [[ "$visible_exists" == true ]] || return 0
  az storage blob download --auth-mode login --account-name "$account" \
    --container-name "$container" --name "$visible" --file "$source" --overwrite -o none
  "$python_bin" - "$source" "$step" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); step=int(sys.argv[2])
assert x['status']=='complete' and int(x['step'])==step,x
assert set(x['sha256'])=={'adapter_model','recurrent_state'},x
PY
  preserved_exists="$(az storage blob exists --auth-mode login --account-name "$account" \
    --container-name "$container" --name "$preserved" --query exists -o tsv 2>/dev/null || true)"
  if [[ "$preserved_exists" != true ]]; then
    az storage blob upload --auth-mode login --account-name "$account" \
      --container-name "$container" --name "$preserved" --file "$source" --overwrite -o none
  fi
  az storage blob download --auth-mode login --account-name "$account" \
    --container-name "$container" --name "$preserved" --file "$verify" --overwrite -o none
  cmp "$source" "$verify"
  az storage blob delete --auth-mode login --account-name "$account" \
    --container-name "$container" --name "$visible" -o none
  visible_exists="$(az storage blob exists --auth-mode login --account-name "$account" \
    --container-name "$container" --name "$visible" --query exists -o tsv)"
  [[ "$visible_exists" == false ]]
  log "preserved non-boundary checkpoint receipt step=$step"
}

while true; do
  heartbeat
  for step in $(seq 250 250 5000); do
    (( step % 4 == 0 )) && continue
    quarantine_step "$step" || log "retrying receipt guard step=$step"
  done
  current="$(phase || true)"
  case "$current" in
    preparing|pre-queued|queued|running|paused|resuming|'') sleep 30 ;;
    *) log "original packaged attempt terminal phase=$current"; break ;;
  esac
done
touch "$state/complete"; heartbeat
