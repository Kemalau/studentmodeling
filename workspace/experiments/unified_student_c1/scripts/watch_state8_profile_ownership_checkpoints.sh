#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
auth=${PROFILE_OWNERSHIP_SNAPSHOT_AUTH:-/home/v-yuejiali/.azure-studentmodeling-feeds-v2}
source_root=${PROFILE_OWNERSHIP_SOURCE_ROOT:-studentmodeling/checkpoints/state8_future_profile_ownership_seed8801_v3}
snapshot_root=${PROFILE_OWNERSHIP_SNAPSHOT_ROOT:-studentmodeling/checkpoint_snapshots/state8_future_profile_ownership_seed8801_v3}
state=${PROFILE_OWNERSHIP_SNAPSHOT_STATE:-$project/results/state8_future_profile_ownership_v3_checkpoint_snapshots_20260820}
interval=${PROFILE_OWNERSHIP_SNAPSHOT_INTERVAL_SECONDS:-120}
account=csnewsandfeeds4150361735
container=amulet

mkdir -p "$state/receipts"
cd "$project" || exit 1
test -d "$auth" || {
  printf 'Azure browser-login cache does not exist: %s\n' "$auth" >&2
  exit 2
}
[[ "$interval" =~ ^[0-9]+$ ]] && (( interval >= 30 )) || {
  printf 'Snapshot interval must be an integer of at least 30 seconds\n' >&2
  exit 2
}
exec 9>"$state/watcher.lock"
flock -n 9 || exit 0
echo $$ > "$state/watcher.pid"
trap 'rm -f "$state/watcher.pid"' EXIT

snapshot_receipt() {
  local blob=$1 step=$2 receipt partial output
  receipt="$state/receipts/checkpoint-$step.json"
  partial="$receipt.partial"
  output="$state/checkpoint-$step.snapshot.json"
  if [[ -s "$output" ]]; then
    return 0
  fi
  rm -f "$partial"
  AZURE_CONFIG_DIR="$auth" timeout 90 az storage blob download \
    --account-name "$account" --container-name "$container" --auth-mode login \
    --name "$blob" --file "$partial" --overwrite --only-show-errors \
    >/dev/null || {
      rm -f "$partial"
      return 1
    }
  python - "$partial" "$step" <<'PY' || {
import json,sys
value=json.load(open(sys.argv[1]))
step=int(sys.argv[2])
assert value.get('status')=='complete',value
assert value.get('step')==step,value
assert set(value.get('sha256') or {})=={'adapter_model','recurrent_state'},value
assert all(isinstance(x,str) and len(x)==64 for x in value['sha256'].values()),value
PY
    rm -f "$partial"
    return 1
  }
  mv "$partial" "$receipt"
  AZURE_CONFIG_DIR="$auth" timeout 1800 \
    /home/v-yuejiali/.venvs/studentmodeling/bin/python \
      scripts/snapshot_state8_checkpoint_blob.py \
      --source-prefix "$source_root/checkpoint-$step" \
      --destination-prefix "$snapshot_root/checkpoint-$step" \
      --receipt "$receipt" --output "$output" \
      >> "$state/watcher.log" 2>&1
}

while true; do
  date -u +%Y-%m-%dT%H:%M:%SZ > "$state/watcher.heartbeat"
  listing="$state/receipt_blobs.tsv.partial"
  if AZURE_CONFIG_DIR="$auth" timeout 90 az storage blob list \
      --account-name "$account" --container-name "$container" --auth-mode login \
      --prefix "$source_root/checkpoint-" \
      --query "[?ends_with(name, '/checkpoint_completion.json')].name" -o tsv \
      > "$listing" 2>> "$state/watcher.log"; then
    mv "$listing" "$state/receipt_blobs.tsv"
    while IFS= read -r blob; do
      [[ "$blob" =~ /checkpoint-([0-9]+)/checkpoint_completion\.json$ ]] || continue
      step=${BASH_REMATCH[1]}
      (( step >= 100 && step <= 1000 && step % 100 == 0 )) || continue
      snapshot_receipt "$blob" "$step" || true
    done < "$state/receipt_blobs.tsv"
  else
    rm -f "$listing"
  fi
  sleep "$interval"
done
