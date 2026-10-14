#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
state="$project/results/state8_profile_ownership_checkpoint400_local_fallback_20260820"
remote_state="$project/results/state8_profile_ownership_checkpoint400_audit_20260820"
auth=${PROFILE_OWNERSHIP_CP400_AUTH:-/home/v-yuejiali/.azure-keyflip-e123-recovery-20260819}
account=csnewsandfeeds4150361735
container=amulet
prefix=studentmodeling/checkpoint_snapshots/state8_future_profile_ownership_seed8801_v3/checkpoint-400
checkpoint="$state/checkpoint-400"
model=/home/v-yuejiali/studentmodeling/models/Qwen3-32B
rows="$project/manifests/dbe_student_profile_v1/test.jsonl"
expected_adapter=e04320134b514946048fa621109a8ae41ebcb4715a985871da1c39c1c3e40d3e
expected_state=6416a45dfd00ba3afc5a6c9cd1546177fea5b3b3aa44cfba6ab2610fbfd79ef5
expected_rows=d0316d845a6e1be2f4d24418ebdcd22adb579aed5858aa103da6427e506f5b03

mkdir -p "$state"
cd "$project" || exit 1
exec 9>"$state/controller.lock"
flock -n 9 || exit 0
echo $$ > "$state/controller.pid"
trap 'rm -f "$state/controller.pid"' EXIT

while true; do
  date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"
  if [[ -s "$remote_state/metrics.json" || -s "$state/metrics.json" ]]; then
    printf 'skipped: an authoritative cp400 metric already exists\n' > "$state/skipped.txt"
    exit 0
  fi
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)"
  if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= 73728 )); then
    break
  fi
  sleep 60
done

test "$(df --output=avail "$state" | tail -1)" -ge 7340032 || exit 20
rm -rf "$checkpoint.partial"
mkdir -p "$checkpoint.partial/adapter"
export AZURE_CONFIG_DIR="$auth"
for pair in \
  adapter/adapter_config.json:adapter/adapter_config.json \
  adapter/adapter_model.safetensors:adapter/adapter_model.safetensors \
  recurrent_state.pt:recurrent_state.pt \
  checkpoint_completion.json:checkpoint_completion.json; do
  remote=${pair%%:*}; local=${pair#*:}
  az storage blob download --account-name "$account" --container-name "$container" \
    --auth-mode login --name "$prefix/$remote" --file "$checkpoint.partial/$local" \
    --overwrite --only-show-errors >/dev/null || exit 21
done
test "$(sha256sum "$checkpoint.partial/adapter/adapter_model.safetensors" | cut -d' ' -f1)" = "$expected_adapter" || exit 22
test "$(sha256sum "$checkpoint.partial/recurrent_state.pt" | cut -d' ' -f1)" = "$expected_state" || exit 23
test "$(sha256sum "$rows" | cut -d' ' -f1)" = "$expected_rows" || exit 24
mv "$checkpoint.partial" "$checkpoint"

source /home/v-yuejiali/.venvs/studentmodeling/bin/activate
python -u scripts/early_audit_state8_profile_ownership_checkpoint.py \
  --model "$model" --adapter "$checkpoint/adapter" \
  --checkpoint "$checkpoint/recurrent_state.pt" --rows "$rows" \
  --output "$state/metrics.json.partial" --state-token-alias '<|vision_pad|>' \
  --max-length 1536 --batch-size 4 --seed 20260820 \
  --permutation-replicates 2000 > "$state/audit.log" 2>&1 || exit 25
python - "$state/metrics.json.partial" "$expected_adapter" "$expected_state" <<'PY' || exit 26
import json,sys
x=json.load(open(sys.argv[1]))
assert x['status']=='complete' and x['checkpoint_step']==400 and x['students']==88,x
assert x['strict_state_only'] is True and x['ownership_permutation_replicates']==2000,x
assert x['artifact_sha256']=={'adapter_model':sys.argv[2],'recurrent_state':sys.argv[3]},x
PY
mv "$state/metrics.json.partial" "$state/metrics.json"
touch "$state/complete"
