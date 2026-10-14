#!/usr/bin/env bash
set -euo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
state="$project/results/state8_profile_ownership_lrfix_v4_smoke_local_fallback_20260821"
remote_prefix=studentmodeling/checkpoints/state8_profile_ownership_lrfix_v4_sparsekc_smoke_r2_seed8801/checkpoint-200
account=csnewsandfeeds4150361735
container=amulet
auth=${PROFILE_OWNERSHIP_LRFIX_V4_AUTH:-/home/v-yuejiali/.azure-studentmodeling-feeds-v2}
model=/home/v-yuejiali/studentmodeling/models/Qwen3-32B
rows="$project/manifests/dbe_student_profile_v1/test.jsonl"
expected_adapter=f23b0dc3a4c840ff3e0ef9192dd1d33700950328d7f3beb992cff4d3687ffd43
expected_state=24c4832dc61cbf53deb2d9ad750ec1edc18226522fdfe35f5d584ca9ff88cd78
expected_rows=d0316d845a6e1be2f4d24418ebdcd22adb579aed5858aa103da6427e506f5b03
checkpoint="$state/checkpoint-200"

mkdir -p "$state"
cd "$project"
exec 9>"$state/controller.lock"
flock -n 9 || exit 0
echo $$ > "$state/controller.pid"
trap 'rm -f "$state/controller.pid"' EXIT
date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"

free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)"
test "$free_mib" -ge 73728
test "$(df --output=avail "$state" | tail -1)" -ge 7340032
test "$(sha256sum "$rows" | cut -d' ' -f1)" = "$expected_rows"

if [[ ! -f "$checkpoint/download_complete" ]]; then
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
      --auth-mode login --name "$remote_prefix/$remote" \
      --file "$checkpoint.partial/$local" --overwrite --only-show-errors \
      >> "$state/download.log" 2>&1
  done
  test "$(sha256sum "$checkpoint.partial/adapter/adapter_model.safetensors" | cut -d' ' -f1)" = "$expected_adapter"
  test "$(sha256sum "$checkpoint.partial/recurrent_state.pt" | cut -d' ' -f1)" = "$expected_state"
  python - "$checkpoint.partial/checkpoint_completion.json" "$expected_adapter" "$expected_state" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x=={'sha256':{'adapter_model':sys.argv[2],'recurrent_state':sys.argv[3]},'status':'complete','step':200},x
PY
  touch "$checkpoint.partial/download_complete"
  mv "$checkpoint.partial" "$checkpoint"
fi

source /home/v-yuejiali/.venvs/studentmodeling/bin/activate
python -u scripts/early_audit_state8_profile_ownership_checkpoint.py \
  --model "$model" --adapter "$checkpoint/adapter" \
  --checkpoint "$checkpoint/recurrent_state.pt" --rows "$rows" \
  --output "$state/metrics.json.partial" --state-token-alias '<|vision_pad|>' \
  --max-length 1536 --batch-size 4 --seed 20260820 \
  --permutation-replicates 2000 > "$state/audit.log" 2>&1
python - "$state/metrics.json.partial" "$expected_adapter" "$expected_state" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['status']=='complete' and x['checkpoint_step']==200 and x['students']==88,x
assert x['supported_student_kc_cells']==634 and x['strict_state_only'] is True,x
assert x['profile_ownership_feature_normalization']=='layernorm',x
assert x['ownership_permutation_replicates']==2000,x
assert x['artifact_sha256']=={'adapter_model':sys.argv[2],'recurrent_state':sys.argv[3]},x
PY
mv "$state/metrics.json.partial" "$state/metrics.json"
date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"
touch "$state/complete"
