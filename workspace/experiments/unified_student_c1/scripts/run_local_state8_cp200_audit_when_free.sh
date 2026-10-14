#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
state="$project/results/state8_future_profile_ownership_v3_checkpoint200_local"
model=/home/v-yuejiali/studentmodeling/models/Qwen3-32B
rows="$project/manifests/dbe_student_profile_v1/test.jsonl"
expected_adapter=9f7f8e82f6dd5cc80522ebfd99ac972aa45b60234251c5f69af6f94c7d8ede58
expected_state=fc07a859f74ad5addbdf6bd4fde86ae3cfbc5063eb822323eafab097012cf7ba

mkdir -p "$state"
cd "$project" || exit 1
exec 9>"$state/local_audit.lock"
flock -n 9 || exit 0
echo $$ > "$state/local_audit_supervisor.pid"
trap 'rm -f "$state/local_audit_supervisor.pid"' EXIT

stable_free_checks=0
while true; do
  date -u +%Y-%m-%dT%H:%M:%SZ > "$state/local_audit.heartbeat"
  [[ -s "$state/metrics.json" ]] && exit 0
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)"
  if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= 43008 )); then
    stable_free_checks=$((stable_free_checks + 1))
  else
    stable_free_checks=0
  fi
  if (( stable_free_checks >= 2 )); then
    break
  fi
  sleep 30
done

test "$(sha256sum "$state/adapter/adapter_model.safetensors" | cut -d' ' -f1)" = "$expected_adapter" || exit 21
test "$(sha256sum "$state/recurrent_state.pt" | cut -d' ' -f1)" = "$expected_state" || exit 22
source /home/v-yuejiali/.venvs/studentmodeling/bin/activate
python -u scripts/early_audit_state8_profile_ownership_checkpoint.py \
  --model "$model" --adapter "$state/adapter" \
  --checkpoint "$state/recurrent_state.pt" --rows "$rows" \
  --output "$state/metrics.json.partial" --state-token-alias '<|vision_pad|>' \
  --max-length 1536 --batch-size 4 --seed 20260820 \
  --permutation-replicates 2000 > "$state/local_audit.log" 2>&1 || exit 23
python - "$state/metrics.json.partial" "$expected_adapter" "$expected_state" <<'PY' || exit 24
import json,sys
x=json.load(open(sys.argv[1]))
assert x['status']=='complete' and x['checkpoint_step']==200 and x['students']==88,x
assert x['strict_state_only'] is True and x['ownership_permutation_replicates']==2000,x
assert x['artifact_sha256']=={'adapter_model':sys.argv[2],'recurrent_state':sys.argv[3]},x
PY
mv "$state/metrics.json.partial" "$state/metrics.json"
touch "$state/complete"
