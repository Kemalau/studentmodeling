#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "STATE8_PROFILE_OWNERSHIP_CHECKPOINT500_AUDIT_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

export VENV_ROOT="${VENV_ROOT:-/tmp/studentmodeling-state8-profile-ownership-audit-venv}"
export PATH="$VENV_ROOT/bin:$PATH"
python(){ "$VENV_ROOT/bin/python" "$@"; }
export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/hf_cache HF_HUB_CACHE=/tmp/hf_cache/hub HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRANSFORMERS_VERBOSITY=error

MODEL=/mnt/amulet/studentmodeling/models/Qwen3-32B
CHECKPOINT_ROOT=/mnt/amulet/studentmodeling/checkpoint_snapshots/state8_future_profile_ownership_seed8801_v3/checkpoint-500
ADAPTER="$CHECKPOINT_ROOT/adapter"
STATE="$CHECKPOINT_ROOT/recurrent_state.pt"
RECEIPT="$CHECKPOINT_ROOT/checkpoint_completion.json"
ROWS=manifests/dbe_student_profile_v1/test.jsonl
EXPECTED_ADAPTER=cf86cd848336be02d188166f5df0d266f52f6ca3165fd57d97b98ad0b1fb1cb4
EXPECTED_STATE=898d90194938b7bc42b109e95ec578a3998bf6ec059bb5b0b82c2a1d35a1e8ef
EXPECTED_ROWS=d0316d845a6e1be2f4d24418ebdcd22adb579aed5858aa103da6427e506f5b03
OUTPUT="$AMLT_OUTPUT_DIR/checkpoint_early_audit/metrics.json"
STABLE=/mnt/amulet/studentmodeling/diagnostics/state8_future_profile_ownership_seed8801_v3/checkpoint-500-early-audit

for path in "$MODEL/model.safetensors.index.json" \
  "$ADAPTER/adapter_config.json" "$ADAPTER/adapter_model.safetensors" \
  "$STATE" "$RECEIPT" "$ROWS"; do
  test -f "$path"
done
test "$(sha256sum "$ADAPTER/adapter_model.safetensors" | cut -d' ' -f1)" = "$EXPECTED_ADAPTER"
test "$(sha256sum "$STATE" | cut -d' ' -f1)" = "$EXPECTED_STATE"
test "$(sha256sum "$ROWS" | cut -d' ' -f1)" = "$EXPECTED_ROWS"
python - "$RECEIPT" "$EXPECTED_ADAPTER" "$EXPECTED_STATE" <<'PY'
import json,sys
receipt=json.load(open(sys.argv[1],encoding='utf-8'))
assert receipt=={
    'sha256':{'adapter_model':sys.argv[2],'recurrent_state':sys.argv[3]},
    'status':'complete','step':500,
},receipt
PY

python -m pytest -q \
  tests/test_early_audit_state8_profile_ownership_checkpoint.py \
  tests/test_evaluate_state8_reusable_interface.py
mkdir -p "$(dirname "$OUTPUT")" "$STABLE"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

python -u scripts/early_audit_state8_profile_ownership_checkpoint.py \
  --model "$MODEL" --adapter "$ADAPTER" --checkpoint "$STATE" \
  --rows "$ROWS" --output "$OUTPUT" --state-token-alias '<|vision_pad|>' \
  --max-length 1536 --batch-size 4 --seed 20260820 \
  --permutation-replicates 2000

python - "$OUTPUT" "$EXPECTED_ADAPTER" "$EXPECTED_STATE" <<'PY'
import json,sys
value=json.load(open(sys.argv[1],encoding='utf-8'))
assert value['stage']=='state8_profile_ownership_checkpoint_early_audit_v1',value
assert value['status']=='complete' and value['diagnostic_only'] is True,value
assert value['checkpoint_step']==500 and value['students']==88,value
assert value['strict_state_only'] is True,value
assert value['artifact_sha256']=={
    'adapter_model':sys.argv[2],'recurrent_state':sys.argv[3]
},value
assert value['ownership_permutation_replicates']==2000,value
PY
cp "$OUTPUT" "$STABLE/metrics.json.partial"
mv "$STABLE/metrics.json.partial" "$STABLE/metrics.json"
