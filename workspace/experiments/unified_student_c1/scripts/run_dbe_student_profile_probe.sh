#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "DBE_STUDENT_PROFILE_PROBE_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

export VENV_ROOT="${VENV_ROOT:-/tmp/studentmodeling-dbe-profile-venv}"
export PATH="$VENV_ROOT/bin:$PATH"
python(){ "$VENV_ROOT/bin/python" "$@"; }
export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/hf_cache HF_HUB_CACHE=/tmp/hf_cache/hub HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRANSFORMERS_VERBOSITY=error

MODEL_REMOTE=/mnt/amulet/studentmodeling/models/Qwen3-32B
MODEL_LOCAL=/tmp/Qwen3-32B
RUN_DIR=/mnt/amulet/studentmodeling/checkpoints/state8_stateonly_repair_seed8401_taskanchor_source_anchor_full_mse1_eedi_same_question_v1_early_label_high_seed8401
ADAPTER="$RUN_DIR/final_adapter"
RECURRENT_STATE="$RUN_DIR/recurrent_state.pt"
EXPECTED_ADAPTER=3bcf7bdc1d6e4de432efbd2bf78551c3d92349e44d7e2b217166ca1305cffe24
EXPECTED_STATE=6985e40ec82e100211a552847b2f4a8601fbdb5785d58fdcb0f0620df10d5762
MANIFEST=manifests/dbe_student_profile_v1

for path in \
  "$ADAPTER/adapter_model.safetensors" "$RECURRENT_STATE" \
  "$MANIFEST/manifest.json" "$MANIFEST/train.jsonl" \
  "$MANIFEST/valid.jsonl" "$MANIFEST/test.jsonl"; do
  test -f "$path"
done
test "$(sha256sum "$ADAPTER/adapter_model.safetensors" | cut -d' ' -f1)" = "$EXPECTED_ADAPTER"
test "$(sha256sum "$RECURRENT_STATE" | cut -d' ' -f1)" = "$EXPECTED_STATE"
test "$(sha256sum "$MANIFEST/train.jsonl" | cut -d' ' -f1)" = 50adec7ecf5fb02cf374fb946cc5c31b5a443b61c3a7a5c75fa0ca6a9523a7d0
test "$(sha256sum "$MANIFEST/valid.jsonl" | cut -d' ' -f1)" = 3d8145a0643464cf8c1b24b87487ffb30b5929d689b90eb7a73add95db88bb89
test "$(sha256sum "$MANIFEST/test.jsonl" | cut -d' ' -f1)" = d0316d845a6e1be2f4d24418ebdcd22adb579aed5858aa103da6427e506f5b03
test "$(sha256sum scripts/evaluate_dbe_student_profile_probe.py | cut -d' ' -f1)" = 9990027a80c1b3bdfebf401f8d761c6f8f792ddd6c3a7b27757f5a93295c259c
test "$(df --output=avail /tmp | tail -1)" -ge 83886080

python -m pytest -q tests/test_build_dbe_student_profile_pilot.py \
  tests/test_evaluate_dbe_student_profile_probe.py
rm -rf "$MODEL_LOCAL"; mkdir -p "$MODEL_LOCAL"
python scripts/copy_model_parallel.py --source "$MODEL_REMOTE" \
  --destination "$MODEL_LOCAL" --workers 16
test -f "$MODEL_LOCAL/model.safetensors.index.json"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

OUT="$AMLT_OUTPUT_DIR/profile_probe"
python -u scripts/evaluate_dbe_student_profile_probe.py \
  --model "$MODEL_LOCAL" --adapter "$ADAPTER" \
  --recurrent-state "$RECURRENT_STATE" --manifest-dir "$MANIFEST" \
  --output-dir "$OUT" --state-token-alias '<|vision_pad|>' \
  --max-length 1536 --state-batch-size 4 --projection-dim 32 --seed 20260816

python - "$OUT/metrics.json" "$EXPECTED_ADAPTER" "$EXPECTED_STATE" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['status']=='complete' and x['students']=={'test':88,'train':663,'valid':84},x
assert x['state_shape']==[64,8,5120] and x['student_disjoint'] is True,x
assert x['raw_history_visible_to_readout'] is False,x
assert x['artifact_sha256']=={'adapter_model':sys.argv[2],'recurrent_state':sys.argv[3]},x
assert set(x['views'])=={'final_layer_mean_slots','mean_layers_mean_slots','layerwise_mean_slots_rp'},x
PY
