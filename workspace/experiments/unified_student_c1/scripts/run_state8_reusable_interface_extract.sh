#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "STATE8_REUSABLE_INTERFACE_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

export VENV_ROOT="${VENV_ROOT:-/tmp/studentmodeling-state8-reusable-interface-venv}"
export PATH="$VENV_ROOT/bin:$PATH"
python(){ "$VENV_ROOT/bin/python" "$@"; }
export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/hf_cache HF_HUB_CACHE=/tmp/hf_cache/hub HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRANSFORMERS_VERBOSITY=error

test "$(sha256sum scripts/evaluate_state8_reusable_interface.py | cut -d' ' -f1)" = d74c6859912e00b0d2ce116c557206e3e8c000a0e890cbc1eb6f7e6ae35d78aa
test "$(sha256sum scripts/evaluate_dbe_student_profile_transport_v3.py | cut -d' ' -f1)" = 2406e41ba64273cb186d1e0af3d41c767bb931f1d631bf30caaaf7792010bbda

SPLIT="${STATE_INTERFACE_SPLIT:?STATE_INTERFACE_SPLIT is required}"
case "$SPLIT" in train|valid|test) ;; *) echo "invalid split: $SPLIT" >&2; exit 2;; esac

MODEL_REMOTE=/mnt/amulet/studentmodeling/models/Qwen3-32B
MODEL_LOCAL=/tmp/Qwen3-32B
RUN_DIR=/mnt/amulet/studentmodeling/checkpoints/state8_student_profile_readout_seed8601_v1
ADAPTER="$RUN_DIR/final_adapter"
STATE="$RUN_DIR/recurrent_state.pt"
EXPECTED_ADAPTER=11e94d1e3b8d9ab14a791e675801b83f0feb745e3a35e840465e0d8424fbaff0
EXPECTED_STATE=8be7045388065c784c02705339ad6c3eb33bc9f42a03aba7db999a8d70a991c0
MANIFEST=manifests/dbe_student_profile_v1
REMOTE_FEATURE_DIR=/mnt/amulet/studentmodeling/diagnostics/state8_reusable_interface_seed8601_v1/features
OUTPUT="$AMLT_OUTPUT_DIR/$SPLIT.npz"

for path in "$MODEL_REMOTE/model.safetensors.index.json" \
  "$ADAPTER/adapter_model.safetensors" "$STATE" "$MANIFEST/$SPLIT.jsonl"; do
  test -f "$path"
done
test "$(sha256sum "$ADAPTER/adapter_model.safetensors" | cut -d' ' -f1)" = "$EXPECTED_ADAPTER"
test "$(sha256sum "$STATE" | cut -d' ' -f1)" = "$EXPECTED_STATE"

python -m pytest -q tests/test_evaluate_state8_reusable_interface.py
rm -rf "$MODEL_LOCAL"
mkdir -p "$MODEL_LOCAL" "$REMOTE_FEATURE_DIR"
python scripts/copy_model_parallel.py --source "$MODEL_REMOTE" \
  --destination "$MODEL_LOCAL" --workers 16
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

python -u scripts/evaluate_state8_reusable_interface.py extract \
  --model "$MODEL_LOCAL" --adapter "$ADAPTER" --recurrent-state "$STATE" \
  --rows-jsonl "$MANIFEST/$SPLIT.jsonl" --split "$SPLIT" --output "$OUTPUT" \
  --state-token-alias '<|vision_pad|>' --max-length 1536 --batch-size 4 \
  --projection-dim 64 --projection-seed 20260818

test -s "$OUTPUT"
test -s "${OUTPUT%.npz}.json"
cp "$OUTPUT" "$REMOTE_FEATURE_DIR/$SPLIT.npz"
cp "${OUTPUT%.npz}.json" "$REMOTE_FEATURE_DIR/$SPLIT.json"
python - "$REMOTE_FEATURE_DIR/$SPLIT.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['status']=='complete' and x['state_token_count']==8,x
assert x['raw_history_visible_to_readout'] is False,x
assert x['student_id_visible_to_readout'] is False,x
if x['split']=='test': assert x['odd_even_views']==['even','odd'],x
PY
