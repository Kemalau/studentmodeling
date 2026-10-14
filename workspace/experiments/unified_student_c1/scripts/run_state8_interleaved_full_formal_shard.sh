#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "STATE8_INTERLEAVED_FULL_FORMAL_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

export VENV_ROOT="${VENV_ROOT:-/tmp/studentmodeling-state8-interleaved-full-eval-venv}"
export PATH="$VENV_ROOT/bin:$PATH"
python(){ "$VENV_ROOT/bin/python" "$@"; }
export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/hf_cache
export HF_HUB_CACHE=/tmp/hf_cache/hub
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error

: "${SHARD_INDEX:?}" "${NUM_SHARDS:?}" "${STATE8_EVAL_RUN_DIR:?}"
test "$NUM_SHARDS" = 16
[[ "$SHARD_INDEX" =~ ^([0-9]|1[0-5])$ ]]
test "$STATE8_EVAL_RUN_DIR" = /mnt/amulet/studentmodeling/checkpoints/state8_stateonly_repair_seed8401_taskanchor_source_anchor_full_mse1_eedi_same_question_v1_early_label_high_seed8401

MODEL_REMOTE=/mnt/amulet/studentmodeling/models/Qwen3-32B
MODEL_LOCAL=/tmp/Qwen3-32B
RAW_EVAL=/mnt/amulet/studentmodeling/formal_evaluation_v1/test.jsonl
RECURRENT_EVAL=/tmp/state8_interleaved_full_eval/test.jsonl
ADAPTER="$STATE8_EVAL_RUN_DIR/final_adapter"
RECURRENT_STATE="$STATE8_EVAL_RUN_DIR/recurrent_state.pt"
OUT="$AMLT_OUTPUT_DIR/eval/formal_shard$SHARD_INDEX"
EXPECTED_ADAPTER=3bcf7bdc1d6e4de432efbd2bf78551c3d92349e44d7e2b217166ca1305cffe24
EXPECTED_STATE=6985e40ec82e100211a552847b2f4a8601fbdb5785d58fdcb0f0620df10d5762

mkdir -p "$HF_HUB_CACHE" "$(dirname "$RECURRENT_EVAL")" "$OUT"
test -f "$ADAPTER/adapter_model.safetensors"
test -f "$RECURRENT_STATE"
test -f "$RAW_EVAL"
test "$(sha256sum "$ADAPTER/adapter_model.safetensors" | cut -d' ' -f1)" = "$EXPECTED_ADAPTER"
test "$(sha256sum "$RECURRENT_STATE" | cut -d' ' -f1)" = "$EXPECTED_STATE"
test "$(sha256sum "$RAW_EVAL" | cut -d' ' -f1)" = bff7726953ae048bedf6dacdbb7500a54cb8de305e406f2497ef1f11c04167f8
test "$(sha256sum scripts/evaluate_recurrent_formal_protocols_diagnosis_first_eedi_current_query.py | cut -d' ' -f1)" = 967711843054673eab67e43a8d5392ac7329a31bf5005cc1ec5be8a60507244f
test "$(sha256sum scripts/build_recurrent_evaluation_manifest.py | cut -d' ' -f1)" = d335b0b97471683eb1f317f21af6f0dd36ccca0fe6a8e495838112785ea85a8e
test "$(sha256sum scripts/copy_model_parallel.py | cut -d' ' -f1)" = 62b37ab70bc2ffc81c8314ca69215da7facdf18dba3f023d83096d1c3526ea4c
test "$(df --output=avail /tmp | tail -1)" -ge 83886080

python scripts/build_recurrent_evaluation_manifest.py \
  --input-jsonl "$RAW_EVAL" --output-jsonl "$RECURRENT_EVAL" \
  --audit-json /tmp/state8_interleaved_full_eval/audit.json \
  --max-updates 4 --seed 20260730
test "$(sha256sum "$RECURRENT_EVAL" | cut -d' ' -f1)" = a368cefb0c49cb6a305ad1a892808361fba2d44d555a0a28ea64e56babe32525

rm -rf "$MODEL_LOCAL"; mkdir -p "$MODEL_LOCAL"
python scripts/copy_model_parallel.py --source "$MODEL_REMOTE" \
  --destination "$MODEL_LOCAL" --workers 16
test -f "$MODEL_LOCAL/model.safetensors.index.json"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

python -u scripts/evaluate_recurrent_formal_protocols_diagnosis_first_eedi_current_query.py \
  --model "$MODEL_LOCAL" --adapter "$ADAPTER" \
  --recurrent-state "$RECURRENT_STATE" --state-architecture layerwise \
  --eval-jsonl "$RECURRENT_EVAL" --output-dir "$OUT" \
  --state-token-alias '<|vision_pad|>' --max-length 1536 --max-new-tokens 192 \
  --state-batch-size 2 --candidate-row-batch-size 2 --generation-batch-size 2 \
  --shard-index "$SHARD_INDEX" --num-shards "$NUM_SHARDS" --seed 8401 \
  --save-unredacted-predictions --t3-natural-schema

python - "$OUT/metrics.json" "$SHARD_INDEX" "$EXPECTED_ADAPTER" "$EXPECTED_STATE" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); index=int(sys.argv[2])
assert x['evaluation_shard']=={'index':index,'count':16},x
assert x['sampled_rows'] in {1354,1355} and x['unsharded_filtered_rows']==21676,x
assert x['state_architecture']=='layerwise' and x['state_shape']==[64,8,5120],x
assert x['raw_history_visible_to_decoder'] is False,x
assert x['persistent_history_kv_cache'] is False,x
assert x['artifact_sha256']=={'adapter_model':sys.argv[3],'recurrent_state':sys.argv[4]},x
PY
cp /tmp/state8_interleaved_full_eval/audit.json "$OUT/manifest_audit.json"

