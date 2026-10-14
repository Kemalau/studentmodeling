#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "STATE8_TASKANCHOR_CONFIRM_FORMAL_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

export VENV_ROOT="${VENV_ROOT:-/tmp/studentmodeling-state8-taskanchor-confirm-venv}"
export PATH="$VENV_ROOT/bin:$PATH"
python(){ "$VENV_ROOT/bin/python" "$@"; }
export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/hf_cache HF_HUB_CACHE=/tmp/hf_cache/hub HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRANSFORMERS_VERBOSITY=error

: "${MILESTONE_STEP:?}" "${MILESTONE_ADAPTER_SHA256:?}"
: "${MILESTONE_STATE_SHA256:?}" "${MILESTONE_RECEIPT_SHA256:?}"
: "${SHARD_INDEX:?}"
test "${NUM_SHARDS:-16}" = 16
case "$MILESTONE_STEP" in 200|400|600|800|1000) ;; *) exit 2 ;; esac

CHECKPOINT="/mnt/amulet/studentmodeling/checkpoints/state8_stateonly_repair_seed2067_taskanchor_matched_cp2000_causal_taskanchor_v1/checkpoint-$MILESTONE_STEP"
MODEL_REMOTE=/mnt/amulet/studentmodeling/models/Qwen3-32B
MODEL_LOCAL=/tmp/Qwen3-32B
RAW_EVAL=/mnt/amulet/studentmodeling/formal_evaluation_v1/test.jsonl
RECURRENT_EVAL=/tmp/state8_taskanchor_confirmation_eval/test.jsonl
OUT="$AMLT_OUTPUT_DIR/eval/formal_shard$SHARD_INDEX"

mkdir -p "$MODEL_LOCAL" "$(dirname "$RECURRENT_EVAL")" "$OUT" "$AMLT_OUTPUT_DIR/metrics"
test "$(df --output=avail /tmp | tail -1)" -ge 83886080
test "$(sha256sum "$RAW_EVAL" | cut -d' ' -f1)" = bff7726953ae048bedf6dacdbb7500a54cb8de305e406f2497ef1f11c04167f8
test "$(sha256sum scripts/evaluate_recurrent_formal_protocols_diagnosis_first.py | cut -d' ' -f1)" = a8bc07a8fe6d2c9bfa9af6d976b53b84321cb4baa9c9fcace3188e370f7732c9

python scripts/audit_state8_taskanchor_milestone.py \
  --checkpoint-root "$CHECKPOINT" --step "$MILESTONE_STEP" \
  --adapter-sha256 "$MILESTONE_ADAPTER_SHA256" \
  --state-sha256 "$MILESTONE_STATE_SHA256" \
  --receipt-sha256 "$MILESTONE_RECEIPT_SHA256" \
  --output "$AMLT_OUTPUT_DIR/metrics/artifact_audit.json"
python scripts/build_recurrent_evaluation_manifest.py --input-jsonl "$RAW_EVAL" \
  --output-jsonl "$RECURRENT_EVAL" \
  --audit-json "$AMLT_OUTPUT_DIR/metrics/manifest_audit.json" \
  --max-updates 4 --seed 20260730
test "$(sha256sum "$RECURRENT_EVAL" | cut -d' ' -f1)" = a368cefb0c49cb6a305ad1a892808361fba2d44d555a0a28ea64e56babe32525

rm -rf "$MODEL_LOCAL"; mkdir -p "$MODEL_LOCAL"
python scripts/copy_model_parallel.py --source "$MODEL_REMOTE" --destination "$MODEL_LOCAL" --workers 16
test -f "$MODEL_LOCAL/model.safetensors.index.json"
python -u scripts/evaluate_recurrent_formal_protocols_diagnosis_first.py \
  --model "$MODEL_LOCAL" --adapter "$CHECKPOINT/adapter" \
  --recurrent-state "$CHECKPOINT/recurrent_state.pt" --state-architecture layerwise \
  --eval-jsonl "$RECURRENT_EVAL" --output-dir "$OUT" \
  --state-token-alias '<|vision_pad|>' --max-length 1536 --max-new-tokens 160 \
  --state-batch-size 2 --candidate-row-batch-size 2 \
  --shard-index "$SHARD_INDEX" --num-shards 16 --seed 42 \
  --save-unredacted-predictions --t3-natural-schema
python - "$OUT/metrics.json" "$SHARD_INDEX" "$MILESTONE_ADAPTER_SHA256" "$MILESTONE_STATE_SHA256" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); index=int(sys.argv[2])
expected={'adapter_model':sys.argv[3],'recurrent_state':sys.argv[4]}
assert x['state_shape']==[64,8,5120] and x['artifact_sha256']==expected,x
assert x['raw_history_visible_to_decoder'] is False,x
assert x['persistent_history_kv_cache'] is False,x
assert x['unsharded_filtered_rows']==21676,x
assert x['evaluation_shard']=={'index':index,'count':16},x
assert x['sampled_rows'] in {1354,1355},x
PY

