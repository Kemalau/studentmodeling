#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "STATE8_MATCHED_CHECKPOINT_T3_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

export VENV_ROOT="${VENV_ROOT:-/tmp/studentmodeling-state8-matched-checkpoint-screen-venv}"
export PATH="$VENV_ROOT/bin:$PATH"
python(){ "$VENV_ROOT/bin/python" "$@"; }
export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/hf_cache HF_HUB_CACHE=/tmp/hf_cache/hub HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRANSFORMERS_VERBOSITY=error
: "${STATE8_MATCHED_CHECKPOINT_ROOT:?}" "${STATE8_MATCHED_CHECKPOINT_STEP:?}"
: "${STATE8_MATCHED_ADAPTER_SHA256:?}" "${STATE8_MATCHED_STATE_SHA256:?}"
: "${MATCHED_CHECKPOINT_RECEIPT_SHA256:?}"

MODEL_REMOTE=/mnt/amulet/studentmodeling/models/Qwen3-32B
MODEL_LOCAL=/tmp/Qwen3-32B
RAW_EVAL=/mnt/amulet/studentmodeling/formal_evaluation_v1/test.jsonl
RECURRENT_EVAL=/tmp/state8_matched_checkpoint_eval/test.jsonl
OUT="$AMLT_OUTPUT_DIR/eval/formal_t3_75"
mkdir -p "$MODEL_LOCAL" "$(dirname "$RECURRENT_EVAL")" "$OUT" "$AMLT_OUTPUT_DIR/metrics"
test "$(sha256sum scripts/evaluate_recurrent_formal_protocols_diagnosis_first.py | cut -d' ' -f1)" = a8bc07a8fe6d2c9bfa9af6d976b53b84321cb4baa9c9fcace3188e370f7732c9
python scripts/audit_state8_matched_checkpoint.py --checkpoint-root "$STATE8_MATCHED_CHECKPOINT_ROOT" \
  --step "$STATE8_MATCHED_CHECKPOINT_STEP" --adapter-sha256 "$STATE8_MATCHED_ADAPTER_SHA256" \
  --state-sha256 "$STATE8_MATCHED_STATE_SHA256" \
  --receipt-sha256 "$MATCHED_CHECKPOINT_RECEIPT_SHA256" \
  --output "$AMLT_OUTPUT_DIR/metrics/artifact_audit.json"
python scripts/build_recurrent_evaluation_manifest.py --input-jsonl "$RAW_EVAL" \
  --output-jsonl "$RECURRENT_EVAL" --audit-json "$AMLT_OUTPUT_DIR/metrics/manifest_audit.json" \
  --max-updates 4 --seed 20260730
test "$(sha256sum "$RECURRENT_EVAL" | cut -d' ' -f1)" = a368cefb0c49cb6a305ad1a892808361fba2d44d555a0a28ea64e56babe32525
rm -rf "$MODEL_LOCAL"; mkdir -p "$MODEL_LOCAL"
python scripts/copy_model_parallel.py --source "$MODEL_REMOTE" --destination "$MODEL_LOCAL" --workers 16
python -u scripts/evaluate_recurrent_formal_protocols_diagnosis_first.py \
  --model "$MODEL_LOCAL" --adapter "$STATE8_MATCHED_CHECKPOINT_ROOT/adapter" \
  --recurrent-state "$STATE8_MATCHED_CHECKPOINT_ROOT/recurrent_state.pt" \
  --state-architecture layerwise --eval-jsonl "$RECURRENT_EVAL" --output-dir "$OUT" \
  --state-token-alias '<|vision_pad|>' --max-samples-per-group 25 --max-length 1536 \
  --max-new-tokens 160 --state-batch-size 2 --candidate-row-batch-size 2 \
  --seed "$STATE8_MATCHED_CHECKPOINT_STEP" --tasks diagnose_misconception \
  --sources eedi_mining malrule --save-unredacted-predictions --t3-natural-schema
python - "$OUT/metrics.json" "$STATE8_MATCHED_ADAPTER_SHA256" "$STATE8_MATCHED_STATE_SHA256" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); expected={'adapter_model':sys.argv[2],'recurrent_state':sys.argv[3]}
assert x['artifact_sha256']==expected and x['state_shape']==[64,8,5120],x
assert x['sampled_rows']==75 and x['unsharded_filtered_rows']==1494,x
assert x['raw_history_visible_to_decoder'] is False and x['persistent_history_kv_cache'] is False,x
assert all(v['exact_match_role']=='auxiliary_only_not_authoritative' for v in x['metrics_by_group'].values()),x
PY
