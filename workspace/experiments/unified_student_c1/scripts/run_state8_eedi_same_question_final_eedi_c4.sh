#!/usr/bin/env bash
set -euo pipefail

export VENV_ROOT="${VENV_ROOT:-/tmp/studentmodeling-state8-final-eval-venv}"
export PATH="$VENV_ROOT/bin:$PATH"
python(){ "$VENV_ROOT/bin/python" "$@"; }
export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/hf_cache HF_HUB_CACHE=/tmp/hf_cache/hub HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRANSFORMERS_VERBOSITY=error
: "${STATE8_FINAL_EVAL_RUN_DIR:?}" "${STATE8_FINAL_EVAL_PROGRESS_NAMESPACE:?}"
RUN_DIR="$STATE8_FINAL_EVAL_RUN_DIR"
LOCAL_MODEL=/tmp/Qwen3-32B
PANEL=manifests/state8_eedi_same_question_c4_final_seed20260811_offset125_n150_v1.jsonl
PANEL_RECEIPT=manifests/state8_eedi_same_question_c4_final_seed20260811_offset125_n150_v1.receipt.json
OUT="$AMLT_OUTPUT_DIR/eval/eedi_same_question_c4_final150"
mkdir -p "$HF_HUB_CACHE" "$OUT"
for path in "$RUN_DIR/final_adapter/adapter_model.safetensors" "$RUN_DIR/recurrent_state.pt" \
  "$RUN_DIR/completion.json" "$RUN_DIR/state8_final_fit_receipt.json" \
  "$RUN_DIR/canonical_publish_receipt.json" "$PANEL" "$PANEL_RECEIPT"; do test -f "$path"; done
test "$(sha256sum "$PANEL" | cut -d' ' -f1)" = 6743c2a29a1c2b576a96e64cd73a1ffccaa6b0364d04fe2e8c692ad315d97d4e
test "$(sha256sum "$PANEL_RECEIPT" | cut -d' ' -f1)" = 4536002e2dbdb2eb2ebd92d8ad227494fb5bed81baa930d4c3ad88a5e2257332
test "$(sha256sum scripts/evaluate_state8_eedi_same_question_c4.py | cut -d' ' -f1)" = 7aabe7cae743d63a566746eab5ba572b51e70e20130f3759dd006536d3543530
readarray -t ARTIFACT_SHA256 < <(python - "$RUN_DIR" <<'PY'
import hashlib,json,pathlib,sys
r=pathlib.Path(sys.argv[1]); fit=json.load(open(r/'state8_final_fit_receipt.json'))
completion=json.load(open(r/'completion.json')); publish=json.load(open(r/'canonical_publish_receipt.json'))
assert fit['selected_replay_variant'].startswith('same_question_'),fit
assert fit['one_epoch_exact'] is True and fit['real_examples']==259845,fit
assert fit['raw_history_visible_to_decoder'] is False,fit
assert fit['persistent_history_kv_cache'] is False,fit
assert fit['history_training_mode']=='strict_state_only',fit
assert fit['history_dropout_counts']=={'1.0':259845},fit
assert completion['sha256']['adapter_model']==fit['adapter_sha256'],(completion,fit)
assert completion['sha256']['recurrent_state']==fit['recurrent_state_sha256'],(completion,fit)
assert publish['artifact_sha256']=={'adapter_model':fit['adapter_sha256'],'recurrent_state':fit['recurrent_state_sha256']},publish
print(fit['adapter_sha256']); print(fit['recurrent_state_sha256'])
PY
)
python - "$PANEL_RECEIPT" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); assert x['status']=='passed' and x['twins']==150,x
assert x['offset']==125 and x['output_pairs']==300,x
assert x['qualified_question_groups']==1002,x
assert x['same_question_query_identical'] is True,x
assert x['current_item_visible_to_decoder'] is True,x
assert x['student_response_visible_to_decoder'] is False,x
assert x['raw_history_visible_to_decoder'] is False,x
PY
rm -rf "$LOCAL_MODEL"; mkdir -p "$LOCAL_MODEL"
python scripts/copy_model_parallel.py --source /mnt/amulet/studentmodeling/models/Qwen3-32B \
  --destination "$LOCAL_MODEL" --workers 16
test -f "$LOCAL_MODEL/model.safetensors.index.json"
python -u scripts/evaluate_state8_eedi_same_question_c4.py \
  --model "$LOCAL_MODEL" --adapter "$RUN_DIR/final_adapter" \
  --recurrent-state "$RUN_DIR/recurrent_state.pt" --state-architecture layerwise \
  --causal-jsonl "$PANEL" --output-dir "$OUT" --state-token-alias '<|vision_pad|>' \
  --max-length 1536 --max-new-tokens 160 --num-pairs 150 --seed 20260811 \
  --expected-adapter-sha256 "${ARTIFACT_SHA256[0]}" \
  --expected-recurrent-state-sha256 "${ARTIFACT_SHA256[1]}"
python - "$OUT/metrics.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); assert x['num_pairs']==150,x
assert x['causal_jsonl_sha256']=='6743c2a29a1c2b576a96e64cd73a1ffccaa6b0364d04fe2e8c692ad315d97d4e',x
assert x['same_question_query_identical'] is True,x
assert x['student_response_visible_to_decoder'] is False,x
assert x['raw_history_visible_to_decoder'] is False,x
assert x['persistent_history_kv_cache'] is False,x
assert x['state_shape']==[64,8,5120],x
PY
