#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "STATE8_EEDI_SAME_QUESTION_CHECKPOINT_SCREEN_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

export VENV_ROOT="${VENV_ROOT:-/tmp/studentmodeling-state8-eedi-same-question-screen-venv}"
export PATH="$VENV_ROOT/bin:$PATH"; python(){ "$VENV_ROOT/bin/python" "$@"; }
export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/hf_cache HF_HUB_CACHE=/tmp/hf_cache/hub HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRANSFORMERS_VERBOSITY=error
: "${CHECKPOINT_STEP:?}" "${CHECKPOINT_ARM:?}" "${CHECKPOINT_ROOT:?}" \
  "${CHECKPOINT_ADAPTER_SHA256:?}" "${CHECKPOINT_STATE_SHA256:?}" \
  "${CHECKPOINT_RECEIPT_SHA256:?}"
[[ "$CHECKPOINT_STEP" == 20 || "$CHECKPOINT_STEP" == 100 ]] || exit 2
[[ "$CHECKPOINT_ARM" == eedi_same_question_updater_mse1 \
   || "$CHECKPOINT_ARM" == eedi_same_question_full_mse1 ]] || exit 2

MODEL_REMOTE=/mnt/amulet/studentmodeling/models/Qwen3-32B
MODEL_LOCAL=/tmp/Qwen3-32B
RAW_EVAL=/mnt/amulet/studentmodeling/formal_evaluation_v1/test.jsonl
RECURRENT_EVAL=/tmp/state8_eedi_same_question_checkpoint_eval/test.jsonl
C4_DATA=/mnt/amulet/studentmodeling/c4_fullopen_20260724/malrule_c1_clean/test_same_template.jsonl
PARENT_PAIRS=manifests/c4_stage_a_gate_pairs_seed42_n150_v1.json
DIAG_PAIRS=/tmp/state8_eedi_same_question_diagnosis_pairs_n20.json
EEDI_PAIRS=manifests/state8_eedi_same_question_c4_valid_seed20260811_offset100_n25_v1.jsonl
FORMAL_OUT="$AMLT_OUTPUT_DIR/eval/formal_t3_75"
MALRULE_OUT="$AMLT_OUTPUT_DIR/eval/malrule_diagnosis_c4_20"
EEDI_OUT="$AMLT_OUTPUT_DIR/eval/eedi_same_question_c4_25"
mkdir -p "$MODEL_LOCAL" "$(dirname "$RECURRENT_EVAL")" "$FORMAL_OUT" \
  "$MALRULE_OUT" "$EEDI_OUT" "$AMLT_OUTPUT_DIR/metrics"

test "$(sha256sum "$RAW_EVAL" | cut -d' ' -f1)" = bff7726953ae048bedf6dacdbb7500a54cb8de305e406f2497ef1f11c04167f8
test "$(sha256sum "$PARENT_PAIRS" | cut -d' ' -f1)" = edcdb934a65252557f6c569eea9ef7a85e5445549deac9fb4829c3f579e1a9b8
test "$(sha256sum "$EEDI_PAIRS" | cut -d' ' -f1)" = 2ba0b47bbdc99188c32542aaaf61a5d1493d2dc805bb44855eadad7d40e8effb
test "$(sha256sum scripts/audit_state8_source_anchor_checkpoint.py | cut -d' ' -f1)" = cbeb91f809d7a769260c6da2ef2fe6333f907ebf88b7732a1ef36a4a3b128792
test "$(sha256sum scripts/evaluate_recurrent_formal_protocols_diagnosis_first_eedi_current_query.py | cut -d' ' -f1)" = 967711843054673eab67e43a8d5392ac7329a31bf5005cc1ec5be8a60507244f
test "$(sha256sum scripts/evaluate_recurrent_diagnosis_c4_diagnosis_first.py | cut -d' ' -f1)" = 660df80664c6012f6eac1d379cc1d611c9f085c91496d1c9cc8e7f7bcb30d7ca
test "$(sha256sum scripts/evaluate_state8_eedi_same_question_c4.py | cut -d' ' -f1)" = 7aabe7cae743d63a566746eab5ba572b51e70e20130f3759dd006536d3543530
python scripts/audit_state8_source_anchor_checkpoint.py \
  --checkpoint-root "$CHECKPOINT_ROOT" --arm "$CHECKPOINT_ARM" --step "$CHECKPOINT_STEP" \
  --adapter-sha256 "$CHECKPOINT_ADAPTER_SHA256" --state-sha256 "$CHECKPOINT_STATE_SHA256" \
  --receipt-sha256 "$CHECKPOINT_RECEIPT_SHA256" --output "$AMLT_OUTPUT_DIR/metrics/artifact_audit.json"
python scripts/build_recurrent_evaluation_manifest.py --input-jsonl "$RAW_EVAL" \
  --output-jsonl "$RECURRENT_EVAL" --audit-json "$AMLT_OUTPUT_DIR/metrics/manifest_audit.json" \
  --max-updates 4 --seed 20260730
test "$(sha256sum "$RECURRENT_EVAL" | cut -d' ' -f1)" = a368cefb0c49cb6a305ad1a892808361fba2d44d555a0a28ea64e56babe32525
python - "$PARENT_PAIRS" "$DIAG_PAIRS" <<'PY'
import hashlib,json,pathlib,sys
source,out=map(pathlib.Path,sys.argv[1:]); value=json.loads(source.read_text())
assert value['num_pairs']==150 and len(value['pairs'])==150,value
value={**value,'num_pairs':20,'pairs':value['pairs'][:20],'diagnostic_only':True,
       'selection':'ordered_prefix','parent_manifest_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
out.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
PY
rm -rf "$MODEL_LOCAL"; mkdir -p "$MODEL_LOCAL"
python scripts/copy_model_parallel.py --source "$MODEL_REMOTE" --destination "$MODEL_LOCAL" --workers 16
python -u scripts/evaluate_recurrent_formal_protocols_diagnosis_first_eedi_current_query.py \
  --model "$MODEL_LOCAL" --adapter "$CHECKPOINT_ROOT/adapter" \
  --recurrent-state "$CHECKPOINT_ROOT/recurrent_state.pt" --state-architecture layerwise \
  --eval-jsonl "$RECURRENT_EVAL" --output-dir "$FORMAL_OUT" --state-token-alias '<|vision_pad|>' \
  --max-samples-per-group 25 --max-length 1536 --max-new-tokens 160 \
  --state-batch-size 2 --candidate-row-batch-size 2 --seed 42 \
  --tasks diagnose_misconception --sources eedi_mining malrule \
  --save-unredacted-predictions --t3-natural-schema
python -u scripts/evaluate_recurrent_diagnosis_c4_diagnosis_first.py \
  --model "$MODEL_LOCAL" --adapter "$CHECKPOINT_ROOT/adapter" \
  --recurrent-state "$CHECKPOINT_ROOT/recurrent_state.pt" --state-architecture layerwise \
  --jsonl "$C4_DATA" --pair-manifest "$DIAG_PAIRS" --output-dir "$MALRULE_OUT" \
  --state-token-alias '<|vision_pad|>' --max-length 1536 --max-new-tokens 160 \
  --num-pairs 20 --seed 42 --expected-adapter-sha256 "$CHECKPOINT_ADAPTER_SHA256" \
  --expected-recurrent-state-sha256 "$CHECKPOINT_STATE_SHA256"
python -u scripts/evaluate_state8_eedi_same_question_c4.py \
  --model "$MODEL_LOCAL" --adapter "$CHECKPOINT_ROOT/adapter" \
  --recurrent-state "$CHECKPOINT_ROOT/recurrent_state.pt" --state-architecture layerwise \
  --causal-jsonl "$EEDI_PAIRS" --output-dir "$EEDI_OUT" --state-token-alias '<|vision_pad|>' \
  --max-length 1536 --max-new-tokens 160 --num-pairs 25 --seed 20260811 \
  --expected-adapter-sha256 "$CHECKPOINT_ADAPTER_SHA256" \
  --expected-recurrent-state-sha256 "$CHECKPOINT_STATE_SHA256"
python - "$AMLT_OUTPUT_DIR" "$CHECKPOINT_STEP" "$CHECKPOINT_ARM" \
  "$CHECKPOINT_ADAPTER_SHA256" "$CHECKPOINT_STATE_SHA256" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); expected={'adapter_model':sys.argv[4],'recurrent_state':sys.argv[5]}
formal=json.load(open(root/'eval/formal_t3_75/metrics.json'))
malrule=json.load(open(root/'eval/malrule_diagnosis_c4_20/metrics.json'))
eedi=json.load(open(root/'eval/eedi_same_question_c4_25/metrics.json'))
for value in (formal,malrule,eedi):
 assert value['artifact_sha256']==expected and value['state_shape']==[64,8,5120],value
 assert value['raw_history_visible_to_decoder'] is False,value
assert formal['sampled_rows']==75 and malrule['num_pairs']==20 and eedi['num_pairs']==25,(formal,malrule,eedi)
assert eedi['same_question_query_identical'] is True,eedi
receipt={'stage':'state8_eedi_same_question_checkpoint_screen_v1','status':'complete',
 'diagnostic_only':True,'checkpoint_step':int(sys.argv[2]),'checkpoint_arm':sys.argv[3],
 'state_token_count':8,'raw_history_visible_to_decoder':False,
 'current_item_visible_to_decoder':True,'student_response_visible_to_decoder':False,
 'semantic_judge_authoritative':True,'exact_match_authoritative':False,
 'artifact_sha256':expected,'protocol':{'t3_rows':75,'malrule_c4_pairs':20,
 'eedi_same_question_c4_pairs':25,'seed':42}}
(root/'metrics/screen_receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
PY
