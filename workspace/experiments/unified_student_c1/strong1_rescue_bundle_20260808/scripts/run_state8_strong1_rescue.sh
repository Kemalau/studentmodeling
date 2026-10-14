#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "STATE8_STRONG1_RESCUE_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

: "${STRONG1_SOURCE_SLUG:?STRONG1_SOURCE_SLUG is required}"
: "${STUDENT_ROOT:?STUDENT_ROOT is required}"
: "${EXPECTED_STUDENT_ADAPTER_SHA256:?EXPECTED_STUDENT_ADAPTER_SHA256 is required}"
: "${EXPECTED_STUDENT_STATE_SHA256:?EXPECTED_STUDENT_STATE_SHA256 is required}"
case "$STRONG1_SOURCE_SLUG" in
  lr5e7_seed58|lr1e6_seed59|single_lr5e7_seed60) ;;
  *) exit 2 ;;
esac
case "${TRAIN_ENTRYPOINT:?TRAIN_ENTRYPOINT is required}" in
  scripts/train_state8_diagnosis_first_causal_repair.py)
    test "${STATE8_REPAIR_LABEL_WEIGHT:?}" = 1.0
    test "${STATE8_CAUSAL_LABEL_WEIGHT:?}" = 1.0 ;;
  scripts/train_state8_task_anchored_causal_repair.py)
    test "${STATE8_REPAIR_LABEL_WEIGHT:?}" = 4.0
    test "${STATE8_CAUSAL_LABEL_WEIGHT:?}" = 4.0 ;;
  *) exit 2 ;;
esac
case "$STUDENT_ROOT" in
  /mnt/amulet/studentmodeling/diagnostics/state8_qualified_teacher_curriculum_highkd_v1_immutable_taxonomy_semantic_v1_strong1_widen8_*) ;;
  *) exit 2 ;;
esac
[[ "$EXPECTED_STUDENT_ADAPTER_SHA256" =~ ^[0-9a-f]{64}$ ]] || exit 2
[[ "$EXPECTED_STUDENT_STATE_SHA256" =~ ^[0-9a-f]{64}$ ]] || exit 2

export STUDENT_MODEL_DIR="$STUDENT_ROOT/model"
export STUDENT_STATE_PATH="$STUDENT_MODEL_DIR/recurrent_state.pt"
source_receipt="$STUDENT_ROOT/metrics/immutable_receipt.json"
normalized_root="/tmp/state8-strong1-rescue-source-$STRONG1_SOURCE_SLUG"
normalized_receipt="$normalized_root/immutable_receipt.json"
mkdir -p "$normalized_root"

python_bin="${VENV_ROOT:?VENV_ROOT is required}/bin/python"
"$python_bin" - "$source_receipt" "$STUDENT_MODEL_DIR/adapter_model.safetensors" \
  "$STUDENT_STATE_PATH" "$normalized_receipt" "$STUDENT_ROOT" \
  "$STRONG1_SOURCE_SLUG" "$EXPECTED_STUDENT_ADAPTER_SHA256" \
  "$EXPECTED_STUDENT_STATE_SHA256" <<'PY'
import hashlib,json,pathlib,sys
receipt,adapter,state,output,root=map(pathlib.Path,sys.argv[1:6])
slug,expected_adapter,expected_state=sys.argv[6:]
value=json.load(receipt.open())
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as handle:
  for chunk in iter(lambda:handle.read(8*1024*1024),b''): h.update(chunk)
 return h.hexdigest()
stage=str(value.get('stage') or '')
prefix='state8_qualified_teacher_curriculum_highkd_immutable_v1_taxonomy_semantic_v1_strong1_widen8_'
assert stage.startswith(prefix),value
assert value.get('status')=='complete' and value.get('micro_steps')==5000,value
assert value.get('state_shape')==[64,8,5120],value
assert value.get('raw_history_visible_to_decoder') is False,value
assert value.get('persistent_history_kv_cache') is False,value
assert value.get('semantic_judge_authoritative') is True,value
assert value.get('exact_label_metric_authoritative') is False,value
assert value.get('taxonomy_label_contract')=='source_aware_no_answer_labels_v1',value
observed={'adapter_model':digest(adapter),'recurrent_state':digest(state)}
assert observed==value.get('sha256'),(observed,value)
assert observed=={'adapter_model':expected_adapter,'recurrent_state':expected_state},observed
assert str(value.get('remote_root'))==str(root),(value.get('remote_root'),root)
normalized={
 'stage':'state8_qualified_teacher_curriculum_highkd_immutable_v1_taxonomy_semantic_v1',
 'status':'complete','state_shape':[64,8,5120],
 'raw_history_visible_to_decoder':False,
 'persistent_history_kv_cache':False,
 'sha256':observed,
 'normalized_from_stage':stage,
 'normalized_from_remote_root':str(root),
 'strong1_source_slug':slug,
}
output.write_text(json.dumps(normalized,indent=2,sort_keys=True)+'\n')
PY
export STUDENT_RECEIPT="$normalized_receipt"

bash scripts/run_state8_stateonly_repair_ab.sh

answer_out="$AMLT_OUTPUT_DIR/eval/answer_c4_150"
mkdir -p "$answer_out"
test -f "$AMLT_OUTPUT_DIR/model/adapter_model.safetensors"
test -f "$AMLT_OUTPUT_DIR/model/recurrent_state.pt"
test -f /tmp/Qwen3-32B/model.safetensors.index.json
test -f manifests/c4_stage_a_gate_pairs_seed42_n150_v1.json
test "$(sha256sum manifests/c4_stage_a_gate_pairs_seed42_n150_v1.json | cut -d' ' -f1)" = edcdb934a65252557f6c569eea9ef7a85e5445549deac9fb4829c3f579e1a9b8
test "$(sha256sum scripts/evaluate_recurrent_causal_gate.py | cut -d' ' -f1)" = e0c9fe1361dfda5eddc14abcd275fbe8e2b92f472f9eaf27468b71f800fafccf

"$python_bin" -u scripts/evaluate_recurrent_causal_gate.py \
  --model /tmp/Qwen3-32B --adapter "$AMLT_OUTPUT_DIR/model" \
  --recurrent-state "$AMLT_OUTPUT_DIR/model/recurrent_state.pt" \
  --state-architecture layerwise \
  --jsonl /mnt/amulet/studentmodeling/c4_fullopen_20260724/malrule_c1_clean/test_same_template.jsonl \
  --pair-manifest manifests/c4_stage_a_gate_pairs_seed42_n150_v1.json \
  --output-json "$answer_out/recurrent_causal_gate.json" \
  --state-token-alias '<|vision_pad|>' --max-length 1536 \
  --num-pairs 150 --seed 42

"$python_bin" - "$answer_out/recurrent_causal_gate.json" \
  "$AMLT_OUTPUT_DIR/metrics/publish_receipt.json" <<'PY'
import json,sys
result=json.load(open(sys.argv[1])); receipt=json.load(open(sys.argv[2]))
assert result['state_shape']==[64,8,5120],result
assert result['raw_history_visible_to_decoder'] is False,result
assert result['persistent_history_kv_cache'] is False,result
assert result['artifact_sha256']==receipt['artifact_sha256'],(result,receipt)
assert result['summary']['all_pairs_n']==150 and result['summary']['clean_n']>0,result
PY
cp "$source_receipt" "$AMLT_OUTPUT_DIR/metrics/strong1_source_immutable_receipt.json"
cp "$normalized_receipt" "$AMLT_OUTPUT_DIR/metrics/strong1_source_normalized_receipt.json"
