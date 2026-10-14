#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "STATE8_TASKANCHOR_MILESTONE_SCREEN_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

export VENV_ROOT="${VENV_ROOT:-/tmp/studentmodeling-state8-taskanchor-milestone-venv}"
export PATH="$VENV_ROOT/bin:$PATH"
python(){ "$VENV_ROOT/bin/python" "$@"; }
export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/hf_cache HF_HUB_CACHE=/tmp/hf_cache/hub HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRANSFORMERS_VERBOSITY=error

: "${MILESTONE_STEP:?}" "${MILESTONE_ADAPTER_SHA256:?}"
: "${MILESTONE_STATE_SHA256:?}" "${MILESTONE_RECEIPT_SHA256:?}"
case "$MILESTONE_STEP" in 200|400|600|800|1000) ;; *) exit 2 ;; esac

CHECKPOINT="/mnt/amulet/studentmodeling/checkpoints/state8_stateonly_repair_seed2067_taskanchor_matched_cp2000_causal_taskanchor_v1/checkpoint-$MILESTONE_STEP"
MODEL_REMOTE=/mnt/amulet/studentmodeling/models/Qwen3-32B
MODEL_LOCAL=/tmp/Qwen3-32B
RAW_EVAL=/mnt/amulet/studentmodeling/formal_evaluation_v1/test.jsonl
RECURRENT_EVAL=/tmp/state8_taskanchor_milestone_eval/test.jsonl
C4_DATA=/mnt/amulet/studentmodeling/c4_fullopen_20260724/malrule_c1_clean/test_same_template.jsonl
PARENT_PAIRS=manifests/c4_stage_a_gate_pairs_seed42_n150_v1.json
ANSWER_PAIRS=/tmp/state8_taskanchor_answer_pairs_n50.json
DIAG_PAIRS=/tmp/state8_taskanchor_diagnosis_pairs_n20.json
FORMAL_OUT="$AMLT_OUTPUT_DIR/eval/formal_t3_75"
ANSWER_OUT="$AMLT_OUTPUT_DIR/eval/answer_c4_50"
DIAG_OUT="$AMLT_OUTPUT_DIR/eval/diagnosis_c4_20"

mkdir -p "$MODEL_LOCAL" "$(dirname "$RECURRENT_EVAL")" "$FORMAL_OUT" \
  "$ANSWER_OUT" "$DIAG_OUT" "$AMLT_OUTPUT_DIR/metrics"
test "$(df --output=avail /tmp | tail -1)" -ge 83886080
test "$(sha256sum "$RAW_EVAL" | cut -d' ' -f1)" = bff7726953ae048bedf6dacdbb7500a54cb8de305e406f2497ef1f11c04167f8
test "$(sha256sum "$PARENT_PAIRS" | cut -d' ' -f1)" = edcdb934a65252557f6c569eea9ef7a85e5445549deac9fb4829c3f579e1a9b8
test "$(sha256sum scripts/evaluate_recurrent_formal_protocols_diagnosis_first.py | cut -d' ' -f1)" = a8bc07a8fe6d2c9bfa9af6d976b53b84321cb4baa9c9fcace3188e370f7732c9
test "$(sha256sum scripts/evaluate_recurrent_causal_gate.py | cut -d' ' -f1)" = 69fd4f2f4943a78a0afe9a701533d9ab034f247f87c7a09327f7523e310b7a52
test "$(sha256sum scripts/evaluate_recurrent_diagnosis_c4_diagnosis_first.py | cut -d' ' -f1)" = 660df80664c6012f6eac1d379cc1d611c9f085c91496d1c9cc8e7f7bcb30d7ca

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
python - "$PARENT_PAIRS" "$ANSWER_PAIRS" "$DIAG_PAIRS" <<'PY'
import hashlib,json,pathlib,sys
source,answer,diagnosis=map(pathlib.Path,sys.argv[1:])
value=json.loads(source.read_text()); assert value['num_pairs']==150 and len(value['pairs'])==150,value
parent=hashlib.sha256(source.read_bytes()).hexdigest()
for path,count in ((answer,50),(diagnosis,20)):
 out={**value,'num_pairs':count,'pairs':value['pairs'][:count],
      'diagnostic_only':True,'selection':'ordered_prefix','parent_manifest_sha256':parent}
 path.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
PY

rm -rf "$MODEL_LOCAL"; mkdir -p "$MODEL_LOCAL"
python scripts/copy_model_parallel.py --source "$MODEL_REMOTE" --destination "$MODEL_LOCAL" --workers 16
test -f "$MODEL_LOCAL/model.safetensors.index.json"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

python -u scripts/evaluate_recurrent_formal_protocols_diagnosis_first.py \
  --model "$MODEL_LOCAL" --adapter "$CHECKPOINT/adapter" \
  --recurrent-state "$CHECKPOINT/recurrent_state.pt" --state-architecture layerwise \
  --eval-jsonl "$RECURRENT_EVAL" --output-dir "$FORMAL_OUT" \
  --state-token-alias '<|vision_pad|>' --max-samples-per-group 25 \
  --max-length 1536 --max-new-tokens 160 --state-batch-size 2 \
  --candidate-row-batch-size 2 --seed 42 --tasks diagnose_misconception \
  --sources eedi_mining malrule --save-unredacted-predictions --t3-natural-schema
python -u scripts/evaluate_recurrent_causal_gate.py \
  --model "$MODEL_LOCAL" --adapter "$CHECKPOINT/adapter" \
  --recurrent-state "$CHECKPOINT/recurrent_state.pt" --state-architecture layerwise \
  --jsonl "$C4_DATA" --pair-manifest "$ANSWER_PAIRS" \
  --output-json "$ANSWER_OUT/recurrent_causal_gate.json" \
  --state-token-alias '<|vision_pad|>' --max-length 1536 --num-pairs 50 --seed 42
python -u scripts/evaluate_recurrent_diagnosis_c4_diagnosis_first.py \
  --model "$MODEL_LOCAL" --adapter "$CHECKPOINT/adapter" \
  --recurrent-state "$CHECKPOINT/recurrent_state.pt" --state-architecture layerwise \
  --jsonl "$C4_DATA" --pair-manifest "$DIAG_PAIRS" --output-dir "$DIAG_OUT" \
  --state-token-alias '<|vision_pad|>' --max-length 1536 --max-new-tokens 160 \
  --num-pairs 20 --seed 42 --expected-adapter-sha256 "$MILESTONE_ADAPTER_SHA256" \
  --expected-recurrent-state-sha256 "$MILESTONE_STATE_SHA256"

python - "$AMLT_OUTPUT_DIR" "$MILESTONE_STEP" \
  "$MILESTONE_ADAPTER_SHA256" "$MILESTONE_STATE_SHA256" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); step=int(sys.argv[2])
expected={'adapter_model':sys.argv[3],'recurrent_state':sys.argv[4]}
formal=json.load(open(root/'eval/formal_t3_75/metrics.json'))
answer=json.load(open(root/'eval/answer_c4_50/recurrent_causal_gate.json'))
diagnosis=json.load(open(root/'eval/diagnosis_c4_20/metrics.json'))
for value in (formal,answer,diagnosis):
 assert value['artifact_sha256']==expected and value['state_shape']==[64,8,5120],value
 assert value['raw_history_visible_to_decoder'] is False,value
 assert value['persistent_history_kv_cache'] is False,value
assert formal['sampled_rows']==75 and answer['summary']['all_pairs_n']==50, (formal,answer)
assert diagnosis['num_pairs']==20 and diagnosis['semantic_judgment_pending'] is True,diagnosis
receipt={'stage':'state8_taskanchor_milestone_screen_v1','status':'complete',
 'diagnostic_only':True,'checkpoint_step':step,'state_token_count':8,
 'raw_history_visible_to_decoder':False,'semantic_judge_authoritative':True,
 'exact_match_authoritative':False,'canonical_label_role':'auxiliary_ordinary_weight',
 'artifact_sha256':expected,'protocol':{'t3_rows':75,'answer_c4_pairs':50,
 'diagnosis_c4_pairs':20,'seed':42}}
(root/'metrics/screen_receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
PY
