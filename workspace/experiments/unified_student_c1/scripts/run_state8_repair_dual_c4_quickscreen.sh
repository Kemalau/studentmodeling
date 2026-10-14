#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "STATE8_REPAIR_DUAL_C4_QUICKSCREEN_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

export VENV_ROOT="${VENV_ROOT:-/tmp/studentmodeling-state8-repair-dual-c4-venv}"
export PATH="$VENV_ROOT/bin:$PATH"
python(){ "$VENV_ROOT/bin/python" "$@"; }
export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/hf_cache HF_HUB_CACHE=/tmp/hf_cache/hub HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRANSFORMERS_VERBOSITY=error

: "${REPAIR_SEED:?}" "${REPAIR_BRANCH:?}" "${CHECKPOINT_STEP:?}"
: "${CHECKPOINT_ROOT:?}" "${CHECKPOINT_ADAPTER_SHA256:?}"
: "${CHECKPOINT_STATE_SHA256:?}" "${CHECKPOINT_RECEIPT_SHA256:?}"
case "$REPAIR_SEED:$REPAIR_BRANCH" in
  8201:causal_teacher_low|8202:causal_teacher_high) ;;
  *) exit 2 ;;
esac
test "$CHECKPOINT_STEP" = 20
expected_root="/mnt/amulet/studentmodeling/checkpoint_snapshots/state8_curriculum_repairs_20260812/seed${REPAIR_SEED}_${REPAIR_BRANCH}/checkpoint-20"
test "$CHECKPOINT_ROOT" = "$expected_root"

MODEL_REMOTE=/mnt/amulet/studentmodeling/models/Qwen3-32B
MODEL_LOCAL=/tmp/Qwen3-32B
C4_DATA=/mnt/amulet/studentmodeling/c4_fullopen_20260724/malrule_c1_clean/test_same_template.jsonl
PARENT_PAIRS=manifests/c4_stage_a_gate_pairs_seed42_n150_v1.json
DIAG_PAIRS=/tmp/state8_repair_dual_c4_pairs_n20.json
EEDI_PAIRS=manifests/state8_eedi_same_question_c4_valid_seed20260811_offset100_n25_v1.jsonl
MALRULE_OUT="$AMLT_OUTPUT_DIR/eval/malrule_diagnosis_c4_20"
EEDI_OUT="$AMLT_OUTPUT_DIR/eval/eedi_same_question_c4_25"
MALRULE_PAIRWISE="$AMLT_OUTPUT_DIR/eval/malrule_pairwise"
EEDI_PAIRWISE="$AMLT_OUTPUT_DIR/eval/eedi_pairwise"
mkdir -p "$MODEL_LOCAL" "$MALRULE_OUT" "$EEDI_OUT" \
  "$MALRULE_PAIRWISE" "$EEDI_PAIRWISE" "$AMLT_OUTPUT_DIR/metrics"

test "$(sha256sum "$PARENT_PAIRS" | cut -d' ' -f1)" = edcdb934a65252557f6c569eea9ef7a85e5445549deac9fb4829c3f579e1a9b8
test "$(sha256sum "$EEDI_PAIRS" | cut -d' ' -f1)" = 2ba0b47bbdc99188c32542aaaf61a5d1493d2dc805bb44855eadad7d40e8effb
test "$(sha256sum scripts/evaluate_recurrent_diagnosis_c4_diagnosis_first.py | cut -d' ' -f1)" = 660df80664c6012f6eac1d379cc1d611c9f085c91496d1c9cc8e7f7bcb30d7ca
test "$(sha256sum scripts/evaluate_state8_eedi_same_question_c4.py | cut -d' ' -f1)" = 7aabe7cae743d63a566746eab5ba572b51e70e20130f3759dd006536d3543530

python scripts/audit_state8_curriculum_repair_checkpoint.py \
  --checkpoint-root "$CHECKPOINT_ROOT" --seed "$REPAIR_SEED" \
  --branch "$REPAIR_BRANCH" --step "$CHECKPOINT_STEP" \
  --adapter-sha256 "$CHECKPOINT_ADAPTER_SHA256" \
  --state-sha256 "$CHECKPOINT_STATE_SHA256" \
  --receipt-sha256 "$CHECKPOINT_RECEIPT_SHA256" \
  --output "$AMLT_OUTPUT_DIR/metrics/artifact_audit.json"

python - "$PARENT_PAIRS" "$DIAG_PAIRS" <<'PY'
import hashlib,json,pathlib,sys
source,out=map(pathlib.Path,sys.argv[1:])
value=json.loads(source.read_text())
assert value['num_pairs']==150 and len(value['pairs'])==150,value
value={**value,'num_pairs':20,'pairs':value['pairs'][:20],
       'diagnostic_only':True,'selection':'ordered_prefix',
       'parent_manifest_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
out.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
PY

rm -rf "$MODEL_LOCAL"; mkdir -p "$MODEL_LOCAL"
python scripts/copy_model_parallel.py --source "$MODEL_REMOTE" \
  --destination "$MODEL_LOCAL" --workers 16
test -f "$MODEL_LOCAL/model.safetensors.index.json"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

python -u scripts/evaluate_recurrent_diagnosis_c4_diagnosis_first.py \
  --model "$MODEL_LOCAL" --adapter "$CHECKPOINT_ROOT/adapter" \
  --recurrent-state "$CHECKPOINT_ROOT/recurrent_state.pt" \
  --state-architecture layerwise --jsonl "$C4_DATA" \
  --pair-manifest "$DIAG_PAIRS" --output-dir "$MALRULE_OUT" \
  --state-token-alias '<|vision_pad|>' --max-length 1536 \
  --max-new-tokens 192 --num-pairs 20 --seed 20 \
  --expected-adapter-sha256 "$CHECKPOINT_ADAPTER_SHA256" \
  --expected-recurrent-state-sha256 "$CHECKPOINT_STATE_SHA256"
python -u scripts/evaluate_state8_eedi_same_question_c4.py \
  --model "$MODEL_LOCAL" --adapter "$CHECKPOINT_ROOT/adapter" \
  --recurrent-state "$CHECKPOINT_ROOT/recurrent_state.pt" \
  --state-architecture layerwise --causal-jsonl "$EEDI_PAIRS" \
  --output-dir "$EEDI_OUT" --state-token-alias '<|vision_pad|>' \
  --max-length 1536 --max-new-tokens 192 --num-pairs 25 --seed 20260811 \
  --expected-adapter-sha256 "$CHECKPOINT_ADAPTER_SHA256" \
  --expected-recurrent-state-sha256 "$CHECKPOINT_STATE_SHA256"

python scripts/build_pairwise_diagnosis_c4_judge_panel.py \
  --generations-jsonl "$MALRULE_OUT/predictions.jsonl" \
  --predictions-jsonl "$MALRULE_PAIRWISE/judge_predictions.jsonl" \
  --eval-jsonl "$MALRULE_PAIRWISE/judge_eval.jsonl" \
  --receipt "$MALRULE_PAIRWISE/panel_receipt.json"
python scripts/build_pairwise_diagnosis_c4_judge_panel.py \
  --generations-jsonl "$EEDI_OUT/predictions.jsonl" \
  --predictions-jsonl "$EEDI_PAIRWISE/judge_predictions.jsonl" \
  --eval-jsonl "$EEDI_PAIRWISE/judge_eval.jsonl" \
  --receipt "$EEDI_PAIRWISE/panel_receipt.json"

python - "$AMLT_OUTPUT_DIR" "$REPAIR_SEED" "$REPAIR_BRANCH" \
  "$CHECKPOINT_ADAPTER_SHA256" "$CHECKPOINT_STATE_SHA256" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1])
expected={'adapter_model':sys.argv[4],'recurrent_state':sys.argv[5]}
mal=json.load(open(root/'eval/malrule_diagnosis_c4_20/metrics.json'))
eedi=json.load(open(root/'eval/eedi_same_question_c4_25/metrics.json'))
for value,count in ((mal,20),(eedi,25)):
 assert value['artifact_sha256']==expected,value
 assert value['state_shape']==[64,8,5120] and value['num_pairs']==count,value
 assert value['raw_history_visible_to_decoder'] is False,value
assert eedi['same_question_query_identical'] is True,eedi
receipt={'stage':'state8_repair_dual_c4_quickscreen_generation_v1',
 'status':'complete','diagnostic_only':True,'repair_seed':int(sys.argv[2]),
 'repair_branch':sys.argv[3],'checkpoint_step':20,'state_token_count':8,
 'raw_history_visible_to_decoder':False,'semantic_judge_authoritative':True,
 'exact_match_authoritative':False,'artifact_sha256':expected,
 'protocol':{'malrule_pairs':20,'eedi_same_question_pairs':25,
             'conditions':['native','noop','samemal','random','swap'],
             'pairwise_native_output_controls':True}}
(root/'metrics/quickscreen_receipt.json').write_text(
 json.dumps(receipt,indent=2,sort_keys=True)+'\n')
PY
