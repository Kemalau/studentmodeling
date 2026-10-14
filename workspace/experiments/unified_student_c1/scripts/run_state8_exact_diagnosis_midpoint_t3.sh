#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "STATE8_EXACT_MIDPOINT_T3_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

export VENV_ROOT="${VENV_ROOT:-/tmp/studentmodeling-state8-exact-midpoint-eval-venv}"
export PATH="$VENV_ROOT/bin:$PATH"
python(){ "$VENV_ROOT/bin/python" "$@"; }
export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/hf_cache
export HF_HUB_CACHE=/tmp/hf_cache/hub
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error

: "${STATE8_EXACT_CHECKPOINT_ROOT:?STATE8_EXACT_CHECKPOINT_ROOT is required}"
: "${EXACT_DIAGNOSIS_FORMAL_SHA256:?EXACT_DIAGNOSIS_FORMAL_SHA256 is required}"
: "${EXACT_DIAGNOSIS_C4_SHA256:?EXACT_DIAGNOSIS_C4_SHA256 is required}"
CHECKPOINT_STEP="${STATE8_EXACT_CHECKPOINT_STEP:-2000}"
case "$CHECKPOINT_STEP" in 2000|3000|4000|5000) ;; *) exit 2 ;; esac
EXPECTED_ROOT=/mnt/amulet/studentmodeling/checkpoints/state8_qualified_teacher_curriculum_seed42_exact5000_rounds1250_kd1_v2_20260805_taxonomy_semantic_v1_exact_diagnosis_first_seed64_single_ga4/checkpoint-$CHECKPOINT_STEP
test "$STATE8_EXACT_CHECKPOINT_ROOT" = "$EXPECTED_ROOT"
test "$(sha256sum scripts/evaluate_recurrent_formal_protocols_diagnosis_first.py | cut -d' ' -f1)" = "$EXACT_DIAGNOSIS_FORMAL_SHA256"
test "$(sha256sum scripts/evaluate_recurrent_diagnosis_c4_diagnosis_first.py | cut -d' ' -f1)" = "$EXACT_DIAGNOSIS_C4_SHA256"

MODEL_REMOTE=/mnt/amulet/studentmodeling/models/Qwen3-32B
MODEL_LOCAL=/tmp/Qwen3-32B
RAW_EVAL=/mnt/amulet/studentmodeling/formal_evaluation_v1/test.jsonl
RECURRENT_EVAL=/tmp/state8_exact_midpoint_eval_u4/test.jsonl
ADAPTER="$STATE8_EXACT_CHECKPOINT_ROOT/adapter"
STATE="$STATE8_EXACT_CHECKPOINT_ROOT/recurrent_state.pt"
RECEIPT="$STATE8_EXACT_CHECKPOINT_ROOT/checkpoint_completion.json"
OUT="$AMLT_OUTPUT_DIR/eval/formal_t3_75"

mkdir -p "$HF_HUB_CACHE" "$MODEL_LOCAL" "$(dirname "$RECURRENT_EVAL")" "$OUT"
for path in "$MODEL_REMOTE/model.safetensors.index.json" "$RAW_EVAL" \
  "$ADAPTER/adapter_model.safetensors" "$STATE" "$RECEIPT"; do test -f "$path"; done
test "$(df --output=avail /tmp | tail -1)" -ge 83886080
test "$(sha256sum "$RAW_EVAL" | cut -d' ' -f1)" = bff7726953ae048bedf6dacdbb7500a54cb8de305e406f2497ef1f11c04167f8

python - "$RECEIPT" "$ADAPTER/adapter_model.safetensors" "$STATE" "$CHECKPOINT_STEP" <<'PY'
import hashlib,json,pathlib,sys,torch
receipt,adapter,state=map(pathlib.Path,sys.argv[1:4]); expected_step=int(sys.argv[4])
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()
observed={'adapter_model':digest(adapter),'recurrent_state':digest(state)}
x=json.load(receipt.open()); assert x=={'sha256':observed,'status':'complete','step':expected_step},(x,observed)
p=torch.load(state,map_location='cpu',weights_only=False,mmap=True); c=p['config']; history=p['history']
assert p['step']==expected_step and len(history)==expected_step,(p['step'],len(history))
assert c['state_token_count']==8 and c['state_architecture']=='layerwise',c
assert c['final_inference_route']=='strict_state_only',c
assert c['diagnosis_field_order']==['diagnosis','evidence','label'],c
assert c['canonical_label_role']=='auxiliary_ordinary_weight',c
assert float(c['label_auxiliary_weight'])==1.0,c
assert c['taxonomy_label_contract']=='source_aware_no_answer_labels_v1',c
schedule=[0.0,0.25,0.5,0.75,1.0]
for index,value in enumerate(schedule):
 expected=max(0,min(1000,expected_step-index*1000))
 assert sum(float(r['history_dropout'])==value for r in history)==expected,(value,expected)
PY

python scripts/build_recurrent_evaluation_manifest.py \
  --input-jsonl "$RAW_EVAL" --output-jsonl "$RECURRENT_EVAL" \
  --audit-json /tmp/state8_exact_midpoint_eval_u4/audit.json \
  --max-updates 4 --seed 20260730
test "$(sha256sum "$RECURRENT_EVAL" | cut -d' ' -f1)" = a368cefb0c49cb6a305ad1a892808361fba2d44d555a0a28ea64e56babe32525

rm -rf "$MODEL_LOCAL"; mkdir -p "$MODEL_LOCAL"
python scripts/copy_model_parallel.py --source "$MODEL_REMOTE" --destination "$MODEL_LOCAL" --workers 16
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python -u scripts/evaluate_recurrent_formal_protocols_diagnosis_first.py \
  --model "$MODEL_LOCAL" --adapter "$ADAPTER" --recurrent-state "$STATE" \
  --state-architecture layerwise --eval-jsonl "$RECURRENT_EVAL" --output-dir "$OUT" \
  --state-token-alias '<|vision_pad|>' --max-samples-per-group 25 \
  --max-length 1536 --max-new-tokens 160 --state-batch-size 2 \
  --candidate-row-batch-size 2 --seed 42 --tasks diagnose_misconception \
  --sources eedi_mining malrule --save-unredacted-predictions --t3-natural-schema
python - "$OUT/metrics.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); assert x['state_shape']==[64,8,5120],x
assert x['raw_history_visible_to_decoder'] is False and x['persistent_history_kv_cache'] is False,x
assert x['sampled_rows']==75,x
for row in x['metrics_by_group'].values():
 assert row['exact_match_role']=='auxiliary_only_not_authoritative',row
PY
cp /tmp/state8_exact_midpoint_eval_u4/audit.json "$OUT/manifest_audit.json"
