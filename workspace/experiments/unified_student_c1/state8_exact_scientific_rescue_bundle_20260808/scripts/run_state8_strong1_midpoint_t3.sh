#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "STATE8_MIDPOINT_T3_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

export VENV_ROOT="${VENV_ROOT:-/tmp/studentmodeling-state8-midpoint-eval-venv}"
export PATH="$VENV_ROOT/bin:$PATH"
python(){ "$VENV_ROOT/bin/python" "$@"; }
export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/hf_cache
export HF_HUB_CACHE=/tmp/hf_cache/hub
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error

: "${STATE8_CHECKPOINT_ROOT:?STATE8_CHECKPOINT_ROOT is required}"
case "$STATE8_CHECKPOINT_ROOT" in
  /mnt/amulet/studentmodeling/checkpoints/state8_qualified_teacher_curriculum_seed42_exact5000_rounds1250_kd1_v2_20260805_taxonomy_semantic_v1_strong1_widen8_single_lr5e7_seed60_single_ga4/checkpoint-2000) ;;
  *) exit 2 ;;
esac

MODEL_REMOTE=/mnt/amulet/studentmodeling/models/Qwen3-32B
MODEL_LOCAL=/tmp/Qwen3-32B
RAW_EVAL=/mnt/amulet/studentmodeling/formal_evaluation_v1/test.jsonl
RECURRENT_EVAL=/tmp/recurrent_formal_evaluation_fixedu4_v2/test.jsonl
ADAPTER="$STATE8_CHECKPOINT_ROOT/adapter"
STATE="$STATE8_CHECKPOINT_ROOT/recurrent_state.pt"
RECEIPT="$STATE8_CHECKPOINT_ROOT/checkpoint_completion.json"
OUT="$AMLT_OUTPUT_DIR/eval/t3_step2000"

mkdir -p "$HF_HUB_CACHE" "$MODEL_LOCAL" "$(dirname "$RECURRENT_EVAL")" "$OUT"
for path in "$MODEL_REMOTE/model.safetensors.index.json" "$RAW_EVAL" \
  "$ADAPTER/adapter_model.safetensors" "$STATE" "$RECEIPT"; do
  test -f "$path"
done
test "$(df --output=avail /tmp | tail -1)" -ge 83886080
test "$(sha256sum "$RAW_EVAL" | cut -d' ' -f1)" = bff7726953ae048bedf6dacdbb7500a54cb8de305e406f2497ef1f11c04167f8

python - "$RECEIPT" "$ADAPTER/adapter_model.safetensors" "$STATE" <<'PY'
import hashlib,json,pathlib,sys,torch
receipt,adapter,state=map(pathlib.Path,sys.argv[1:])
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()
observed={'adapter_model':digest(adapter),'recurrent_state':digest(state)}
x=json.load(receipt.open()); assert x=={'sha256':observed,'status':'complete','step':2000},(x,observed)
p=torch.load(state,map_location='cpu',weights_only=False,mmap=True); c=p['config']; history=p['history']
assert p['step']==2000 and len(history)==2000,(p['step'],len(history))
assert c['state_token_count']==8 and c['state_architecture']=='layerwise',c
assert c['final_inference_route']=='strict_state_only',c
assert c['taxonomy_label_contract']=='source_aware_no_answer_labels_v1',c
assert sum(float(r['history_dropout'])==0.0 for r in history)==1000
assert sum(float(r['history_dropout'])==0.25 for r in history)==1000
PY

python scripts/build_recurrent_evaluation_manifest.py \
  --input-jsonl "$RAW_EVAL" --output-jsonl "$RECURRENT_EVAL" \
  --audit-json /tmp/recurrent_formal_evaluation_fixedu4_v2/audit.json \
  --max-updates 4 --seed 20260730
test "$(sha256sum "$RECURRENT_EVAL" | cut -d' ' -f1)" = a368cefb0c49cb6a305ad1a892808361fba2d44d555a0a28ea64e56babe32525

rm -rf "$MODEL_LOCAL"; mkdir -p "$MODEL_LOCAL"
python scripts/copy_model_parallel.py --source "$MODEL_REMOTE" --destination "$MODEL_LOCAL" --workers 16
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python -u scripts/evaluate_recurrent_formal_protocols_taxonomy.py \
  --model "$MODEL_LOCAL" --adapter "$ADAPTER" --recurrent-state "$STATE" \
  --state-architecture layerwise --eval-jsonl "$RECURRENT_EVAL" --output-dir "$OUT" \
  --state-token-alias '<|vision_pad|>' --max-samples-per-group 25 \
  --max-length 1536 --max-new-tokens 128 --state-batch-size 2 \
  --candidate-row-batch-size 2 --seed 42 --tasks diagnose_misconception \
  --sources eedi_mining malrule --save-unredacted-predictions --t3-natural-schema
python - "$OUT/metrics.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); assert x['state_shape']==[64,8,5120],x
assert x['raw_history_visible_to_decoder'] is False and x['persistent_history_kv_cache'] is False,x
assert x['sampled_rows']==75,x
PY
cp /tmp/recurrent_formal_evaluation_fixedu4_v2/audit.json "$OUT/manifest_audit.json"
