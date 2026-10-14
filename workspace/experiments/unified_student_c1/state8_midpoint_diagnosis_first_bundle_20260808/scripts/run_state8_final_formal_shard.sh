#!/usr/bin/env bash
set -euo pipefail

export VENV_ROOT="${VENV_ROOT:-/tmp/studentmodeling-state8-final-eval-venv}"
export PATH="$VENV_ROOT/bin:$PATH"
python(){ "$VENV_ROOT/bin/python" "$@"; }
export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/hf_cache
export HF_HUB_CACHE=/tmp/hf_cache/hub
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error
export LOCAL_MODEL=/tmp/Qwen3-32B
export RAW_EVAL=/mnt/amulet/studentmodeling/formal_evaluation_v1/test.jsonl
export RECURRENT_EVAL=/tmp/state8_final_formal_eval_u4_v1/test.jsonl
export RUN_DIR=/mnt/amulet/studentmodeling/checkpoints/state8_qualified_unified_full259845_seed42_v1_20260805
: "${SHARD_INDEX:?SHARD_INDEX is required}"
test "${NUM_SHARDS:-16}" = 16
export NUM_SHARDS=16
export OUT="$AMLT_OUTPUT_DIR/eval/formal_shard$SHARD_INDEX"

mkdir -p "$HF_HUB_CACHE" "$(dirname "$RECURRENT_EVAL")" "$OUT"
test -f "$RUN_DIR/final_adapter/adapter_model.safetensors"
test -f "$RUN_DIR/recurrent_state.pt"
test -f "$RUN_DIR/completion.json"
test -f "$RUN_DIR/state8_final_fit_receipt.json"
test -f "$RUN_DIR/canonical_publish_receipt.json"
test -f "$RAW_EVAL"
test "$(df --output=avail /tmp | tail -1)" -ge 83886080
test "$(sha256sum "$RAW_EVAL" | cut -d' ' -f1)" = bff7726953ae048bedf6dacdbb7500a54cb8de305e406f2497ef1f11c04167f8
test "$(sha256sum scripts/evaluate_recurrent_formal_protocols.py | cut -d' ' -f1)" = 262370592e018641b99d6282858d8990b38314c05bb6e3bbc645823834cf73ed
test "$(sha256sum scripts/state8_natural_curriculum_view.py | cut -d' ' -f1)" = 98c961180d2428b2c68ef9399996ed991d5bd34120a3046b4f56fb9ba4e09682
test "$(sha256sum scripts/build_recurrent_evaluation_manifest.py | cut -d' ' -f1)" = d335b0b97471683eb1f317f21af6f0dd36ccca0fe6a8e495838112785ea85a8e

FORMAL_EVALUATOR=scripts/evaluate_recurrent_formal_protocols.py
SELECTED_VARIANT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_replay_variant"])' "$RUN_DIR/state8_final_fit_receipt.json")"
if [[ "$SELECTED_VARIANT" == highcausal_taxonomy \
      || "$SELECTED_VARIANT" == taxonomy_curriculum_t3heavy \
      || "$SELECTED_VARIANT" == taxonomy_curriculum_balanced \
      || "$SELECTED_VARIANT" == taxonomy_curriculum_t3heavy_diag \
      || "$SELECTED_VARIANT" == taxonomy_curriculum_balanced_diag ]]; then
  FORMAL_EVALUATOR=scripts/evaluate_recurrent_formal_protocols_taxonomy.py
  test -f "$RUN_DIR/taxonomy_causal_repair.json"
  test "$(sha256sum "$FORMAL_EVALUATOR" | cut -d' ' -f1)" = 648baa998672f30652ad3e68f114c2648c5bf839dcddd5b1e0acae47400e5b22
  test "$(sha256sum scripts/state8_taxonomy_contract.py | cut -d' ' -f1)" = 1ca32112e76c795502f3cb1edbedcc7220f093c133fae8d20ff5cab789e512ed
fi

python - "$RUN_DIR" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); completion=json.load(open(root/'completion.json'))
receipt=json.load(open(root/'state8_final_fit_receipt.json'))
published=json.load(open(root/'canonical_publish_receipt.json'))
def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()
observed={'adapter_model':digest(root/'final_adapter/adapter_model.safetensors'),
          'recurrent_state':digest(root/'recurrent_state.pt')}
assert completion['status']=='complete' and completion['diagnostic'] is False,completion
assert completion['micro_steps']==259845 and completion['exact_history_support'] is True,completion
assert receipt['status']=='final_fit_complete' and receipt['one_epoch_exact'] is True,receipt
assert receipt['stage']=='state8_qualified_unified_final_fit_v1',receipt
assert len(receipt['teacher_adapter_sha256'])==64 and receipt['teacher_gate_required'] is True,receipt
assert receipt['state_shape']==[64,8,5120],receipt
assert receipt['raw_history_visible_to_decoder'] is False,receipt
assert receipt['history_training_mode']=='strict_state_only',receipt
assert receipt['history_dropout_counts']=={'1.0':259845},receipt
assert float(receipt['online_kd_weight'])==1.0,receipt
assert float(receipt['label_auxiliary_weight'])==4.0,receipt
assert published['stage']=='state8_canonical_final_checkpoint_publish_v1',published
assert published['status']=='complete',published
assert published['selected_replay_variant']==receipt['selected_replay_variant'],(published,receipt)
assert published['artifact_sha256']==observed,(published,observed)
assert completion['sha256']['adapter_model']==observed['adapter_model'],(completion,observed)
assert completion['sha256']['recurrent_state']==observed['recurrent_state'],(completion,observed)
assert receipt['adapter_sha256']==observed['adapter_model'],(receipt,observed)
assert receipt['recurrent_state_sha256']==observed['recurrent_state'],(receipt,observed)
PY

python -m pytest -q tests/test_evaluate_recurrent_formal_protocols.py \
  tests/test_state8_natural_curriculum_view.py
python scripts/build_recurrent_evaluation_manifest.py \
  --input-jsonl "$RAW_EVAL" --output-jsonl "$RECURRENT_EVAL" \
  --audit-json /tmp/state8_final_formal_eval_u4_v1/audit.json \
  --max-updates 4 --seed 20260730
test "$(sha256sum "$RECURRENT_EVAL" | cut -d' ' -f1)" = a368cefb0c49cb6a305ad1a892808361fba2d44d555a0a28ea64e56babe32525

rm -rf "$LOCAL_MODEL"; mkdir -p "$LOCAL_MODEL"
python scripts/copy_model_parallel.py \
  --source /mnt/amulet/studentmodeling/models/Qwen3-32B \
  --destination "$LOCAL_MODEL" --workers 16
test -f "$LOCAL_MODEL/model.safetensors.index.json"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python -u "$FORMAL_EVALUATOR" \
  --model "$LOCAL_MODEL" --adapter "$RUN_DIR/final_adapter" \
  --recurrent-state "$RUN_DIR/recurrent_state.pt" --state-architecture layerwise \
  --eval-jsonl "$RECURRENT_EVAL" --output-dir "$OUT" \
  --state-token-alias '<|vision_pad|>' --max-length 1536 --max-new-tokens 128 \
  --state-batch-size 2 --candidate-row-batch-size 2 \
  --shard-index "$SHARD_INDEX" --num-shards "$NUM_SHARDS" --seed 42 \
  --save-unredacted-predictions --t3-natural-schema
python - "$OUT/metrics.json" "$SHARD_INDEX" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); index=int(sys.argv[2])
assert x['state_architecture']=='layerwise' and x['state_shape']==[64,8,5120],x
assert x['raw_history_visible_to_decoder'] is False,x
assert x['persistent_history_kv_cache'] is False,x
assert x['unsharded_filtered_rows']==21676,x
assert x['evaluation_shard']=={'index':index,'count':16},x
assert x['sampled_rows'] in {1354,1355},x
assert len(x['artifact_sha256']['adapter_model'])==64,x
assert len(x['artifact_sha256']['recurrent_state'])==64,x
PY
cp /tmp/state8_final_formal_eval_u4_v1/audit.json "$OUT/manifest_audit.json"
