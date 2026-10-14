#!/usr/bin/env bash
set -euo pipefail

: "${PROBE_VARIANT:?PROBE_VARIANT is required}"
: "${PROBE_ROUND:?PROBE_ROUND is required}"
case "$PROBE_VARIANT" in t3heavy|t3heavy_diag) ;; *) exit 2 ;; esac
[[ "$PROBE_ROUND" == 100 ]] || exit 2

export VENV_ROOT="${VENV_ROOT:-/tmp/studentmodeling-state8-consolidation-probe-venv}"
export PATH="$VENV_ROOT/bin:$PATH"
python(){ "$VENV_ROOT/bin/python" "$@"; }
export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/hf_cache
export HF_HUB_CACHE=/tmp/hf_cache/hub
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error

export LOCAL_MODEL=/tmp/Qwen3-32B
export CHECKPOINT="/mnt/amulet/studentmodeling/diagnostics/state8_taxonomy_consolidation_milestones_20260808/$PROBE_VARIANT/round-$PROBE_ROUND"
export RAW_EVAL=/mnt/amulet/studentmodeling/formal_evaluation_v1/test.jsonl
export RECURRENT_EVAL="/tmp/state8_consolidation_${PROBE_VARIANT}_r${PROBE_ROUND}_u4/test.jsonl"
export C4_DATA=/mnt/amulet/studentmodeling/c4_fullopen_20260724/malrule_c1_clean/test_same_template.jsonl
export PAIRS=manifests/c4_stage_a_gate_pairs_seed42_n150_v1.json
export T3_OUT="$AMLT_OUTPUT_DIR/eval/t3_screen"
export C4_OUT="$AMLT_OUTPUT_DIR/eval/diagnosis_c4_10"

mkdir -p "$HF_HUB_CACHE" "$(dirname "$RECURRENT_EVAL")" "$T3_OUT" "$C4_OUT"
test "$(df --output=avail /tmp | tail -1)" -ge 83886080
for path in checkpoint_completion.json adapter/adapter_model.safetensors \
  adapter/adapter_config.json recurrent_state.pt; do
  test -f "$CHECKPOINT/$path"
done
test -f "$RAW_EVAL"
test -f "$C4_DATA"
test -f "$PAIRS"
test "$(sha256sum "$RAW_EVAL" | cut -d' ' -f1)" = bff7726953ae048bedf6dacdbb7500a54cb8de305e406f2497ef1f11c04167f8
test "$(sha256sum "$PAIRS" | cut -d' ' -f1)" = edcdb934a65252557f6c569eea9ef7a85e5445549deac9fb4829c3f579e1a9b8
test "$(sha256sum scripts/evaluate_recurrent_formal_protocols.py | cut -d' ' -f1)" = 262370592e018641b99d6282858d8990b38314c05bb6e3bbc645823834cf73ed
test "$(sha256sum scripts/evaluate_recurrent_formal_protocols_taxonomy.py | cut -d' ' -f1)" = 648baa998672f30652ad3e68f114c2648c5bf839dcddd5b1e0acae47400e5b22
test "$(sha256sum scripts/evaluate_recurrent_diagnosis_c4.py | cut -d' ' -f1)" = e7421a850c2327d8df1436073742e71f796bec41c87f68403b7b00b60fac4664
test "$(sha256sum scripts/evaluate_recurrent_diagnosis_c4_taxonomy.py | cut -d' ' -f1)" = c3d34960d0f6caff3265b772686bd84222637f628d5fce000215d076c5b2a355
test "$(sha256sum scripts/state8_taxonomy_contract.py | cut -d' ' -f1)" = 1ca32112e76c795502f3cb1edbedcc7220f093c133fae8d20ff5cab789e512ed

python - "$CHECKPOINT" "$PROBE_ROUND" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); expected_round=int(sys.argv[2])
receipt=json.load(open(root/'checkpoint_completion.json'))
def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()
observed={'adapter_model':digest(root/'adapter/adapter_model.safetensors'),
          'recurrent_state':digest(root/'recurrent_state.pt')}
assert receipt['status']=='complete' and receipt['round']==expected_round,receipt
assert receipt['world_size']==4,receipt
assert observed==receipt['sha256'],(observed,receipt)
PY

python -m pytest -q tests/test_evaluate_recurrent_formal_protocols.py \
  tests/test_evaluate_recurrent_diagnosis_c4.py tests/test_state8_taxonomy_contract.py
python scripts/build_recurrent_evaluation_manifest.py \
  --input-jsonl "$RAW_EVAL" --output-jsonl "$RECURRENT_EVAL" \
  --audit-json "$(dirname "$RECURRENT_EVAL")/audit.json" \
  --max-updates 4 --seed 20260730
test "$(sha256sum "$RECURRENT_EVAL" | cut -d' ' -f1)" = a368cefb0c49cb6a305ad1a892808361fba2d44d555a0a28ea64e56babe32525

rm -rf "$LOCAL_MODEL"; mkdir -p "$LOCAL_MODEL"
python scripts/copy_model_parallel.py \
  --source /mnt/amulet/studentmodeling/models/Qwen3-32B \
  --destination "$LOCAL_MODEL" --workers 16
test -f "$LOCAL_MODEL/model.safetensors.index.json"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

python -u scripts/evaluate_recurrent_formal_protocols_taxonomy.py \
  --model "$LOCAL_MODEL" --adapter "$CHECKPOINT/adapter" \
  --recurrent-state "$CHECKPOINT/recurrent_state.pt" \
  --state-architecture layerwise --eval-jsonl "$RECURRENT_EVAL" \
  --output-dir "$T3_OUT" --state-token-alias '<|vision_pad|>' \
  --max-samples-per-group 25 --max-length 1536 --max-new-tokens 128 \
  --state-batch-size 2 --candidate-row-batch-size 2 --seed 42 \
  --tasks diagnose_misconception --sources eedi_mining malrule \
  --save-unredacted-predictions --t3-natural-schema

python -u scripts/evaluate_recurrent_diagnosis_c4_taxonomy.py \
  --model "$LOCAL_MODEL" --adapter "$CHECKPOINT/adapter" \
  --recurrent-state "$CHECKPOINT/recurrent_state.pt" \
  --state-architecture layerwise --jsonl "$C4_DATA" --pair-manifest "$PAIRS" \
  --output-dir "$C4_OUT" --state-token-alias '<|vision_pad|>' \
  --max-length 1536 --max-new-tokens 160 --num-pairs 10 --seed 42

python - "$T3_OUT/metrics.json" "$C4_OUT/metrics.json" "$CHECKPOINT/checkpoint_completion.json" <<'PY'
import json,sys
t3,c4,receipt=map(lambda p:json.load(open(p)),sys.argv[1:])
expected=receipt['sha256']
assert t3['state_architecture']=='layerwise' and t3['state_shape']==[64,8,5120],t3
assert t3['raw_history_visible_to_decoder'] is False,t3
assert t3['persistent_history_kv_cache'] is False and t3['sampled_rows']==75,t3
assert t3['artifact_sha256']==expected,(t3,expected)
assert c4['state_architecture']=='layerwise' and c4['state_shape']==[64,8,5120],c4
assert c4['raw_history_visible_to_decoder'] is False,c4
assert c4['persistent_history_kv_cache'] is False and c4['num_pairs']==10,c4
assert c4['conditions']==['native','noop','samemal','random','swap'],c4
assert c4['artifact_sha256']==expected,(c4,expected)
PY
cp "$CHECKPOINT/checkpoint_completion.json" "$AMLT_OUTPUT_DIR/checkpoint_completion.json"
cp "$(dirname "$RECURRENT_EVAL")/audit.json" "$AMLT_OUTPUT_DIR/manifest_audit.json"
