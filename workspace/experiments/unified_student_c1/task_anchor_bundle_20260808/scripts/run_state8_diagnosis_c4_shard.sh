#!/usr/bin/env bash
set -euo pipefail

export VENV_ROOT="${VENV_ROOT:-/tmp/studentmodeling-state8-c4-shard-venv}"
export PATH="$VENV_ROOT/bin:$PATH"
python(){ "$VENV_ROOT/bin/python" "$@"; }
export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/hf_cache
export HF_HUB_CACHE=/tmp/hf_cache/hub
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error

: "${PAIR_START:?PAIR_START is required}"
: "${PAIR_COUNT:?PAIR_COUNT is required}"
test "$PAIR_COUNT" = 10
case "$PAIR_START" in 0|10|20|30|40) ;; *) exit 2 ;; esac

MODEL_REMOTE=/mnt/amulet/studentmodeling/models/Qwen3-32B
MODEL_LOCAL=/tmp/Qwen3-32B
ADAPTER=/mnt/amulet/studentmodeling/checkpoints/state8_stateonly_repair_seed45_trajectory01_semantic_causal_strongbase_v1/final_adapter
STATE=/mnt/amulet/studentmodeling/checkpoints/state8_stateonly_repair_seed45_trajectory01_semantic_causal_strongbase_v1/recurrent_state.pt
DATA=/mnt/amulet/studentmodeling/c4_fullopen_20260724/malrule_c1_clean/test_same_template.jsonl
PAIRS=manifests/c4_stage_a_gate_pairs_seed42_n150_v1.json
OUT="$AMLT_OUTPUT_DIR/eval/diagnosis_c4_shard_${PAIR_START}"

for path in "$MODEL_REMOTE/model.safetensors.index.json" \
  "$ADAPTER/adapter_model.safetensors" "$STATE" "$DATA" "$PAIRS"; do
  test -f "$path"
done
test "$(sha256sum "$ADAPTER/adapter_model.safetensors" | cut -d' ' -f1)" = 449162955d459e0d60ce3613ca9318ca417eef4986844f37803ac68200a04e69
test "$(sha256sum "$STATE" | cut -d' ' -f1)" = 43c16db184b69bf01bd7d63d8b2b9b05ffda215c932376d600a4dc2e35e5e6ad
test "$(sha256sum "$PAIRS" | cut -d' ' -f1)" = edcdb934a65252557f6c569eea9ef7a85e5445549deac9fb4829c3f579e1a9b8

mkdir -p "$HF_HUB_CACHE" "$MODEL_LOCAL" "$OUT"
python scripts/copy_model_parallel.py \
  --source "$MODEL_REMOTE" --destination "$MODEL_LOCAL" --workers 16
test -f "$MODEL_LOCAL/model.safetensors.index.json"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

python -m pytest -q tests/test_evaluate_recurrent_diagnosis_c4.py \
  tests/test_merge_state8_diagnosis_c4_shards.py
python -u scripts/evaluate_recurrent_diagnosis_c4.py \
  --model "$MODEL_LOCAL" --adapter "$ADAPTER" --recurrent-state "$STATE" \
  --state-architecture layerwise --jsonl "$DATA" --pair-manifest "$PAIRS" \
  --output-dir "$OUT" --state-token-alias '<|vision_pad|>' \
  --max-length 1536 --max-new-tokens 160 --num-pairs 50 \
  --pair-start "$PAIR_START" --pair-count "$PAIR_COUNT" --seed 42 \
  --expected-adapter-sha256 449162955d459e0d60ce3613ca9318ca417eef4986844f37803ac68200a04e69 \
  --expected-recurrent-state-sha256 43c16db184b69bf01bd7d63d8b2b9b05ffda215c932376d600a4dc2e35e5e6ad

python - "$OUT/metrics.json" "$PAIR_START" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); start=int(sys.argv[2])
assert x['num_pairs']==10 and x['manifest_prefix_pairs']==50,x
assert x['pair_start']==start and x['pair_stop']==start+10,x
assert x['raw_history_visible_to_decoder'] is False,x
assert x['artifact_sha256']=={
 'adapter_model':'449162955d459e0d60ce3613ca9318ca417eef4986844f37803ac68200a04e69',
 'recurrent_state':'43c16db184b69bf01bd7d63d8b2b9b05ffda215c932376d600a4dc2e35e5e6ad'},x
PY
