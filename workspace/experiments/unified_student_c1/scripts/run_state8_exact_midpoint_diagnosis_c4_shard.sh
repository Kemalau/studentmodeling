#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "STATE8_EXACT_MIDPOINT_DIAGNOSIS_C4_SHARD_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

export VENV_ROOT="${VENV_ROOT:-/tmp/studentmodeling-state8-exact-c4-resilient-venv}"
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
: "${STATE8_EXACT_CHECKPOINT_ROOT:?STATE8_EXACT_CHECKPOINT_ROOT is required}"
: "${STATE8_EXACT_ADAPTER_SHA256:?STATE8_EXACT_ADAPTER_SHA256 is required}"
: "${STATE8_EXACT_STATE_SHA256:?STATE8_EXACT_STATE_SHA256 is required}"
: "${EXACT_DIAGNOSIS_C4_SHA256:?EXACT_DIAGNOSIS_C4_SHA256 is required}"
test "$PAIR_COUNT" = 10
case "$PAIR_START" in 0|10|20|30|40) ;; *) exit 2 ;; esac

EXPECTED_ROOT=/mnt/amulet/studentmodeling/checkpoints/state8_qualified_teacher_curriculum_seed42_exact5000_rounds1250_kd1_v2_20260805_taxonomy_semantic_v1_exact_diagnosis_first_seed64_single_ga4/checkpoint-2000
EXPECTED_ADAPTER=a80766e8dae4881fc8b4fb8dcebd14b4e95d059bf3246b120fd050c856c4d587
EXPECTED_STATE=9088dbc32437b1feb5b27bf1a4dd6502bff99b15d608c10977feb6f5cef10426
test "$STATE8_EXACT_CHECKPOINT_ROOT" = "$EXPECTED_ROOT"
test "$STATE8_EXACT_ADAPTER_SHA256" = "$EXPECTED_ADAPTER"
test "$STATE8_EXACT_STATE_SHA256" = "$EXPECTED_STATE"
test "$(sha256sum scripts/evaluate_recurrent_diagnosis_c4_diagnosis_first.py | cut -d' ' -f1)" = "$EXACT_DIAGNOSIS_C4_SHA256"

MODEL_REMOTE=/mnt/amulet/studentmodeling/models/Qwen3-32B
MODEL_LOCAL=/tmp/Qwen3-32B
ADAPTER="$STATE8_EXACT_CHECKPOINT_ROOT/adapter"
STATE="$STATE8_EXACT_CHECKPOINT_ROOT/recurrent_state.pt"
RECEIPT="$STATE8_EXACT_CHECKPOINT_ROOT/checkpoint_completion.json"
DATA=/mnt/amulet/studentmodeling/c4_fullopen_20260724/malrule_c1_clean/test_same_template.jsonl
PAIRS=manifests/c4_stage_a_gate_pairs_seed42_n150_v1.json
OUT="$AMLT_OUTPUT_DIR/eval/diagnosis_c4_shard_${PAIR_START}"

mkdir -p "$HF_HUB_CACHE" "$MODEL_LOCAL" "$OUT"
for path in "$MODEL_REMOTE/model.safetensors.index.json" "$ADAPTER/adapter_model.safetensors" \
  "$STATE" "$RECEIPT" "$DATA" "$PAIRS"; do test -f "$path"; done
test "$(sha256sum "$ADAPTER/adapter_model.safetensors" | cut -d' ' -f1)" = "$EXPECTED_ADAPTER"
test "$(sha256sum "$STATE" | cut -d' ' -f1)" = "$EXPECTED_STATE"
test "$(sha256sum "$PAIRS" | cut -d' ' -f1)" = edcdb934a65252557f6c569eea9ef7a85e5445549deac9fb4829c3f579e1a9b8
test "$(df --output=avail /tmp | tail -1)" -ge 83886080
python - "$RECEIPT" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x=={'sha256':{
 'adapter_model':'a80766e8dae4881fc8b4fb8dcebd14b4e95d059bf3246b120fd050c856c4d587',
 'recurrent_state':'9088dbc32437b1feb5b27bf1a4dd6502bff99b15d608c10977feb6f5cef10426'},
 'status':'complete','step':2000},x
PY

rm -rf "$MODEL_LOCAL"; mkdir -p "$MODEL_LOCAL"
python scripts/copy_model_parallel.py --source "$MODEL_REMOTE" --destination "$MODEL_LOCAL" --workers 16
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
python -u scripts/evaluate_recurrent_diagnosis_c4_diagnosis_first.py \
  --model "$MODEL_LOCAL" --adapter "$ADAPTER" --recurrent-state "$STATE" \
  --state-architecture layerwise --jsonl "$DATA" --pair-manifest "$PAIRS" \
  --output-dir "$OUT" --state-token-alias '<|vision_pad|>' --max-length 1536 \
  --max-new-tokens 160 --num-pairs 50 --pair-start "$PAIR_START" \
  --pair-count "$PAIR_COUNT" --seed 42 \
  --expected-adapter-sha256 "$EXPECTED_ADAPTER" \
  --expected-recurrent-state-sha256 "$EXPECTED_STATE"

python - "$OUT/metrics.json" "$PAIR_START" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); start=int(sys.argv[2])
assert x['num_pairs']==10 and x['manifest_prefix_pairs']==50,x
assert x['pair_start']==start and x['pair_stop']==start+10,x
assert x['state_shape']==[64,8,5120],x
assert x['raw_history_visible_to_decoder'] is False,x
assert x['persistent_history_kv_cache'] is False,x
assert x['artifact_sha256']=={
 'adapter_model':'a80766e8dae4881fc8b4fb8dcebd14b4e95d059bf3246b120fd050c856c4d587',
 'recurrent_state':'9088dbc32437b1feb5b27bf1a4dd6502bff99b15d608c10977feb6f5cef10426'},x
PY
