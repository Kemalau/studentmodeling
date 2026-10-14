#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "STATE8_EXACT_MIDPOINT_ANSWER_C4_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

export VENV_ROOT="${VENV_ROOT:-/tmp/studentmodeling-state8-exact-c4-resilient-venv}"
export PATH="$VENV_ROOT/bin:$PATH"
python(){ "$VENV_ROOT/bin/python" "$@"; }
export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/hf_cache
export HF_HUB_CACHE=/tmp/hf_cache/hub
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error

: "${STATE8_EXACT_CHECKPOINT_ROOT:?STATE8_EXACT_CHECKPOINT_ROOT is required}"
: "${STATE8_EXACT_ADAPTER_SHA256:?STATE8_EXACT_ADAPTER_SHA256 is required}"
: "${STATE8_EXACT_STATE_SHA256:?STATE8_EXACT_STATE_SHA256 is required}"
: "${STATE8_EXACT_ANSWER_PROGRESS_DIR:?STATE8_EXACT_ANSWER_PROGRESS_DIR is required}"
: "${STATE8_EXACT_ANSWER_PROGRESS_KEY:?STATE8_EXACT_ANSWER_PROGRESS_KEY is required}"
: "${STATE8_EXACT_ANSWER_C4_SHA256:?STATE8_EXACT_ANSWER_C4_SHA256 is required}"

EXPECTED_ROOT=/mnt/amulet/studentmodeling/checkpoints/state8_qualified_teacher_curriculum_seed42_exact5000_rounds1250_kd1_v2_20260805_taxonomy_semantic_v1_exact_diagnosis_first_seed64_single_ga4/checkpoint-2000
EXPECTED_ADAPTER=a80766e8dae4881fc8b4fb8dcebd14b4e95d059bf3246b120fd050c856c4d587
EXPECTED_STATE=9088dbc32437b1feb5b27bf1a4dd6502bff99b15d608c10977feb6f5cef10426
EXPECTED_PROGRESS=/mnt/amulet/studentmodeling/eval_progress/state8_exact_diagnosis_midpoint_step2000_answer_c4_a80766e8_9088dbc3
EXPECTED_KEY=state8-exact-midpoint-step2000-answer-c4-a80766e8-9088dbc3-pairs50-seed42-v1
test "$STATE8_EXACT_CHECKPOINT_ROOT" = "$EXPECTED_ROOT"
test "$STATE8_EXACT_ADAPTER_SHA256" = "$EXPECTED_ADAPTER"
test "$STATE8_EXACT_STATE_SHA256" = "$EXPECTED_STATE"
test "$STATE8_EXACT_ANSWER_PROGRESS_DIR" = "$EXPECTED_PROGRESS"
test "$STATE8_EXACT_ANSWER_PROGRESS_KEY" = "$EXPECTED_KEY"
test "$(sha256sum scripts/evaluate_recurrent_causal_gate.py | cut -d' ' -f1)" = "$STATE8_EXACT_ANSWER_C4_SHA256"

MODEL_REMOTE=/mnt/amulet/studentmodeling/models/Qwen3-32B
MODEL_LOCAL=/tmp/Qwen3-32B
ADAPTER="$STATE8_EXACT_CHECKPOINT_ROOT/adapter"
STATE="$STATE8_EXACT_CHECKPOINT_ROOT/recurrent_state.pt"
RECEIPT="$STATE8_EXACT_CHECKPOINT_ROOT/checkpoint_completion.json"
DATA=/mnt/amulet/studentmodeling/c4_fullopen_20260724/malrule_c1_clean/test_same_template.jsonl
PAIRS=manifests/c4_stage_a_gate_pairs_seed42_n150_v1.json
OUT="$AMLT_OUTPUT_DIR/eval/answer_c4_50"

mkdir -p "$HF_HUB_CACHE" "$MODEL_LOCAL" "$OUT" "$STATE8_EXACT_ANSWER_PROGRESS_DIR"
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
python -u scripts/evaluate_recurrent_causal_gate.py \
  --model "$MODEL_LOCAL" --adapter "$ADAPTER" --recurrent-state "$STATE" \
  --state-architecture layerwise --jsonl "$DATA" --pair-manifest "$PAIRS" \
  --output-json "$OUT/recurrent_causal_gate.json" --state-token-alias '<|vision_pad|>' \
  --max-length 1536 --num-pairs 50 --allow-manifest-prefix --seed 42 \
  --progress-dir "$STATE8_EXACT_ANSWER_PROGRESS_DIR" \
  --progress-key "$STATE8_EXACT_ANSWER_PROGRESS_KEY"

python - "$OUT/recurrent_causal_gate.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['state_shape']==[64,8,5120] and x['summary']['all_pairs_n']==50,x
assert x['raw_history_visible_to_decoder'] is False,x
assert x['persistent_history_kv_cache'] is False,x
assert x['resumable_progress'] is True,x
assert x['artifact_sha256']=={
 'adapter_model':'a80766e8dae4881fc8b4fb8dcebd14b4e95d059bf3246b120fd050c856c4d587',
 'recurrent_state':'9088dbc32437b1feb5b27bf1a4dd6502bff99b15d608c10977feb6f5cef10426'},x
PY
