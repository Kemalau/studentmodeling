#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "STATE8_LABEL_FIRST_EVALONLY_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

: "${EVAL_KIND:?}"
case "$EVAL_KIND" in readout|natural_c4) ;; *) exit 2 ;; esac

export VENV_ROOT="${VENV_ROOT:-/tmp/studentmodeling-state8-label-first-evalonly-venv}"
export PATH="$VENV_ROOT/bin:$PATH"
python(){ "$VENV_ROOT/bin/python" "$@"; }
export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/hf_cache HF_HUB_CACHE=/tmp/hf_cache/hub HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRANSFORMERS_VERBOSITY=error
export STATE8_DIAGNOSIS_FIELD_ORDER=label_first

MODEL_REMOTE=/mnt/amulet/studentmodeling/models/Qwen3-32B
MODEL_LOCAL=/tmp/Qwen3-32B
RUN_DIR=/mnt/amulet/studentmodeling/checkpoints/state8_label_first_repair_seed8501_v1
ADAPTER="$RUN_DIR/final_adapter"
RECURRENT_STATE="$RUN_DIR/recurrent_state.pt"
EXPECTED_ADAPTER=ff247c06229c78c8b873f1717d100bfc7f99b2e2b953c7c428228961c8db80df
EXPECTED_STATE=7648c37fc508c0a594b18bfddd848a591756efecf1eef73f125f49497d45890b
PANEL=manifests/state8_eedi_same_question_c4_valid_seed20260811_offset100_n25_v1.jsonl
PANEL_SHA256=2ba0b47bbdc99188c32542aaaf61a5d1493d2dc805bb44855eadad7d40e8effb

mkdir -p "$HF_HUB_CACHE" "$AMLT_OUTPUT_DIR/eval"
for path in "$ADAPTER/adapter_model.safetensors" "$RECURRENT_STATE" \
  "$RUN_DIR/completion.json" "$RUN_DIR/run_config.json" "$PANEL"; do
  test -f "$path"
done
test "$(sha256sum "$ADAPTER/adapter_model.safetensors" | cut -d' ' -f1)" = "$EXPECTED_ADAPTER"
test "$(sha256sum "$RECURRENT_STATE" | cut -d' ' -f1)" = "$EXPECTED_STATE"
test "$(sha256sum "$PANEL" | cut -d' ' -f1)" = "$PANEL_SHA256"
test "$(sha256sum scripts/copy_model_parallel.py | cut -d' ' -f1)" = 62b37ab70bc2ffc81c8314ca69215da7facdf18dba3f023d83096d1c3526ea4c
test "$(sha256sum scripts/evaluate_state8_eedi_label_readout_c4.py | cut -d' ' -f1)" = fb550be116d2587d52d6900cc1075d2ea47a32d5ebe38230f816a4910d46f3b6
test "$(sha256sum scripts/evaluate_state8_eedi_same_question_c4.py | cut -d' ' -f1)" = c4ebe26d6c273298335ae04cb4fc86de075782c1840e2509cb46a5421d3496eb
test "$(df --output=avail /tmp | tail -1)" -ge 83886080

python - "$RUN_DIR" "$EXPECTED_ADAPTER" "$EXPECTED_STATE" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); adapter,state=sys.argv[2:]
c=json.load(open(root/'completion.json')); r=json.load(open(root/'run_config.json'))
assert c['status']=='complete' and c['micro_steps']==500 and c['optimizer_steps']==125,c
assert c['sha256']['adapter_model']==adapter and c['sha256']['recurrent_state']==state,c
assert c['history_integrity_pass'] is True and c['all_causal_pairs_covered'] is True,c
assert r['state_token_count']==8 and r['final_inference_route']=='strict_state_only',r
assert r['diagnosis_field_order']==['label','diagnosis','evidence'],r
assert r['early_label_query_mode']=='state_only' and r['raw_history_visible_to_decoder'] is False,r
PY

python -m pytest -q tests/test_evaluate_state8_eedi_label_readout_c4.py \
  tests/test_evaluate_state8_eedi_same_question_c4.py
rm -rf "$MODEL_LOCAL"; mkdir -p "$MODEL_LOCAL"
python scripts/copy_model_parallel.py --source "$MODEL_REMOTE" \
  --destination "$MODEL_LOCAL" --workers 16
test -f "$MODEL_LOCAL/model.safetensors.index.json"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

if [[ "$EVAL_KIND" == readout ]]; then
  OUT="$AMLT_OUTPUT_DIR/eval/eedi_label_readout_25"
  python -u scripts/evaluate_state8_eedi_label_readout_c4.py \
    --model "$MODEL_LOCAL" --adapter "$ADAPTER" --recurrent-state "$RECURRENT_STATE" \
    --causal-jsonl "$PANEL" --output-dir "$OUT" --state-architecture layerwise \
    --state-token-alias '<|vision_pad|>' --max-length 1536 --max-new-tokens 24 \
    --num-pairs 25 --seed 20260811 --expected-adapter-sha256 "$EXPECTED_ADAPTER" \
    --expected-recurrent-state-sha256 "$EXPECTED_STATE"
  python - "$OUT/metrics.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); assert x['status']=='complete' and x['num_pairs']==25,x
assert x['state_shape']==[64,8,5120] and x['label_readout_query_mode']=='state_only',x
assert x['raw_history_visible_to_decoder'] is False and x['current_item_visible_to_decoder'] is False,x
PY
else
  OUT="$AMLT_OUTPUT_DIR/eval/eedi_natural_c4_25"
  PROGRESS=/mnt/amulet/studentmodeling/eval_progress/state8_label_first_seed8501_evalonly_v1/eedi_natural_c4_25
  mkdir -p "$OUT" "$PROGRESS"
  python -u scripts/evaluate_state8_eedi_same_question_c4.py \
    --model "$MODEL_LOCAL" --adapter "$ADAPTER" --recurrent-state "$RECURRENT_STATE" \
    --state-architecture layerwise --causal-jsonl "$PANEL" --output-dir "$OUT" \
    --state-token-alias '<|vision_pad|>' --max-length 1536 --max-new-tokens 192 \
    --num-pairs 25 --seed 20260811 --progress-dir "$PROGRESS" \
    --progress-key "adapter=$EXPECTED_ADAPTER:state=$EXPECTED_STATE:label-first-natural-c4-v1" \
    --expected-adapter-sha256 "$EXPECTED_ADAPTER" \
    --expected-recurrent-state-sha256 "$EXPECTED_STATE"
  python - "$OUT/metrics.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); assert x['num_pairs']==25 and x['resumable_progress'] is True,x
assert x['state_shape']==[64,8,5120] and x['same_question_query_identical'] is True,x
assert x['raw_history_visible_to_decoder'] is False and x['student_response_visible_to_decoder'] is False,x
PY
fi
