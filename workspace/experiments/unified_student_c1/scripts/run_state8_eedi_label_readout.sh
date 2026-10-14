#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "STATE8_EEDI_LABEL_READOUT_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

: "${CHECKPOINT_ROOT:?}" "${CHECKPOINT_ADAPTER_SHA256:?}"
: "${CHECKPOINT_STATE_SHA256:?}" "${READOUT_TAG:?}"
[[ "$CHECKPOINT_ADAPTER_SHA256" =~ ^[0-9a-f]{64}$ ]] || exit 2
[[ "$CHECKPOINT_STATE_SHA256" =~ ^[0-9a-f]{64}$ ]] || exit 2
[[ "$READOUT_TAG" =~ ^[a-z0-9_-]+$ ]] || exit 2
case "$CHECKPOINT_ROOT" in
  /mnt/amulet/studentmodeling/checkpoint_snapshots/state8_curriculum_repairs_20260812/seed*_*/checkpoint-*) ;;
  *) echo "Unapproved checkpoint root: $CHECKPOINT_ROOT" >&2; exit 2 ;;
esac

export VENV_ROOT="${VENV_ROOT:-/tmp/studentmodeling-state8-label-readout-venv}"
export PATH="$VENV_ROOT/bin:$PATH"
python(){ "$VENV_ROOT/bin/python" "$@"; }
export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/hf_cache HF_HUB_CACHE=/tmp/hf_cache/hub HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRANSFORMERS_VERBOSITY=error

MODEL_REMOTE=/mnt/amulet/studentmodeling/models/Qwen3-32B
MODEL_LOCAL=/tmp/Qwen3-32B
PANEL=manifests/state8_eedi_same_question_c4_valid_seed20260811_offset100_n25_v1.jsonl
PANEL_SHA256=2ba0b47bbdc99188c32542aaaf61a5d1493d2dc805bb44855eadad7d40e8effb
OUT="$AMLT_OUTPUT_DIR/eval/eedi_label_readout_25"
mkdir -p "$MODEL_LOCAL" "$OUT"

for path in "$CHECKPOINT_ROOT/adapter/adapter_model.safetensors" \
  "$CHECKPOINT_ROOT/recurrent_state.pt" "$CHECKPOINT_ROOT/checkpoint_completion.json" \
  "$PANEL"; do
  test -f "$path"
done
test "$(sha256sum "$CHECKPOINT_ROOT/adapter/adapter_model.safetensors" | cut -d' ' -f1)" = "$CHECKPOINT_ADAPTER_SHA256"
test "$(sha256sum "$CHECKPOINT_ROOT/recurrent_state.pt" | cut -d' ' -f1)" = "$CHECKPOINT_STATE_SHA256"
test "$(sha256sum "$PANEL" | cut -d' ' -f1)" = "$PANEL_SHA256"
test "$(sha256sum scripts/evaluate_state8_eedi_label_readout_c4.py | cut -d' ' -f1)" = 187bb388511c75348699d6c29d801a9e969467af2d5b2d059238c563ab9d6e6d

python - "$CHECKPOINT_ROOT/checkpoint_completion.json" "$CHECKPOINT_ADAPTER_SHA256" \
  "$CHECKPOINT_STATE_SHA256" <<'PY'
import json,sys
receipt=json.load(open(sys.argv[1]))
assert receipt['status']=='complete',receipt
assert receipt['sha256']=={'adapter_model':sys.argv[2],'recurrent_state':sys.argv[3]},receipt
assert int(receipt['step']) in {20,100,240,500},receipt
PY

rm -rf "$MODEL_LOCAL"; mkdir -p "$MODEL_LOCAL"
python scripts/copy_model_parallel.py --source "$MODEL_REMOTE" \
  --destination "$MODEL_LOCAL" --workers 16
test -f "$MODEL_LOCAL/model.safetensors.index.json"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

python -u scripts/evaluate_state8_eedi_label_readout_c4.py \
  --model "$MODEL_LOCAL" --adapter "$CHECKPOINT_ROOT/adapter" \
  --recurrent-state "$CHECKPOINT_ROOT/recurrent_state.pt" \
  --causal-jsonl "$PANEL" --output-dir "$OUT" \
  --state-architecture layerwise --state-token-alias '<|vision_pad|>' \
  --max-length 1536 --max-new-tokens 24 --num-pairs 25 --seed 20260811 \
  --expected-adapter-sha256 "$CHECKPOINT_ADAPTER_SHA256" \
  --expected-recurrent-state-sha256 "$CHECKPOINT_STATE_SHA256"

python - "$OUT/metrics.json" "$READOUT_TAG" "$PANEL_SHA256" <<'PY'
import json,sys
value=json.load(open(sys.argv[1]))
assert value['stage']=='state8_eedi_label_readout_c4_v1' and value['status']=='complete',value
assert value['diagnostic_only'] is True,value
assert value['semantic_judge_authoritative'] is True,value
assert value['exact_identifier_authoritative'] is False,value
assert value['num_pairs']==25 and value['state_shape']==[64,8,5120],value
assert value['causal_jsonl_sha256']==sys.argv[3],value
assert value['raw_history_visible_to_decoder'] is False,value
assert value['student_response_visible_to_decoder'] is False,value
assert value['target_candidates_visible_to_decoder'] is False,value
value['readout_tag']=sys.argv[2]
open(sys.argv[1],'w').write(json.dumps(value,indent=2,sort_keys=True)+'\n')
PY
