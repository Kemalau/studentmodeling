#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "STATE8_SOURCE_ANCHOR_ANSWER_C4_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

export VENV_ROOT="${VENV_ROOT:-/tmp/studentmodeling-state8-source-anchor-answer-c4-venv}"
export PATH="$VENV_ROOT/bin:$PATH"
python(){ "$VENV_ROOT/bin/python" "$@"; }
export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/hf_cache HF_HUB_CACHE=/tmp/hf_cache/hub HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRANSFORMERS_VERBOSITY=error

: "${SOURCE_ANCHOR_ARM:?}" "${SOURCE_ANCHOR_JOB:?}"
: "${SOURCE_ANCHOR_ADAPTER_SHA256:?}" "${SOURCE_ANCHOR_STATE_SHA256:?}"
case "$SOURCE_ANCHOR_ARM" in full_mse1|updater_mse1|full_mse2) ;; *) exit 2 ;; esac
case "$SOURCE_ANCHOR_JOB" in
  state8-source-anchor-full-mse1-seed3071|state8-source-anchor-updater-mse1-seed3072|state8-source-anchor-full-mse2-seed3073) ;;
  *) exit 2 ;;
esac

ROOT="/mnt/amulet/studentmodeling/diagnostics/state8_stateonly_repair_seed"
case "$SOURCE_ANCHOR_ARM" in
  full_mse1) REMOTE="$ROOT"'3071_taskanchor_source_anchor_full_mse1_v1' ;;
  updater_mse1) REMOTE="$ROOT"'3072_taskanchor_source_anchor_updater_mse1_v1' ;;
  full_mse2) REMOTE="$ROOT"'3073_taskanchor_source_anchor_full_mse2_v1' ;;
esac
MODEL_REMOTE=/mnt/amulet/studentmodeling/models/Qwen3-32B
MODEL_LOCAL=/tmp/Qwen3-32B
C4_DATA=/mnt/amulet/studentmodeling/c4_fullopen_20260724/malrule_c1_clean/test_same_template.jsonl
PAIRS=manifests/c4_stage_a_gate_pairs_seed42_n150_v1.json
OUT="$AMLT_OUTPUT_DIR/eval/answer_c4_seed42_n150"
PROGRESS="/mnt/amulet/studentmodeling/eval_progress/source_anchor_${SOURCE_ANCHOR_ARM}/answer_c4_seed42_n150_v1"
mkdir -p "$MODEL_LOCAL" "$OUT" "$PROGRESS" "$AMLT_OUTPUT_DIR/metrics"

test "$(sha256sum scripts/evaluate_recurrent_causal_gate.py | cut -d' ' -f1)" = 69fd4f2f4943a78a0afe9a701533d9ab034f247f87c7a09327f7523e310b7a52
test "$(sha256sum scripts/copy_model_parallel.py | cut -d' ' -f1)" = 62b37ab70bc2ffc81c8314ca69215da7facdf18dba3f023d83096d1c3526ea4c
test "$(sha256sum "$PAIRS" | cut -d' ' -f1)" = edcdb934a65252557f6c569eea9ef7a85e5445549deac9fb4829c3f579e1a9b8
for path in "$REMOTE/model/adapter_model.safetensors" "$REMOTE/model/recurrent_state.pt" \
  "$REMOTE/metrics/publish_receipt.json"; do test -f "$path"; done
test "$(sha256sum "$REMOTE/model/adapter_model.safetensors" | cut -d' ' -f1)" = "$SOURCE_ANCHOR_ADAPTER_SHA256"
test "$(sha256sum "$REMOTE/model/recurrent_state.pt" | cut -d' ' -f1)" = "$SOURCE_ANCHOR_STATE_SHA256"
python - "$REMOTE/metrics/publish_receipt.json" "$SOURCE_ANCHOR_ADAPTER_SHA256" "$SOURCE_ANCHOR_STATE_SHA256" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); expected={'adapter_model':sys.argv[2],'recurrent_state':sys.argv[3]}
assert x['status']=='complete' and x['state_shape']==[64,8,5120],x
assert x['raw_history_visible_to_decoder'] is False and x['history_dropout']==1.0,x
assert x['artifact_sha256']==expected,x
PY

rm -rf "$MODEL_LOCAL"; mkdir -p "$MODEL_LOCAL"
python scripts/copy_model_parallel.py --source "$MODEL_REMOTE" --destination "$MODEL_LOCAL" --workers 16
python -u scripts/evaluate_recurrent_causal_gate.py \
  --model "$MODEL_LOCAL" --adapter "$REMOTE/model" \
  --recurrent-state "$REMOTE/model/recurrent_state.pt" --state-architecture layerwise \
  --jsonl "$C4_DATA" --pair-manifest "$PAIRS" \
  --output-json "$OUT/recurrent_causal_gate.json" \
  --state-token-alias '<|vision_pad|>' --max-length 1536 --num-pairs 150 \
  --seed 42 --progress-dir "$PROGRESS" \
  --progress-key "arm=$SOURCE_ANCHOR_ARM:adapter=$SOURCE_ANCHOR_ADAPTER_SHA256:state=$SOURCE_ANCHOR_STATE_SHA256:n150:v1"
python - "$OUT/recurrent_causal_gate.json" "$SOURCE_ANCHOR_ADAPTER_SHA256" "$SOURCE_ANCHOR_STATE_SHA256" <<'PY'
import json,pathlib,sys
x=json.load(open(sys.argv[1])); expected={'adapter_model':sys.argv[2],'recurrent_state':sys.argv[3]}
assert x['artifact_sha256']==expected and x['state_shape']==[64,8,5120],x
assert x['summary']['all_pairs_n']==150,x
assert x['raw_history_visible_to_decoder'] is False,x
assert x['persistent_history_kv_cache'] is False,x
receipt={'stage':'state8_source_anchor_answer_c4_confirmation_v1','status':'complete',
 'pair_seed':42,'num_pairs':150,'semantic_judge_not_applicable':True,
 'exact_match_not_used_for_diagnosis':True,'artifact_sha256':expected,
 'raw_history_visible_to_decoder':False,'persistent_history_kv_cache':False,
 'summary':x['summary']}
path=pathlib.Path(sys.argv[1]).parents[2]/'metrics'/'confirmation_receipt.json'
path.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
PY
