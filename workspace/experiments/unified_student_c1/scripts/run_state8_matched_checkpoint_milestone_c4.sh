#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "STATE8_MATCHED_MILESTONE_C4_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

export VENV_ROOT="${VENV_ROOT:-/tmp/studentmodeling-state8-matched-milestone-screen-venv}"
export PATH="$VENV_ROOT/bin:$PATH"
python(){ "$VENV_ROOT/bin/python" "$@"; }
export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/hf_cache HF_HUB_CACHE=/tmp/hf_cache/hub HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRANSFORMERS_VERBOSITY=error
: "${STATE8_MATCHED_CHECKPOINT_ROOT:?}" "${STATE8_MATCHED_CHECKPOINT_STEP:?}"
: "${STATE8_MATCHED_ADAPTER_SHA256:?}" "${STATE8_MATCHED_STATE_SHA256:?}"
: "${MATCHED_CHECKPOINT_RECEIPT_SHA256:?}"

MODEL_REMOTE=/mnt/amulet/studentmodeling/models/Qwen3-32B
MODEL_LOCAL=/tmp/Qwen3-32B
DATA=/mnt/amulet/studentmodeling/c4_fullopen_20260724/malrule_c1_clean/test_same_template.jsonl
PARENT_PAIRS=manifests/c4_stage_a_gate_pairs_seed42_n150_v1.json
ANSWER_PAIRS=/tmp/state8_matched_milestone_answer_pairs_n50.json
DIAG_PAIRS=/tmp/state8_matched_milestone_diagnosis_pairs_n20.json
ANSWER_OUT="$AMLT_OUTPUT_DIR/eval/answer_c4_50"
DIAG_OUT="$AMLT_OUTPUT_DIR/eval/diagnosis_c4_20"
PROGRESS=/mnt/amulet/studentmodeling/eval_progress/matched_checkpoint${STATE8_MATCHED_CHECKPOINT_STEP}/answer_c4_seed42
mkdir -p "$MODEL_LOCAL" "$ANSWER_OUT" "$DIAG_OUT" "$PROGRESS" "$AMLT_OUTPUT_DIR/metrics"

test "$(sha256sum "$PARENT_PAIRS" | cut -d' ' -f1)" = edcdb934a65252557f6c569eea9ef7a85e5445549deac9fb4829c3f579e1a9b8
test "$(sha256sum scripts/evaluate_recurrent_causal_gate.py | cut -d' ' -f1)" = 69fd4f2f4943a78a0afe9a701533d9ab034f247f87c7a09327f7523e310b7a52
test "$(sha256sum scripts/evaluate_recurrent_diagnosis_c4_diagnosis_first.py | cut -d' ' -f1)" = 660df80664c6012f6eac1d379cc1d611c9f085c91496d1c9cc8e7f7bcb30d7ca
python scripts/audit_state8_matched_checkpoint.py --checkpoint-root "$STATE8_MATCHED_CHECKPOINT_ROOT" \
  --step "$STATE8_MATCHED_CHECKPOINT_STEP" --adapter-sha256 "$STATE8_MATCHED_ADAPTER_SHA256" \
  --state-sha256 "$STATE8_MATCHED_STATE_SHA256" --receipt-sha256 "$MATCHED_CHECKPOINT_RECEIPT_SHA256" \
  --output "$AMLT_OUTPUT_DIR/metrics/artifact_audit.json"

python - "$PARENT_PAIRS" "$ANSWER_PAIRS" "$DIAG_PAIRS" <<'PY'
import hashlib,json,pathlib,sys
source,answer,diagnosis=map(pathlib.Path,sys.argv[1:])
value=json.loads(source.read_text()); assert value['num_pairs']==150 and len(value['pairs'])==150,value
sha=hashlib.sha256(source.read_bytes()).hexdigest()
for path,count in ((answer,50),(diagnosis,20)):
 out={**value,'num_pairs':count,'pairs':value['pairs'][:count],'diagnostic_only':True,
      'selection':'ordered_prefix','parent_manifest_sha256':sha}
 path.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
PY

rm -rf "$MODEL_LOCAL"; mkdir -p "$MODEL_LOCAL"
python scripts/copy_model_parallel.py --source "$MODEL_REMOTE" --destination "$MODEL_LOCAL" --workers 16
python -u scripts/evaluate_recurrent_causal_gate.py --model "$MODEL_LOCAL" \
  --adapter "$STATE8_MATCHED_CHECKPOINT_ROOT/adapter" \
  --recurrent-state "$STATE8_MATCHED_CHECKPOINT_ROOT/recurrent_state.pt" \
  --state-architecture layerwise --jsonl "$DATA" --pair-manifest "$ANSWER_PAIRS" \
  --output-json "$ANSWER_OUT/recurrent_causal_gate.json" --state-token-alias '<|vision_pad|>' \
  --max-length 1536 --num-pairs 50 --seed 42 --progress-dir "$PROGRESS" \
  --progress-key "step=${STATE8_MATCHED_CHECKPOINT_STEP}:adapter=${STATE8_MATCHED_ADAPTER_SHA256}:state=${STATE8_MATCHED_STATE_SHA256}:n50:v1"
python -u scripts/evaluate_recurrent_diagnosis_c4_diagnosis_first.py --model "$MODEL_LOCAL" \
  --adapter "$STATE8_MATCHED_CHECKPOINT_ROOT/adapter" \
  --recurrent-state "$STATE8_MATCHED_CHECKPOINT_ROOT/recurrent_state.pt" \
  --state-architecture layerwise --jsonl "$DATA" --pair-manifest "$DIAG_PAIRS" \
  --output-dir "$DIAG_OUT" --state-token-alias '<|vision_pad|>' --max-length 1536 \
  --max-new-tokens 160 --num-pairs 20 --seed "$STATE8_MATCHED_CHECKPOINT_STEP" \
  --expected-adapter-sha256 "$STATE8_MATCHED_ADAPTER_SHA256" \
  --expected-recurrent-state-sha256 "$STATE8_MATCHED_STATE_SHA256"
python - "$ANSWER_OUT/recurrent_causal_gate.json" "$DIAG_OUT/metrics.json" \
  "$STATE8_MATCHED_ADAPTER_SHA256" "$STATE8_MATCHED_STATE_SHA256" <<'PY'
import json,sys
a=json.load(open(sys.argv[1])); d=json.load(open(sys.argv[2]))
expected={'adapter_model':sys.argv[3],'recurrent_state':sys.argv[4]}
assert a['artifact_sha256']==expected and d['artifact_sha256']==expected,(a,d)
assert a['summary']['all_pairs_n']==50 and d['num_pairs']==20,(a,d)
assert a['state_shape']==[64,8,5120] and d['state_shape']==[64,8,5120],(a,d)
assert a['raw_history_visible_to_decoder'] is False and d['raw_history_visible_to_decoder'] is False,(a,d)
assert a['persistent_history_kv_cache'] is False and d['persistent_history_kv_cache'] is False,(a,d)
assert d['conditions']==['native','noop','samemal','random','swap'] and d['semantic_judgment_pending'] is True,d
PY
