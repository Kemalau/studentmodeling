#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "STATE8_MATCHED_CHECKPOINT_ANSWER_C4_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR
export VENV_ROOT="${VENV_ROOT:-/tmp/studentmodeling-state8-matched-checkpoint-screen-venv}"
export PATH="$VENV_ROOT/bin:$PATH"; python(){ "$VENV_ROOT/bin/python" "$@"; }
export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/hf_cache HF_HUB_CACHE=/tmp/hf_cache/hub HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRANSFORMERS_VERBOSITY=error
: "${STATE8_MATCHED_CHECKPOINT_ROOT:?}" "${STATE8_MATCHED_CHECKPOINT_STEP:?}"
: "${STATE8_MATCHED_ADAPTER_SHA256:?}" "${STATE8_MATCHED_STATE_SHA256:?}"
: "${MATCHED_CHECKPOINT_RECEIPT_SHA256:?}"
MODEL_REMOTE=/mnt/amulet/studentmodeling/models/Qwen3-32B; MODEL_LOCAL=/tmp/Qwen3-32B
DATA=/mnt/amulet/studentmodeling/c4_fullopen_20260724/malrule_c1_clean/test_same_template.jsonl
PARENT=manifests/c4_stage_a_gate_pairs_seed42_n150_v1.json; PAIRS=/tmp/matched_pairs_n50.json
OUT="$AMLT_OUTPUT_DIR/eval/answer_c4_50"; PROGRESS=/mnt/amulet/studentmodeling/eval_progress/matched_checkpoint${STATE8_MATCHED_CHECKPOINT_STEP}/answer_c4_seed42
mkdir -p "$MODEL_LOCAL" "$OUT" "$PROGRESS" "$AMLT_OUTPUT_DIR/metrics"
python scripts/audit_state8_matched_checkpoint.py --checkpoint-root "$STATE8_MATCHED_CHECKPOINT_ROOT" \
  --step "$STATE8_MATCHED_CHECKPOINT_STEP" --adapter-sha256 "$STATE8_MATCHED_ADAPTER_SHA256" \
  --state-sha256 "$STATE8_MATCHED_STATE_SHA256" \
  --receipt-sha256 "$MATCHED_CHECKPOINT_RECEIPT_SHA256" \
  --output "$AMLT_OUTPUT_DIR/metrics/artifact_audit.json"
python - "$PARENT" "$PAIRS" <<'PY'
import hashlib,json,pathlib,sys
s,p=map(pathlib.Path,sys.argv[1:]); x=json.loads(s.read_text()); assert x['num_pairs']==150
o={**x,'num_pairs':50,'pairs':x['pairs'][:50],'selection':'ordered_prefix','diagnostic_only':True,
   'parent_manifest_sha256':hashlib.sha256(s.read_bytes()).hexdigest()}
p.write_text(json.dumps(o,indent=2,sort_keys=True)+'\n')
PY
rm -rf "$MODEL_LOCAL"; mkdir -p "$MODEL_LOCAL"
python scripts/copy_model_parallel.py --source "$MODEL_REMOTE" --destination "$MODEL_LOCAL" --workers 16
python -u scripts/evaluate_recurrent_causal_gate.py --model "$MODEL_LOCAL" \
  --adapter "$STATE8_MATCHED_CHECKPOINT_ROOT/adapter" \
  --recurrent-state "$STATE8_MATCHED_CHECKPOINT_ROOT/recurrent_state.pt" \
  --state-architecture layerwise --jsonl "$DATA" --pair-manifest "$PAIRS" \
  --output-json "$OUT/recurrent_causal_gate.json" --state-token-alias '<|vision_pad|>' \
  --max-length 1536 --num-pairs 50 --seed 42 --progress-dir "$PROGRESS" \
  --progress-key "step=${STATE8_MATCHED_CHECKPOINT_STEP}:adapter=${STATE8_MATCHED_ADAPTER_SHA256}:state=${STATE8_MATCHED_STATE_SHA256}:n50"
python - "$OUT/recurrent_causal_gate.json" "$STATE8_MATCHED_ADAPTER_SHA256" "$STATE8_MATCHED_STATE_SHA256" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); expected={'adapter_model':sys.argv[2],'recurrent_state':sys.argv[3]}
assert x['artifact_sha256']==expected and x['state_shape']==[64,8,5120],x
assert x['summary']['all_pairs_n']==50 and x['raw_history_visible_to_decoder'] is False,x
PY
