#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "STATE8_STRONG1_WIDEN8_EVAL_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

export VENV_ROOT="${VENV_ROOT:-/tmp/studentmodeling-state8-strong1-widen8-eval-venv}"
export PATH="$VENV_ROOT/bin:$PATH"
python(){ "$VENV_ROOT/bin/python" "$@"; }
export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/hf_cache
export HF_HUB_CACHE=/tmp/hf_cache/hub
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error

: "${CANDIDATE_SLUG:?CANDIDATE_SLUG is required}"
: "${STATE8_EVAL_ROOT:?STATE8_EVAL_ROOT is required}"
case "$CANDIDATE_SLUG" in
  lr5e7_seed58|lr1e6_seed59|single_lr5e7_seed60) ;;
  *) exit 2 ;;
esac
case "$STATE8_EVAL_ROOT" in
  /mnt/amulet/studentmodeling/diagnostics/state8_qualified_teacher_curriculum_highkd_v1_immutable_taxonomy_semantic_v1_strong1_widen8_*) ;;
  *) exit 2 ;;
esac

MODEL_REMOTE=/mnt/amulet/studentmodeling/models/Qwen3-32B
MODEL_LOCAL=/tmp/Qwen3-32B
ADAPTER="$STATE8_EVAL_ROOT/model"
STATE="$STATE8_EVAL_ROOT/model/recurrent_state.pt"
RECEIPT="$STATE8_EVAL_ROOT/metrics/immutable_receipt.json"
DATA=/mnt/amulet/studentmodeling/c4_fullopen_20260724/malrule_c1_clean/test_same_template.jsonl
PAIRS=manifests/c4_stage_a_gate_pairs_seed42_n150_v1.json
OUT="$AMLT_OUTPUT_DIR/eval"
RAW_OUT="$OUT/answer_c4_seed42_n150"
DIAG_OUT="$OUT/diagnosis_c4_seed42_n50"
PROGRESS="/mnt/amulet/studentmodeling/eval_progress/state8_strong1_widen8_${CANDIDATE_SLUG}_answer_c4_seed42"

mkdir -p "$HF_HUB_CACHE" "$MODEL_LOCAL" "$RAW_OUT" "$DIAG_OUT" "$PROGRESS"
for path in "$MODEL_REMOTE/model.safetensors.index.json" \
  "$ADAPTER/adapter_model.safetensors" "$STATE" "$RECEIPT" "$DATA" "$PAIRS"; do
  test -f "$path"
done
test "$(sha256sum "$PAIRS" | cut -d' ' -f1)" = edcdb934a65252557f6c569eea9ef7a85e5445549deac9fb4829c3f579e1a9b8
test "$(df --output=avail /tmp | tail -1)" -ge 83886080

readarray -t ARTIFACT_SHA256 < <(python - "$RECEIPT" "$ADAPTER/adapter_model.safetensors" "$STATE" <<'PY'
import hashlib,json,pathlib,sys
receipt,adapter,state=map(pathlib.Path,sys.argv[1:])
x=json.load(receipt.open())
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
 return h.hexdigest()
assert x['status']=='complete' and x['state_shape']==[64,8,5120],x
assert x['raw_history_visible_to_decoder'] is False,x
assert x['persistent_history_kv_cache'] is False,x
assert x['semantic_judge_authoritative'] is True,x
assert x['exact_label_metric_authoritative'] is False,x
assert x['taxonomy_label_contract']=='source_aware_no_answer_labels_v1',x
observed={'adapter_model':digest(adapter),'recurrent_state':digest(state)}
assert observed==x['sha256'],(observed,x)
print(observed['adapter_model']); print(observed['recurrent_state'])
PY
)
test "${#ARTIFACT_SHA256[@]}" = 2
export EXPECTED_ADAPTER_SHA256="${ARTIFACT_SHA256[0]}"
export EXPECTED_RECURRENT_STATE_SHA256="${ARTIFACT_SHA256[1]}"
export PROGRESS_KEY="candidate=$CANDIDATE_SLUG:adapter=$EXPECTED_ADAPTER_SHA256:state=$EXPECTED_RECURRENT_STATE_SHA256"

python -m pytest -q tests/test_evaluate_recurrent_causal_gate.py \
  tests/test_evaluate_recurrent_diagnosis_c4.py
rm -rf "$MODEL_LOCAL"; mkdir -p "$MODEL_LOCAL"
python scripts/copy_model_parallel.py \
  --source "$MODEL_REMOTE" --destination "$MODEL_LOCAL" --workers 16
test -f "$MODEL_LOCAL/model.safetensors.index.json"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

python -u scripts/evaluate_recurrent_causal_gate.py \
  --model "$MODEL_LOCAL" --adapter "$ADAPTER" --recurrent-state "$STATE" \
  --state-architecture layerwise --jsonl "$DATA" --pair-manifest "$PAIRS" \
  --output-json "$RAW_OUT/recurrent_causal_gate.json" \
  --state-token-alias '<|vision_pad|>' --max-length 1536 --num-pairs 150 \
  --seed 42 --progress-dir "$PROGRESS" --progress-key "$PROGRESS_KEY"

python -u scripts/evaluate_recurrent_diagnosis_c4_taxonomy.py \
  --model "$MODEL_LOCAL" --adapter "$ADAPTER" --recurrent-state "$STATE" \
  --state-architecture layerwise --jsonl "$DATA" --pair-manifest "$PAIRS" \
  --output-dir "$DIAG_OUT" --state-token-alias '<|vision_pad|>' \
  --max-length 1536 --max-new-tokens 160 --num-pairs 50 --seed 42 \
  --expected-adapter-sha256 "$EXPECTED_ADAPTER_SHA256" \
  --expected-recurrent-state-sha256 "$EXPECTED_RECURRENT_STATE_SHA256"

python - "$RAW_OUT/recurrent_causal_gate.json" "$DIAG_OUT/metrics.json" "$CANDIDATE_SLUG" <<'PY'
import json,sys
raw=json.load(open(sys.argv[1])); diag=json.load(open(sys.argv[2])); slug=sys.argv[3]
assert raw['pair_seed']==42 and raw['state_shape']==[64,8,5120],raw
assert raw['raw_history_visible_to_decoder'] is False,raw
assert raw['persistent_history_kv_cache'] is False,raw
assert raw['summary']['all_pairs_n']==150 and raw['summary']['clean_n']>0,raw
assert diag['num_pairs']==50 and diag['pair_seed']==42,diag
assert diag['state_shape']==[64,8,5120],diag
assert diag['raw_history_visible_to_decoder'] is False,diag
assert diag['persistent_history_kv_cache'] is False,diag
assert diag['semantic_judgment_pending'] is True,diag
assert slug in {'lr5e7_seed58','lr1e6_seed59','single_lr5e7_seed60'}
PY
