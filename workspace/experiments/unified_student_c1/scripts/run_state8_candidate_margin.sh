#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "STATE8_CANDIDATE_MARGIN_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

: "${AUDIT_MODE:?}" "${ADAPTER_ROOT:?}"
case "$AUDIT_MODE" in student|teacher) ;; *) exit 2 ;; esac
case "$ADAPTER_ROOT" in
  /mnt/amulet/studentmodeling/checkpoint_snapshots/state8_diagnosis_first_eedi_current_query_curriculum_20260812/checkpoint-3000/adapter) ;;
  /mnt/amulet/studentmodeling/diagnostics/state8_diagnosis_first_full_history_teacher_v1/model) ;;
  *) echo "Unapproved adapter root: $ADAPTER_ROOT" >&2; exit 2 ;;
esac

if [[ "$AUDIT_MODE" == student ]]; then
  : "${ADAPTER_SHA256:?}"
  [[ "$ADAPTER_SHA256" =~ ^[0-9a-f]{64}$ ]] || exit 2
else
  : "${ADAPTER_RECEIPT:?}"
  test "$ADAPTER_RECEIPT" = /mnt/amulet/studentmodeling/diagnostics/state8_diagnosis_first_full_history_teacher_v1/metrics/publish_receipt.json
  test -f "$ADAPTER_RECEIPT"
  ADAPTER_SHA256="$(python - "$ADAPTER_RECEIPT" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['stage']=='state8_diagnosis_first_full_history_teacher_publish_v1',x
assert x['status'] in {'screen_complete','complete'},x
value=str(x['adapter_sha256'])
assert len(value)==64 and set(value)<=set('0123456789abcdef'),value
print(value)
PY
)"
fi

export VENV_ROOT="${VENV_ROOT:-/tmp/studentmodeling-state8-candidate-margin-venv}"
export PATH="$VENV_ROOT/bin:$PATH"
python(){ "$VENV_ROOT/bin/python" "$@"; }
export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/hf_cache HF_HUB_CACHE=/tmp/hf_cache/hub HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRANSFORMERS_VERBOSITY=error

MODEL_REMOTE=/mnt/amulet/studentmodeling/models/Qwen3-32B
MODEL_LOCAL=/tmp/Qwen3-32B
PANEL=manifests/state8_eedi_same_question_c4_valid_seed20260811_offset100_n25_v1.jsonl
PANEL_SHA256=2ba0b47bbdc99188c32542aaaf61a5d1493d2dc805bb44855eadad7d40e8effb
OUT="$AMLT_OUTPUT_DIR/eval/$AUDIT_MODE"
mkdir -p "$MODEL_LOCAL" "$OUT"

test -f "$ADAPTER_ROOT/adapter_model.safetensors"
test -f "$ADAPTER_ROOT/adapter_config.json"
test -f "$PANEL"
test "$(sha256sum "$ADAPTER_ROOT/adapter_model.safetensors" | cut -d' ' -f1)" = "$ADAPTER_SHA256"
test "$(sha256sum "$PANEL" | cut -d' ' -f1)" = "$PANEL_SHA256"

state_args=()
if [[ "$AUDIT_MODE" == student ]]; then
  : "${RECURRENT_STATE:?}" "${RECURRENT_STATE_SHA256:?}"
  [[ "$RECURRENT_STATE_SHA256" =~ ^[0-9a-f]{64}$ ]] || exit 2
  test "$RECURRENT_STATE" = /mnt/amulet/studentmodeling/checkpoint_snapshots/state8_diagnosis_first_eedi_current_query_curriculum_20260812/checkpoint-3000/recurrent_state.pt
  test -f "$RECURRENT_STATE"
  test "$(sha256sum "$RECURRENT_STATE" | cut -d' ' -f1)" = "$RECURRENT_STATE_SHA256"
  state_args=(--recurrent-state "$RECURRENT_STATE" \
    --expected-recurrent-state-sha256 "$RECURRENT_STATE_SHA256")
fi

rm -rf "$MODEL_LOCAL"; mkdir -p "$MODEL_LOCAL"
python scripts/copy_model_parallel.py --source "$MODEL_REMOTE" \
  --destination "$MODEL_LOCAL" --workers 16
test -f "$MODEL_LOCAL/model.safetensors.index.json"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

python -m pytest -q tests/test_evaluate_state8_candidate_margin.py
python -u scripts/evaluate_state8_candidate_margin.py \
  --model "$MODEL_LOCAL" --adapter "$ADAPTER_ROOT" \
  "${state_args[@]}" --mode "$AUDIT_MODE" \
  --causal-jsonl "$PANEL" --output-dir "$OUT" \
  --state-token-alias '<|vision_pad|>' --state-token-count 8 \
  --max-length 1536 --num-pairs 25 \
  --expected-adapter-sha256 "$ADAPTER_SHA256"

python - "$OUT/metrics.json" "$AUDIT_MODE" "$PANEL_SHA256" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); mode=sys.argv[2]
assert x['stage']=='state8_eedi_candidate_margin_audit_v1',x
assert x['status']=='complete' and x['mode']==mode,x
assert x['num_pairs']==25,x
assert x['candidate_protocol']=='pairwise_complete_label_mean_token_nll',x
assert x['raw_history_visible_to_final_decoder'] is False,x
assert x['artifact_sha256']['causal_jsonl']==sys.argv[3],x
expected={'student_recurrent','student_direct'} if mode=='student' else {'teacher_full_context','teacher_direct'}
assert set(x['state_paths'])==expected,x
for value in x['state_paths'].values():
 assert value['pairs']==25,value
 assert 0 <= value['state_accuracy'] <= 1,value
 assert 0 <= value['positive_margin_rate'] <= 1,value
PY
