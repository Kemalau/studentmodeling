#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "STATE8_STRUCTURED_CANDIDATE_MARGIN_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

export VENV_ROOT="${VENV_ROOT:-/tmp/studentmodeling-state8-structured-margin-venv}"
export PATH="$VENV_ROOT/bin:$PATH"
python(){ "$VENV_ROOT/bin/python" "$@"; }
export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/hf_cache HF_HUB_CACHE=/tmp/hf_cache/hub HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRANSFORMERS_VERBOSITY=error

MODEL_REMOTE=/mnt/amulet/studentmodeling/models/Qwen3-32B
MODEL_LOCAL=/tmp/Qwen3-32B
STUDENT=/mnt/amulet/studentmodeling/checkpoint_snapshots/state8_diagnosis_first_eedi_current_query_curriculum_20260812/checkpoint-3000
STUDENT_ADAPTER_SHA=566bd6a488523bd59610767e019c53c5f1609b760a1ab1d1fcf6cbaabc95cabd
STUDENT_STATE_SHA=5e0b0d40ee5dee2429f641270cdbb83e66b96172cd53e3336e84ac9b6564ac73
TEACHER=/mnt/amulet/studentmodeling/diagnostics/state8_diagnosis_first_full_history_teacher_v1
TEACHER_RECEIPT="$TEACHER/metrics/publish_receipt.json"
PANEL=manifests/state8_eedi_same_question_c4_valid_seed20260811_offset100_n25_v1.jsonl
PANEL_SHA=2ba0b47bbdc99188c32542aaaf61a5d1493d2dc805bb44855eadad7d40e8effb
OUT="$AMLT_OUTPUT_DIR/eval/structured"

for path in "$STUDENT/adapter/adapter_model.safetensors" \
  "$STUDENT/recurrent_state.pt" "$STUDENT/checkpoint_completion.json" \
  "$TEACHER/model/adapter_model.safetensors" "$TEACHER_RECEIPT" "$PANEL"; do
  test -f "$path"
done
test "$(sha256sum "$STUDENT/adapter/adapter_model.safetensors" | cut -d' ' -f1)" = "$STUDENT_ADAPTER_SHA"
test "$(sha256sum "$STUDENT/recurrent_state.pt" | cut -d' ' -f1)" = "$STUDENT_STATE_SHA"
test "$(sha256sum "$PANEL" | cut -d' ' -f1)" = "$PANEL_SHA"
TEACHER_SHA="$(python - "$TEACHER_RECEIPT" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['stage']=='state8_diagnosis_first_full_history_teacher_publish_v1',x
assert x['status'] in {'screen_complete','complete'},x
value=str(x['adapter_sha256'])
assert len(value)==64 and set(value)<=set('0123456789abcdef'),value
print(value)
PY
)"
test "$(sha256sum "$TEACHER/model/adapter_model.safetensors" | cut -d' ' -f1)" = "$TEACHER_SHA"

rm -rf "$MODEL_LOCAL"; mkdir -p "$MODEL_LOCAL" "$OUT"
python scripts/copy_model_parallel.py --source "$MODEL_REMOTE" \
  --destination "$MODEL_LOCAL" --workers 16
test -f "$MODEL_LOCAL/model.safetensors.index.json"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

python -m pytest -q tests/test_evaluate_state8_candidate_margin.py
python -u scripts/evaluate_state8_candidate_margin.py \
  --model "$MODEL_LOCAL" --adapter "$STUDENT/adapter" \
  --recurrent-state "$STUDENT/recurrent_state.pt" --mode student \
  --target-protocol structured \
  --external-teacher-adapter "$TEACHER/model" \
  --expected-external-teacher-sha256 "$TEACHER_SHA" \
  --causal-jsonl "$PANEL" --output-dir "$OUT" \
  --state-token-alias '<|vision_pad|>' --state-token-count 8 \
  --max-length 1536 --num-pairs 25 \
  --expected-adapter-sha256 "$STUDENT_ADAPTER_SHA" \
  --expected-recurrent-state-sha256 "$STUDENT_STATE_SHA"

python - "$OUT/metrics.json" "$PANEL_SHA" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['stage']=='state8_eedi_candidate_margin_audit_v1' and x['status']=='complete',x
assert x['target_protocol']=='structured' and x['num_pairs']==25,x
assert x['candidate_protocol']=='pairwise_complete_structured_target_mean_token_nll',x
assert x['artifact_sha256']['causal_jsonl']==sys.argv[2],x
assert set(x['state_paths'])=={
 'student_recurrent','student_direct',
 'external_teacher_direct_to_student_decoder','teacher_full_context'},x
for value in x['state_paths'].values():
 assert value['pairs']==25,value
 assert 0 <= value['state_accuracy'] <= 1,value
 assert 0 <= value['positive_margin_rate'] <= 1,value
PY
