#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "STATE8_EEDI_SAME_QUESTION_ANSWER_C4_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

export VENV_ROOT="${VENV_ROOT:-/tmp/studentmodeling-state8-eedi-same-question-answer-c4-venv}"
export PATH="$VENV_ROOT/bin:$PATH"
python(){ "$VENV_ROOT/bin/python" "$@"; }
export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/hf_cache HF_HUB_CACHE=/tmp/hf_cache/hub HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRANSFORMERS_VERBOSITY=error

: "${SAME_QUESTION_ARM:?}" "${SAME_QUESTION_REMOTE_ROOT:?}" \
  "${SAME_QUESTION_ARTIFACT_LAYOUT:?}" \
  "${SAME_QUESTION_ADAPTER_SHA256:?}" "${SAME_QUESTION_STATE_SHA256:?}"
REMOTE="$SAME_QUESTION_REMOTE_ROOT"
case "$SAME_QUESTION_ARTIFACT_LAYOUT" in
  repair_publish)
    case "$SAME_QUESTION_ARM" in updater|full|repair_*) ;; *) exit 2 ;; esac
    case "$REMOTE" in /mnt/amulet/studentmodeling/diagnostics/state8_stateonly_repair_seed*) ;; *) exit 2 ;; esac
    ADAPTER="$REMOTE/model"; STATE="$REMOTE/model/recurrent_state.pt"
    RECEIPT="$REMOTE/metrics/publish_receipt.json" ;;
  curriculum_snapshot)
    case "$SAME_QUESTION_ARM" in curriculum_generic_cp*|curriculum_current_query_cp*) ;; *) exit 2 ;; esac
    case "$REMOTE" in /mnt/amulet/studentmodeling/checkpoint_snapshots/state8_diagnosis_first_eedi_*/checkpoint-*) ;; *) exit 2 ;; esac
    ADAPTER="$REMOTE/adapter"; STATE="$REMOTE/recurrent_state.pt"
    RECEIPT="$REMOTE/checkpoint_completion.json" ;;
  repair_checkpoint_snapshot)
    case "$SAME_QUESTION_ARM" in repair_checkpoint_*) ;; *) exit 2 ;; esac
    case "$REMOTE" in /mnt/amulet/studentmodeling/checkpoint_snapshots/state8_curriculum_repairs_20260812/*/checkpoint-*) ;; *) exit 2 ;; esac
    ADAPTER="$REMOTE/adapter"; STATE="$REMOTE/recurrent_state.pt"
    RECEIPT="$REMOTE/checkpoint_completion.json" ;;
  *) exit 2 ;;
esac

MODEL_REMOTE=/mnt/amulet/studentmodeling/models/Qwen3-32B
MODEL_LOCAL=/tmp/Qwen3-32B
C4_DATA=/mnt/amulet/studentmodeling/c4_fullopen_20260724/malrule_c1_clean/test_same_template.jsonl
PAIRS=manifests/c4_stage_a_gate_pairs_seed42_n150_v1.json
OUT="$AMLT_OUTPUT_DIR/eval/answer_c4_seed42_n150"
PROGRESS="/mnt/amulet/studentmodeling/eval_progress/same_question_${SAME_QUESTION_ARM}/answer_c4_seed42_n150_v1"
mkdir -p "$MODEL_LOCAL" "$OUT" "$PROGRESS" "$AMLT_OUTPUT_DIR/metrics"

for path in "$ADAPTER/adapter_model.safetensors" "$STATE" "$RECEIPT" \
  "$C4_DATA" "$PAIRS"; do test -f "$path"; done
test "$(sha256sum "$ADAPTER/adapter_model.safetensors" | cut -d' ' -f1)" = "$SAME_QUESTION_ADAPTER_SHA256"
test "$(sha256sum "$STATE" | cut -d' ' -f1)" = "$SAME_QUESTION_STATE_SHA256"
python - "$RECEIPT" "$SAME_QUESTION_ARTIFACT_LAYOUT" \
  "$SAME_QUESTION_ADAPTER_SHA256" "$SAME_QUESTION_STATE_SHA256" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); layout=sys.argv[2]
expected={'adapter_model':sys.argv[3],'recurrent_state':sys.argv[4]}
if layout=='repair_publish':
 assert x['stage']=='state8_stateonly_repair_pilot_publish_v1' and x['status']=='complete',x
 assert x['state_shape']==[64,8,5120] and x['raw_history_visible_to_decoder'] is False,x
 assert x['history_dropout']==1.0 and x['online_teacher_kd'] is True,x
 assert x['artifact_sha256']==expected,x
elif layout=='curriculum_snapshot':
 assert x['status']=='complete' and int(x['step']) in {2000,3000,4000,5000},x
 assert x['sha256']==expected,x
else:
 assert x['status']=='complete' and int(x['step']) in {20,100,250,500},x
 assert x['sha256']==expected,x
PY

rm -rf "$MODEL_LOCAL"; mkdir -p "$MODEL_LOCAL"
python scripts/copy_model_parallel.py --source "$MODEL_REMOTE" --destination "$MODEL_LOCAL" --workers 16
python -u scripts/evaluate_recurrent_causal_gate.py \
  --model "$MODEL_LOCAL" --adapter "$ADAPTER" \
  --recurrent-state "$STATE" --state-architecture layerwise \
  --jsonl "$C4_DATA" --pair-manifest "$PAIRS" \
  --output-json "$OUT/recurrent_causal_gate.json" \
  --state-token-alias '<|vision_pad|>' --max-length 1536 --num-pairs 150 \
  --seed 42 --progress-dir "$PROGRESS" \
  --progress-key "arm=$SAME_QUESTION_ARM:adapter=$SAME_QUESTION_ADAPTER_SHA256:state=$SAME_QUESTION_STATE_SHA256:n150:v1"
python - "$OUT/recurrent_causal_gate.json" "$SAME_QUESTION_ADAPTER_SHA256" "$SAME_QUESTION_STATE_SHA256" <<'PY'
import json,pathlib,sys
x=json.load(open(sys.argv[1])); expected={'adapter_model':sys.argv[2],'recurrent_state':sys.argv[3]}
assert x['artifact_sha256']==expected and x['state_shape']==[64,8,5120],x
assert x['summary']['all_pairs_n']==150 and x['pair_seed']==42,x
assert x['raw_history_visible_to_decoder'] is False,x
assert x['persistent_history_kv_cache'] is False,x
receipt={'stage':'state8_eedi_same_question_answer_c4_confirmation_v1','status':'complete',
 'pair_seed':42,'num_pairs':150,'artifact_sha256':expected,
 'raw_history_visible_to_decoder':False,'persistent_history_kv_cache':False,
 'summary':x['summary']}
path=pathlib.Path(sys.argv[1]).parents[2]/'metrics'/'confirmation_receipt.json'
path.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
PY
