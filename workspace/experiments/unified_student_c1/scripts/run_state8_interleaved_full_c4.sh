#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "STATE8_INTERLEAVED_FULL_C4_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

export VENV_ROOT="${VENV_ROOT:-/tmp/studentmodeling-state8-interleaved-full-eval-venv}"
export PATH="$VENV_ROOT/bin:$PATH"
python(){ "$VENV_ROOT/bin/python" "$@"; }
export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/hf_cache HF_HUB_CACHE=/tmp/hf_cache/hub HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRANSFORMERS_VERBOSITY=error

: "${C4_KIND:?}" "${STATE8_EVAL_RUN_DIR:?}" "${STATE8_EVAL_PROGRESS_NAMESPACE:?}"
case "$C4_KIND" in malrule|eedi) ;; *) exit 2 ;; esac
test "$STATE8_EVAL_RUN_DIR" = /mnt/amulet/studentmodeling/checkpoints/state8_stateonly_repair_seed8401_taskanchor_source_anchor_full_mse1_eedi_same_question_v1_early_label_high_seed8401
test "$STATE8_EVAL_PROGRESS_NAMESPACE" = state8_interleaved_seed8401_full_eval_20260814

MODEL_REMOTE=/mnt/amulet/studentmodeling/models/Qwen3-32B
MODEL_LOCAL=/tmp/Qwen3-32B
ADAPTER="$STATE8_EVAL_RUN_DIR/final_adapter"
RECURRENT_STATE="$STATE8_EVAL_RUN_DIR/recurrent_state.pt"
EXPECTED_ADAPTER=3bcf7bdc1d6e4de432efbd2bf78551c3d92349e44d7e2b217166ca1305cffe24
EXPECTED_STATE=6985e40ec82e100211a552847b2f4a8601fbdb5785d58fdcb0f0620df10d5762

mkdir -p "$HF_HUB_CACHE"
test -f "$ADAPTER/adapter_model.safetensors"
test -f "$RECURRENT_STATE"
test "$(sha256sum "$ADAPTER/adapter_model.safetensors" | cut -d' ' -f1)" = "$EXPECTED_ADAPTER"
test "$(sha256sum "$RECURRENT_STATE" | cut -d' ' -f1)" = "$EXPECTED_STATE"
test "$(sha256sum scripts/copy_model_parallel.py | cut -d' ' -f1)" = 62b37ab70bc2ffc81c8314ca69215da7facdf18dba3f023d83096d1c3526ea4c
test "$(df --output=avail /tmp | tail -1)" -ge 83886080

rm -rf "$MODEL_LOCAL"; mkdir -p "$MODEL_LOCAL"
python scripts/copy_model_parallel.py --source "$MODEL_REMOTE" \
  --destination "$MODEL_LOCAL" --workers 16
test -f "$MODEL_LOCAL/model.safetensors.index.json"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

if [[ "$C4_KIND" == malrule ]]; then
  : "${PAIR_SEED:?}" "${PAIR_MANIFEST_SHA256:?}"
  case "$PAIR_SEED:$PAIR_MANIFEST_SHA256" in
    13:9856e643848d11a834761e3d28fb078f6eb70fa6c42fc468c3eb1b3a1aafd693|\
    42:edcdb934a65252557f6c569eea9ef7a85e5445549deac9fb4829c3f579e1a9b8|\
    97:15ed710992abd71c37bdfa2246a5a38b5eb422e70ecbfb6da933be013d6d9118) ;;
    *) exit 2 ;;
  esac
  DATA=/mnt/amulet/studentmodeling/c4_fullopen_20260724/malrule_c1_clean/test_same_template.jsonl
  PAIRS="manifests/c4_stage_a_gate_pairs_seed${PAIR_SEED}_n150_v1.json"
  OUT="$AMLT_OUTPUT_DIR/eval/c4_seed$PAIR_SEED"
  DIAG="$AMLT_OUTPUT_DIR/eval/diagnosis_c4_seed$PAIR_SEED"
  PROGRESS="/mnt/amulet/studentmodeling/eval_progress/${STATE8_EVAL_PROGRESS_NAMESPACE}/c4_seed${PAIR_SEED}_v1"
  mkdir -p "$OUT" "$DIAG" "$PROGRESS"
  test -f "$DATA"; test -f "$PAIRS"
  test "$(sha256sum "$PAIRS" | cut -d' ' -f1)" = "$PAIR_MANIFEST_SHA256"
  test "$(sha256sum scripts/evaluate_recurrent_causal_gate.py | cut -d' ' -f1)" = 69fd4f2f4943a78a0afe9a701533d9ab034f247f87c7a09327f7523e310b7a52
  test "$(sha256sum scripts/evaluate_recurrent_diagnosis_c4_diagnosis_first.py | cut -d' ' -f1)" = 660df80664c6012f6eac1d379cc1d611c9f085c91496d1c9cc8e7f7bcb30d7ca
  python -u scripts/evaluate_recurrent_causal_gate.py \
    --model "$MODEL_LOCAL" --adapter "$ADAPTER" --recurrent-state "$RECURRENT_STATE" \
    --state-architecture layerwise --jsonl "$DATA" --pair-manifest "$PAIRS" \
    --output-json "$OUT/recurrent_causal_gate.json" --state-token-alias '<|vision_pad|>' \
    --max-length 1536 --num-pairs 150 --seed "$PAIR_SEED" \
    --progress-dir "$PROGRESS" \
    --progress-key "adapter=$EXPECTED_ADAPTER:state=$EXPECTED_STATE:interleaved-full-v1"
  python -u scripts/evaluate_recurrent_diagnosis_c4_diagnosis_first.py \
    --model "$MODEL_LOCAL" --adapter "$ADAPTER" --recurrent-state "$RECURRENT_STATE" \
    --state-architecture layerwise --jsonl "$DATA" --pair-manifest "$PAIRS" \
    --output-dir "$DIAG" --state-token-alias '<|vision_pad|>' \
    --max-length 1536 --max-new-tokens 192 --num-pairs 150 --seed "$PAIR_SEED" \
    --expected-adapter-sha256 "$EXPECTED_ADAPTER" \
    --expected-recurrent-state-sha256 "$EXPECTED_STATE"
  python - "$OUT/recurrent_causal_gate.json" "$DIAG/metrics.json" "$PAIR_SEED" <<'PY'
import json,sys
a=json.load(open(sys.argv[1])); d=json.load(open(sys.argv[2])); seed=int(sys.argv[3])
expected={'adapter_model':'3bcf7bdc1d6e4de432efbd2bf78551c3d92349e44d7e2b217166ca1305cffe24','recurrent_state':'6985e40ec82e100211a552847b2f4a8601fbdb5785d58fdcb0f0620df10d5762'}
assert a['pair_seed']==seed and a['summary']['all_pairs_n']==150,a
assert d['pair_seed']==seed and d['num_pairs']==150,d
for x in (a,d):
 assert x['artifact_sha256']==expected and x['state_shape']==[64,8,5120],x
 assert x['raw_history_visible_to_decoder'] is False and x['persistent_history_kv_cache'] is False,x
PY
else
  PANEL=manifests/state8_eedi_same_question_c4_final_seed20260811_offset125_n150_v1.jsonl
  RECEIPT=manifests/state8_eedi_same_question_c4_final_seed20260811_offset125_n150_v1.receipt.json
  OUT="$AMLT_OUTPUT_DIR/eval/eedi_same_question_c4_final150"
  mkdir -p "$OUT"
  test "$(sha256sum "$PANEL" | cut -d' ' -f1)" = 6743c2a29a1c2b576a96e64cd73a1ffccaa6b0364d04fe2e8c692ad315d97d4e
  test "$(sha256sum "$RECEIPT" | cut -d' ' -f1)" = 4536002e2dbdb2eb2ebd92d8ad227494fb5bed81baa930d4c3ad88a5e2257332
  test "$(sha256sum scripts/evaluate_state8_eedi_same_question_c4.py | cut -d' ' -f1)" = 4b9911ce6051a04340ce2b1a2e0b8cb1806b4dac20000a7e5a9ed130aac82352
  python - "$RECEIPT" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); assert x['status']=='passed' and x['twins']==150,x
assert x['offset']==125 and x['same_question_query_identical'] is True,x
assert x['student_response_visible_to_decoder'] is False and x['raw_history_visible_to_decoder'] is False,x
PY
  python -u scripts/evaluate_state8_eedi_same_question_c4.py \
    --model "$MODEL_LOCAL" --adapter "$ADAPTER" --recurrent-state "$RECURRENT_STATE" \
    --state-architecture layerwise --causal-jsonl "$PANEL" --output-dir "$OUT" \
    --state-token-alias '<|vision_pad|>' --max-length 1536 --max-new-tokens 192 \
    --num-pairs 150 --seed 20260811 \
    --progress-dir "/mnt/amulet/studentmodeling/eval_progress/${STATE8_EVAL_PROGRESS_NAMESPACE}/eedi_same_question_c4_v1" \
    --progress-key "adapter=$EXPECTED_ADAPTER:state=$EXPECTED_STATE:interleaved-full-eedi-v1" \
    --expected-adapter-sha256 "$EXPECTED_ADAPTER" \
    --expected-recurrent-state-sha256 "$EXPECTED_STATE"
  python - "$OUT/metrics.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); assert x['num_pairs']==150,x
assert x['artifact_sha256']=={'adapter_model':'3bcf7bdc1d6e4de432efbd2bf78551c3d92349e44d7e2b217166ca1305cffe24','recurrent_state':'6985e40ec82e100211a552847b2f4a8601fbdb5785d58fdcb0f0620df10d5762'},x
assert x['same_question_query_identical'] is True and x['student_response_visible_to_decoder'] is False,x
assert x['raw_history_visible_to_decoder'] is False and x['persistent_history_kv_cache'] is False,x
assert x['resumable_progress'] is True,x
PY
fi
