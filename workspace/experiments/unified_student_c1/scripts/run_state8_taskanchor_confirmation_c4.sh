#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "STATE8_TASKANCHOR_CONFIRM_C4_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

export VENV_ROOT="${VENV_ROOT:-/tmp/studentmodeling-state8-taskanchor-confirm-venv}"
export PATH="$VENV_ROOT/bin:$PATH"
python(){ "$VENV_ROOT/bin/python" "$@"; }
export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/hf_cache HF_HUB_CACHE=/tmp/hf_cache/hub HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRANSFORMERS_VERBOSITY=error

: "${MILESTONE_STEP:?}" "${MILESTONE_ADAPTER_SHA256:?}"
: "${MILESTONE_STATE_SHA256:?}" "${MILESTONE_RECEIPT_SHA256:?}"
: "${PAIR_SEED:?}" "${PAIR_MANIFEST_SHA256:?}"
case "$MILESTONE_STEP" in 200|400|600|800|1000) ;; *) exit 2 ;; esac
case "$PAIR_SEED" in 13|42|97) ;; *) exit 2 ;; esac

CHECKPOINT="/mnt/amulet/studentmodeling/checkpoints/state8_stateonly_repair_seed2067_taskanchor_matched_cp2000_causal_taskanchor_v1/checkpoint-$MILESTONE_STEP"
MODEL_REMOTE=/mnt/amulet/studentmodeling/models/Qwen3-32B
MODEL_LOCAL=/tmp/Qwen3-32B
C4_DATA=/mnt/amulet/studentmodeling/c4_fullopen_20260724/malrule_c1_clean/test_same_template.jsonl
PAIR_MANIFEST="manifests/c4_stage_a_gate_pairs_seed${PAIR_SEED}_n150_v1.json"
OUT="$AMLT_OUTPUT_DIR/eval/c4_seed$PAIR_SEED"
DIAGNOSIS_OUT="$AMLT_OUTPUT_DIR/eval/diagnosis_c4_seed$PAIR_SEED"
PROGRESS="/mnt/amulet/studentmodeling/eval_progress/state8_taskanchor_confirmation_step${MILESTONE_STEP}/c4_seed${PAIR_SEED}_v1"

mkdir -p "$MODEL_LOCAL" "$OUT" "$DIAGNOSIS_OUT" "$PROGRESS" "$AMLT_OUTPUT_DIR/metrics"
test "$(df --output=avail /tmp | tail -1)" -ge 83886080
test "$(sha256sum "$PAIR_MANIFEST" | cut -d' ' -f1)" = "$PAIR_MANIFEST_SHA256"
test "$(sha256sum scripts/evaluate_recurrent_causal_gate.py | cut -d' ' -f1)" = 69fd4f2f4943a78a0afe9a701533d9ab034f247f87c7a09327f7523e310b7a52
test "$(sha256sum scripts/evaluate_recurrent_diagnosis_c4_diagnosis_first.py | cut -d' ' -f1)" = 660df80664c6012f6eac1d379cc1d611c9f085c91496d1c9cc8e7f7bcb30d7ca

python scripts/audit_state8_taskanchor_milestone.py \
  --checkpoint-root "$CHECKPOINT" --step "$MILESTONE_STEP" \
  --adapter-sha256 "$MILESTONE_ADAPTER_SHA256" \
  --state-sha256 "$MILESTONE_STATE_SHA256" \
  --receipt-sha256 "$MILESTONE_RECEIPT_SHA256" \
  --output "$AMLT_OUTPUT_DIR/metrics/artifact_audit.json"
rm -rf "$MODEL_LOCAL"; mkdir -p "$MODEL_LOCAL"
python scripts/copy_model_parallel.py --source "$MODEL_REMOTE" --destination "$MODEL_LOCAL" --workers 16
test -f "$MODEL_LOCAL/model.safetensors.index.json"

python -u scripts/evaluate_recurrent_causal_gate.py \
  --model "$MODEL_LOCAL" --adapter "$CHECKPOINT/adapter" \
  --recurrent-state "$CHECKPOINT/recurrent_state.pt" --state-architecture layerwise \
  --jsonl "$C4_DATA" --pair-manifest "$PAIR_MANIFEST" \
  --output-json "$OUT/recurrent_causal_gate.json" \
  --state-token-alias '<|vision_pad|>' --max-length 1536 --num-pairs 150 \
  --seed "$PAIR_SEED" --progress-dir "$PROGRESS" \
  --progress-key "step=$MILESTONE_STEP:adapter=$MILESTONE_ADAPTER_SHA256:state=$MILESTONE_STATE_SHA256"
python -u scripts/evaluate_recurrent_diagnosis_c4_diagnosis_first.py \
  --model "$MODEL_LOCAL" --adapter "$CHECKPOINT/adapter" \
  --recurrent-state "$CHECKPOINT/recurrent_state.pt" --state-architecture layerwise \
  --jsonl "$C4_DATA" --pair-manifest "$PAIR_MANIFEST" \
  --output-dir "$DIAGNOSIS_OUT" --state-token-alias '<|vision_pad|>' \
  --max-length 1536 --max-new-tokens 160 --num-pairs 150 --seed "$PAIR_SEED" \
  --expected-adapter-sha256 "$MILESTONE_ADAPTER_SHA256" \
  --expected-recurrent-state-sha256 "$MILESTONE_STATE_SHA256"
python - "$OUT/recurrent_causal_gate.json" "$DIAGNOSIS_OUT/metrics.json" \
  "$PAIR_SEED" "$MILESTONE_ADAPTER_SHA256" "$MILESTONE_STATE_SHA256" <<'PY'
import json,sys
a,d=map(lambda p:json.load(open(p)),sys.argv[1:3]); seed=int(sys.argv[3])
expected={'adapter_model':sys.argv[4],'recurrent_state':sys.argv[5]}
for x in (a,d):
 assert x['pair_seed']==seed and x['state_shape']==[64,8,5120],x
 assert x['artifact_sha256']==expected,x
 assert x['raw_history_visible_to_decoder'] is False,x
 assert x['persistent_history_kv_cache'] is False,x
assert a['summary']['all_pairs_n']==150,a
assert d['num_pairs']==150 and d['semantic_judgment_pending'] is True,d
PY

