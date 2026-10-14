#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "STATE8_MATCHED_CHECKPOINT_DIAGNOSIS_C4_SHARD_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR
export VENV_ROOT="${VENV_ROOT:-/tmp/studentmodeling-state8-matched-checkpoint-screen-venv}"
export PATH="$VENV_ROOT/bin:$PATH"; python(){ "$VENV_ROOT/bin/python" "$@"; }
export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/hf_cache HF_HUB_CACHE=/tmp/hf_cache/hub HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRANSFORMERS_VERBOSITY=error
: "${STATE8_MATCHED_CHECKPOINT_ROOT:?}" "${STATE8_MATCHED_CHECKPOINT_STEP:?}"
: "${STATE8_MATCHED_ADAPTER_SHA256:?}" "${STATE8_MATCHED_STATE_SHA256:?}"
: "${MATCHED_CHECKPOINT_RECEIPT_SHA256:?}"
: "${PAIR_START:?}" "${PAIR_COUNT:?}"; test "$PAIR_COUNT" = 10
case "$PAIR_START" in 0|10|20|30|40) ;; *) exit 2 ;; esac
MODEL_REMOTE=/mnt/amulet/studentmodeling/models/Qwen3-32B; MODEL_LOCAL=/tmp/Qwen3-32B
DATA=/mnt/amulet/studentmodeling/c4_fullopen_20260724/malrule_c1_clean/test_same_template.jsonl
PAIRS=manifests/c4_stage_a_gate_pairs_seed42_n150_v1.json
OUT="$AMLT_OUTPUT_DIR/eval/diagnosis_c4_shard_${PAIR_START}"
mkdir -p "$MODEL_LOCAL" "$OUT" "$AMLT_OUTPUT_DIR/metrics"
python scripts/audit_state8_matched_checkpoint.py --checkpoint-root "$STATE8_MATCHED_CHECKPOINT_ROOT" \
  --step "$STATE8_MATCHED_CHECKPOINT_STEP" --adapter-sha256 "$STATE8_MATCHED_ADAPTER_SHA256" \
  --state-sha256 "$STATE8_MATCHED_STATE_SHA256" \
  --receipt-sha256 "$MATCHED_CHECKPOINT_RECEIPT_SHA256" \
  --output "$AMLT_OUTPUT_DIR/metrics/artifact_audit.json"
rm -rf "$MODEL_LOCAL"; mkdir -p "$MODEL_LOCAL"
python scripts/copy_model_parallel.py --source "$MODEL_REMOTE" --destination "$MODEL_LOCAL" --workers 16
python -u scripts/evaluate_recurrent_diagnosis_c4_diagnosis_first.py \
  --model "$MODEL_LOCAL" --adapter "$STATE8_MATCHED_CHECKPOINT_ROOT/adapter" \
  --recurrent-state "$STATE8_MATCHED_CHECKPOINT_ROOT/recurrent_state.pt" \
  --state-architecture layerwise --jsonl "$DATA" --pair-manifest "$PAIRS" \
  --output-dir "$OUT" --state-token-alias '<|vision_pad|>' --max-length 1536 \
  --max-new-tokens 160 --num-pairs 50 --pair-start "$PAIR_START" --pair-count 10 \
  --seed 42 --expected-adapter-sha256 "$STATE8_MATCHED_ADAPTER_SHA256" \
  --expected-recurrent-state-sha256 "$STATE8_MATCHED_STATE_SHA256"
python - "$OUT/metrics.json" "$PAIR_START" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); start=int(sys.argv[2])
assert x['num_pairs']==10 and x['pair_start']==start and x['pair_stop']==start+10,x
assert x['state_shape']==[64,8,5120] and x['raw_history_visible_to_decoder'] is False,x
PY
python - "$OUT/predictions.jsonl" <<'PY'
import json,sys
rows=[json.loads(line) for line in open(sys.argv[1]) if line.strip()]
assert len(rows)==50, len(rows)
assert all('For this Malrule diagnosis' in row['decode_prompt'] for row in rows)
assert all('For this Eedi diagnosis' not in row['decode_prompt'] for row in rows)
PY
