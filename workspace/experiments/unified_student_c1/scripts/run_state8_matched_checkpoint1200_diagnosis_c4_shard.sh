#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "STATE8_MATCHED_CHECKPOINT1200_DIAGNOSIS_C4_SHARD_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

export VENV_ROOT="${VENV_ROOT:-/tmp/studentmodeling-state8-matched-checkpoint1200-c4-shards-venv}"
export PATH="$VENV_ROOT/bin:$PATH"
python(){ "$VENV_ROOT/bin/python" "$@"; }
export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/hf_cache
export HF_HUB_CACHE=/tmp/hf_cache/hub
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error

: "${PAIR_START:?PAIR_START is required}"
: "${PAIR_COUNT:?PAIR_COUNT is required}"
test "$PAIR_COUNT" = 10
case "$PAIR_START" in 0|10|20|30|40) ;; *) exit 2 ;; esac

CHECKPOINT=/mnt/amulet/studentmodeling/checkpoints/state8_qualified_teacher_curriculum_seed42_exact5000_rounds1250_kd1_v2_20260805_taxonomy_semantic_v1_diagnosis_first_matched_teacher_seed67_single_ga4/checkpoint-1200
EXPECTED_ADAPTER=27fcdc000fcf0d410004aad84f78b314fc9bc7d227385aa62c67f9eabeedc356
EXPECTED_STATE=7a99e2f15379549d3caef6f0d764b97716e143c4881e5dec4e2a54caf184d12e
MODEL_REMOTE=/mnt/amulet/studentmodeling/models/Qwen3-32B
MODEL_LOCAL=/tmp/Qwen3-32B
ADAPTER="$CHECKPOINT/adapter"
STATE="$CHECKPOINT/recurrent_state.pt"
RECEIPT="$CHECKPOINT/checkpoint_completion.json"
DATA=/mnt/amulet/studentmodeling/c4_fullopen_20260724/malrule_c1_clean/test_same_template.jsonl
PAIRS=manifests/c4_stage_a_gate_pairs_seed42_n150_v1.json
OUT="$AMLT_OUTPUT_DIR/eval/diagnosis_c4_shard_${PAIR_START}"

mkdir -p "$HF_HUB_CACHE" "$MODEL_LOCAL" "$OUT"
for path in "$MODEL_REMOTE/model.safetensors.index.json" "$ADAPTER/adapter_model.safetensors" \
  "$STATE" "$RECEIPT" "$DATA" "$PAIRS"; do test -f "$path"; done
test "$(sha256sum "$ADAPTER/adapter_model.safetensors" | cut -d' ' -f1)" = "$EXPECTED_ADAPTER"
test "$(sha256sum "$STATE" | cut -d' ' -f1)" = "$EXPECTED_STATE"
test "$(sha256sum "$PAIRS" | cut -d' ' -f1)" = edcdb934a65252557f6c569eea9ef7a85e5445549deac9fb4829c3f579e1a9b8
test "$(sha256sum scripts/evaluate_recurrent_diagnosis_c4_diagnosis_first.py | cut -d' ' -f1)" = 660df80664c6012f6eac1d379cc1d611c9f085c91496d1c9cc8e7f7bcb30d7ca
test "$(df --output=avail /tmp | tail -1)" -ge 83886080
python - "$RECEIPT" "$EXPECTED_ADAPTER" "$EXPECTED_STATE" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x=={'sha256':{'adapter_model':sys.argv[2],'recurrent_state':sys.argv[3]},
           'status':'complete','step':1200},x
PY

rm -rf "$MODEL_LOCAL"; mkdir -p "$MODEL_LOCAL"
python scripts/copy_model_parallel.py --source "$MODEL_REMOTE" --destination "$MODEL_LOCAL" --workers 16
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
python -u scripts/evaluate_recurrent_diagnosis_c4_diagnosis_first.py \
  --model "$MODEL_LOCAL" --adapter "$ADAPTER" --recurrent-state "$STATE" \
  --state-architecture layerwise --jsonl "$DATA" --pair-manifest "$PAIRS" \
  --output-dir "$OUT" --state-token-alias '<|vision_pad|>' --max-length 1536 \
  --max-new-tokens 160 --num-pairs 50 --pair-start "$PAIR_START" \
  --pair-count "$PAIR_COUNT" --seed 42 \
  --expected-adapter-sha256 "$EXPECTED_ADAPTER" \
  --expected-recurrent-state-sha256 "$EXPECTED_STATE"

python - "$OUT/metrics.json" "$PAIR_START" "$EXPECTED_ADAPTER" "$EXPECTED_STATE" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); start=int(sys.argv[2])
assert x['num_pairs']==10 and x['manifest_prefix_pairs']==50,x
assert x['pair_start']==start and x['pair_stop']==start+10,x
assert x['state_shape']==[64,8,5120],x
assert x['raw_history_visible_to_decoder'] is False,x
assert x['persistent_history_kv_cache'] is False,x
assert x['artifact_sha256']=={'adapter_model':sys.argv[3],'recurrent_state':sys.argv[4]},x
PY
python - "$OUT/predictions.jsonl" <<'PY'
import json,sys
rows=[json.loads(line) for line in open(sys.argv[1]) if line.strip()]
assert len(rows)==50, len(rows)
assert all('For this Malrule diagnosis' in row['decode_prompt'] for row in rows)
assert all('For this Eedi diagnosis' not in row['decode_prompt'] for row in rows)
PY
