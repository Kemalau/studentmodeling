#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "STATE8_MATCHED_FINAL_DIAGNOSIS_C4_CORRECTED_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

export VENV_ROOT=/tmp/studentmodeling-matched-final-diagnosis-c4-corrected-venv
export PATH="$VENV_ROOT/bin:$PATH"; python(){ "$VENV_ROOT/bin/python" "$@"; }
export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/hf_cache HF_HUB_CACHE=/tmp/hf_cache/hub HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRANSFORMERS_VERBOSITY=error
: "${EXPECTED_ADAPTER_SHA256:?}" "${EXPECTED_STATE_SHA256:?}" "${EXPECTED_RECEIPT_SHA256:?}"
ROOT=/mnt/amulet/studentmodeling/diagnostics/state8_qualified_teacher_curriculum_highkd_v1_immutable_taxonomy_semantic_v1_diagnosis_first_matched_teacher_seed67_single_ga4
MODEL_REMOTE=/mnt/amulet/studentmodeling/models/Qwen3-32B
MODEL_LOCAL=/tmp/Qwen3-32B
DATA=/mnt/amulet/studentmodeling/c4_fullopen_20260724/malrule_c1_clean/test_same_template.jsonl
PAIRS=manifests/c4_stage_a_gate_pairs_seed42_n150_v1.json
OUT="$AMLT_OUTPUT_DIR/eval/diagnosis_c4_20_corrected"
mkdir -p "$MODEL_LOCAL" "$OUT" "$AMLT_OUTPUT_DIR/metrics"
for path in "$ROOT/model/adapter_model.safetensors" "$ROOT/model/recurrent_state.pt" \
  "$ROOT/metrics/immutable_receipt.json" "$DATA" "$PAIRS"; do test -f "$path"; done
test "$(sha256sum scripts/evaluate_recurrent_diagnosis_c4_diagnosis_first.py | cut -d' ' -f1)" = 660df80664c6012f6eac1d379cc1d611c9f085c91496d1c9cc8e7f7bcb30d7ca
test "$(sha256sum "$PAIRS" | cut -d' ' -f1)" = edcdb934a65252557f6c569eea9ef7a85e5445549deac9fb4829c3f579e1a9b8
test "$(df --output=avail /tmp | tail -1)" -ge 83886080
python - "$ROOT" "$EXPECTED_ADAPTER_SHA256" "$EXPECTED_STATE_SHA256" "$EXPECTED_RECEIPT_SHA256" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1])
def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
 return h.hexdigest()
receipt=root/'metrics/immutable_receipt.json'
assert digest(receipt)==sys.argv[4]
x=json.load(open(receipt)); observed={
 'adapter_model':digest(root/'model/adapter_model.safetensors'),
 'recurrent_state':digest(root/'model/recurrent_state.pt')}
assert observed=={'adapter_model':sys.argv[2],'recurrent_state':sys.argv[3]},observed
assert x['status']=='complete' and x['state_shape']==[64,8,5120],x
assert x['sha256']==observed,x
assert x['raw_history_visible_to_decoder'] is False,x
assert x['persistent_history_kv_cache'] is False,x
PY
rm -rf "$MODEL_LOCAL"; mkdir -p "$MODEL_LOCAL"
python scripts/copy_model_parallel.py --source "$MODEL_REMOTE" --destination "$MODEL_LOCAL" --workers 16
python -u scripts/evaluate_recurrent_diagnosis_c4_diagnosis_first.py \
  --model "$MODEL_LOCAL" --adapter "$ROOT/model" \
  --recurrent-state "$ROOT/model/recurrent_state.pt" --state-architecture layerwise \
  --jsonl "$DATA" --pair-manifest "$PAIRS" --output-dir "$OUT" \
  --state-token-alias '<|vision_pad|>' --max-length 1536 --max-new-tokens 192 \
  --num-pairs 20 --seed 67 --expected-adapter-sha256 "$EXPECTED_ADAPTER_SHA256" \
  --expected-recurrent-state-sha256 "$EXPECTED_STATE_SHA256"
python - "$OUT/metrics.json" "$OUT/predictions.jsonl" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); rows=[json.loads(line) for line in open(sys.argv[2]) if line.strip()]
assert x['num_pairs']==20 and x['state_shape']==[64,8,5120],x
assert x['raw_history_visible_to_decoder'] is False and x['persistent_history_kv_cache'] is False,x
assert len(rows)==100,len(rows)
assert all('For this Malrule diagnosis' in row['decode_prompt'] for row in rows)
assert all('For this Eedi diagnosis' not in row['decode_prompt'] for row in rows)
PY
cp "$ROOT/metrics/immutable_receipt.json" "$AMLT_OUTPUT_DIR/metrics/immutable_receipt.json"
