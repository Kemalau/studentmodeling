#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "STATE8_MATCHED_CHECKPOINT1200_T3_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

export VENV_ROOT="${VENV_ROOT:-/tmp/studentmodeling-state8-matched-checkpoint1200-screen-venv}"
export PATH="$VENV_ROOT/bin:$PATH"
python(){ "$VENV_ROOT/bin/python" "$@"; }
export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/hf_cache
export HF_HUB_CACHE=/tmp/hf_cache/hub
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error

CHECKPOINT=/mnt/amulet/studentmodeling/checkpoints/state8_qualified_teacher_curriculum_seed42_exact5000_rounds1250_kd1_v2_20260805_taxonomy_semantic_v1_diagnosis_first_matched_teacher_seed67_single_ga4/checkpoint-1200
EXPECTED_ADAPTER=27fcdc000fcf0d410004aad84f78b314fc9bc7d227385aa62c67f9eabeedc356
EXPECTED_STATE=7a99e2f15379549d3caef6f0d764b97716e143c4881e5dec4e2a54caf184d12e
MODEL_REMOTE=/mnt/amulet/studentmodeling/models/Qwen3-32B
MODEL_LOCAL=/tmp/Qwen3-32B
RAW_EVAL=/mnt/amulet/studentmodeling/formal_evaluation_v1/test.jsonl
RECURRENT_EVAL=/tmp/state8_matched_checkpoint1200_eval/test.jsonl
OUT="$AMLT_OUTPUT_DIR/eval/formal_t3_75"

mkdir -p "$HF_HUB_CACHE" "$MODEL_LOCAL" "$(dirname "$RECURRENT_EVAL")" "$OUT" "$AMLT_OUTPUT_DIR/metrics"
for path in "$MODEL_REMOTE/model.safetensors.index.json" "$RAW_EVAL" \
  "$CHECKPOINT/adapter/adapter_model.safetensors" "$CHECKPOINT/recurrent_state.pt" \
  "$CHECKPOINT/checkpoint_completion.json"; do test -f "$path"; done
test "$(df --output=avail /tmp | tail -1)" -ge 83886080
test "$(sha256sum scripts/evaluate_recurrent_formal_protocols_diagnosis_first.py | cut -d' ' -f1)" = a8bc07a8fe6d2c9bfa9af6d976b53b84321cb4baa9c9fcace3188e370f7732c9
test "$(sha256sum scripts/build_recurrent_evaluation_manifest.py | cut -d' ' -f1)" = d335b0b97471683eb1f317f21af6f0dd36ccca0fe6a8e495838112785ea85a8e

python - "$CHECKPOINT" "$EXPECTED_ADAPTER" "$EXPECTED_STATE" <<'PY'
import hashlib,json,pathlib,sys,torch
root=pathlib.Path(sys.argv[1]); expected={'adapter_model':sys.argv[2],'recurrent_state':sys.argv[3]}
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
 return h.hexdigest()
observed={'adapter_model':digest(root/'adapter/adapter_model.safetensors'),
          'recurrent_state':digest(root/'recurrent_state.pt')}
assert observed==expected,(observed,expected)
receipt=json.load(open(root/'checkpoint_completion.json'))
assert receipt=={'sha256':expected,'status':'complete','step':1200},receipt
p=torch.load(root/'recurrent_state.pt',map_location='cpu',weights_only=False,mmap=True)
c=p['config']; h=p['history']
assert p['step']==1200 and len(h)==1200,(p['step'],len(h))
assert c['state_architecture']=='layerwise' and c['state_token_count']==8,c
assert c['history_dropout_schedule']==[0.0,0.25,0.5,0.75,1.0],c
assert c['final_inference_route']=='strict_state_only',c
assert c.get('raw_history_visible_to_decoder',False) is False,c
assert c['diagnosis_field_order']==['diagnosis','evidence','label'],c
assert c['canonical_label_role']=='auxiliary_ordinary_weight',c
assert float(c['label_auxiliary_weight'])==1.0,c
assert sum(float(r['history_dropout'])==0.0 for r in h)==1000,h[-1]
assert sum(float(r['history_dropout'])==0.25 for r in h)==200,h[-1]
PY

python scripts/build_recurrent_evaluation_manifest.py \
  --input-jsonl "$RAW_EVAL" --output-jsonl "$RECURRENT_EVAL" \
  --audit-json /tmp/state8_matched_checkpoint1200_eval/audit.json \
  --max-updates 4 --seed 20260730
test "$(sha256sum "$RECURRENT_EVAL" | cut -d' ' -f1)" = a368cefb0c49cb6a305ad1a892808361fba2d44d555a0a28ea64e56babe32525

rm -rf "$MODEL_LOCAL"; mkdir -p "$MODEL_LOCAL"
python scripts/copy_model_parallel.py --source "$MODEL_REMOTE" --destination "$MODEL_LOCAL" --workers 16
python -u scripts/evaluate_recurrent_formal_protocols_diagnosis_first.py \
  --model "$MODEL_LOCAL" --adapter "$CHECKPOINT/adapter" \
  --recurrent-state "$CHECKPOINT/recurrent_state.pt" --state-architecture layerwise \
  --eval-jsonl "$RECURRENT_EVAL" --output-dir "$OUT" --state-token-alias '<|vision_pad|>' \
  --max-samples-per-group 25 --max-length 1536 --max-new-tokens 160 \
  --state-batch-size 2 --candidate-row-batch-size 2 --seed 1200 \
  --tasks diagnose_misconception --sources eedi_mining malrule \
  --save-unredacted-predictions --t3-natural-schema
python - "$OUT/metrics.json" "$EXPECTED_ADAPTER" "$EXPECTED_STATE" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); expected={'adapter_model':sys.argv[2],'recurrent_state':sys.argv[3]}
assert x['artifact_sha256']==expected,(x,expected)
assert x['state_shape']==[64,8,5120] and x['sampled_rows']==75,x
assert x['raw_history_visible_to_decoder'] is False,x
assert x['persistent_history_kv_cache'] is False,x
assert all(v['exact_match_role']=='auxiliary_only_not_authoritative' for v in x['metrics_by_group'].values()),x
PY
cp /tmp/state8_matched_checkpoint1200_eval/audit.json "$AMLT_OUTPUT_DIR/metrics/manifest_audit.json"
cp "$CHECKPOINT/checkpoint_completion.json" "$AMLT_OUTPUT_DIR/metrics/checkpoint_completion.json"

