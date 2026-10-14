#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "STATE8_MATCHED_CHECKPOINT1200_C4_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

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
DATA=/mnt/amulet/studentmodeling/c4_fullopen_20260724/malrule_c1_clean/test_same_template.jsonl
PARENT_PAIRS=manifests/c4_stage_a_gate_pairs_seed42_n150_v1.json
PAIRS=/tmp/state8_matched_checkpoint1200_pairs_seed42_n50.json
ANSWER_OUT="$AMLT_OUTPUT_DIR/eval/answer_c4_50"
DIAG_OUT="$AMLT_OUTPUT_DIR/eval/diagnosis_c4_50"

mkdir -p "$HF_HUB_CACHE" "$MODEL_LOCAL" "$ANSWER_OUT" "$DIAG_OUT" "$AMLT_OUTPUT_DIR/metrics"
for path in "$MODEL_REMOTE/model.safetensors.index.json" "$DATA" "$PARENT_PAIRS" \
  "$CHECKPOINT/adapter/adapter_model.safetensors" "$CHECKPOINT/recurrent_state.pt" \
  "$CHECKPOINT/checkpoint_completion.json"; do test -f "$path"; done
test "$(df --output=avail /tmp | tail -1)" -ge 83886080
test "$(sha256sum "$PARENT_PAIRS" | cut -d' ' -f1)" = edcdb934a65252557f6c569eea9ef7a85e5445549deac9fb4829c3f579e1a9b8
test "$(sha256sum scripts/evaluate_recurrent_causal_gate.py | cut -d' ' -f1)" = 69fd4f2f4943a78a0afe9a701533d9ab034f247f87c7a09327f7523e310b7a52
test "$(sha256sum scripts/evaluate_recurrent_diagnosis_c4_diagnosis_first.py | cut -d' ' -f1)" = 660df80664c6012f6eac1d379cc1d611c9f085c91496d1c9cc8e7f7bcb30d7ca

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
assert json.load(open(root/'checkpoint_completion.json'))=={'sha256':expected,'status':'complete','step':1200}
p=torch.load(root/'recurrent_state.pt',map_location='cpu',weights_only=False,mmap=True); c=p['config']; h=p['history']
assert p['step']==1200 and len(h)==1200,(p['step'],len(h))
assert c['state_architecture']=='layerwise' and c['state_token_count']==8,c
assert c['final_inference_route']=='strict_state_only',c
assert c.get('raw_history_visible_to_decoder',False) is False,c
assert c['diagnosis_field_order']==['diagnosis','evidence','label'],c
assert sum(float(r['history_dropout'])==0.0 for r in h)==1000,h[-1]
assert sum(float(r['history_dropout'])==0.25 for r in h)==200,h[-1]
PY

python - "$PARENT_PAIRS" "$PAIRS" <<'PY'
import hashlib,json,pathlib,sys
source,path=map(pathlib.Path,sys.argv[1:]); value=json.loads(source.read_text())
assert value['num_pairs']==150 and len(value['pairs'])==150,value
out={**value,'num_pairs':50,'pairs':value['pairs'][:50],'diagnostic_only':True,
     'selection':'ordered_prefix','parent_manifest_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
path.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
PY

rm -rf "$MODEL_LOCAL"; mkdir -p "$MODEL_LOCAL"
python scripts/copy_model_parallel.py --source "$MODEL_REMOTE" --destination "$MODEL_LOCAL" --workers 16
python -u scripts/evaluate_recurrent_causal_gate.py \
  --model "$MODEL_LOCAL" --adapter "$CHECKPOINT/adapter" \
  --recurrent-state "$CHECKPOINT/recurrent_state.pt" --state-architecture layerwise \
  --jsonl "$DATA" --pair-manifest "$PAIRS" \
  --output-json "$ANSWER_OUT/recurrent_causal_gate.json" --state-token-alias '<|vision_pad|>' \
  --max-length 1536 --num-pairs 50 --seed 42
python -u scripts/evaluate_recurrent_diagnosis_c4_diagnosis_first.py \
  --model "$MODEL_LOCAL" --adapter "$CHECKPOINT/adapter" \
  --recurrent-state "$CHECKPOINT/recurrent_state.pt" --state-architecture layerwise \
  --jsonl "$DATA" --pair-manifest "$PAIRS" --output-dir "$DIAG_OUT" \
  --state-token-alias '<|vision_pad|>' --max-length 1536 --max-new-tokens 160 \
  --num-pairs 50 --seed 42 --expected-adapter-sha256 "$EXPECTED_ADAPTER" \
  --expected-recurrent-state-sha256 "$EXPECTED_STATE"
python - "$ANSWER_OUT/recurrent_causal_gate.json" "$DIAG_OUT/metrics.json" "$EXPECTED_ADAPTER" "$EXPECTED_STATE" <<'PY'
import json,sys
a=json.load(open(sys.argv[1])); d=json.load(open(sys.argv[2])); expected={'adapter_model':sys.argv[3],'recurrent_state':sys.argv[4]}
assert a['artifact_sha256']==expected and d['artifact_sha256']==expected,(a,d)
assert a['state_shape']==[64,8,5120] and a['summary']['all_pairs_n']==50,a
assert d['state_shape']==[64,8,5120] and d['num_pairs']==50,d
assert a['raw_history_visible_to_decoder'] is False and d['raw_history_visible_to_decoder'] is False,(a,d)
assert a['persistent_history_kv_cache'] is False and d['persistent_history_kv_cache'] is False,(a,d)
assert d['conditions']==['native','noop','samemal','random','swap'],d
assert d['semantic_judgment_pending'] is True,d
PY
cp "$CHECKPOINT/checkpoint_completion.json" "$AMLT_OUTPUT_DIR/metrics/checkpoint_completion.json"
