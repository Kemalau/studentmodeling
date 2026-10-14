#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "STATE8_EXACT_MIDPOINT_C4_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

export VENV_ROOT="${VENV_ROOT:-/tmp/studentmodeling-state8-exact-midpoint-eval-venv}"
export PATH="$VENV_ROOT/bin:$PATH"
python(){ "$VENV_ROOT/bin/python" "$@"; }
export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/hf_cache
export HF_HUB_CACHE=/tmp/hf_cache/hub
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error

: "${STATE8_EXACT_CHECKPOINT_ROOT:?STATE8_EXACT_CHECKPOINT_ROOT is required}"
: "${EXACT_DIAGNOSIS_FORMAL_SHA256:?EXACT_DIAGNOSIS_FORMAL_SHA256 is required}"
: "${EXACT_DIAGNOSIS_C4_SHA256:?EXACT_DIAGNOSIS_C4_SHA256 is required}"
CHECKPOINT_STEP="${STATE8_EXACT_CHECKPOINT_STEP:-2000}"
case "$CHECKPOINT_STEP" in 2000|3000|4000|5000) ;; *) exit 2 ;; esac
EXPECTED_ROOT=/mnt/amulet/studentmodeling/checkpoints/state8_qualified_teacher_curriculum_seed42_exact5000_rounds1250_kd1_v2_20260805_taxonomy_semantic_v1_exact_diagnosis_first_seed64_single_ga4/checkpoint-$CHECKPOINT_STEP
test "$STATE8_EXACT_CHECKPOINT_ROOT" = "$EXPECTED_ROOT"
test "$(sha256sum scripts/evaluate_recurrent_formal_protocols_diagnosis_first.py | cut -d' ' -f1)" = "$EXACT_DIAGNOSIS_FORMAL_SHA256"
test "$(sha256sum scripts/evaluate_recurrent_diagnosis_c4_diagnosis_first.py | cut -d' ' -f1)" = "$EXACT_DIAGNOSIS_C4_SHA256"

MODEL_REMOTE=/mnt/amulet/studentmodeling/models/Qwen3-32B
MODEL_LOCAL=/tmp/Qwen3-32B
ADAPTER="$STATE8_EXACT_CHECKPOINT_ROOT/adapter"
STATE="$STATE8_EXACT_CHECKPOINT_ROOT/recurrent_state.pt"
RECEIPT="$STATE8_EXACT_CHECKPOINT_ROOT/checkpoint_completion.json"
DATA=/mnt/amulet/studentmodeling/c4_fullopen_20260724/malrule_c1_clean/test_same_template.jsonl
PARENT_PAIRS=manifests/c4_stage_a_gate_pairs_seed42_n150_v1.json
PAIRS=/tmp/state8_exact_diagnosis_midpoint_pairs_seed42_n50.json
ANSWER_OUT="$AMLT_OUTPUT_DIR/eval/answer_c4_50"
DIAG_OUT="$AMLT_OUTPUT_DIR/eval/diagnosis_c4_50"

mkdir -p "$HF_HUB_CACHE" "$MODEL_LOCAL" "$ANSWER_OUT" "$DIAG_OUT"
for path in "$MODEL_REMOTE/model.safetensors.index.json" "$ADAPTER/adapter_model.safetensors" \
  "$STATE" "$RECEIPT" "$DATA" "$PARENT_PAIRS"; do test -f "$path"; done
test "$(sha256sum "$PARENT_PAIRS" | cut -d' ' -f1)" = edcdb934a65252557f6c569eea9ef7a85e5445549deac9fb4829c3f579e1a9b8
test "$(df --output=avail /tmp | tail -1)" -ge 83886080

python - "$PARENT_PAIRS" "$PAIRS" <<'PY'
import hashlib,json,pathlib,sys
source,path=map(pathlib.Path,sys.argv[1:])
value=json.loads(source.read_text(encoding='utf-8'))
assert value['num_pairs']==150 and len(value['pairs'])==150,value
out={**value,'num_pairs':50,'pairs':value['pairs'][:50],
     'diagnostic_only':True,'selection':'ordered_prefix',
     'parent_manifest_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
temporary=path.with_suffix(path.suffix+'.partial')
temporary.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
temporary.replace(path)
PY
python - "$PARENT_PAIRS" "$PAIRS" <<'PY'
import hashlib,json,pathlib,sys
parent,child=map(pathlib.Path,sys.argv[1:]); p=json.load(parent.open()); c=json.load(child.open())
assert c['num_pairs']==50 and c['pairs']==p['pairs'][:50],c
assert c['selection']=='ordered_prefix' and c['diagnostic_only'] is True,c
assert c['parent_manifest_sha256']==hashlib.sha256(parent.read_bytes()).hexdigest(),c
PY

readarray -t ARTIFACT_SHA256 < <(python - "$RECEIPT" "$ADAPTER/adapter_model.safetensors" "$STATE" "$CHECKPOINT_STEP" <<'PY'
import hashlib,json,pathlib,sys,torch
receipt,adapter,state=map(pathlib.Path,sys.argv[1:4]); expected_step=int(sys.argv[4])
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()
observed={'adapter_model':digest(adapter),'recurrent_state':digest(state)}
x=json.load(receipt.open()); assert x=={'sha256':observed,'status':'complete','step':expected_step},(x,observed)
p=torch.load(state,map_location='cpu',weights_only=False,mmap=True); c=p['config']; history=p['history']
assert p['step']==expected_step and len(history)==expected_step,(p['step'],len(history))
assert c['state_token_count']==8 and c['state_architecture']=='layerwise',c
assert c['final_inference_route']=='strict_state_only',c
assert c['diagnosis_field_order']==['diagnosis','evidence','label'],c
assert c['canonical_label_role']=='auxiliary_ordinary_weight',c
assert float(c['label_auxiliary_weight'])==1.0,c
schedule=[0.0,0.25,0.5,0.75,1.0]
for index,value in enumerate(schedule):
 expected=max(0,min(1000,expected_step-index*1000))
 assert sum(float(r['history_dropout'])==value for r in history)==expected,(value,expected)
print(observed['adapter_model']); print(observed['recurrent_state'])
PY
)
test "${#ARTIFACT_SHA256[@]}" = 2

rm -rf "$MODEL_LOCAL"; mkdir -p "$MODEL_LOCAL"
python scripts/copy_model_parallel.py --source "$MODEL_REMOTE" --destination "$MODEL_LOCAL" --workers 16
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python -u scripts/evaluate_recurrent_causal_gate.py \
  --model "$MODEL_LOCAL" --adapter "$ADAPTER" --recurrent-state "$STATE" \
  --state-architecture layerwise --jsonl "$DATA" --pair-manifest "$PAIRS" \
  --output-json "$ANSWER_OUT/recurrent_causal_gate.json" --state-token-alias '<|vision_pad|>' \
  --max-length 1536 --num-pairs 50 --seed 42
python -u scripts/evaluate_recurrent_diagnosis_c4_diagnosis_first.py \
  --model "$MODEL_LOCAL" --adapter "$ADAPTER" --recurrent-state "$STATE" \
  --state-architecture layerwise --jsonl "$DATA" --pair-manifest "$PAIRS" \
  --output-dir "$DIAG_OUT" --state-token-alias '<|vision_pad|>' --max-length 1536 \
  --max-new-tokens 160 --num-pairs 50 --seed 42 \
  --expected-adapter-sha256 "${ARTIFACT_SHA256[0]}" \
  --expected-recurrent-state-sha256 "${ARTIFACT_SHA256[1]}"
python - "$ANSWER_OUT/recurrent_causal_gate.json" "$DIAG_OUT/metrics.json" <<'PY'
import json,sys
a=json.load(open(sys.argv[1])); d=json.load(open(sys.argv[2]))
assert a['state_shape']==[64,8,5120] and a['summary']['all_pairs_n']==50,a
assert a['raw_history_visible_to_decoder'] is False and a['persistent_history_kv_cache'] is False,a
assert d['state_shape']==[64,8,5120] and d['num_pairs']==50,d
assert d['raw_history_visible_to_decoder'] is False and d['persistent_history_kv_cache'] is False,d
assert d['semantic_judgment_pending'] is True,d
PY
