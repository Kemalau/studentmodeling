#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "STATE8_XES_PROFILE_CONFIRMATORY_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

export VENV_ROOT="${VENV_ROOT:-/tmp/studentmodeling-state8-xes-profile-confirmatory-venv}"
export PATH="$VENV_ROOT/bin:$PATH"
python(){ "$VENV_ROOT/bin/python" "$@"; }
export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/hf_cache HF_HUB_CACHE=/tmp/hf_cache/hub HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRANSFORMERS_VERBOSITY=error

test "$(sha256sum scripts/evaluate_state8_reusable_interface.py | cut -d' ' -f1)" = d74c6859912e00b0d2ce116c557206e3e8c000a0e890cbc1eb6f7e6ae35d78aa
test "$(sha256sum scripts/verify_xes_profile_confirmatory_freeze.py | cut -d' ' -f1)" = c0ec3bd97a1b642592e356d24ff430f2b55e561872aa4b743fa15ea59eedba36

MODEL_REMOTE=/mnt/amulet/studentmodeling/models/Qwen3-32B
MODEL_LOCAL=/tmp/Qwen3-32B
RUN_DIR=/mnt/amulet/studentmodeling/checkpoints/state8_profile_ownership_lrfix_v4_formal_world4_seed8801
ADAPTER="$RUN_DIR/final_adapter"
STATE="$RUN_DIR/recurrent_state.pt"
DEVELOPMENT_METRICS=/mnt/amulet/studentmodeling/diagnostics/state8_profile_ownership_lrfix_v4_formal_world4_seed8801/eval/reusable_interface/metrics.json
MANIFEST=manifests/xes_state8_profile_confirmatory_v1
FEATURES="$AMLT_OUTPUT_DIR/features"
ANALYSIS="$AMLT_OUTPUT_DIR/analysis"

for path in "$MODEL_REMOTE/model.safetensors.index.json" \
  "$ADAPTER/adapter_model.safetensors" "$STATE" "$RUN_DIR/completion.json" \
  "$DEVELOPMENT_METRICS" \
  "$MANIFEST/manifest.json" "$MANIFEST/train.jsonl" "$MANIFEST/valid.jsonl" \
  "$MANIFEST/test.jsonl"; do
  test -f "$path"
done

# The frozen manifest verifier reads hashes/cardinalities only.  It never fits,
# scores, or prints confirmatory test metrics.
python scripts/verify_xes_profile_confirmatory_freeze.py \
  --manifest-dir "$MANIFEST" > "$AMLT_OUTPUT_DIR/freeze_verification.json"
python - "$RUN_DIR/completion.json" "$STATE" "$ADAPTER/adapter_model.safetensors" \
  "$DEVELOPMENT_METRICS" "$AMLT_OUTPUT_DIR/checkpoint_binding.json" <<'PY'
import hashlib,json,sys
completion=json.load(open(sys.argv[1]))
assert completion['status']=='complete' and completion['micro_steps']==1000,completion
assert completion['history_integrity_pass'] is True,completion
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()
metrics=json.load(open(sys.argv[4]))
assert metrics['scientific_gate_pass'] is True and all(metrics['gates'].values()),metrics
bindings={
    (value['checkpoint']['adapter'],value['checkpoint']['recurrent_state'])
    for value in metrics['feature_artifacts'].values()
}
assert len(bindings)==1,bindings
adapter_hash,state_hash=next(iter(bindings))
assert sha(sys.argv[2])==state_hash
assert sha(sys.argv[3])==adapter_hash
receipt={
    'status':'frozen_checkpoint_bound',
    'adapter_sha256':adapter_hash,
    'recurrent_state_sha256':state_hash,
    'development_scientific_gate_pass':True,
    'development_test_labels_used_for_selection':False,
}
open(sys.argv[5],'w').write(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps(receipt,sort_keys=True))
PY

rm -rf "$MODEL_LOCAL" "$FEATURES" "$ANALYSIS"
mkdir -p "$MODEL_LOCAL" "$FEATURES" "$ANALYSIS"
python scripts/copy_model_parallel.py --source "$MODEL_REMOTE" \
  --destination "$MODEL_LOCAL" --workers 16
test -f "$MODEL_LOCAL/model.safetensors.index.json"

# Fit the fixed linear readout and select ridge alpha on official train/valid
# students only.  Official XES test is touched once, after the checkpoint and
# protocol above are frozen.
for split in train valid test; do
  python -u scripts/evaluate_state8_reusable_interface.py extract \
    --model "$MODEL_LOCAL" --adapter "$ADAPTER" --recurrent-state "$STATE" \
    --rows-jsonl "$MANIFEST/$split.jsonl" --split "$split" \
    --output "$FEATURES/$split.npz" --state-token-alias '<|vision_pad|>' \
    --max-length 1536 --batch-size 4 --projection-dim 64 \
    --projection-seed 20260818
done

python -u scripts/evaluate_state8_reusable_interface.py analyze \
  --manifest-dir "$MANIFEST" --feature-dir "$FEATURES" \
  --output-dir "$ANALYSIS" --bootstrap-replicates 5000 \
  --permutation-replicates 2000 --seed 20260820

python - "$ANALYSIS/metrics.json" "$AMLT_OUTPUT_DIR/freeze_verification.json" <<'PY'
import json,sys
metrics=json.load(open(sys.argv[1])); freeze=json.load(open(sys.argv[2]))
assert freeze['status']=='pass' and freeze['test_metrics_inspected_by_this_verifier'] is False,freeze
assert metrics['status']=='complete' and metrics['students']=={'train':800,'valid':200,'test':466},metrics
assert metrics['student_disjoint'] is True and metrics['chronological_future_targets'] is True,metrics
assert metrics['raw_history_visible_to_readout'] is False,metrics
assert metrics['student_id_visible_to_readout'] is False,metrics
assert metrics['future_support_visible_to_readout'] is False,metrics
assert metrics['readout_selection']['selection_split']=='validation_students_only',metrics
assert metrics['readout_selection']['test_labels_used_for_selection'] is False,metrics
print(json.dumps({'status':'confirmatory_complete','scientific_gate_pass':metrics['scientific_gate_pass'],'gates':metrics['gates']},sort_keys=True))
PY
