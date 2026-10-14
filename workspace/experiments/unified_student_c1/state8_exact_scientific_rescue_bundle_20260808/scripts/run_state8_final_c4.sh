#!/usr/bin/env bash
set -euo pipefail

export VENV_ROOT="${VENV_ROOT:-/tmp/studentmodeling-state8-final-eval-venv}"
export PATH="$VENV_ROOT/bin:$PATH"
python(){ "$VENV_ROOT/bin/python" "$@"; }
export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/hf_cache
export HF_HUB_CACHE=/tmp/hf_cache/hub
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error
export LOCAL_MODEL=/tmp/Qwen3-32B
export RUN_DIR=/mnt/amulet/studentmodeling/checkpoints/state8_qualified_unified_full259845_seed42_v1_20260805
if [[ -n "${STATE8_FINAL_EVAL_RUN_DIR:-}" ]]; then export RUN_DIR="$STATE8_FINAL_EVAL_RUN_DIR"; fi
export C4_DATA=/mnt/amulet/studentmodeling/c4_fullopen_20260724/malrule_c1_clean
: "${PAIR_SEED:?PAIR_SEED is required}"
: "${PAIR_MANIFEST_SHA256:?PAIR_MANIFEST_SHA256 is required}"
export PAIR_MANIFEST="manifests/c4_stage_a_gate_pairs_seed${PAIR_SEED}_n150_v1.json"
export FINAL_EVAL_PROGRESS_NAMESPACE="${STATE8_FINAL_EVAL_PROGRESS_NAMESPACE:-state8_final_full259845}"
export C4_PROGRESS_DIR="/mnt/amulet/studentmodeling/eval_progress/${FINAL_EVAL_PROGRESS_NAMESPACE}/c4_seed${PAIR_SEED}_v1"
export OUT="$AMLT_OUTPUT_DIR/eval/c4_seed$PAIR_SEED"
export DIAGNOSIS_OUT="$AMLT_OUTPUT_DIR/eval/diagnosis_c4_seed$PAIR_SEED"

mkdir -p "$HF_HUB_CACHE" "$C4_PROGRESS_DIR" "$OUT" "$DIAGNOSIS_OUT"
test -f "$RUN_DIR/final_adapter/adapter_model.safetensors"
test -f "$RUN_DIR/recurrent_state.pt"
test -f "$RUN_DIR/completion.json"
test -f "$RUN_DIR/state8_final_fit_receipt.json"
test -f "$RUN_DIR/canonical_publish_receipt.json"
test -f "$C4_DATA/test_same_template.jsonl"
test -f "$PAIR_MANIFEST"
test "$(df --output=avail /tmp | tail -1)" -ge 83886080
test "$(sha256sum "$PAIR_MANIFEST" | cut -d' ' -f1)" = "$PAIR_MANIFEST_SHA256"
test "$(sha256sum scripts/evaluate_recurrent_causal_gate.py | cut -d' ' -f1)" = e0c9fe1361dfda5eddc14abcd275fbe8e2b92f472f9eaf27468b71f800fafccf
test "$(sha256sum scripts/evaluate_recurrent_diagnosis_c4.py | cut -d' ' -f1)" = e7421a850c2327d8df1436073742e71f796bec41c87f68403b7b00b60fac4664

DIAGNOSIS_EVALUATOR=scripts/evaluate_recurrent_diagnosis_c4.py
SELECTED_VARIANT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_replay_variant"])' "$RUN_DIR/state8_final_fit_receipt.json")"
if [[ "$SELECTED_VARIANT" == highcausal_taxonomy \
      || "$SELECTED_VARIANT" == taxonomy_curriculum_t3heavy \
      || "$SELECTED_VARIANT" == taxonomy_curriculum_balanced \
      || "$SELECTED_VARIANT" == taxonomy_curriculum_t3heavy_diag \
      || "$SELECTED_VARIANT" == taxonomy_curriculum_balanced_diag \
      || "$SELECTED_VARIANT" == task_anchor_base \
      || "$SELECTED_VARIANT" == task_anchor_c4strong \
      || "$SELECTED_VARIANT" == strong1_widen8_lr5e7_seed58 \
      || "$SELECTED_VARIANT" == strong1_widen8_lr1e6_seed59 \
      || "$SELECTED_VARIANT" == strong1_widen8_single_lr5e7_seed60 \
      || "$SELECTED_VARIANT" == strong1_rescue_taxonomy_joint_seed61 \
      || "$SELECTED_VARIANT" == strong1_rescue_taskanchor_c4strong_seed62 \
      || "$SELECTED_VARIANT" == strong1_midpoint_diagnosis_first_seed63 \
      || "$SELECTED_VARIANT" == strong1_exact_diagnosis_curriculum_seed65 ]]; then
  DIAGNOSIS_EVALUATOR=scripts/evaluate_recurrent_diagnosis_c4_taxonomy.py
  test -f "$RUN_DIR/taxonomy_causal_repair.json"
  test "$(sha256sum "$DIAGNOSIS_EVALUATOR" | cut -d' ' -f1)" = c3d34960d0f6caff3265b772686bd84222637f628d5fce000215d076c5b2a355
  test "$(sha256sum scripts/state8_taxonomy_contract.py | cut -d' ' -f1)" = 1ca32112e76c795502f3cb1edbedcc7220f093c133fae8d20ff5cab789e512ed
fi
if [[ "$SELECTED_VARIANT" == strong1_rescue_taxonomy_joint_seed61 \
      || "$SELECTED_VARIANT" == strong1_midpoint_diagnosis_first_seed63 \
      || "$SELECTED_VARIANT" == strong1_exact_diagnosis_curriculum_seed65 ]]; then
  DIAGNOSIS_EVALUATOR=scripts/evaluate_recurrent_diagnosis_c4_diagnosis_first.py
  test -f "$RUN_DIR/diagnosis_first_causal_repair.json"
  test "$(sha256sum "$DIAGNOSIS_EVALUATOR" | cut -d' ' -f1)" = 7eef90e079d698c94f1f8655a1df3fe6bd93cedbb175ba4e7082a49bb8069c8b
fi

readarray -t ARTIFACT_SHA256 < <(python - "$RUN_DIR" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); c=json.load(open(root/'completion.json')); r=json.load(open(root/'state8_final_fit_receipt.json'))
p=json.load(open(root/'canonical_publish_receipt.json'))
assert c['status']=='complete' and c['micro_steps']==259845 and c['exact_history_support'] is True,c
assert r['status']=='final_fit_complete' and r['state_shape']==[64,8,5120],r
assert r['stage']=='state8_qualified_unified_final_fit_v1',r
assert len(r['teacher_adapter_sha256'])==64 and r['teacher_gate_required'] is True,r
assert r['adapter_sha256']==c['sha256']['adapter_model'],(r,c)
assert r['recurrent_state_sha256']==c['sha256']['recurrent_state'],(r,c)
assert p['stage']=='state8_canonical_final_checkpoint_publish_v1' and p['status']=='complete',p
assert p['selected_replay_variant']==r['selected_replay_variant'],(p,r)
assert p['artifact_sha256']=={'adapter_model':r['adapter_sha256'],'recurrent_state':r['recurrent_state_sha256']},(p,r)
assert r['causal_supervision']==['prospective_answer','diagnosis_natural_schema'],r
assert r['causal_output_schema']=='label_diagnosis_evidence_v1',r
assert r['canonical_label_auxiliary_supervision'] is True,r
assert r['history_training_mode']=='strict_state_only',r
assert r['history_dropout_counts']=={'1.0':259845},r
assert float(r['online_kd_weight'])==1.0,r
expected_label_weight=1.0 if r['selected_replay_variant'] in {'strong1_rescue_taxonomy_joint_seed61','strong1_midpoint_diagnosis_first_seed63','strong1_exact_diagnosis_curriculum_seed65'} else 4.0
assert float(r['label_auxiliary_weight'])==expected_label_weight,r
if r['selected_replay_variant'] in {'strong1_rescue_taxonomy_joint_seed61','strong1_midpoint_diagnosis_first_seed63','strong1_exact_diagnosis_curriculum_seed65'}:
 assert r['diagnosis_field_order']==['diagnosis','evidence','label'],r
 assert r['canonical_label_role']=='auxiliary_ordinary_weight',r
print(r['adapter_sha256']); print(r['recurrent_state_sha256'])
PY
)
test "${#ARTIFACT_SHA256[@]}" = 2
export EXPECTED_ADAPTER_SHA256="${ARTIFACT_SHA256[0]}"
export EXPECTED_RECURRENT_STATE_SHA256="${ARTIFACT_SHA256[1]}"
export C4_PROGRESS_KEY="adapter=$EXPECTED_ADAPTER_SHA256:state=$EXPECTED_RECURRENT_STATE_SHA256"
python -m pytest -q tests/test_evaluate_recurrent_causal_gate.py
rm -rf "$LOCAL_MODEL"; mkdir -p "$LOCAL_MODEL"
python scripts/copy_model_parallel.py \
  --source /mnt/amulet/studentmodeling/models/Qwen3-32B \
  --destination "$LOCAL_MODEL" --workers 16
test -f "$LOCAL_MODEL/model.safetensors.index.json"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python -u scripts/evaluate_recurrent_causal_gate.py \
  --model "$LOCAL_MODEL" --adapter "$RUN_DIR/final_adapter" \
  --recurrent-state "$RUN_DIR/recurrent_state.pt" --state-architecture layerwise \
  --jsonl "$C4_DATA/test_same_template.jsonl" --pair-manifest "$PAIR_MANIFEST" \
  --output-json "$OUT/recurrent_causal_gate.json" \
  --state-token-alias '<|vision_pad|>' --max-length 1536 --num-pairs 150 \
  --seed "$PAIR_SEED" --progress-dir "$C4_PROGRESS_DIR" --progress-key "$C4_PROGRESS_KEY"
python - "$OUT/recurrent_causal_gate.json" "$PAIR_SEED" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); seed=int(sys.argv[2])
assert x['pair_seed']==seed and x['state_shape']==[64,8,5120],x
assert x['raw_history_visible_to_decoder'] is False,x
assert x['persistent_history_kv_cache'] is False,x
assert x['summary']['all_pairs_n']==150 and x['summary']['clean_n']>0,x
assert len(x['artifact_sha256']['adapter_model'])==64,x
assert len(x['artifact_sha256']['recurrent_state'])==64,x
PY
python -u "$DIAGNOSIS_EVALUATOR" \
  --model "$LOCAL_MODEL" --adapter "$RUN_DIR/final_adapter" \
  --recurrent-state "$RUN_DIR/recurrent_state.pt" --state-architecture layerwise \
  --jsonl "$C4_DATA/test_same_template.jsonl" --pair-manifest "$PAIR_MANIFEST" \
  --output-dir "$DIAGNOSIS_OUT" --state-token-alias '<|vision_pad|>' \
  --max-length 1536 --max-new-tokens 160 --num-pairs 150 --seed "$PAIR_SEED" \
  --expected-adapter-sha256 "$EXPECTED_ADAPTER_SHA256" \
  --expected-recurrent-state-sha256 "$EXPECTED_RECURRENT_STATE_SHA256"
python - "$DIAGNOSIS_OUT/metrics.json" "$PAIR_SEED" \
  "$EXPECTED_ADAPTER_SHA256" "$EXPECTED_RECURRENT_STATE_SHA256" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); seed=int(sys.argv[2])
assert x['stage']=='state8_closed_diagnosis_c4_smoke_v1',x
assert x['num_pairs']==150 and x['pair_seed']==seed,x
assert x['state_shape']==[64,8,5120],x
assert x['raw_history_visible_to_decoder'] is False,x
assert x['persistent_history_kv_cache'] is False,x
assert x['artifact_sha256']=={
 'adapter_model':sys.argv[3],'recurrent_state':sys.argv[4]},x
assert x['semantic_judgment_pending'] is True,x
PY
