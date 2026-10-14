#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "STATE8_EXACT_RESCUE_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

: "${VENV_ROOT:?VENV_ROOT is required}"
: "${EXACT_SOURCE_ROOT:?EXACT_SOURCE_ROOT is required}"
: "${EXACT_CURRICULUM_ROOT:?EXACT_CURRICULUM_ROOT is required}"
: "${EXPECTED_STUDENT_ADAPTER_SHA256:?EXPECTED_STUDENT_ADAPTER_SHA256 is required}"
: "${EXPECTED_STUDENT_STATE_SHA256:?EXPECTED_STUDENT_STATE_SHA256 is required}"
: "${STATE8_COMPOSED_ENTRYPOINT_SHA256:?STATE8_COMPOSED_ENTRYPOINT_SHA256 is required}"
: "${STATE8_RETENTION_ENTRYPOINT_SHA256:?STATE8_RETENTION_ENTRYPOINT_SHA256 is required}"
: "${STATE8_DIAGNOSIS_FIRST_ENTRYPOINT_SHA256:?STATE8_DIAGNOSIS_FIRST_ENTRYPOINT_SHA256 is required}"
: "${STATE8_RESCUE_MODE:?STATE8_RESCUE_MODE is required}"
[[ "$EXACT_SOURCE_ROOT" == \
  /mnt/amulet/studentmodeling/diagnostics/state8_stateonly_repair_seed65_trajectory01_exact_teacher_curriculum_c4_seed65_v1 ]] \
  || exit 2
[[ "$EXACT_CURRICULUM_ROOT" == \
  /mnt/amulet/studentmodeling/diagnostics/state8_qualified_teacher_curriculum_highkd_v1_immutable_taxonomy_semantic_v1_exact_diagnosis_first_seed64_single_ga4 ]] \
  || exit 2
[[ "$EXPECTED_STUDENT_ADAPTER_SHA256" =~ ^[0-9a-f]{64}$ ]] || exit 2
[[ "$EXPECTED_STUDENT_STATE_SHA256" =~ ^[0-9a-f]{64}$ ]] || exit 2

export STUDENT_ROOT="$EXACT_SOURCE_ROOT"
export STUDENT_MODEL_DIR="$STUDENT_ROOT/model"
export STUDENT_STATE_PATH="$STUDENT_MODEL_DIR/recurrent_state.pt"
export STUDENT_RECEIPT="$STUDENT_ROOT/metrics/publish_receipt.json"
case "$STATE8_RESCUE_MODE" in
  semantic_preserve)
    export TRAIN_ENTRYPOINT=scripts/train_state8_diagnosis_first_causal_repair.py
    export REPAIR_ARM=trajectory01 ;;
  causal_taskanchor)
    export TRAIN_ENTRYPOINT=scripts/train_state8_diagnosis_first_task_anchored_causal_repair.py
    export REPAIR_ARM=taskanchor ;;
  retention_preserve)
    export TRAIN_ENTRYPOINT=scripts/train_state8_diagnosis_first_retention_anchored_causal_repair.py
    export REPAIR_ARM=taskanchor ;;
  *) echo "Unknown STATE8_RESCUE_MODE=$STATE8_RESCUE_MODE" >&2; exit 2 ;;
esac
export REQUESTED_WORLD_SIZE=1
export GRADIENT_ACCUMULATION_STEPS=4
export STATE8_REPAIR_SAVE_STEPS=200
export STATE8_REPAIR_RECOVERY_SAVE_STEPS=200
export STATE8_REPAIR_LABEL_WEIGHT=1.0
export STATE8_CAUSAL_LABEL_WEIGHT=1.0
export STATE8_RANDOM_NEUTRAL=1
export STATE8_CAUSAL_DIAGNOSIS_ONLY=0
export STATE8_LAYERWISE_TRANSITION_ONLY=0
export STATE8_UPDATER_ONLY=0
export STATE8_FORMAL_SCOPE=unified
export STATE8_REPAIR_MANIFEST=manifests/layerwise_task_retention_staged_t3_t6_seed42_v1.json
export STATE8_REPAIR_MANIFEST_SHA256=a4eb41b2130f7e0810eeff36b5f3d16b8dd2c8631c50dbd25216f0793af23557
if [[ "$STATE8_RESCUE_MODE" == retention_preserve ]]; then
  export STATE8_REPAIR_MANIFEST=manifests/layerwise_eedi_open_t3_t6_repair_balanced_seed42_v1.json
  export STATE8_REPAIR_MANIFEST_SHA256=08945e930bc8b9a55e57a41d4656612e3dfce99b8544c88068110a74a1a22da9
fi
export STATE8_T3_ANCHOR_SCHEDULE="${STATE8_T3_ANCHOR_SCHEDULE:-1.0,0.5,0.25}"
export STATE8_T6_ANCHOR_SCHEDULE="${STATE8_T6_ANCHOR_SCHEDULE:-0.5,0.75,1.0}"
export STATE8_OTHER_ANCHOR_WEIGHT="${STATE8_OTHER_ANCHOR_WEIGHT:-0.1}"

test "$(sha256sum scripts/train_state8_diagnosis_first_task_anchored_causal_repair.py | cut -d' ' -f1)" \
  = "$STATE8_COMPOSED_ENTRYPOINT_SHA256"
test "$(sha256sum scripts/train_state8_diagnosis_first_causal_repair.py | cut -d' ' -f1)" \
  = "$STATE8_DIAGNOSIS_FIRST_ENTRYPOINT_SHA256"
test "$(sha256sum scripts/train_state8_diagnosis_first_retention_anchored_causal_repair.py | cut -d' ' -f1)" \
  = "$STATE8_RETENTION_ENTRYPOINT_SHA256"

# The established launcher is shared with already-running jobs and therefore
# remains immutable.  Render a fail-closed exact-rescue variant in /tmp that
# only adds the composed entrypoint and the current frozen trainer hash.
python - scripts/run_state8_stateonly_repair_ab.sh \
  /tmp/run_state8_exact_scientific_rescue.rendered.sh \
  "$STATE8_COMPOSED_ENTRYPOINT_SHA256" "$STATE8_RETENTION_ENTRYPOINT_SHA256" <<'PY'
import hashlib,pathlib,sys
source,output=map(pathlib.Path,sys.argv[1:3]); composed=sys.argv[3]; retention=sys.argv[4]
raw=source.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='2d52d2d84b64dcc1d177583ddb96b006e6afdeee74cb2d3241325ae170a059b7'
text=raw.decode()
text=text.replace(
 "test \"$(sha256sum scripts/train_recurrent_student_m4.py | cut -d' ' -f1)\" = ae502968872ff6597e73bcc12cb3ff9a5bbb0fe9d7a763383f6e76bc392de42c",
 "test \"$(sha256sum scripts/train_recurrent_student_m4.py | cut -d' ' -f1)\" = a67662448408a587f891d32d4c5170fdca0435adac47554268edf5a3972330b5",
)
old='''  scripts/train_state8_diagnosis_first_causal_repair.py)
    test "$(sha256sum "$TRAIN_ENTRYPOINT" | cut -d' ' -f1)" = "${STATE8_DIAGNOSIS_FIRST_ENTRYPOINT_SHA256:?}" 
    test "$(sha256sum scripts/state8_taxonomy_contract.py | cut -d' ' -f1)" = 1ca32112e76c795502f3cb1edbedcc7220f093c133fae8d20ff5cab789e512ed
    test "${STATE8_REPAIR_LABEL_WEIGHT:-}" = 1.0
    test "${STATE8_CAUSAL_LABEL_WEIGHT:-}" = 1.0 ;;'''
new=old+f'''
  scripts/train_state8_diagnosis_first_task_anchored_causal_repair.py)
    test "$(sha256sum "$TRAIN_ENTRYPOINT" | cut -d' ' -f1)" = "${{STATE8_COMPOSED_ENTRYPOINT_SHA256:?}}"
    test "$(sha256sum scripts/state8_taxonomy_contract.py | cut -d' ' -f1)" = 1ca32112e76c795502f3cb1edbedcc7220f093c133fae8d20ff5cab789e512ed
    test "${{STATE8_REPAIR_LABEL_WEIGHT:-}}" = 1.0
    test "${{STATE8_CAUSAL_LABEL_WEIGHT:-}}" = 1.0 ;;'''
new+=f'''
  scripts/train_state8_diagnosis_first_retention_anchored_causal_repair.py)
    test "$(sha256sum "$TRAIN_ENTRYPOINT" | cut -d' ' -f1)" = "${{STATE8_RETENTION_ENTRYPOINT_SHA256:?}}"
    test "$(sha256sum scripts/state8_taxonomy_contract.py | cut -d' ' -f1)" = 1ca32112e76c795502f3cb1edbedcc7220f093c133fae8d20ff5cab789e512ed
    test "${{STATE8_REPAIR_LABEL_WEIGHT:-}}" = 1.0
    test "${{STATE8_CAUSAL_LABEL_WEIGHT:-}}" = 1.0 ;;'''
assert text.count(old)==1
text=text.replace(old,new)
needle='|| "$TRAIN_ENTRYPOINT" == scripts/train_state8_diagnosis_first_causal_repair.py'
text=text.replace(needle,needle+' \\\n      || "$TRAIN_ENTRYPOINT" == scripts/train_state8_diagnosis_first_task_anchored_causal_repair.py')
text=text.replace(
 '|| "$TRAIN_ENTRYPOINT" == scripts/train_state8_diagnosis_first_task_anchored_causal_repair.py',
 '|| "$TRAIN_ENTRYPOINT" == scripts/train_state8_diagnosis_first_task_anchored_causal_repair.py \\'
 + chr(10)
 + '      || "$TRAIN_ENTRYPOINT" == scripts/train_state8_diagnosis_first_retention_anchored_causal_repair.py',
)
old='''if [[ "$TRAIN_ENTRYPOINT" == scripts/train_state8_task_anchored_causal_repair.py ]]; then
  python -m pytest -q tests/test_train_state8_task_anchored_causal_repair.py
fi'''
new=old+'''
if [[ "$TRAIN_ENTRYPOINT" == scripts/train_state8_diagnosis_first_task_anchored_causal_repair.py ]]; then
  python -m pytest -q tests/test_train_state8_task_anchored_causal_repair.py \\
    tests/test_train_state8_diagnosis_first_causal_repair.py \\
    tests/test_train_state8_diagnosis_first_task_anchored_causal_repair.py
fi'''
new+='''
if [[ "$TRAIN_ENTRYPOINT" == scripts/train_state8_diagnosis_first_retention_anchored_causal_repair.py ]]; then
  python -m pytest -q tests/test_train_state8_diagnosis_first_causal_repair.py \\
    tests/test_train_state8_diagnosis_first_retention_anchored_causal_repair.py
fi'''
assert text.count(old)==1; text=text.replace(old,new)
old="""if [[ "$TRAIN_ENTRYPOINT" == scripts/train_state8_diagnosis_first_causal_repair.py ]]; then
  FORMAL_EVALUATOR=scripts/evaluate_recurrent_formal_protocols_diagnosis_first.py"""
new="""if [[ "$TRAIN_ENTRYPOINT" == scripts/train_state8_diagnosis_first_causal_repair.py \\
      || "$TRAIN_ENTRYPOINT" == scripts/train_state8_diagnosis_first_task_anchored_causal_repair.py \\
      || "$TRAIN_ENTRYPOINT" == scripts/train_state8_diagnosis_first_retention_anchored_causal_repair.py ]]; then
  FORMAL_EVALUATOR=scripts/evaluate_recurrent_formal_protocols_diagnosis_first.py"""
assert text.count(old)==1; text=text.replace(old,new)
old="""if [[ "$TRAIN_ENTRYPOINT" == scripts/train_state8_task_anchored_causal_repair.py ]]; then
  test -f "$RUN_DIR/task_anchored_causal_repair.json"
fi"""
new=old+"""
if [[ "$TRAIN_ENTRYPOINT" == scripts/train_state8_diagnosis_first_task_anchored_causal_repair.py ]]; then
  test -f "$RUN_DIR/diagnosis_first_task_anchored_causal_repair.json"
fi"""
new+="""
if [[ "$TRAIN_ENTRYPOINT" == scripts/train_state8_diagnosis_first_retention_anchored_causal_repair.py ]]; then
  test -f "$RUN_DIR/diagnosis_first_retention_anchored_causal_repair.json"
fi"""
assert text.count(old)==1; text=text.replace(old,new)
old="""                        'train_state8_diagnosis_first_causal_repair.py')):"""
new="""                        'train_state8_diagnosis_first_causal_repair.py',
                        'train_state8_diagnosis_first_task_anchored_causal_repair.py',
                        'train_state8_diagnosis_first_retention_anchored_causal_repair.py')):"""
assert text.count(old)==1; text=text.replace(old,new)
output.write_text(text)
print(hashlib.sha256(text.encode()).hexdigest())
PY
bash -n /tmp/run_state8_exact_scientific_rescue.rendered.sh

python - "$STUDENT_RECEIPT" "$STUDENT_MODEL_DIR/adapter_model.safetensors" \
  "$STUDENT_STATE_PATH" "$EXPECTED_STUDENT_ADAPTER_SHA256" \
  "$EXPECTED_STUDENT_STATE_SHA256" <<'PY'
import hashlib,json,sys
receipt=json.load(open(sys.argv[1]))
def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
 return h.hexdigest()
observed={'adapter_model':digest(sys.argv[2]),'recurrent_state':digest(sys.argv[3])}
assert receipt['stage']=='state8_stateonly_repair_pilot_publish_v1' and receipt['status']=='complete',receipt
assert receipt['train_entrypoint']=='scripts/train_state8_diagnosis_first_causal_repair.py',receipt
assert receipt['state_shape']==[64,8,5120] and receipt['raw_history_visible_to_decoder'] is False,receipt
assert observed==receipt['artifact_sha256'],(observed,receipt)
assert observed=={'adapter_model':sys.argv[4],'recurrent_state':sys.argv[5]},observed
PY

bash /tmp/run_state8_exact_scientific_rescue.rendered.sh

test -f "$AMLT_OUTPUT_DIR/metrics/diagnosis_first_causal_repair.json"
if [[ "$STATE8_RESCUE_MODE" == causal_taskanchor ]]; then
  test -f "$AMLT_OUTPUT_DIR/metrics/diagnosis_first_task_anchored_causal_repair.json"
fi
if [[ "$STATE8_RESCUE_MODE" == retention_preserve ]]; then
  test -f "$AMLT_OUTPUT_DIR/metrics/diagnosis_first_retention_anchored_causal_repair.json"
fi
# Bind every rescue candidate to the same audited five-stage curriculum used
# by the exact parent.  These receipts are required by the common promotion
# selector; copying metadata does not expose history to the decoder.
cp "$EXACT_CURRICULUM_ROOT/metrics/immutable_receipt.json" \
  "$AMLT_OUTPUT_DIR/metrics/immutable_receipt.json"
cp "$EXACT_CURRICULUM_ROOT/metrics/diagnosis_first_teacher_curriculum.json" \
  "$AMLT_OUTPUT_DIR/metrics/diagnosis_first_teacher_curriculum.json"
answer_out="$AMLT_OUTPUT_DIR/eval/answer_c4_150"
mkdir -p "$answer_out"
"$VENV_ROOT/bin/python" -u scripts/evaluate_recurrent_causal_gate.py \
  --model /tmp/Qwen3-32B --adapter "$AMLT_OUTPUT_DIR/model" \
  --recurrent-state "$AMLT_OUTPUT_DIR/model/recurrent_state.pt" \
  --state-architecture layerwise \
  --jsonl /mnt/amulet/studentmodeling/c4_fullopen_20260724/malrule_c1_clean/test_same_template.jsonl \
  --pair-manifest manifests/c4_stage_a_gate_pairs_seed42_n150_v1.json \
  --output-json "$answer_out/recurrent_causal_gate.json" \
  --state-token-alias '<|vision_pad|>' --max-length 1536 --num-pairs 150 --seed 42
"$VENV_ROOT/bin/python" - "$answer_out/recurrent_causal_gate.json" \
  "$AMLT_OUTPUT_DIR/metrics/publish_receipt.json" <<'PY'
import json,sys
answer,publish=map(lambda p:json.load(open(p)),sys.argv[1:])
assert answer['state_shape']==[64,8,5120] and answer['raw_history_visible_to_decoder'] is False,answer
assert answer['persistent_history_kv_cache'] is False,answer
assert answer['artifact_sha256']==publish['artifact_sha256'],(answer,publish)
assert answer['summary']['all_pairs_n']==150,answer
PY
