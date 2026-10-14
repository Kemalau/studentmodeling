#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "STATE8_MATCHED_CP2000_RESCUE_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

: "${VENV_ROOT:?VENV_ROOT is required}"
: "${MATCHED_CHECKPOINT_ROOT:?MATCHED_CHECKPOINT_ROOT is required}"
: "${EXPECTED_STUDENT_ADAPTER_SHA256:?EXPECTED_STUDENT_ADAPTER_SHA256 is required}"
: "${EXPECTED_STUDENT_STATE_SHA256:?EXPECTED_STUDENT_STATE_SHA256 is required}"
: "${STATE8_RESCUE_MODE:?STATE8_RESCUE_MODE is required}"
: "${RUN_TAG:?RUN_TAG is required}"

expected_root=/mnt/amulet/studentmodeling/checkpoints/state8_qualified_teacher_curriculum_seed42_exact5000_rounds1250_kd1_v2_20260805_taxonomy_semantic_v1_diagnosis_first_matched_teacher_seed67_single_ga4/checkpoint-2000
teacher_root=/mnt/amulet/studentmodeling/diagnostics/state8_diagnosis_first_full_history_teacher_v1
[[ "$MATCHED_CHECKPOINT_ROOT" == "$expected_root" ]] || exit 2
[[ "$EXPECTED_STUDENT_ADAPTER_SHA256" =~ ^[0-9a-f]{64}$ ]] || exit 2
[[ "$EXPECTED_STUDENT_STATE_SHA256" =~ ^[0-9a-f]{64}$ ]] || exit 2
[[ "$RUN_TAG" =~ ^[a-z0-9_]+$ ]] || exit 2

export STUDENT_ROOT="$MATCHED_CHECKPOINT_ROOT"
export STUDENT_MODEL_DIR="$STUDENT_ROOT/adapter"
export STUDENT_STATE_PATH="$STUDENT_ROOT/recurrent_state.pt"
export STUDENT_RECEIPT="$STUDENT_ROOT/checkpoint_completion.json"
export REQUESTED_WORLD_SIZE=1
export GRADIENT_ACCUMULATION_STEPS=4
# Feeds has twice terminated this one-A100 job after roughly 35 minutes for
# infrastructure capability failures.  Twenty micro-steps is a completed
# GA=4 boundary and keeps each retry resumable without changing the objective.
export STATE8_REPAIR_SAVE_STEPS=20
export STATE8_REPAIR_RECOVERY_SAVE_STEPS=20
export STATE8_REPAIR_LABEL_WEIGHT=1.0
export STATE8_CAUSAL_LABEL_WEIGHT=1.0
export STATE8_RANDOM_NEUTRAL=1
export STATE8_CAUSAL_DIAGNOSIS_ONLY=0
export STATE8_LAYERWISE_TRANSITION_ONLY=0
export STATE8_UPDATER_ONLY=0
export STATE8_FORMAL_SCOPE=unified
export STATE8_REPAIR_MANIFEST=manifests/layerwise_task_retention_staged_t3_t6_seed42_v1.json
export STATE8_REPAIR_MANIFEST_SHA256=a4eb41b2130f7e0810eeff36b5f3d16b8dd2c8631c50dbd25216f0793af23557
export STATE8_T3_ANCHOR_SCHEDULE="${STATE8_T3_ANCHOR_SCHEDULE:-1.0,0.5,0.25}"
export STATE8_OTHER_ANCHOR_WEIGHT="${STATE8_OTHER_ANCHOR_WEIGHT:-0.1}"

case "$STATE8_RESCUE_MODE" in
  semantic_preserve)
    export TRAIN_ENTRYPOINT=scripts/train_state8_diagnosis_first_causal_repair.py
    export REPAIR_ARM=trajectory01
    export TRAIN_SEED=2066
    export STATE8_CAUSAL_REPEAT_FRACTION=0.50
    export STATE8_CAUSAL_WEIGHT=2.0
    export STATE8_CAUSAL_MARGIN=0.5
    export STATE8_GEOMETRY_WEIGHT=0.10
    export STATE8_EEDI_GEOMETRY_WEIGHT=0.05
    ;;
  causal_taskanchor)
    export TRAIN_ENTRYPOINT=scripts/train_state8_diagnosis_first_task_anchored_causal_repair.py
    export REPAIR_ARM=taskanchor
    export TRAIN_SEED=2067
    export STATE8_CAUSAL_REPEAT_FRACTION=1.0
    export STATE8_CAUSAL_WEIGHT=4.0
    export STATE8_CAUSAL_MARGIN=0.75
    export STATE8_GEOMETRY_WEIGHT=0.20
    export STATE8_EEDI_GEOMETRY_WEIGHT=0.10
    ;;
  *) echo "Unknown STATE8_RESCUE_MODE=$STATE8_RESCUE_MODE" >&2; exit 2 ;;
esac
export STATE8_DIFFERENT_MAX_COSINE=0.90

test "$(sha256sum scripts/run_state8_stateonly_repair_ab.sh | cut -d' ' -f1)" = 2d52d2d84b64dcc1d177583ddb96b006e6afdeee74cb2d3241325ae170a059b7
test "$(sha256sum scripts/train_recurrent_student_m4.py | cut -d' ' -f1)" = a67662448408a587f891d32d4c5170fdca0435adac47554268edf5a3972330b5
test "$(sha256sum scripts/select_latest_valid_single_checkpoint.py | cut -d' ' -f1)" = 224736e342bb55bd54ad284eb0907a880dccf9bb906e4d22d71b91edb618d243
test "$(sha256sum scripts/stage_distributed_resume_bundle.py | cut -d' ' -f1)" = c89b26b4e5b69d60774e2fceed57e7c7a635d4b5d403e7e0f1e952127ada6708
test "$(sha256sum scripts/train_state8_diagnosis_first_causal_repair.py | cut -d' ' -f1)" = 7d638013f932f29f52cffeea709b96872df8ba0960b9bb4e56ed23cc3475b216
test "$(sha256sum scripts/train_state8_diagnosis_first_task_anchored_causal_repair.py | cut -d' ' -f1)" = eb9837127af9fe3761f6fa57b78a3e9ba3373a223706f0a4921749eab61e35a3
test "$(sha256sum scripts/evaluate_recurrent_causal_gate.py | cut -d' ' -f1)" = 69fd4f2f4943a78a0afe9a701533d9ab034f247f87c7a09327f7523e310b7a52
test "$(sha256sum scripts/evaluate_recurrent_formal_protocols_diagnosis_first.py | cut -d' ' -f1)" = a8bc07a8fe6d2c9bfa9af6d976b53b84321cb4baa9c9fcace3188e370f7732c9
test "$(sha256sum scripts/evaluate_recurrent_diagnosis_c4_diagnosis_first.py | cut -d' ' -f1)" = 660df80664c6012f6eac1d379cc1d611c9f085c91496d1c9cc8e7f7bcb30d7ca

for path in \
  "$STUDENT_MODEL_DIR/adapter_model.safetensors" \
  "$STUDENT_STATE_PATH" "$STUDENT_RECEIPT" \
  "$teacher_root/model/adapter_model.safetensors" \
  "$teacher_root/metrics/teacher_gate.json" \
  "$teacher_root/metrics/teacher_provenance_gate.json"; do
  test -f "$path"
done

"$VENV_ROOT/bin/python" - "$STUDENT_RECEIPT" "$STUDENT_MODEL_DIR/adapter_model.safetensors" \
  "$STUDENT_STATE_PATH" "$EXPECTED_STUDENT_ADAPTER_SHA256" \
  "$EXPECTED_STUDENT_STATE_SHA256" "$teacher_root" <<'PY'
import hashlib,json,pathlib,sys
receipt=json.load(open(sys.argv[1])); teacher=pathlib.Path(sys.argv[6])
def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
 return h.hexdigest()
observed={'adapter_model':digest(sys.argv[2]),'recurrent_state':digest(sys.argv[3])}
assert receipt['status']=='complete' and receipt['step']==2000,receipt
assert receipt['sha256']==observed,(receipt,observed)
assert observed=={'adapter_model':sys.argv[4],'recurrent_state':sys.argv[5]},observed
gate=json.load(open(teacher/'metrics/teacher_gate.json'))
provenance=json.load(open(teacher/'metrics/teacher_provenance_gate.json'))
assert gate['status']=='complete' and gate['pass'] is True,gate
assert gate['support']==75 and sum(x['support'] for x in gate['by_group'].values())==75,gate
assert all(x['pass'] is True for x in gate['by_group'].values()),gate
assert gate['judge_backend']=='trapi' and gate['semantic_judge_authoritative'] is True,gate
assert gate['exact_match_authoritative'] is False,gate
assert provenance['status']=='complete' and provenance['teacher_gate_pass'] is True,provenance
assert provenance['output_schema']=='diagnosis_evidence_label_v1',provenance
assert provenance['diagnosis_field_order']==['diagnosis','evidence','label'],provenance
assert provenance['semantic_judge_authoritative'] is True,provenance
assert provenance['exact_match_authoritative'] is False,provenance
assert provenance['teacher_adapter_sha256']==digest(teacher/'model/adapter_model.safetensors'),provenance
PY

# Render the established state-only launcher without mutating the source used
# by already-running jobs.  Every replacement is cardinality checked so recipe
# drift fails before a GPU optimizer step.
"$VENV_ROOT/bin/python" - scripts/run_state8_stateonly_repair_ab.sh \
  /tmp/run_state8_matched_checkpoint2000_rescue.rendered.sh "$teacher_root" <<'PY'
import hashlib,pathlib,sys
source,output=map(pathlib.Path,sys.argv[1:3]); teacher_root=sys.argv[3]
raw=source.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='2d52d2d84b64dcc1d177583ddb96b006e6afdeee74cb2d3241325ae170a059b7'
text=raw.decode()
def replace(old,new,count=1):
 global text
 assert text.count(old)==count,(old,text.count(old),count)
 text=text.replace(old,new)
replace(
 "test \"$(sha256sum scripts/train_recurrent_student_m4.py | cut -d' ' -f1)\" = ae502968872ff6597e73bcc12cb3ff9a5bbb0fe9d7a763383f6e76bc392de42c",
 "test \"$(sha256sum scripts/train_recurrent_student_m4.py | cut -d' ' -f1)\" = a67662448408a587f891d32d4c5170fdca0435adac47554268edf5a3972330b5",
)
replace(
 '''elif [[ "$REQUESTED_WORLD_SIZE" == 1 ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  export RANK=0 WORLD_SIZE=1 LOCAL_RANK=0
  selected=""''',
 '''elif [[ "$REQUESTED_WORLD_SIZE" == 1 ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  export RANK=0 WORLD_SIZE=1 LOCAL_RANK=0
  # Scope automatic single-device recovery to this rendered rescue only.
  test "$STATE8_REPAIR_SAVE_STEPS" = "$STATE8_REPAIR_RECOVERY_SAVE_STEPS"
  selected="$(python scripts/select_latest_valid_single_checkpoint.py \\
    --run-dir "$RUN_DIR" --max-step 1000 \\
    --save-steps "$STATE8_REPAIR_SAVE_STEPS" \\
    --planned-optimizer-steps 250 \\
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS")"''',
)
replace(
 'export TEACHER_ROOT=/mnt/amulet/studentmodeling/diagnostics/state8_full_history_natural_teacher_v1',
 f'export TEACHER_ROOT={teacher_root}',
)
replace(
 "assert config['output_schema']=='label_diagnosis_evidence_v1',config",
 "assert config['output_schema'] in {'label_diagnosis_evidence_v1','diagnosis_evidence_label_v1'},config",
)
replace(
 "assert tg['stage']=='state8_independent_full_history_teacher_gate_v1' and tg['pass'] is True,tg\n"
 "assert tg['support']==1494 and tg['exact_match_is_authoritative'] is False,tg",
 "assert tg['stage']=='state8_diagnosis_first_full_history_teacher_screen_v1' and tg['status']=='complete' and tg['pass'] is True,tg\n"
 "assert tg['support']==75 and sum(x['support'] for x in tg['by_group'].values())==75,tg\n"
 "assert all(x['pass'] is True for x in tg['by_group'].values()),tg\n"
 "assert tg['judge_backend']=='trapi' and tg['semantic_judge_authoritative'] is True,tg\n"
 "assert tg['exact_match_authoritative'] is False,tg",
)
old='''  scripts/train_state8_diagnosis_first_causal_repair.py)
    test "$(sha256sum "$TRAIN_ENTRYPOINT" | cut -d' ' -f1)" = "${STATE8_DIAGNOSIS_FIRST_ENTRYPOINT_SHA256:?}" 
    test "$(sha256sum scripts/state8_taxonomy_contract.py | cut -d' ' -f1)" = 1ca32112e76c795502f3cb1edbedcc7220f093c133fae8d20ff5cab789e512ed
    test "${STATE8_REPAIR_LABEL_WEIGHT:-}" = 1.0
    test "${STATE8_CAUSAL_LABEL_WEIGHT:-}" = 1.0 ;;'''
new=old+'''
  scripts/train_state8_diagnosis_first_task_anchored_causal_repair.py)
    test "$(sha256sum "$TRAIN_ENTRYPOINT" | cut -d' ' -f1)" = "${STATE8_COMPOSED_ENTRYPOINT_SHA256:?}"
    test "$(sha256sum scripts/state8_taxonomy_contract.py | cut -d' ' -f1)" = 1ca32112e76c795502f3cb1edbedcc7220f093c133fae8d20ff5cab789e512ed
    test "${STATE8_REPAIR_LABEL_WEIGHT:-}" = 1.0
    test "${STATE8_CAUSAL_LABEL_WEIGHT:-}" = 1.0 ;;'''
replace(old,new)
old="""elif si.get('status')=='complete' and si.get('sha256'):
 expected={key:si['sha256'][key] for key in observed}
 assert si['micro_steps']>=1000 and si['real_examples_seen_once'] is True,si
else: raise AssertionError(si)"""
new="""elif si.get('status')=='complete' and si.get('step')==2000 and si.get('sha256'):
 expected={key:si['sha256'][key] for key in observed}
elif si.get('status')=='complete' and si.get('sha256'):
 expected={key:si['sha256'][key] for key in observed}
 assert si['micro_steps']>=1000 and si['real_examples_seen_once'] is True,si
else: raise AssertionError(si)"""
replace(old,new)
needle='|| "$TRAIN_ENTRYPOINT" == scripts/train_state8_diagnosis_first_causal_repair.py'
text=text.replace(needle,needle+' \\\n      || "$TRAIN_ENTRYPOINT" == scripts/train_state8_diagnosis_first_task_anchored_causal_repair.py')
old='''if [[ "$TRAIN_ENTRYPOINT" == scripts/train_state8_task_anchored_causal_repair.py ]]; then
  python -m pytest -q tests/test_train_state8_task_anchored_causal_repair.py
fi'''
new=old+'''
if [[ "$TRAIN_ENTRYPOINT" == scripts/train_state8_diagnosis_first_task_anchored_causal_repair.py ]]; then
  python -m pytest -q tests/test_train_state8_diagnosis_first_causal_repair.py \\
    tests/test_train_state8_diagnosis_first_task_anchored_causal_repair.py
fi'''
replace(old,new)
old='''if [[ "$TRAIN_ENTRYPOINT" == scripts/train_state8_task_anchored_causal_repair.py ]]; then
  test -f "$RUN_DIR/task_anchored_causal_repair.json"
fi'''
new=old+'''
if [[ "$TRAIN_ENTRYPOINT" == scripts/train_state8_diagnosis_first_task_anchored_causal_repair.py ]]; then
  test -f "$RUN_DIR/diagnosis_first_task_anchored_causal_repair.json"
fi'''
replace(old,new)
replace(
 "                        'train_state8_diagnosis_first_causal_repair.py')):",
 "                        'train_state8_diagnosis_first_causal_repair.py',\n                        'train_state8_diagnosis_first_task_anchored_causal_repair.py')):",
)
old='''if [[ "$TRAIN_ENTRYPOINT" == scripts/train_state8_diagnosis_first_causal_repair.py ]]; then
  FORMAL_EVALUATOR=scripts/evaluate_recurrent_formal_protocols_diagnosis_first.py'''
new='''if [[ "$TRAIN_ENTRYPOINT" == scripts/train_state8_diagnosis_first_causal_repair.py \\
      || "$TRAIN_ENTRYPOINT" == scripts/train_state8_diagnosis_first_task_anchored_causal_repair.py ]]; then
  FORMAL_EVALUATOR=scripts/evaluate_recurrent_formal_protocols_diagnosis_first.py'''
replace(old,new)
output.write_text(text)
print(hashlib.sha256(text.encode()).hexdigest())
PY
bash -n /tmp/run_state8_matched_checkpoint2000_rescue.rendered.sh

export STATE8_DIAGNOSIS_FIRST_ENTRYPOINT_SHA256=7d638013f932f29f52cffeea709b96872df8ba0960b9bb4e56ed23cc3475b216
export STATE8_COMPOSED_ENTRYPOINT_SHA256=eb9837127af9fe3761f6fa57b78a3e9ba3373a223706f0a4921749eab61e35a3
export STATE8_DIAGNOSIS_FIRST_FORMAL_SHA256=a8bc07a8fe6d2c9bfa9af6d976b53b84321cb4baa9c9fcace3188e370f7732c9
export STATE8_DIAGNOSIS_FIRST_C4_SHA256=660df80664c6012f6eac1d379cc1d611c9f085c91496d1c9cc8e7f7bcb30d7ca
bash /tmp/run_state8_matched_checkpoint2000_rescue.rendered.sh

stable_root="/mnt/amulet/studentmodeling/diagnostics/state8_stateonly_repair_seed${TRAIN_SEED}_${REPAIR_ARM}_${RUN_TAG}"
answer_out="$AMLT_OUTPUT_DIR/eval/answer_c4_50"
mkdir -p "$answer_out"
readarray -t artifact_hashes < <("$VENV_ROOT/bin/python" - "$stable_root/metrics/publish_receipt.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); print(x['artifact_sha256']['adapter_model']); print(x['artifact_sha256']['recurrent_state'])
PY
)
test "${#artifact_hashes[@]}" = 2
"$VENV_ROOT/bin/python" -u scripts/evaluate_recurrent_causal_gate.py \
  --model /tmp/Qwen3-32B --adapter "$stable_root/model" \
  --recurrent-state "$stable_root/model/recurrent_state.pt" \
  --state-architecture layerwise \
  --jsonl /mnt/amulet/studentmodeling/c4_fullopen_20260724/malrule_c1_clean/test_same_template.jsonl \
  --pair-manifest manifests/c4_stage_a_gate_pairs_seed42_n150_v1.json \
  --output-json "$answer_out/recurrent_causal_gate.json" \
  --state-token-alias '<|vision_pad|>' --max-length 1536 \
  --num-pairs 50 --allow-manifest-prefix --seed 42
"$VENV_ROOT/bin/python" - "$answer_out/recurrent_causal_gate.json" \
  "${artifact_hashes[0]}" "${artifact_hashes[1]}" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); assert x['summary']['all_pairs_n']==50,x
assert x['state_shape']==[64,8,5120] and x['raw_history_visible_to_decoder'] is False,x
assert x['persistent_history_kv_cache'] is False,x
assert x['artifact_sha256']=={'adapter_model':sys.argv[2],'recurrent_state':sys.argv[3]},x
PY

cp "$teacher_root/metrics/teacher_gate.json" "$AMLT_OUTPUT_DIR/metrics/matched_teacher_gate.json"
cp "$teacher_root/metrics/teacher_provenance_gate.json" "$AMLT_OUTPUT_DIR/metrics/matched_teacher_provenance_gate.json"
"$VENV_ROOT/bin/python" - "$AMLT_OUTPUT_DIR/metrics/rescue_receipt.json" \
  "$STATE8_RESCUE_MODE" "$RUN_TAG" "$EXPECTED_STUDENT_ADAPTER_SHA256" \
  "$EXPECTED_STUDENT_STATE_SHA256" "${artifact_hashes[0]}" "${artifact_hashes[1]}" <<'PY'
import json,pathlib,sys
out={'stage':'state8_matched_checkpoint2000_causal_rescue_v1','status':'complete',
 'diagnostic_only':True,'source_checkpoint_step':2000,'rescue_mode':sys.argv[2],
 'run_tag':sys.argv[3],'source_artifact_sha256':{'adapter_model':sys.argv[4],'recurrent_state':sys.argv[5]},
 'artifact_sha256':{'adapter_model':sys.argv[6],'recurrent_state':sys.argv[7]},
 'state_token_count':8,'history_dropout_schedule':[1.0],
 'raw_history_visible_to_decoder':False,'final_inference_route':'strict_state_only',
 'semantic_judge_authoritative':True,'exact_match_authoritative':False,
 'canonical_label_role':'auxiliary_ordinary_weight'}
pathlib.Path(sys.argv[1]).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
PY
