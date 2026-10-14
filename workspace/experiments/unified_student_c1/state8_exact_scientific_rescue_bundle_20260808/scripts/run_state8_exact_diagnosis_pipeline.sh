#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "STATE8_EXACT_PIPELINE_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

: "${VENV_ROOT:?VENV_ROOT is required}"
: "${STATE8_DIAGNOSIS_FIRST_CURRICULUM_SHA256:?STATE8_DIAGNOSIS_FIRST_CURRICULUM_SHA256 is required}"
: "${STATE8_DIAGNOSIS_FIRST_ENTRYPOINT_SHA256:?STATE8_DIAGNOSIS_FIRST_ENTRYPOINT_SHA256 is required}"
: "${STATE8_DIAGNOSIS_FIRST_FORMAL_SHA256:?STATE8_DIAGNOSIS_FIRST_FORMAL_SHA256 is required}"
: "${STATE8_DIAGNOSIS_FIRST_C4_SHA256:?STATE8_DIAGNOSIS_FIRST_C4_SHA256 is required}"

bash scripts/run_state8_exact_diagnosis_curriculum.sh

curriculum_root=/mnt/amulet/studentmodeling/diagnostics/state8_qualified_teacher_curriculum_highkd_v1_immutable_taxonomy_semantic_v1_exact_diagnosis_first_seed64_single_ga4
curriculum_receipt="$curriculum_root/metrics/immutable_receipt.json"
curriculum_completion="$curriculum_root/metrics/completion.json"
test -f "$curriculum_root/model/adapter_model.safetensors"
test -f "$curriculum_root/model/recurrent_state.pt"
test -f "$curriculum_receipt"
test -f "$curriculum_completion"

readarray -t artifact_sha < <("$VENV_ROOT/bin/python" - "$curriculum_receipt" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['status']=='complete' and x['state_shape']==[64,8,5120],x
assert x['raw_history_visible_to_decoder'] is False,x
assert x['semantic_judge_authoritative'] is True,x
assert x['exact_label_metric_authoritative'] is False,x
assert x['diagnosis_field_order']==['diagnosis','evidence','label'],x
assert x['canonical_label_role']=='auxiliary_ordinary_weight',x
print(x['sha256']['adapter_model']); print(x['sha256']['recurrent_state'])
PY
)
test "${#artifact_sha[@]}" = 2

export STUDENT_ROOT="$curriculum_root"
export STUDENT_MODEL_DIR="$curriculum_root/model"
export STUDENT_STATE_PATH="$curriculum_root/model/recurrent_state.pt"
# The immutable receipt keeps the extended diagnosis-first audit, while the
# base completion receipt is the source format already accepted by the frozen
# state-only continuation loader (status, hashes, micro-steps, epoch audit).
export STUDENT_RECEIPT="$curriculum_completion"
export EXPECTED_STUDENT_ADAPTER_SHA256="${artifact_sha[0]}"
export EXPECTED_STUDENT_STATE_SHA256="${artifact_sha[1]}"
export TRAIN_ENTRYPOINT=scripts/train_state8_diagnosis_first_causal_repair.py
export REPAIR_ARM=trajectory01
export REQUESTED_WORLD_SIZE=1
export GRADIENT_ACCUMULATION_STEPS=4
export STATE8_REPAIR_SAVE_STEPS=200
export STATE8_REPAIR_RECOVERY_SAVE_STEPS=200
export RUN_TAG=exact_teacher_curriculum_c4_seed65_v1
export TRAIN_SEED=65
export STATE8_REPAIR_LABEL_WEIGHT=1.0
export STATE8_CAUSAL_LABEL_WEIGHT=1.0
export STATE8_RANDOM_NEUTRAL=1
export STATE8_GEOMETRY_WEIGHT=0.10
export STATE8_EEDI_GEOMETRY_WEIGHT=0.05
export STATE8_DIFFERENT_MAX_COSINE=0.90
export STATE8_CAUSAL_DIAGNOSIS_ONLY=0
export STATE8_CAUSAL_REPEAT_FRACTION=0.50
export STATE8_CAUSAL_WEIGHT=2.0
export STATE8_CAUSAL_MARGIN=0.5
export STATE8_LAYERWISE_TRANSITION_ONLY=0
export STATE8_UPDATER_ONLY=0
export STATE8_FORMAL_SCOPE=unified
export STATE8_REPAIR_MANIFEST=manifests/layerwise_task_retention_staged_t3_t6_seed42_v1.json
export STATE8_REPAIR_MANIFEST_SHA256=a4eb41b2130f7e0810eeff36b5f3d16b8dd2c8631c50dbd25216f0793af23557

# Render the repair launcher against the current trainer hash.  The scientific
# recipe is unchanged; this only updates the stale bundle-era source pin.
python - scripts/run_state8_stateonly_repair_ab.sh \
  /tmp/run_state8_exact_diagnosis_c4.rendered.sh <<'PY'
import hashlib,pathlib,sys
source,output=map(pathlib.Path,sys.argv[1:])
raw=source.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='2d52d2d84b64dcc1d177583ddb96b006e6afdeee74cb2d3241325ae170a059b7'
text=raw.decode()
old="test \"$(sha256sum scripts/train_recurrent_student_m4.py | cut -d' ' -f1)\" = ae502968872ff6597e73bcc12cb3ff9a5bbb0fe9d7a763383f6e76bc392de42c"
new="test \"$(sha256sum scripts/train_recurrent_student_m4.py | cut -d' ' -f1)\" = a67662448408a587f891d32d4c5170fdca0435adac47554268edf5a3972330b5"
assert text.count(old)==1
text=text.replace(old,new)
old='''elif [[ "$REQUESTED_WORLD_SIZE" == 1 ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  export RANK=0 WORLD_SIZE=1 LOCAL_RANK=0
  selected=""'''
new='''elif [[ "$REQUESTED_WORLD_SIZE" == 1 ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  export RANK=0 WORLD_SIZE=1 LOCAL_RANK=0
  selected="$(python scripts/select_latest_valid_single_checkpoint.py \\
    --run-dir "$RUN_DIR" --max-step 1000 \\
    --save-steps "$STATE8_REPAIR_SAVE_STEPS" \\
    --planned-optimizer-steps 250 --gradient-accumulation-steps 4)"'''
assert text.count(old)==1
text=text.replace(old,new)
output.write_text(text)
print(hashlib.sha256(output.read_bytes()).hexdigest())
PY
bash -n /tmp/run_state8_exact_diagnosis_c4.rendered.sh
bash /tmp/run_state8_exact_diagnosis_c4.rendered.sh

final_root=/mnt/amulet/studentmodeling/diagnostics/state8_stateonly_repair_seed65_trajectory01_exact_teacher_curriculum_c4_seed65_v1
test -f "$final_root/metrics/publish_receipt.json"
answer_c4="$AMLT_OUTPUT_DIR/eval/answer_c4_150"
mkdir -p "$answer_c4"
"$VENV_ROOT/bin/python" -u scripts/evaluate_recurrent_causal_gate.py \
  --model /tmp/Qwen3-32B --adapter "$final_root/model" \
  --recurrent-state "$final_root/model/recurrent_state.pt" \
  --state-architecture layerwise \
  --jsonl /mnt/amulet/studentmodeling/c4_fullopen_20260724/malrule_c1_clean/test_same_template.jsonl \
  --pair-manifest manifests/c4_stage_a_gate_pairs_seed42_n150_v1.json \
  --output-json "$answer_c4/recurrent_causal_gate.json" \
  --state-token-alias '<|vision_pad|>' --max-length 1536 \
  --num-pairs 150 --seed 42
"$VENV_ROOT/bin/python" - "$curriculum_receipt" "$final_root/metrics/publish_receipt.json" <<'PY'
import json,sys
curriculum,final=map(lambda p:json.load(open(p)),sys.argv[1:])
assert curriculum['diagnosis_field_order']==['diagnosis','evidence','label'],curriculum
assert curriculum['canonical_label_role']=='auxiliary_ordinary_weight',curriculum
assert final['status']=='complete' and final['state_shape']==[64,8,5120],final
assert final['raw_history_visible_to_decoder'] is False,final
assert final['history_dropout']==1.0 and final['online_teacher_kd'] is True,final
PY
"$VENV_ROOT/bin/python" - "$answer_c4/recurrent_causal_gate.json" \
  "$final_root/metrics/publish_receipt.json" <<'PY'
import json,sys
answer,publish=map(lambda p:json.load(open(p)),sys.argv[1:])
assert answer['state_shape']==[64,8,5120],answer
assert answer['raw_history_visible_to_decoder'] is False,answer
assert answer['persistent_history_kv_cache'] is False,answer
assert answer['artifact_sha256']==publish['artifact_sha256'],(answer,publish)
assert answer['summary']['all_pairs_n']==150,answer
PY
