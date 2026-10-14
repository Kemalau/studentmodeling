#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "STATE8_EXACT_MILESTONE_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

export VENV_ROOT="${VENV_ROOT:-/tmp/studentmodeling-state8-exact-milestone-venv}"
export PATH="$VENV_ROOT/bin:$PATH"
python(){ "$VENV_ROOT/bin/python" "$@"; }
export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/hf_cache
export HF_HUB_CACHE=/tmp/hf_cache/hub
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error

: "${MILESTONE_STEP:?MILESTONE_STEP is required}"
case "$MILESTONE_STEP" in 200|400|600|800) ;; *) exit 2 ;; esac

export EXACT_RUN=/mnt/amulet/studentmodeling/checkpoints/state8_stateonly_repair_seed65_trajectory01_exact_teacher_curriculum_c4_seed65_v1
export CURRICULUM_ROOT=/mnt/amulet/studentmodeling/diagnostics/state8_qualified_teacher_curriculum_highkd_v1_immutable_taxonomy_semantic_v1_exact_diagnosis_first_seed64_single_ga4
export CHECKPOINT="$EXACT_RUN/checkpoint-$MILESTONE_STEP"
export STABLE_ROOT="/mnt/amulet/studentmodeling/diagnostics/state8_exact_consolidation_milestone_step${MILESTONE_STEP}_v1"
export LOCAL_MODEL=/tmp/Qwen3-32B
export RAW_EVAL=/mnt/amulet/studentmodeling/formal_evaluation_v1/test.jsonl
export RECURRENT_EVAL=/tmp/state8_exact_milestone_formal_u4/test.jsonl
export C4_DATA=/mnt/amulet/studentmodeling/c4_fullopen_20260724/malrule_c1_clean
export PAIRS=manifests/c4_stage_a_gate_pairs_seed42_n150_v1.json
export FORMAL_OUT="$AMLT_OUTPUT_DIR/eval/formal_t3_300"
export ANSWER_OUT="$AMLT_OUTPUT_DIR/eval/answer_c4_150"
export DIAGNOSIS_OUT="$AMLT_OUTPUT_DIR/eval/diagnosis_c4_50"

mkdir -p "$HF_HUB_CACHE" "$(dirname "$RECURRENT_EVAL")" "$FORMAL_OUT" \
  "$ANSWER_OUT" "$DIAGNOSIS_OUT" "$AMLT_OUTPUT_DIR/model" \
  "$AMLT_OUTPUT_DIR/metrics" "$STABLE_ROOT/model" "$STABLE_ROOT/metrics"
test "$(df --output=avail /tmp | tail -1)" -ge 83886080
for path in \
  "$CHECKPOINT/checkpoint_completion.json" \
  "$CHECKPOINT/adapter/adapter_model.safetensors" \
  "$CHECKPOINT/recurrent_state.pt" \
  "$CURRICULUM_ROOT/metrics/immutable_receipt.json" \
  "$CURRICULUM_ROOT/metrics/diagnosis_first_teacher_curriculum.json" \
  "$RAW_EVAL" "$C4_DATA/test_same_template.jsonl" "$PAIRS" \
  /mnt/amulet/studentmodeling/models/Qwen3-32B/model.safetensors.index.json; do
  test -f "$path"
done
test "$(sha256sum "$RAW_EVAL" | cut -d' ' -f1)" = bff7726953ae048bedf6dacdbb7500a54cb8de305e406f2497ef1f11c04167f8
test "$(sha256sum "$PAIRS" | cut -d' ' -f1)" = edcdb934a65252557f6c569eea9ef7a85e5445549deac9fb4829c3f579e1a9b8

python - "$CHECKPOINT" "$CURRICULUM_ROOT" "$MILESTONE_STEP" \
  "$AMLT_OUTPUT_DIR/metrics/checkpoint_audit.json" <<'PY'
import collections,hashlib,json,pathlib,sys,torch
checkpoint=pathlib.Path(sys.argv[1]); curriculum=pathlib.Path(sys.argv[2])
step=int(sys.argv[3]); output=pathlib.Path(sys.argv[4])
def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
 return h.hexdigest()
observed={
 'adapter_model':digest(checkpoint/'adapter/adapter_model.safetensors'),
 'recurrent_state':digest(checkpoint/'recurrent_state.pt'),
}
receipt=json.load(open(checkpoint/'checkpoint_completion.json'))
assert receipt=={'status':'complete','step':step,'sha256':observed},(receipt,observed)
payload=torch.load(checkpoint/'recurrent_state.pt',map_location='cpu',weights_only=False,mmap=True)
config=payload['config']; history=list(payload['history'])
assert payload['step']==step and len(history)==step,(payload['step'],len(history),step)
assert config['distributed'] is False and config['distributed_world_size']==1,config
assert config['planned_optimizer_steps']==250 and config['gradient_accumulation_steps']==4,config
assert config['full_micro_steps']==1000 and config['state_token_count']==8,config
assert config['state_architecture']=='layerwise' and config['final_inference_route']=='strict_state_only',config
assert config['history_dropout_schedule']==[1.0],config
assert config['diagnosis_field_order']==['diagnosis','evidence','label'],config
assert config['canonical_label_role']=='auxiliary_ordinary_weight',config
assert float(config['label_auxiliary_weight'])==1.0,config
assert all(bool(row['history_dropped']) and float(row['history_dropout'])==1.0 for row in history),history[:2]
assert [int(row['global_position']) for row in history]==list(range(step)),history[:2]
curriculum_receipt=json.load(open(curriculum/'metrics/immutable_receipt.json'))
assert curriculum_receipt['status']=='complete' and curriculum_receipt['state_shape']==[64,8,5120],curriculum_receipt
assert curriculum_receipt['raw_history_visible_to_decoder'] is False,curriculum_receipt
value={
 'stage':'state8_exact_consolidation_milestone_audit_v1','status':'complete',
 'checkpoint_step':step,'completed_optimizer_steps':step//4,
 'planned_micro_steps':1000,'planned_optimizer_steps':250,
 'consolidation_fraction':step/1000,'history_dropout_counts':{'1.0':step},
 'state_shape':[64,8,5120],'raw_history_visible_to_decoder':False,
 'persistent_history_kv_cache':False,'artifact_sha256':observed,
 'train_entrypoint':'scripts/train_state8_diagnosis_first_causal_repair.py',
}
output.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
PY

python scripts/build_recurrent_evaluation_manifest.py \
  --input-jsonl "$RAW_EVAL" --output-jsonl "$RECURRENT_EVAL" \
  --audit-json /tmp/state8_exact_milestone_formal_u4/audit.json \
  --max-updates 4 --seed 20260730
test "$(sha256sum "$RECURRENT_EVAL" | cut -d' ' -f1)" = a368cefb0c49cb6a305ad1a892808361fba2d44d555a0a28ea64e56babe32525

rm -rf "$LOCAL_MODEL"; mkdir -p "$LOCAL_MODEL"
python scripts/copy_model_parallel.py \
  --source /mnt/amulet/studentmodeling/models/Qwen3-32B \
  --destination "$LOCAL_MODEL" --workers 16
test -f "$LOCAL_MODEL/model.safetensors.index.json"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

python -u scripts/evaluate_recurrent_formal_protocols_diagnosis_first.py \
  --model "$LOCAL_MODEL" --adapter "$CHECKPOINT/adapter" \
  --recurrent-state "$CHECKPOINT/recurrent_state.pt" --state-architecture layerwise \
  --eval-jsonl "$RECURRENT_EVAL" --output-dir "$FORMAL_OUT" \
  --state-token-alias '<|vision_pad|>' --max-samples-per-group 25 \
  --max-length 1536 --max-new-tokens 160 --state-batch-size 2 \
  --candidate-row-batch-size 2 --seed 42 --save-unredacted-predictions \
  --t3-natural-schema

python -u scripts/evaluate_recurrent_causal_gate.py \
  --model "$LOCAL_MODEL" --adapter "$CHECKPOINT/adapter" \
  --recurrent-state "$CHECKPOINT/recurrent_state.pt" --state-architecture layerwise \
  --jsonl "$C4_DATA/test_same_template.jsonl" --pair-manifest "$PAIRS" \
  --output-json "$ANSWER_OUT/recurrent_causal_gate.json" \
  --state-token-alias '<|vision_pad|>' --max-length 1536 --num-pairs 150 --seed 42

readarray -t ARTIFACT_SHA < <(python - "$AMLT_OUTPUT_DIR/metrics/checkpoint_audit.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); print(x['artifact_sha256']['adapter_model']); print(x['artifact_sha256']['recurrent_state'])
PY
)
python -u scripts/evaluate_recurrent_diagnosis_c4_diagnosis_first.py \
  --model "$LOCAL_MODEL" --adapter "$CHECKPOINT/adapter" \
  --recurrent-state "$CHECKPOINT/recurrent_state.pt" --state-architecture layerwise \
  --jsonl "$C4_DATA/test_same_template.jsonl" --pair-manifest "$PAIRS" \
  --output-dir "$DIAGNOSIS_OUT" --state-token-alias '<|vision_pad|>' \
  --max-length 1536 --max-new-tokens 160 --num-pairs 50 --seed 42 \
  --expected-adapter-sha256 "${ARTIFACT_SHA[0]}" \
  --expected-recurrent-state-sha256 "${ARTIFACT_SHA[1]}"

python - "$AMLT_OUTPUT_DIR" "$MILESTONE_STEP" <<'PY'
import json,pathlib,shutil,sys
root=pathlib.Path(sys.argv[1]); step=int(sys.argv[2])
audit=json.load(open(root/'metrics/checkpoint_audit.json')); artifacts=audit['artifact_sha256']
formal=json.load(open(root/'eval/formal_t3_300/metrics.json'))
answer=json.load(open(root/'eval/answer_c4_150/recurrent_causal_gate.json'))
diagnosis=json.load(open(root/'eval/diagnosis_c4_50/metrics.json'))
for name,value in [('formal',formal),('answer',answer),('diagnosis',diagnosis)]:
 assert value['artifact_sha256']==artifacts,(name,value,artifacts)
 assert value['state_shape']==[64,8,5120] and value['raw_history_visible_to_decoder'] is False,value
 assert value['persistent_history_kv_cache'] is False,value
assert formal['sampled_rows']==487,formal
assert answer['summary']['all_pairs_n']==150,answer
assert diagnosis['num_pairs']==50 and diagnosis['semantic_judgment_pending'] is True,diagnosis
completion={
 'stage':'state8_exact_consolidation_milestone_completion_v1','status':'checkpoint_complete',
 'diagnostic':True,'checkpoint_step':step,'micro_steps':step,'optimizer_steps':step//4,
 'planned_micro_steps':1000,'planned_optimizer_steps':250,
 'real_examples_seen_once':False,'early_stop_candidate_only':True,'sha256':artifacts,
}
(root/'metrics/completion.json').write_text(json.dumps(completion,indent=2,sort_keys=True)+'\n')
publish={
 'stage':'state8_stateonly_repair_pilot_publish_v1','status':'complete',
 'repair_arm':'trajectory01','state_shape':[64,8,5120],
 'raw_history_visible_to_decoder':False,'history_dropout':1.0,'online_teacher_kd':True,
 'train_entrypoint':'scripts/train_state8_diagnosis_first_causal_repair.py',
 'consolidation_early_stop':True,'checkpoint_step':step,'artifact_sha256':artifacts,
}
(root/'metrics/publish_receipt.json').write_text(json.dumps(publish,indent=2,sort_keys=True)+'\n')
recipe={
 'stage':'state8_diagnosis_first_causal_repair_milestone_v1','status':'complete',
 'checkpoint_step':step,'state_token_count':8,'field_order':['diagnosis','evidence','label'],
 'semantic_fields_primary':True,'canonical_label_role':'auxiliary_ordinary_weight',
 'label_auxiliary_weight':1.0,'raw_history_visible_to_decoder':False,
 'final_inference_route':'strict_state_only','artifact_sha256':artifacts,
}
(root/'metrics/diagnosis_first_causal_repair.json').write_text(json.dumps(recipe,indent=2,sort_keys=True)+'\n')
PY

cp "$CURRICULUM_ROOT/metrics/immutable_receipt.json" "$AMLT_OUTPUT_DIR/metrics/immutable_receipt.json"
cp "$CURRICULUM_ROOT/metrics/diagnosis_first_teacher_curriculum.json" \
  "$AMLT_OUTPUT_DIR/metrics/diagnosis_first_teacher_curriculum.json"
cp /tmp/state8_exact_milestone_formal_u4/audit.json "$AMLT_OUTPUT_DIR/metrics/formal_manifest_audit.json"
cp -r "$CHECKPOINT/adapter/." "$AMLT_OUTPUT_DIR/model/"
cp "$CHECKPOINT/recurrent_state.pt" "$AMLT_OUTPUT_DIR/model/recurrent_state.pt"

# Publish a hash-bound remote initialization root, but never call it the final
# model: promotion remains conditional on the downstream joint semantic gate.
rm -rf "$STABLE_ROOT/model" "$STABLE_ROOT/metrics"
mkdir -p "$STABLE_ROOT/model" "$STABLE_ROOT/metrics"
cp -r "$AMLT_OUTPUT_DIR/model/." "$STABLE_ROOT/model/"
cp -r "$AMLT_OUTPUT_DIR/metrics/." "$STABLE_ROOT/metrics/"

python - "$STABLE_ROOT" "$MILESTONE_STEP" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); step=int(sys.argv[2])
def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
 return h.hexdigest()
observed={'adapter_model':digest(root/'model/adapter_model.safetensors'),
          'recurrent_state':digest(root/'model/recurrent_state.pt')}
publish=json.load(open(root/'metrics/publish_receipt.json'))
assert publish['consolidation_early_stop'] is True and publish['checkpoint_step']==step,publish
assert publish['artifact_sha256']==observed,(publish,observed)
PY
