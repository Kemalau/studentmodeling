#!/usr/bin/env bash
set -euo pipefail

export VENV_ROOT="${VENV_ROOT:-/tmp/studentmodeling-state8-diagnosis-rescue-venv}"
export PATH="$VENV_ROOT/bin:$PATH"
python(){ "$VENV_ROOT/bin/python" "$@"; }
export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/hf_cache
export HF_HUB_CACHE=/tmp/hf_cache/hub
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error
export LOCAL_MODEL=/tmp/Qwen3-32B
export RAW_FULL=/mnt/amulet/studentmodeling/full_trainplusvalid_no_t4_strict_t6/train.jsonl
export RAW_SPLIT=/tmp/state8_diagnosis_rescue_raw_split
export RECURRENT_SPLIT=/tmp/state8_diagnosis_rescue_recurrent_split
export FULL_JSONL="$RECURRENT_SPLIT/final_fit_trainvalid.jsonl"
export SUBSET=/tmp/state8_diagnosis_rescue_train1000.jsonl
export RAW_EVAL=/mnt/amulet/studentmodeling/formal_evaluation_v1/test.jsonl
export RECURRENT_EVAL=/tmp/state8_diagnosis_rescue_formal_u4/test.jsonl
export C4_DATA=/mnt/amulet/studentmodeling/c4_fullopen_20260724/malrule_c1_clean
export CAUSAL_BASE=/tmp/state8_diagnosis_rescue_cf4_base
export CAUSAL_SAME_RULE=/tmp/state8_diagnosis_rescue_cf4_same_rule
export CAUSAL_SUBSET=/tmp/state8_diagnosis_rescue_cf4_subset250
export CAUSAL_DATA=/tmp/state8_diagnosis_rescue_cf4_answer_diagnosis
export STUDENT_ROOT=/mnt/amulet/studentmodeling/diagnostics/state8_natural_dualpath_samerule_v5_immutable
export TEACHER_ROOT=/mnt/amulet/studentmodeling/diagnostics/state8_full_history_natural_teacher_v1
export RUN_DIR=/mnt/amulet/studentmodeling/checkpoints/state8_diagnosis_causal_rescue_seed43_v1_20260806
export MANIFEST=manifests/state8_diagnosis_causal_rescue_seed43_v1.json
export PAIRS=manifests/c4_stage_a_gate_pairs_seed42_n150_v1.json
export FORMAL_OUT="$AMLT_OUTPUT_DIR/eval/formal_screen"
export DIAGNOSIS_OUT="$AMLT_OUTPUT_DIR/eval/diagnosis_c4"

mkdir -p "$HF_HUB_CACHE" "$RAW_SPLIT" "$RECURRENT_SPLIT" \
  "$(dirname "$RECURRENT_EVAL")" "$CAUSAL_BASE" "$CAUSAL_SAME_RULE" \
  "$CAUSAL_SUBSET" "$CAUSAL_DATA" "$RUN_DIR" "$AMLT_OUTPUT_DIR/model" \
  "$AMLT_OUTPUT_DIR/metrics" "$FORMAL_OUT" "$DIAGNOSIS_OUT"
test "$(df --output=avail /tmp | tail -1)" -ge 125829120
for path in "$RAW_FULL" "$RAW_EVAL" "$C4_DATA/train.jsonl" \
  "$C4_DATA/val.jsonl" "$C4_DATA/test_same_template.jsonl" \
  "$STUDENT_ROOT/model/adapter_model.safetensors" \
  "$STUDENT_ROOT/model/recurrent_state.pt" \
  "$STUDENT_ROOT/metrics/immutable_receipt.json" \
  "$TEACHER_ROOT/model/adapter_model.safetensors" \
  "$TEACHER_ROOT/metrics/teacher_gate.json" "$MANIFEST" "$PAIRS"; do
  test -f "$path"
done
test "$(sha256sum "$RAW_FULL" | cut -d' ' -f1)" = aeabd3b4dfef3c4d92ac68583879c91448ef90f58304cedf32e470b586ca4542
test "$(sha256sum "$RAW_EVAL" | cut -d' ' -f1)" = bff7726953ae048bedf6dacdbb7500a54cb8de305e406f2497ef1f11c04167f8
test "$(sha256sum "$MANIFEST" | cut -d' ' -f1)" = 8023658c77474b31159d74881c9db38f56bb173344aa5d9599c4e3b9a0e63df4
test "$(sha256sum "$PAIRS" | cut -d' ' -f1)" = edcdb934a65252557f6c569eea9ef7a85e5445549deac9fb4829c3f579e1a9b8
test "$(sha256sum scripts/train_recurrent_student_m4.py | cut -d' ' -f1)" = ae502968872ff6597e73bcc12cb3ff9a5bbb0fe9d7a763383f6e76bc392de42c
test "$(sha256sum scripts/train_recurrent_student_m4_state8_external_teacher_curriculum.py | cut -d' ' -f1)" = 8e08385392811e48cfc8cf6335b8e3951641a7a9445e140e37d9d9872d515c6f
test "$(sha256sum scripts/augment_recurrent_causal_diagnosis.py | cut -d' ' -f1)" = b74684001e54bf0c9b023540c76a269708f1e8fa529cfec703a4439288590926
test "$(sha256sum scripts/evaluate_recurrent_diagnosis_c4.py | cut -d' ' -f1)" = 80d9f7f5320b997f656bf5bc61ee10c7c31ee2d3de2c7e7b06bba66dd5dbac4a
test "$(sha256sum scripts/select_latest_valid_distributed_checkpoint.py | cut -d' ' -f1)" = f038885f5bc054e4c326b3819dfd9e20a3f6b13f1145c1005f79a97fdf41d54b
test "$(sha256sum scripts/stage_distributed_resume_bundle.py | cut -d' ' -f1)" = c89b26b4e5b69d60774e2fceed57e7c7a635d4b5d403e7e0f1e952127ada6708

export STUDENTMODELING_EXTERNAL_STATE_TEACHER_ADAPTER="$TEACHER_ROOT/model"
export STUDENTMODELING_EXTERNAL_STATE_TEACHER_SHA256="$(python - "$STUDENT_ROOT" "$TEACHER_ROOT" <<'PY'
import hashlib,json,pathlib,sys
student=pathlib.Path(sys.argv[1]); teacher=pathlib.Path(sys.argv[2])
def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
 return h.hexdigest()
si=json.load(open(student/'metrics/immutable_receipt.json'))
tg=json.load(open(teacher/'metrics/teacher_gate.json'))
s={'adapter_model':digest(student/'model/adapter_model.safetensors'),
   'recurrent_state':digest(student/'model/recurrent_state.pt')}
t=digest(teacher/'model/adapter_model.safetensors')
assert si['stage']=='state8_v5_immutable_publish_v1' and si['sha256']==s,si
assert si['state_shape']==[64,8,5120] and si['raw_history_visible_to_decoder'] is False,si
assert tg['stage']=='state8_independent_full_history_teacher_gate_v1' and tg['pass'] is True,tg
assert tg['support']==1494 and tg['exact_match_is_authoritative'] is False,tg
print(t)
PY
)"
test "${#STUDENTMODELING_EXTERNAL_STATE_TEACHER_SHA256}" = 64

python -m pytest -q tests/test_augment_recurrent_causal_diagnosis.py \
  tests/test_evaluate_recurrent_diagnosis_c4.py \
  tests/test_state8_external_teacher_curriculum.py \
  tests/test_train_recurrent_student_m4.py

exec 8>/tmp/state8-diagnosis-rescue-data.lock; flock -x 8
if [[ ! -f /tmp/state8-diagnosis-rescue-data.complete ]]; then
  rm -rf "$RAW_SPLIT" "$RECURRENT_SPLIT" "$(dirname "$RECURRENT_EVAL")" \
    "$CAUSAL_BASE" "$CAUSAL_SAME_RULE" "$CAUSAL_SUBSET" "$CAUSAL_DATA"
  mkdir -p "$RAW_SPLIT" "$RECURRENT_SPLIT" "$(dirname "$RECURRENT_EVAL")" \
    "$CAUSAL_BASE" "$CAUSAL_SAME_RULE" "$CAUSAL_SUBSET" "$CAUSAL_DATA"
  python scripts/split_full_mix_by_original_split.py \
    --input-jsonl "$RAW_FULL" --output-dir "$RAW_SPLIT"
  python scripts/build_unified_recurrent_manifest.py \
    --train-jsonl "$RAW_SPLIT/train.jsonl" --valid-jsonl "$RAW_SPLIT/valid.jsonl" \
    --output-dir "$RECURRENT_SPLIT" --max-updates 4 --seed 20260730
  python scripts/merge_recurrent_full_corpus.py \
    --train-jsonl "$RECURRENT_SPLIT/train.jsonl" --valid-jsonl "$RECURRENT_SPLIT/valid.jsonl" \
    --output-jsonl "$FULL_JSONL" --manifest "$RECURRENT_SPLIT/final_fit_manifest.json"
  python scripts/build_recurrent_evaluation_manifest.py \
    --input-jsonl "$RAW_EVAL" --output-jsonl "$RECURRENT_EVAL" \
    --audit-json /tmp/state8_diagnosis_rescue_formal_u4/audit.json \
    --max-updates 4 --seed 20260730
  python scripts/materialize_recurrent_index_manifest.py \
    --source-jsonl "$FULL_JSONL" --manifest "$MANIFEST" \
    --output-jsonl "$SUBSET" --formal-eval-jsonl "$RECURRENT_EVAL"
  python scripts/build_recurrent_causal_pairs.py \
    --train-source "$C4_DATA/train.jsonl" --valid-source "$C4_DATA/val.jsonl" \
    --train-pairs manifests/cf4_balanced_paired_state_swap_train_seed20260731_v1.json \
    --valid-pairs manifests/cf4_balanced_paired_state_swap_valid_seed20260801_v1.json \
    --output-dir "$CAUSAL_BASE" --max-updates 4
  python scripts/augment_recurrent_causal_same_rule.py \
    --train-source "$C4_DATA/train.jsonl" --valid-source "$C4_DATA/val.jsonl" \
    --causal-train "$CAUSAL_BASE/train.jsonl" --causal-valid "$CAUSAL_BASE/valid.jsonl" \
    --output-dir "$CAUSAL_SAME_RULE" --seed 20260804 --max-updates 4
  python scripts/subset_state8_causal_twins.py \
    --input-jsonl "$CAUSAL_SAME_RULE/train.jsonl" \
    --output-jsonl "$CAUSAL_SUBSET/train.jsonl" --twins 125 --seed 43
  cp "$CAUSAL_SAME_RULE/valid.jsonl" "$CAUSAL_SUBSET/valid.jsonl"
  python scripts/augment_recurrent_causal_diagnosis.py \
    --train-jsonl "$CAUSAL_SUBSET/train.jsonl" \
    --valid-jsonl "$CAUSAL_SUBSET/valid.jsonl" --output-dir "$CAUSAL_DATA"
  touch /tmp/state8-diagnosis-rescue-data.complete
fi
flock -u 8
test "$(sha256sum "$FULL_JSONL" | cut -d' ' -f1)" = 9961f746cb17e6c314e182d3453b6157022fb48b8de44d2e68bc8e87a1131ccd
test "$(sha256sum "$RECURRENT_EVAL" | cut -d' ' -f1)" = a368cefb0c49cb6a305ad1a892808361fba2d44d555a0a28ea64e56babe32525
python - "$SUBSET.receipt.json" "$CAUSAL_DATA/augmentation_receipt.json" <<'PY'
import json,sys
s=json.load(open(sys.argv[1])); c=json.load(open(sys.argv[2]))
assert s['rows']==1000 and s['formal_example_id_overlap']==0,s
assert c['train']['input_pairs']==250 and c['train']['output_pairs']==500,c
assert c['train']['target_kind_counts']=={
 'diagnosis_natural_schema':250,'prospective_answer':250},c
assert c['train']['canonical_label_auxiliary_supervision'] is True,c
assert c['train']['raw_history_visible_to_decoder'] is False,c
PY

exec 7>/tmp/state8-diagnosis-rescue-model.lock; flock -x 7
if [[ ! -f /tmp/state8-diagnosis-rescue-model.complete ]]; then
  rm -rf "$LOCAL_MODEL"; mkdir -p "$LOCAL_MODEL"
  python scripts/copy_model_parallel.py \
    --source /mnt/amulet/studentmodeling/models/Qwen3-32B \
    --destination "$LOCAL_MODEL" --workers 16
  touch /tmp/state8-diagnosis-rescue-model.complete
fi
flock -u 7
test -f "$LOCAL_MODEL/model.safetensors.index.json"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

export CUDA_VISIBLE_DEVICES="${OMPI_COMM_WORLD_LOCAL_RANK:?MPI local rank is required}"
export RANK="${OMPI_COMM_WORLD_RANK:?MPI rank is required}"
export WORLD_SIZE="${OMPI_COMM_WORLD_SIZE:?MPI world size is required}"
export LOCAL_RANK=0
test "$WORLD_SIZE" = 4

resume_checkpoint=""
selected="$(python scripts/select_latest_valid_distributed_checkpoint.py \
  --run-dir "$RUN_DIR" --expected-world-size 4 --max-round 250 \
  --save-steps 25 --recovery-save-steps 25 --probe-expected-cadence)"
if [[ -n "$selected" ]]; then
  printf 'resume_checkpoint_selected=%s\n' "$selected"
else
  printf 'resume_checkpoint_selected=none; fresh_start=true\n'
fi
if [[ -n "$selected" ]]; then
  resume_key="$(python - "$selected" <<'PY'
import hashlib,os,sys
p=os.path.realpath(sys.argv[1]); s=os.stat(p)
print(hashlib.sha256(f'{p}:{s.st_size}:{s.st_mtime_ns}'.encode()).hexdigest())
PY
)"
  local_resume_dir="/tmp/state8-diagnosis-rescue-resume/$resume_key"
  mkdir -p "$local_resume_dir"
  exec 6>"$local_resume_dir/stage.lock"; flock -x 6
  python scripts/stage_distributed_resume_bundle.py \
    --source-state "$selected" --destination "$local_resume_dir"
  flock -u 6
  resume_checkpoint="$local_resume_dir/recurrent_state.pt"
fi
if [[ -n "$resume_checkpoint" ]]; then
  INIT_ARGS=(--init-recurrent-state "$STUDENT_ROOT/model/recurrent_state.pt" \
             --resume-from-checkpoint "$resume_checkpoint")
else
  INIT_ARGS=(--init-recurrent-state "$STUDENT_ROOT/model/recurrent_state.pt")
fi

python -u scripts/train_recurrent_student_m4_state8_external_teacher_curriculum.py \
  --model "$LOCAL_MODEL" --init-adapter "$STUDENT_ROOT/model" \
  "${INIT_ARGS[@]}" \
  --state-architecture layerwise --layerwise-operation-checkpointing \
  --layerwise-transition-rank 256 --state-token-count 8 \
  --train-jsonl "$SUBSET" --valid-jsonl "$RECURRENT_SPLIT/valid.jsonl" \
  --causal-train-jsonl "$CAUSAL_DATA/train.jsonl" \
  --causal-valid-jsonl "$CAUSAL_DATA/valid.jsonl" \
  --output-dir "$RUN_DIR" --state-token-alias '<|vision_pad|>' \
  --history-dropout-curriculum --dual-path-curriculum \
  --online-kd-weight 0.5 --online-kd-temperature 1.0 --online-kd-topk 256 \
  --max-length 1536 --direct-offload-min-tokens 1 \
  --learning-rate 1e-6 --lr-scheduler cosine --warmup-ratio 0.03 \
  --generation-weight 1.0 --behavior-kd-weight 0.0 \
  --consistency-weight 0.0 --direct-replay-weight 0.0 \
  --causal-weight 2.0 --causal-margin 0.5 --causal-preference-reduction sum \
  --causal-objective-mode specificity_triplet --causal-repeat-fraction 0.50 \
  --normalize-causal-exposure --gradient-accumulation-steps 1 \
  --eval-limit 300 --causal-eval-limit 165 --save-steps 25 \
  --checkpoint-retention 2 --distributed-mpi --distributed-world-size 4 --seed 43

if [[ "${OMPI_COMM_WORLD_RANK:-0}" != 0 ]]; then exit 0; fi
test -f "$RUN_DIR/completion.json"
test -f "$RUN_DIR/final_adapter/adapter_model.safetensors"
test -f "$RUN_DIR/recurrent_state.pt"
python - "$RUN_DIR" <<'PY'
import collections,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); c=json.load(open(root/'completion.json'))
cfg=json.load(open(root/'run_config.json')); h=json.load(open(root/'trainer_history.json'))
assert c['status']=='complete' and c['micro_steps']==1000,c
assert c['all_causal_pairs_covered'] is True and c['causal_pairs_seen']==500,c
assert cfg['state_token_count']==8 and cfg['final_inference_route']=='strict_state_only',cfg
assert cfg['causal_objective_mode']=='specificity_triplet',cfg
assert collections.Counter(float(x['history_dropout']) for x in h)=={
 0.0:200,0.25:200,0.5:200,0.75:200,1.0:200},h[:2]
PY

python -u scripts/evaluate_recurrent_formal_protocols.py \
  --model "$LOCAL_MODEL" --adapter "$RUN_DIR/final_adapter" \
  --recurrent-state "$RUN_DIR/recurrent_state.pt" --state-architecture layerwise \
  --eval-jsonl "$RECURRENT_EVAL" --output-dir "$FORMAL_OUT" \
  --state-token-alias '<|vision_pad|>' --max-samples-per-group 25 \
  --max-length 1536 --max-new-tokens 128 --state-batch-size 2 \
  --candidate-row-batch-size 2 --seed 43 --save-unredacted-predictions \
  --t3-natural-schema
readarray -t ARTIFACT_SHA256 < <(python - "$RUN_DIR/completion.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); print(x['sha256']['adapter_model']); print(x['sha256']['recurrent_state'])
PY
)
python -u scripts/evaluate_recurrent_diagnosis_c4.py \
  --model "$LOCAL_MODEL" --adapter "$RUN_DIR/final_adapter" \
  --recurrent-state "$RUN_DIR/recurrent_state.pt" --state-architecture layerwise \
  --jsonl "$C4_DATA/test_same_template.jsonl" --pair-manifest "$PAIRS" \
  --output-dir "$DIAGNOSIS_OUT" --state-token-alias '<|vision_pad|>' \
  --max-length 1536 --max-new-tokens 160 --num-pairs 150 --seed 42 \
  --expected-adapter-sha256 "${ARTIFACT_SHA256[0]}" \
  --expected-recurrent-state-sha256 "${ARTIFACT_SHA256[1]}"

cp -r "$RUN_DIR/final_adapter/." "$AMLT_OUTPUT_DIR/model/"
cp "$RUN_DIR/recurrent_state.pt" "$AMLT_OUTPUT_DIR/model/recurrent_state.pt"
cp "$RUN_DIR"/*.json "$AMLT_OUTPUT_DIR/metrics/"
cp "$SUBSET.receipt.json" "$AMLT_OUTPUT_DIR/metrics/training_subset_receipt.json"
cp "$CAUSAL_DATA/augmentation_receipt.json" "$AMLT_OUTPUT_DIR/metrics/causal_answer_diagnosis_augmentation.json"
