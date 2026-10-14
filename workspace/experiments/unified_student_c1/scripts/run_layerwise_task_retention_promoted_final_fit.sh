#!/usr/bin/env bash
set -euo pipefail

export VENV_ROOT="${VENV_ROOT:-/tmp/studentmodeling-layerwise-promoted-final-venv}"
export PATH="$VENV_ROOT/bin:$PATH"
hash -r
python() { "$VENV_ROOT/bin/python" "$@"; }
for _venv_probe in $(seq 1 120); do
  if "$VENV_ROOT/bin/python" -c 'import accelerate,bitsandbytes,peft,safetensors,transformers' >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
"$VENV_ROOT/bin/python" -c 'import accelerate,bitsandbytes,peft,safetensors,transformers'
export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/hf_cache
export HF_HUB_CACHE=/tmp/hf_cache/hub
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error
export LOCAL_MODEL=/tmp/Qwen3-32B
export RAW_FULL=/mnt/amulet/studentmodeling/full_trainplusvalid_no_t4_strict_t6/train.jsonl
export RAW_SPLIT=/tmp/layerwise_promoted_full_trainvalid_v1
export RECURRENT_SPLIT=/tmp/layerwise_recurrent_full259845_u4_v1
export FINAL_JSONL="$RECURRENT_SPLIT/final_fit_trainvalid.jsonl"
export C4_DATA=/mnt/amulet/studentmodeling/c4_fullopen_20260724/malrule_c1_clean
export CAUSAL_DATA=/tmp/layerwise_final_cf4_u4_v1
export INIT_RUN_DIR=/mnt/amulet/studentmodeling/checkpoints/layerwise_trajectory_curriculum_r256_scale_seed42_n200_20260731
export INIT_ADAPTER=/mnt/amulet/studentmodeling/checkpoints/state_curriculum_stage_a_cf1_alias_visionpad_seed42_20260727/final_adapter
export CACHE_ROOT=/mnt/amulet/studentmodeling/world_model_v1/final_fit/cache_full259845_top256_seed42_v2_16way
export CHECKPOINT=/mnt/amulet/studentmodeling/diagnostics/layerwise_unified_finalfit_seed42_20260801_snapshot/checkpoint-round-750
export INITIAL_CHECKPOINT_SHA256=598901f8b8fc0901c6a1da6eaa32866dc1fd7dbfe762d34483b10dbaf23853fe
export PROMOTION=/mnt/amulet/studentmodeling/layerwise_world_model_v1/task_retention_promotion/selected.json
export T6_CALIBRATION=/mnt/amulet/studentmodeling/layerwise_world_model_v1/task_retention_promotion/t6_calibration.json

mkdir -p "$HF_HUB_CACHE" "$RAW_SPLIT" "$RECURRENT_SPLIT" "$CAUSAL_DATA" \
  "$AMLT_OUTPUT_DIR/model" "$AMLT_OUTPUT_DIR/metrics"
test "$(df --output=avail /tmp | tail -1)" -ge 125829120
test -f "$RAW_FULL"
test -f "$C4_DATA/train.jsonl"
test -f "$C4_DATA/val.jsonl"
test -f "$INIT_RUN_DIR/completion.json"
test -f "$CHECKPOINT/adapter/adapter_model.safetensors"
test -f "$CHECKPOINT/recurrent_state.pt"
test -f "$CHECKPOINT/checkpoint_completion.json"
test -f "$PROMOTION"
test -f "$CACHE_ROOT/integrity.json"
test -f "$CACHE_ROOT/manifest.json"

test "$(sha256sum scripts/train_recurrent_student_m4_promoted.py | cut -d' ' -f1)" = ebf21ce4c0dbdf8b99c0a945d92aeec9d236a55d304b91912539034a8a6db81f
test -f scripts/train_recurrent_student_m4_promoted_open_t3.py
test "$(sha256sum scripts/promoted_checkpoint_cadence.py | cut -d' ' -f1)" = 58d974c91a04a9aee9271ef33ede187aafef84fd127ae08f15cc4dc64bb5d495
test "$(sha256sum scripts/stage_distributed_resume_bundle.py | cut -d' ' -f1)" = c89b26b4e5b69d60774e2fceed57e7c7a635d4b5d403e7e0f1e952127ada6708
test -f scripts/train_recurrent_student_m4_promoted_open_t3_selective_kd.py
test -f scripts/train_recurrent_student_m4_promoted_open_t3_two_stage.py
test "$(sha256sum scripts/train_recurrent_student_m4_promoted_open_t3_two_stage.py | cut -d' ' -f1)" = b52332455504ff5528989d8a71af2cd2edcd37a3425d8bedd03745d6248c3191
test "$(sha256sum scripts/train_recurrent_student_m4_promoted_open_t3.py | cut -d' ' -f1)" = 3ee69fa74b056a6ca15774dacb52a1cfd402d4baf21b0d333a8e7d9f001c4650
test "$(sha256sum scripts/train_recurrent_student_m4_promoted_open_t3_selective_kd.py | cut -d' ' -f1)" = 3fb17bffef5414655175165b26f67a7105d96243963bffd45f42102423b39ed5
test "$(sha256sum scripts/distributed_epoch.py | cut -d' ' -f1)" = 20f85e37512965d560761de26d99d3329b15c16de679d34ad460a1407bad2f9a
test "$(sha256sum scripts/audit_layerwise_distributed_checkpoint.py | cut -d' ' -f1)" = 6690ece8935c6343bec7136ad34a9899b72a866c7bde6286ca26b2c82890007f
test "$(sha256sum scripts/frozen_layerwise_finalfit/layerwise_recurrent_student_model.py | cut -d' ' -f1)" = c80357282e600b31f68f55d8ab434ad349c0e9eeeeb222b52dbadae7caea11ca
test "$(sha256sum tests/frozen_layerwise_finalfit/test_layerwise_recurrent_student_model.py | cut -d' ' -f1)" = a995f82f364eb3247ff3e17873cc720bbce7a2a8f52813c9369421fa61d3c274
cp scripts/frozen_layerwise_finalfit/layerwise_recurrent_student_model.py scripts/layerwise_recurrent_student_model.py
cp tests/frozen_layerwise_finalfit/test_layerwise_recurrent_student_model.py tests/test_layerwise_recurrent_student_model.py
python -m unittest discover -s tests -p 'test_train_recurrent_student_m4.py' -v
python -m unittest discover -s tests -p 'test_layerwise_recurrent_student_model.py' -v
python -m unittest discover -s tests -p 'test_distributed_epoch.py' -v

readarray -t promotion_values < <(python - "$PROMOTION" <<'PY'
import hashlib, json, sys
path = sys.argv[1]
x = json.load(open(path, encoding='utf-8'))
assert x['stage'] == 'layerwise_task_retention_promoted_order_v1', x
assert x['branch'] in {'equal_t3_t6', 'equal_t3_t6_open_t3', 'equal_t3_t6_open_t3_two_stage', 't3_heavy', 'eedi_dense', 'staged_t3_t6', 'eedi_full', 'eedi_full_open_t3'}, x
assert x['selected_loss_profile'] in {'standard','state_kd','state_path'},x
selected_lr=float(x['selected_learning_rate'])
selected_causal_weight=float(x['selected_causal_weight'])
expected_candidate=(f"{x['branch']}__{x['selected_loss_profile']}__lr{selected_lr:.0e}" if x['branch'] in {'eedi_full','eedi_full_open_t3'} or selected_lr != 5e-7 else f"{x['branch']}__{x['selected_loss_profile']}")
assert x['selected_candidate_id']==expected_candidate,x
assert selected_lr in ({1e-6,2e-6,5e-6} if x['branch'] in {'eedi_full','eedi_full_open_t3'} or selected_lr != 5e-7 else {5e-7}),x
assert selected_causal_weight==2.0,x
assert x['one_epoch_permutation'] is True, x
assert x['completed_prefix_unchanged'] is True, x
assert x['formal_example_id_overlap'] == 0, x
assert x['source_checkpoint_round'] == 750, x
assert x['source_world_size'] == 16, x
assert x['completed_prefix_rows'] == 12000, x
assert x['epoch_rows'] == 259845, x
expected_rows=(5000 if x['branch'] in {'eedi_full','eedi_full_open_t3'} else 2000 if x['branch']=='equal_t3_t6_open_t3_two_stage' else 1000)
assert x['promoted_rows'] == expected_rows, x
if x['branch']=='equal_t3_t6_open_t3_two_stage':
    assert x['curriculum_mode']=='open_t3_formation_then_causal_task_consolidation',x
    assert x['formation_learning_rate']==5e-6,x
    assert x['formation_transition_global_position']==13008,x
assert len(x['real_order']) == 259845, x
print(x['branch'])
print(x['selected_loss_profile'])
print(x['selected_candidate_id'])
print(x['selected_learning_rate'])
print(x['selected_causal_weight'])
print(x['real_order_sha256'])
print(hashlib.sha256(open(path, 'rb').read()).hexdigest())
print(x.get('t6_decision_threshold',''))
print(x.get('t6_calibration_artifact_sha256',''))
PY
)
export PROMOTION_BRANCH="${promotion_values[0]}"
export PROMOTION_LOSS_PROFILE="${promotion_values[1]}"
export PROMOTION_CANDIDATE="${promotion_values[2]}"
export PROMOTION_LEARNING_RATE="${promotion_values[3]}"
export PROMOTION_CAUSAL_WEIGHT="${promotion_values[4]}"
export PROMOTION_ORDER_SHA256="${promotion_values[5]}"
export PROMOTION_SHA256="${promotion_values[6]}"
export T6_DECISION_THRESHOLD="${promotion_values[7]}"
export T6_CALIBRATION_SHA256="${promotion_values[8]}"
if [[ -n "$T6_DECISION_THRESHOLD" ]]; then
  test -f "$T6_CALIBRATION"
  test "$(sha256sum "$T6_CALIBRATION" | cut -d' ' -f1)" = "$T6_CALIBRATION_SHA256"
  python - "$T6_CALIBRATION" "$T6_DECISION_THRESHOLD" <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
assert x['stage']=='layerwise_t6_threshold_calibration_v1' and x['status']=='frozen',x
assert float(x['threshold'])==float(sys.argv[2]),x
PY
fi
ACTIVE_LEARNING_RATE="$PROMOTION_LEARNING_RATE"
case "$PROMOTION_LOSS_PROFILE" in
  standard) BEHAVIOR_KD_WEIGHT=0.5; DIRECT_REPLAY_WEIGHT=0.25 ;;
  state_kd) BEHAVIOR_KD_WEIGHT=0.5; DIRECT_REPLAY_WEIGHT=0.0 ;;
  state_path) BEHAVIOR_KD_WEIGHT=0.0; DIRECT_REPLAY_WEIGHT=0.0 ;;
  *) exit 2 ;;
esac
if [[ "$PROMOTION_BRANCH" == equal_t3_t6_open_t3_two_stage ]]; then
  # The selected profile is the consolidation target.  The exact-epoch run
  # first replays the proven high-LR state-KD formation phase, then the trainer
  # atomically switches to the selected low-LR objective at global row 13008.
  ACTIVE_LEARNING_RATE=5e-6
  BEHAVIOR_KD_WEIGHT=0.5
  DIRECT_REPLAY_WEIGHT=0.0
fi
if [[ "$BEHAVIOR_KD_WEIGHT" == "0.0" ]]; then
  export STUDENTMODELING_TEACHER_CACHE_UNUSED=1
else
  unset STUDENTMODELING_TEACHER_CACHE_UNUSED
fi
export RUN_DIR="/mnt/amulet/studentmodeling/checkpoints/layerwise_unified_world_model_seed42_taskretention_${PROMOTION_BRANCH}_${PROMOTION_LOSS_PROFILE}_v1_20260802"
mkdir -p "$RUN_DIR"

exec 9>/tmp/layerwise-promoted-checkpoint-audit.lock
flock -x 9
if [[ ! -f /tmp/layerwise-promoted-checkpoint-audit.complete ]]; then
  python scripts/audit_layerwise_distributed_checkpoint.py \
    --checkpoint "$CHECKPOINT/recurrent_state.pt" --expected-round 750 \
    --output /tmp/layerwise-promoted-checkpoint-audit.json
  python - /tmp/layerwise-promoted-checkpoint-audit.json <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
assert x['status']=='passed' and x['exact_resume_ready'] is True,x
assert x['checkpoint_round']==750 and x['world_size']==16,x
assert x['history_rows']==12000 and x['causal_exposures']==12992,x
PY
  touch /tmp/layerwise-promoted-checkpoint-audit.complete
fi
flock -u 9

exec 8>/tmp/layerwise-promoted-data.lock
flock -x 8
if [[ ! -f /tmp/layerwise-promoted-data.complete ]]; then
  rm -rf "$RAW_SPLIT" "$RECURRENT_SPLIT" "$CAUSAL_DATA"
  mkdir -p "$RAW_SPLIT" "$RECURRENT_SPLIT" "$CAUSAL_DATA"
  python scripts/split_full_mix_by_original_split.py \
    --input-jsonl "$RAW_FULL" --output-dir "$RAW_SPLIT"
  python scripts/build_unified_recurrent_manifest.py \
    --train-jsonl "$RAW_SPLIT/train.jsonl" --valid-jsonl "$RAW_SPLIT/valid.jsonl" \
    --output-dir "$RECURRENT_SPLIT" --max-updates 4 --seed 20260730
  python scripts/audit_unified_recurrent_manifest.py \
    --train-jsonl "$RECURRENT_SPLIT/train.jsonl" --valid-jsonl "$RECURRENT_SPLIT/valid.jsonl" \
    --output-json "$RECURRENT_SPLIT/audit.json"
  python scripts/merge_recurrent_full_corpus.py \
    --train-jsonl "$RECURRENT_SPLIT/train.jsonl" --valid-jsonl "$RECURRENT_SPLIT/valid.jsonl" \
    --output-jsonl "$FINAL_JSONL" --manifest "$RECURRENT_SPLIT/final_fit_manifest.json"
  python scripts/build_recurrent_causal_pairs.py \
    --train-source "$C4_DATA/train.jsonl" --valid-source "$C4_DATA/val.jsonl" \
    --train-pairs manifests/cf4_balanced_paired_state_swap_train_seed20260731_v1.json \
    --valid-pairs manifests/cf4_balanced_paired_state_swap_valid_seed20260801_v1.json \
    --output-dir "$CAUSAL_DATA" --max-updates 4
  touch /tmp/layerwise-promoted-data.complete
fi
flock -u 8
test "$(sha256sum "$FINAL_JSONL" | cut -d' ' -f1)" = 9961f746cb17e6c314e182d3453b6157022fb48b8de44d2e68bc8e87a1131ccd
test "$(sha256sum "$RECURRENT_SPLIT/valid.jsonl" | cut -d' ' -f1)" = 78bfce75576dc3873cbb6459669a952adeedb8b77bda15f281af9519e2506d14
test "$(sha256sum "$CAUSAL_DATA/train.jsonl" | cut -d' ' -f1)" = 30f68f54120e5a9ada0a37b1dba2b6480de0f8eaeac98d0b7ed7c627429f9766
test "$(sha256sum "$CAUSAL_DATA/valid.jsonl" | cut -d' ' -f1)" = 8e038c3a60c2e1b420779cd54eb203b12ae5d51544c87a805fe19d47fdbe294e

exec 7>/tmp/layerwise-promoted-model.lock
flock -x 7
if [[ ! -f /tmp/layerwise-promoted-model.complete ]]; then
  rm -rf "$LOCAL_MODEL"; mkdir -p "$LOCAL_MODEL"
  python scripts/copy_model_parallel.py \
    --source /mnt/amulet/studentmodeling/models/Qwen3-32B \
    --destination "$LOCAL_MODEL" --workers 16
  test -f "$LOCAL_MODEL/model.safetensors.index.json"
  touch /tmp/layerwise-promoted-model.complete
fi
flock -u 7

export CUDA_VISIBLE_DEVICES="${OMPI_COMM_WORLD_LOCAL_RANK:?MPI local rank is required}"
export RANK="${OMPI_COMM_WORLD_RANK:?MPI rank is required}"
export WORLD_SIZE="${OMPI_COMM_WORLD_SIZE:?MPI world size is required}"
export LOCAL_RANK=0

resume_checkpoint="$CHECKPOINT/recurrent_state.pt"
selected_checkpoint="$(python scripts/select_latest_valid_distributed_checkpoint.py \
  --run-dir "$RUN_DIR" --expected-world-size 16 --max-round 16241 --save-steps 250 \
  --recovery-save-steps 10)"
if [[ -n "$selected_checkpoint" ]]; then
  resume_checkpoint="$selected_checkpoint"
fi

# A distributed checkpoint is about 4.6 GB.  Reading it independently from the
# Blob-backed mount on all 16 ranks creates roughly 73 GB of synchronized remote
# I/O before the first optimizer step.  Stage one verified copy per node and let
# all eight local ranks reuse it.  The local lock and cache live on each node's
# private /tmp, while the source remains the authoritative remote artifact.
resume_key="$(python - "$resume_checkpoint" <<'PY'
import hashlib, os, sys
path=os.path.realpath(sys.argv[1])
stat=os.stat(path)
print(hashlib.sha256(f'{path}:{stat.st_size}:{stat.st_mtime_ns}'.encode()).hexdigest())
PY
)"
local_resume_dir="/tmp/layerwise-promoted-resume/$resume_key"
local_resume="$local_resume_dir/recurrent_state.pt"
mkdir -p "$local_resume_dir"
exec 6>"$local_resume_dir/stage.lock"
flock -x 6
stage_args=(--source-state "$resume_checkpoint" --destination "$local_resume_dir")
if [[ "$resume_checkpoint" == "$CHECKPOINT/recurrent_state.pt" ]]; then
  stage_args+=(--expected-state-sha256 "$INITIAL_CHECKPOINT_SHA256")
fi
python scripts/stage_distributed_resume_bundle.py "${stage_args[@]}"
flock -u 6
resume_args=(--resume-from-checkpoint "$local_resume")

TRAIN_ENTRYPOINT=scripts/train_recurrent_student_m4_promoted.py
if [[ "$PROMOTION_BRANCH" == equal_t3_t6_open_t3_two_stage ]]; then
  TRAIN_ENTRYPOINT=scripts/train_recurrent_student_m4_promoted_open_t3_two_stage.py
elif [[ "$PROMOTION_BRANCH" == eedi_full_open_t3 || "$PROMOTION_BRANCH" == equal_t3_t6_open_t3 ]]; then
  case "$PROMOTION_LOSS_PROFILE" in
    state_path)
      TRAIN_ENTRYPOINT=scripts/train_recurrent_student_m4_promoted_open_t3.py
      ;;
    state_kd)
      TRAIN_ENTRYPOINT=scripts/train_recurrent_student_m4_promoted_open_t3_selective_kd.py
      ;;
    *) exit 2 ;;
  esac
fi

python -u "$TRAIN_ENTRYPOINT" \
  --model "$LOCAL_MODEL" --init-adapter "$INIT_ADAPTER" --init-run-dir "$INIT_RUN_DIR" \
  --state-architecture layerwise --layerwise-operation-checkpointing \
  --layerwise-transition-rank 256 \
  --train-jsonl "$FINAL_JSONL" --valid-jsonl "$RECURRENT_SPLIT/valid.jsonl" \
  --causal-train-jsonl "$CAUSAL_DATA/train.jsonl" \
  --causal-valid-jsonl "$CAUSAL_DATA/valid.jsonl" \
  --teacher-cache "$CACHE_ROOT" --behavior-kd-weight "$BEHAVIOR_KD_WEIGHT" --behavior-kd-temperature 1.0 \
  --output-dir "$RUN_DIR" --state-token-alias '<|vision_pad|>' \
  --max-length 1536 --learning-rate "$ACTIVE_LEARNING_RATE" --lr-scheduler cosine --warmup-ratio 0.03 \
  --generation-weight 1.0 --consistency-weight 0.25 --direct-replay-weight "$DIRECT_REPLAY_WEIGHT" \
  --causal-weight "$PROMOTION_CAUSAL_WEIGHT" --causal-margin 0.5 --causal-preference-reduction sum \
  --causal-objective-mode specificity_triplet --causal-repeat-fraction 0.05 \
  --normalize-causal-exposure --gradient-accumulation-steps 1 \
  --eval-limit 300 --causal-eval-limit 165 --save-steps 250 --checkpoint-retention 2 \
  --recovery-checkpoint-first-offset 10 --recovery-checkpoint-interval 20 \
  --promoted-real-order-manifest "$PROMOTION" \
  --distributed-mpi --distributed-world-size 16 --seed 42 \
  "${resume_args[@]}"

if [[ "${OMPI_COMM_WORLD_RANK:-0}" != 0 ]]; then exit 0; fi
test -f "$RUN_DIR/completion.json"
test -f "$RUN_DIR/final_adapter/adapter_model.safetensors"
test -f "$RUN_DIR/recurrent_state.pt"
python - "$RUN_DIR" "$PROMOTION_BRANCH" "$PROMOTION_LOSS_PROFILE" "$PROMOTION_CANDIDATE" "$PROMOTION_LEARNING_RATE" "$PROMOTION_CAUSAL_WEIGHT" "$PROMOTION_SHA256" "$PROMOTION_ORDER_SHA256" <<'PY'
import json,os,pathlib,sys
root=pathlib.Path(sys.argv[1])
completion=json.load(open(root/'completion.json',encoding='utf-8'))
config=json.load(open(root/'run_config.json',encoding='utf-8'))
assert completion['status']=='complete' and completion['diagnostic'] is False,completion
assert completion['micro_steps']==259845 and completion['real_examples_seen']==259845,completion
assert completion['real_examples_seen_once'] is True and completion['exact_history_support'] is True,completion
assert completion['causal_exposure_count_matches_plan'] is True,completion
assert completion['all_causal_pairs_covered'] is True,completion
assert config['promoted_real_order_applied'] is True,config
assert config['promoted_real_order_source_round']==750,config
assert config['promoted_real_order_manifest_sha256']==sys.argv[7],config
assert abs(float(config['learning_rate'])-float(sys.argv[5]))<1e-15,config
assert abs(float(config['promoted_selected_learning_rate'])-float(sys.argv[5]))<1e-15,config
assert float(config['causal_weight'])==float(sys.argv[6])==2.0,config
assert float(config['promoted_selected_causal_weight'])==float(sys.argv[6]),config
expected_weights={'standard':(0.5,0.25),'state_kd':(0.5,0.0),'state_path':(0.0,0.0)}[sys.argv[3]]
assert (config['behavior_kd_weight'],config['direct_replay_weight'])==expected_weights,config
expected_view=(
 'open_t3_selective_kd'
 if sys.argv[2] in {'eedi_full_open_t3','equal_t3_t6_open_t3','equal_t3_t6_open_t3_two_stage'} and (sys.argv[3]=='state_kd' or sys.argv[2]=='equal_t3_t6_open_t3_two_stage')
 else ('open_t3_no_candidate_enumeration' if sys.argv[2] in {'eedi_full_open_t3','equal_t3_t6_open_t3'} else 'standard')
)
assert config['training_view']==expected_view and config['promoted_training_view']==expected_view,config
if sys.argv[2]=='equal_t3_t6_open_t3_two_stage':
    assert config['promoted_two_stage_enabled'] is True,config
    assert config['promoted_two_stage_phase']=='consolidation',config
    assert config['promoted_two_stage_transition_applied'] is True,config
    assert config['promoted_two_stage_transition_global_position']==13008,config
    assert config['promoted_two_stage_formation_learning_rate']==5e-6,config
    assert config['promoted_two_stage_consolidation_learning_rate']==float(sys.argv[5]),config
    transition=json.load(open(root/'two_stage_transition_receipt.json',encoding='utf-8'))
    assert transition['global_position']==13008 and transition['optimizer_moments_preserved'] is True,transition
receipt={
 'status':'final_fit_complete','stage':'layerwise_task_retention_promoted_final_fit',
 'branch':sys.argv[2],'loss_profile':sys.argv[3],'candidate_id':sys.argv[4],
 'learning_rate':float(sys.argv[5]),
 'causal_weight':float(sys.argv[6]),
 'run_dir':str(root),'state_architecture':'layerwise',
 'state_shape':[64,5120],'real_examples':259845,'one_epoch_exact':True,
 'promoted_source_round':750,'promoted_order_sha256':sys.argv[8],
 'optimizer_steps':completion['optimizer_steps'],'raw_history_visible_to_decoder':False,
 'persistent_history_kv_cache':False,
 'two_stage_curriculum':sys.argv[2]=='equal_t3_t6_open_t3_two_stage',
}
if os.environ.get('T6_DECISION_THRESHOLD'):
    receipt.update({
      't6_decision_threshold':float(os.environ['T6_DECISION_THRESHOLD']),
      't6_calibration_artifact_sha256':os.environ['T6_CALIBRATION_SHA256'],
      't6_threshold_selection_split':'unused_original_validation',
      't6_threshold_frozen_before_final_fit':True,
    })
(root/'promoted_final_fit_receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps(receipt,indent=2,sort_keys=True))
PY

cp -r "$RUN_DIR/final_adapter/." "$AMLT_OUTPUT_DIR/model/"
cp "$RUN_DIR/recurrent_state.pt" "$AMLT_OUTPUT_DIR/model/recurrent_state.pt"
cp "$RUN_DIR/completion.json" "$AMLT_OUTPUT_DIR/metrics/completion.json"
cp "$RUN_DIR/promoted_final_fit_receipt.json" "$AMLT_OUTPUT_DIR/metrics/promoted_final_fit_receipt.json"
cp "$RUN_DIR/run_config.json" "$AMLT_OUTPUT_DIR/metrics/run_config.json"
cp "$RUN_DIR/training_metrics.json" "$AMLT_OUTPUT_DIR/metrics/training_metrics.json"
cp "$RUN_DIR/validation_metrics.json" "$AMLT_OUTPUT_DIR/metrics/validation_metrics.json"
cp "$RUN_DIR/causal_validation_metrics.json" "$AMLT_OUTPUT_DIR/metrics/causal_validation_metrics.json"
cp "$PROMOTION" "$AMLT_OUTPUT_DIR/metrics/promoted_real_order_manifest.json"
if [[ -n "$T6_DECISION_THRESHOLD" ]]; then
  cp "$T6_CALIBRATION" "$AMLT_OUTPUT_DIR/metrics/t6_calibration.json"
fi
if [[ "$PROMOTION_BRANCH" == eedi_full_open_t3 || "$PROMOTION_BRANCH" == equal_t3_t6_open_t3 || "$PROMOTION_BRANCH" == equal_t3_t6_open_t3_two_stage ]]; then
  if [[ "$PROMOTION_LOSS_PROFILE" == state_kd || "$PROMOTION_BRANCH" == equal_t3_t6_open_t3_two_stage ]]; then
    test -f "$RUN_DIR/promoted_open_t3_selective_kd_training_view.json"
    cp "$RUN_DIR/promoted_open_t3_selective_kd_training_view.json" \
      "$AMLT_OUTPUT_DIR/metrics/promoted_open_t3_selective_kd_training_view.json"
  else
    test -f "$RUN_DIR/promoted_open_t3_training_view.json"
    cp "$RUN_DIR/promoted_open_t3_training_view.json" "$AMLT_OUTPUT_DIR/metrics/promoted_open_t3_training_view.json"
  fi
fi
if [[ "$PROMOTION_BRANCH" == equal_t3_t6_open_t3_two_stage ]]; then
  test -f "$RUN_DIR/two_stage_transition_receipt.json"
  cp "$RUN_DIR/two_stage_transition_receipt.json" "$AMLT_OUTPUT_DIR/metrics/two_stage_transition_receipt.json"
fi
cp /tmp/layerwise-promoted-checkpoint-audit.json "$AMLT_OUTPUT_DIR/metrics/source_checkpoint_audit.json"
cp "$RECURRENT_SPLIT/audit.json" "$AMLT_OUTPUT_DIR/metrics/data_audit.json"
cp "$RECURRENT_SPLIT/final_fit_manifest.json" "$AMLT_OUTPUT_DIR/metrics/final_fit_manifest.json"
if [[ -f "$RUN_DIR/gradient_audit.json" ]]; then
  cp "$RUN_DIR/gradient_audit.json" "$AMLT_OUTPUT_DIR/metrics/gradient_audit.json"
fi
