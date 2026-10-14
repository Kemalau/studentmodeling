#!/usr/bin/env bash
set -euo pipefail

export VENV_ROOT="${VENV_ROOT:-/tmp/studentmodeling-promoted-finalize-venv}"
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
export RAW_SPLIT=/tmp/layerwise_promoted_finalize_full_trainvalid_v1
export RECURRENT_SPLIT=/tmp/layerwise_promoted_finalize_recurrent_full259845_u4_v1
export FINAL_JSONL="$RECURRENT_SPLIT/final_fit_trainvalid.jsonl"
export C4_DATA=/mnt/amulet/studentmodeling/c4_fullopen_20260724/malrule_c1_clean
export CAUSAL_DATA=/tmp/layerwise_promoted_finalize_cf4_u4_v1
export PROMOTION=/mnt/amulet/studentmodeling/layerwise_world_model_v1/task_retention_promotion/selected.json
export T6_CALIBRATION=/mnt/amulet/studentmodeling/layerwise_world_model_v1/task_retention_promotion/t6_calibration.json
export RUN_DIR=/mnt/amulet/studentmodeling/checkpoints/layerwise_unified_world_model_seed42_taskretention_eedi_full_open_t3_state_path_v1_20260802
export CHECKPOINT="$RUN_DIR/checkpoint-round-16241"
export SOURCE_CHECKPOINT=/mnt/amulet/studentmodeling/diagnostics/layerwise_unified_finalfit_seed42_20260801_snapshot/checkpoint-round-750

mkdir -p "$HF_HUB_CACHE" "$RAW_SPLIT" "$RECURRENT_SPLIT" "$CAUSAL_DATA" \
  "$AMLT_OUTPUT_DIR/model" "$AMLT_OUTPUT_DIR/metrics"
test "$(df --output=avail /tmp | tail -1)" -ge 125829120
test -f "$RAW_FULL"
test -f "$C4_DATA/train.jsonl"
test -f "$C4_DATA/val.jsonl"
test -f "$PROMOTION"
test -f "$T6_CALIBRATION"
test -f "$CHECKPOINT/adapter/adapter_model.safetensors"
test -f "$CHECKPOINT/recurrent_state.pt"
test -f "$CHECKPOINT/checkpoint_completion.json"
test -f "$SOURCE_CHECKPOINT/recurrent_state.pt"
test -f "$SOURCE_CHECKPOINT/checkpoint_completion.json"

test "$(sha256sum scripts/train_recurrent_student_m4_promoted.py | cut -d' ' -f1)" = ebf21ce4c0dbdf8b99c0a945d92aeec9d236a55d304b91912539034a8a6db81f
test "$(sha256sum scripts/train_recurrent_student_m4_promoted_open_t3.py | cut -d' ' -f1)" = 3ee69fa74b056a6ca15774dacb52a1cfd402d4baf21b0d333a8e7d9f001c4650
test "$(sha256sum scripts/audit_layerwise_distributed_checkpoint.py | cut -d' ' -f1)" = 6690ece8935c6343bec7136ad34a9899b72a866c7bde6286ca26b2c82890007f
test "$(sha256sum scripts/frozen_layerwise_finalfit/layerwise_recurrent_student_model.py | cut -d' ' -f1)" = c80357282e600b31f68f55d8ab434ad349c0e9eeeeb222b52dbadae7caea11ca
test "$(sha256sum tests/frozen_layerwise_finalfit/test_layerwise_recurrent_student_model.py | cut -d' ' -f1)" = a995f82f364eb3247ff3e17873cc720bbce7a2a8f52813c9369421fa61d3c274
cp scripts/frozen_layerwise_finalfit/layerwise_recurrent_student_model.py scripts/layerwise_recurrent_student_model.py
cp tests/frozen_layerwise_finalfit/test_layerwise_recurrent_student_model.py tests/test_layerwise_recurrent_student_model.py
python -m unittest discover -s tests -p 'test_train_recurrent_student_m4.py' -v
python -m unittest discover -s tests -p 'test_layerwise_recurrent_student_model.py' -v

readarray -t promotion_values < <(python - "$PROMOTION" "$T6_CALIBRATION" <<'PY'
import hashlib,json,sys
promotion=json.load(open(sys.argv[1],encoding='utf-8'))
calibration=json.load(open(sys.argv[2],encoding='utf-8'))
assert promotion['stage']=='layerwise_task_retention_promoted_order_v1',promotion
assert promotion['branch']=='eedi_full_open_t3',promotion
assert promotion['selected_loss_profile']=='state_path',promotion
assert promotion['selected_candidate_id']=='eedi_full_open_t3__state_path__lr2e-06',promotion
assert float(promotion['selected_learning_rate'])==2e-6,promotion
assert float(promotion['selected_causal_weight'])==2.0,promotion
assert promotion['source_checkpoint_round']==750 and promotion['source_world_size']==16,promotion
assert promotion['epoch_rows']==259845 and promotion['one_epoch_permutation'] is True,promotion
assert promotion['formal_example_id_overlap']==0,promotion
assert calibration['stage']=='layerwise_t6_threshold_calibration_v1',calibration
assert calibration['status']=='frozen',calibration
assert float(calibration['threshold'])==float(promotion['t6_decision_threshold']),promotion
digest=hashlib.sha256(open(sys.argv[2],'rb').read()).hexdigest()
assert digest==promotion['t6_calibration_artifact_sha256'],promotion
print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())
print(promotion['real_order_sha256'])
print(promotion['t6_decision_threshold'])
print(digest)
PY
)
export PROMOTION_SHA256="${promotion_values[0]}"
export PROMOTION_ORDER_SHA256="${promotion_values[1]}"
export T6_DECISION_THRESHOLD="${promotion_values[2]}"
export T6_CALIBRATION_SHA256="${promotion_values[3]}"

python scripts/audit_layerwise_distributed_checkpoint.py \
  --checkpoint "$SOURCE_CHECKPOINT/recurrent_state.pt" --expected-round 750 \
  --output /tmp/promoted-source-checkpoint-audit.json

python - "$CHECKPOINT" "$PROMOTION_SHA256" "$PROMOTION_ORDER_SHA256" <<'PY'
import hashlib,json,pathlib,sys,torch
root=pathlib.Path(sys.argv[1])
receipt=json.load(open(root/'checkpoint_completion.json',encoding='utf-8'))
payload=torch.load(root/'recurrent_state.pt',map_location='cpu',weights_only=False)
def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(8*1024*1024),b''): value.update(chunk)
    return value.hexdigest()
actual={'adapter_model':digest(root/'adapter/adapter_model.safetensors'),'recurrent_state':digest(root/'recurrent_state.pt')}
assert receipt=={'status':'complete','round':16241,'world_size':16,'sha256':actual},receipt
assert payload.get('distributed') is True and payload.get('round')==16241,payload
assert payload.get('world_size')==16,payload
assert (payload.get('scheduler') or {}).get('last_epoch')==16241,payload.get('scheduler')
history=payload.get('history') or []
assert len(history)==259845,len(history)
assert [int(row['global_position']) for row in history]==list(range(259845))
assert len({str(row['example_id']) for row in history})==259845
assert len(payload.get('real_order') or [])==259845
assert set(payload['real_order'])==set(range(259845))
assert len(payload.get('causal_schedule') or {})==12992
config=payload.get('config') or {}
expected={
 'state_architecture':'layerwise','state_token_alias':'<|vision_pad|>',
 'layerwise_transition_rank':256,'layerwise_operation_checkpointing':True,
 'generation_weight':1.0,'behavior_kd_weight':0.0,'consistency_weight':0.25,
 'direct_replay_weight':0.0,'causal_weight':2.0,'causal_margin':0.5,
 'causal_preference_reduction':'sum','causal_objective_mode':'specificity_triplet',
 'causal_repeat_fraction':0.05,'normalize_causal_exposure':True,
 'gradient_accumulation_steps':1,'lr_scheduler':'cosine','warmup_ratio':0.03,
 'learning_rate':2e-6,'seed':42,'distributed_world_size':16,
 'real_train_rows':259845,'promoted_real_order_applied':True,
 'promoted_real_order_source_round':750,
 'promoted_selected_candidate_id':'eedi_full_open_t3__state_path__lr2e-06',
 'promoted_selected_loss_profile':'state_path','promoted_selected_learning_rate':2e-6,
 'promoted_selected_causal_weight':2.0,'training_view':'open_t3_no_candidate_enumeration',
}
drift={key:{'expected':value,'actual':config.get(key)} for key,value in expected.items() if config.get(key)!=value}
assert not drift,drift
assert config.get('promoted_real_order_manifest_sha256')==sys.argv[2],config
manifest=json.load(open('/mnt/amulet/studentmodeling/layerwise_world_model_v1/task_retention_promotion/selected.json',encoding='utf-8'))
assert manifest['real_order_sha256']==sys.argv[3],manifest
assert payload['real_order']==manifest['real_order'],(len(payload['real_order']),len(manifest['real_order']))
audit={'stage':'layerwise_promoted_final_checkpoint_audit_v1','status':'passed','round':16241,'world_size':16,'history_rows':len(history),'unique_examples':259845,'causal_exposures':12992,'optimizer_step_free_finalization_ready':True,'sha256':actual}
path=pathlib.Path('/tmp/promoted-final-checkpoint-audit.json'); path.write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n')
print(json.dumps(audit,sort_keys=True))
PY

rm -rf "$RAW_SPLIT" "$RECURRENT_SPLIT" "$CAUSAL_DATA"
mkdir -p "$RAW_SPLIT" "$RECURRENT_SPLIT" "$CAUSAL_DATA"
python scripts/split_full_mix_by_original_split.py --input-jsonl "$RAW_FULL" --output-dir "$RAW_SPLIT"
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
test "$(sha256sum "$FINAL_JSONL" | cut -d' ' -f1)" = 9961f746cb17e6c314e182d3453b6157022fb48b8de44d2e68bc8e87a1131ccd
test "$(sha256sum "$RECURRENT_SPLIT/valid.jsonl" | cut -d' ' -f1)" = 78bfce75576dc3873cbb6459669a952adeedb8b77bda15f281af9519e2506d14
test "$(sha256sum "$CAUSAL_DATA/train.jsonl" | cut -d' ' -f1)" = 30f68f54120e5a9ada0a37b1dba2b6480de0f8eaeac98d0b7ed7c627429f9766
test "$(sha256sum "$CAUSAL_DATA/valid.jsonl" | cut -d' ' -f1)" = 8e038c3a60c2e1b420779cd54eb203b12ae5d51544c87a805fe19d47fdbe294e

rm -rf "$LOCAL_MODEL"; mkdir -p "$LOCAL_MODEL"
python scripts/copy_model_parallel.py \
  --source /mnt/amulet/studentmodeling/models/Qwen3-32B \
  --destination "$LOCAL_MODEL" --workers 16
test -f "$LOCAL_MODEL/model.safetensors.index.json"

python -u scripts/train_recurrent_student_m4_promoted_open_t3.py \
  --model "$LOCAL_MODEL" --init-adapter "$CHECKPOINT/adapter" \
  --state-architecture layerwise --layerwise-operation-checkpointing \
  --layerwise-transition-rank 256 \
  --train-jsonl "$FINAL_JSONL" --valid-jsonl "$RECURRENT_SPLIT/valid.jsonl" \
  --causal-train-jsonl "$CAUSAL_DATA/train.jsonl" --causal-valid-jsonl "$CAUSAL_DATA/valid.jsonl" \
  --output-dir "$RUN_DIR" --state-token-alias '<|vision_pad|>' --max-length 1536 \
  --learning-rate 2e-6 --lr-scheduler cosine --warmup-ratio 0.03 \
  --generation-weight 1.0 --behavior-kd-weight 0.0 --consistency-weight 0.25 \
  --direct-replay-weight 0.0 --causal-weight 2.0 --causal-margin 0.5 \
  --causal-preference-reduction sum --causal-objective-mode specificity_triplet \
  --causal-repeat-fraction 0.05 --normalize-causal-exposure \
  --gradient-accumulation-steps 1 --eval-limit 300 --causal-eval-limit 165 \
  --save-steps 250 --checkpoint-retention 2 --distributed-world-size 16 --seed 42 \
  --resume-from-checkpoint "$CHECKPOINT/recurrent_state.pt" \
  --validation-only-from-distributed-checkpoint --finalize-complete-distributed-epoch

test -f "$RUN_DIR/completion.json"
test -f "$RUN_DIR/final_adapter/adapter_model.safetensors"
test -f "$RUN_DIR/recurrent_state.pt"
python - "$RUN_DIR" <<'PY'
import json,os,pathlib,sys
root=pathlib.Path(sys.argv[1]); completion=json.load(open(root/'completion.json',encoding='utf-8')); config=json.load(open(root/'run_config.json',encoding='utf-8'))
assert completion['status']=='complete' and completion['diagnostic'] is False,completion
assert completion['micro_steps']==259845 and completion['optimizer_steps']==16241,completion
assert completion['real_examples_seen_once'] is True and completion['exact_history_support'] is True,completion
assert completion['causal_exposure_count_matches_plan'] is True and completion['all_causal_pairs_covered'] is True,completion
assert config['validation_only_from_distributed_checkpoint'] is True,config
assert config['finalize_complete_distributed_epoch'] is True,config
assert config['state_architecture']=='layerwise' and config['distributed_world_size']==16,config
receipt={
 'status':'final_fit_complete','stage':'layerwise_task_retention_promoted_final_fit',
 'branch':'eedi_full_open_t3','loss_profile':'state_path',
 'candidate_id':'eedi_full_open_t3__state_path__lr2e-06','learning_rate':2e-6,
 'causal_weight':2.0,'run_dir':str(root),'state_architecture':'layerwise',
 'state_shape':[64,5120],'real_examples':259845,'one_epoch_exact':True,
 'promoted_source_round':750,'promoted_order_sha256':os.environ['PROMOTION_ORDER_SHA256'],
 'optimizer_steps':completion['optimizer_steps'],'raw_history_visible_to_decoder':False,
 'persistent_history_kv_cache':False,
 't6_decision_threshold':float(os.environ['T6_DECISION_THRESHOLD']),
 't6_calibration_artifact_sha256':os.environ['T6_CALIBRATION_SHA256'],
 't6_threshold_selection_split':'unused_original_validation',
 't6_threshold_frozen_before_final_fit':True,
 'recovery_mode':'single_A100_validation_from_exact_promoted_final_checkpoint',
}
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
cp "$RUN_DIR/promoted_open_t3_training_view.json" "$AMLT_OUTPUT_DIR/metrics/promoted_open_t3_training_view.json"
cp "$PROMOTION" "$AMLT_OUTPUT_DIR/metrics/promoted_real_order_manifest.json"
cp "$T6_CALIBRATION" "$AMLT_OUTPUT_DIR/metrics/t6_calibration.json"
cp /tmp/promoted-source-checkpoint-audit.json "$AMLT_OUTPUT_DIR/metrics/source_checkpoint_audit.json"
cp /tmp/promoted-final-checkpoint-audit.json "$AMLT_OUTPUT_DIR/metrics/final_checkpoint_audit.json"
cp "$RECURRENT_SPLIT/audit.json" "$AMLT_OUTPUT_DIR/metrics/data_audit.json"
cp "$RECURRENT_SPLIT/final_fit_manifest.json" "$AMLT_OUTPUT_DIR/metrics/final_fit_manifest.json"
if [[ -f "$RUN_DIR/gradient_audit.json" ]]; then
  cp "$RUN_DIR/gradient_audit.json" "$AMLT_OUTPUT_DIR/metrics/gradient_audit.json"
fi
