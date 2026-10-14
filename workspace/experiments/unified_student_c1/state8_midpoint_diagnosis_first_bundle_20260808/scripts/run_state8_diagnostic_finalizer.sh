#!/usr/bin/env bash
set -euo pipefail

export VENV_ROOT="${VENV_ROOT:-/tmp/studentmodeling-state8-finalizer-venv}"
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
export RAW_SPLIT=/tmp/state8_diag_finalize_full_trainvalid_v1
export RECURRENT_SPLIT=/tmp/state8_diag_finalize_recurrent_u4_v1
export FINAL_JSONL="$RECURRENT_SPLIT/final_fit_trainvalid.jsonl"
export C4_DATA=/mnt/amulet/studentmodeling/c4_fullopen_20260724/malrule_c1_clean
export CAUSAL_BASE=/tmp/state8_diag_finalize_cf4_base_v1
export CAUSAL_DATA=/tmp/state8_diag_finalize_cf4_same_rule_v1
export SOURCE_ROOT=/mnt/amulet/studentmodeling/diagnostics/layerwise_task_retention_service_recovery/eedi_dense/state_path/same_rule/lr2e-6_start12000_stop3250_cc484d7f7b6db02cfdf1e530866e029a721896d9c57103313e581f504c7bc16e_b57160eeaff758ee9b56109e9660dd3662c77db684dc4fab351747cbd836d471
export CHECKPOINT="$SOURCE_ROOT/checkpoint-round-3250"
export RUN_DIR=/mnt/amulet/studentmodeling/diagnostics/state8_natural_dualpath_samerule_v5_finalized_20260805
export STABLE_DIR=/mnt/amulet/studentmodeling/diagnostics/state8_natural_dualpath_samerule_v5_immutable
export RAW_EVAL=/mnt/amulet/studentmodeling/formal_evaluation_v1/test.jsonl
export RECURRENT_EVAL=/tmp/state8_diag_finalize_formal_eval_u4_v1/test.jsonl
export C4_PROGRESS_DIR=/mnt/amulet/studentmodeling/eval_progress/state8_natural_dualpath_samerule_v5_finalizer/c4_seed42_v1

mkdir -p "$HF_HUB_CACHE" "$RAW_SPLIT" "$RECURRENT_SPLIT" "$CAUSAL_BASE" "$CAUSAL_DATA" \
  "$RUN_DIR" "$C4_PROGRESS_DIR" "$AMLT_OUTPUT_DIR/model" "$AMLT_OUTPUT_DIR/metrics" \
  "$STABLE_DIR/model" "$STABLE_DIR/metrics" \
  "$AMLT_OUTPUT_DIR/eval/c4" "$AMLT_OUTPUT_DIR/eval/formal_screen" "$AMLT_OUTPUT_DIR/eval/t6_full"
test "$(df --output=avail /tmp | tail -1)" -ge 125829120
test -f "$CHECKPOINT/adapter/adapter_model.safetensors"
test -f "$CHECKPOINT/recurrent_state.pt"
test -f "$CHECKPOINT/checkpoint_completion.json"
test "$(sha256sum scripts/train_recurrent_student_m4.py | cut -d' ' -f1)" = b0a726cdd416af47d18ec15e82a4f4b24bb32707a48fc9ecaf6dd2d05626c9f2
test "$(sha256sum scripts/train_recurrent_student_m4_state8_natural_curriculum.py | cut -d' ' -f1)" = 503846f21c919143b5b79304eca4c8e5445c506fc76b54035b3a967922141799

python - "$CHECKPOINT" <<'PY'
import hashlib,json,pathlib,sys,torch
root=pathlib.Path(sys.argv[1]); receipt=json.load(open(root/'checkpoint_completion.json'))
def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()
observed={'adapter_model':digest(root/'adapter/adapter_model.safetensors'),
          'recurrent_state':digest(root/'recurrent_state.pt')}
assert receipt=={'status':'complete','round':3250,'world_size':4,'sha256':observed},receipt
payload=torch.load(root/'recurrent_state.pt',map_location='cpu',weights_only=False)
assert payload.get('distributed') is True and payload.get('round')==3250,payload
assert payload.get('world_size')==4,payload
history=payload.get('history') or []
assert len(history)==1000,len(history)
assert [int(row['global_position']) for row in history]==list(range(12000,13000)),history[:2]
assert len({str(row['example_id']) for row in history})==1000
config=payload.get('config') or {}
assert config['state_architecture']=='layerwise' and config['state_token_count']==8,config
assert config['history_dropout_curriculum'] is True and config['dual_path_curriculum'] is True,config
assert config['output_schema']=='label_diagnosis_evidence_v1',config
assert config['causal_objective_mode']=='specificity_triplet',config
PY

rm -rf "$RAW_SPLIT" "$RECURRENT_SPLIT" "$CAUSAL_BASE" "$CAUSAL_DATA"
mkdir -p "$RAW_SPLIT" "$RECURRENT_SPLIT" "$CAUSAL_BASE" "$CAUSAL_DATA"
python scripts/split_full_mix_by_original_split.py --input-jsonl "$RAW_FULL" --output-dir "$RAW_SPLIT"
python scripts/build_unified_recurrent_manifest.py \
  --train-jsonl "$RAW_SPLIT/train.jsonl" --valid-jsonl "$RAW_SPLIT/valid.jsonl" \
  --output-dir "$RECURRENT_SPLIT" --max-updates 4 --seed 20260730
python scripts/merge_recurrent_full_corpus.py \
  --train-jsonl "$RECURRENT_SPLIT/train.jsonl" --valid-jsonl "$RECURRENT_SPLIT/valid.jsonl" \
  --output-jsonl "$FINAL_JSONL" --manifest "$RECURRENT_SPLIT/final_fit_manifest.json"
python scripts/build_recurrent_causal_pairs.py \
  --train-source "$C4_DATA/train.jsonl" --valid-source "$C4_DATA/val.jsonl" \
  --train-pairs manifests/cf4_balanced_paired_state_swap_train_seed20260731_v1.json \
  --valid-pairs manifests/cf4_balanced_paired_state_swap_valid_seed20260801_v1.json \
  --output-dir "$CAUSAL_BASE" --max-updates 4
python scripts/augment_recurrent_causal_same_rule.py \
  --train-source "$C4_DATA/train.jsonl" --valid-source "$C4_DATA/val.jsonl" \
  --causal-train "$CAUSAL_BASE/train.jsonl" --causal-valid "$CAUSAL_BASE/valid.jsonl" \
  --output-dir "$CAUSAL_DATA" --seed 20260804 --max-updates 4
test "$(sha256sum "$FINAL_JSONL" | cut -d' ' -f1)" = 9961f746cb17e6c314e182d3453b6157022fb48b8de44d2e68bc8e87a1131ccd
test "$(sha256sum "$RECURRENT_SPLIT/valid.jsonl" | cut -d' ' -f1)" = 78bfce75576dc3873cbb6459669a952adeedb8b77bda15f281af9519e2506d14
test "$(sha256sum "$CAUSAL_DATA/train.jsonl" | cut -d' ' -f1)" = a170b085858b656432725cd6e835703cb195c469d8374c3abfd9f0a51f909f24
test "$(sha256sum "$CAUSAL_DATA/valid.jsonl" | cut -d' ' -f1)" = d1f97a387b6be30467918813475edac6ac0d38e5923767f27f3aca792b130696

rm -rf "$LOCAL_MODEL"; mkdir -p "$LOCAL_MODEL"
python scripts/copy_model_parallel.py \
  --source /mnt/amulet/studentmodeling/models/Qwen3-32B \
  --destination "$LOCAL_MODEL" --workers 16
test -f "$LOCAL_MODEL/model.safetensors.index.json"

python -u scripts/train_recurrent_student_m4_state8_natural_curriculum.py \
  --model "$LOCAL_MODEL" --init-adapter "$CHECKPOINT/adapter" \
  --init-recurrent-state "$CHECKPOINT/recurrent_state.pt" \
  --state-architecture layerwise --layerwise-operation-checkpointing \
  --layerwise-transition-rank 256 --state-token-count 8 \
  --train-jsonl "$FINAL_JSONL" --valid-jsonl "$RECURRENT_SPLIT/valid.jsonl" \
  --causal-train-jsonl "$CAUSAL_DATA/train.jsonl" --causal-valid-jsonl "$CAUSAL_DATA/valid.jsonl" \
  --output-dir "$RUN_DIR" --state-token-alias '<|vision_pad|>' \
  --history-dropout-curriculum --dual-path-curriculum \
  --online-kd-weight 0.5 --online-kd-temperature 1.0 --online-kd-topk 256 \
  --max-length 1536 --learning-rate 2e-6 --lr-scheduler constant \
  --generation-weight 1.0 --behavior-kd-weight 0.0 --consistency-weight 0.25 \
  --direct-replay-weight 0.0 --causal-weight 2.0 --causal-margin 0.5 \
  --causal-preference-reduction sum --causal-objective-mode specificity_triplet \
  --causal-repeat-fraction 0.05 --normalize-causal-exposure \
  --gradient-accumulation-steps 1 --eval-limit 300 --causal-eval-limit 165 \
  --save-steps 50 --checkpoint-retention 2 --distributed-world-size 4 \
  --diagnostic-start-global-position 12000 --diagnostic-stop-after-round 3250 \
  --diagnostic-index-manifest manifests/layerwise_task_retention_eedi_dense_seed42_v1.json \
  --diagnostic-allow-causal-data-transition \
  --resume-from-checkpoint "$CHECKPOINT/recurrent_state.pt" \
  --validation-only-from-distributed-checkpoint --seed 42

test -f "$RUN_DIR/completion.json"
test -f "$RUN_DIR/final_adapter/adapter_model.safetensors"
test -f "$RUN_DIR/recurrent_state.pt"
python - "$RUN_DIR" <<'PY'
import collections,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); c=json.load(open(root/'completion.json')); h=json.load(open(root/'trainer_history.json'))
assert c['status']=='diagnostic_complete' and c['diagnostic'] is True,c
assert c['micro_steps']==1000 and c['optimizer_steps']==250,c
assert c['history_integrity_pass'] is True and c['history_rows']==1000,c
assert c['history_global_position_start']==12000 and c['history_global_position_end']==12999,c
assert collections.Counter(float(x['history_dropout']) for x in h)=={0.0:200,0.25:200,0.5:200,0.75:200,1.0:200}
PY

export C4_PROGRESS_KEY="$(python - "$RUN_DIR/completion.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); print(f"adapter={x['sha256']['adapter_model']}:state={x['sha256']['recurrent_state']}")
PY
)"
python -u scripts/evaluate_recurrent_causal_gate.py \
  --model "$LOCAL_MODEL" --adapter "$RUN_DIR/final_adapter" \
  --recurrent-state "$RUN_DIR/recurrent_state.pt" --state-architecture layerwise \
  --jsonl "$C4_DATA/test_same_template.jsonl" \
  --pair-manifest manifests/c4_stage_a_gate_pairs_seed42_n150_v1.json \
  --output-json "$AMLT_OUTPUT_DIR/eval/c4/recurrent_causal_gate.json" \
  --state-token-alias '<|vision_pad|>' --max-length 1536 --num-pairs 150 --seed 42 \
  --progress-dir "$C4_PROGRESS_DIR" --progress-key "$C4_PROGRESS_KEY"
python scripts/build_recurrent_evaluation_manifest.py \
  --input-jsonl "$RAW_EVAL" --output-jsonl "$RECURRENT_EVAL" \
  --audit-json /tmp/state8_diag_finalize_formal_eval_u4_v1/audit.json \
  --max-updates 4 --seed 20260730
test "$(sha256sum "$RECURRENT_EVAL" | cut -d' ' -f1)" = a368cefb0c49cb6a305ad1a892808361fba2d44d555a0a28ea64e56babe32525
python -u scripts/evaluate_recurrent_formal_protocols.py \
  --model "$LOCAL_MODEL" --adapter "$RUN_DIR/final_adapter" \
  --recurrent-state "$RUN_DIR/recurrent_state.pt" --state-architecture layerwise \
  --eval-jsonl "$RECURRENT_EVAL" --output-dir "$AMLT_OUTPUT_DIR/eval/formal_screen" \
  --state-token-alias '<|vision_pad|>' --max-samples-per-group 25 \
  --max-length 1536 --max-new-tokens 128 --state-batch-size 2 \
  --candidate-row-batch-size 2 --seed 42 --save-unredacted-predictions --t3-natural-schema
python -u scripts/evaluate_recurrent_formal_protocols.py \
  --model "$LOCAL_MODEL" --adapter "$RUN_DIR/final_adapter" \
  --recurrent-state "$RUN_DIR/recurrent_state.pt" --state-architecture layerwise \
  --eval-jsonl "$RECURRENT_EVAL" --output-dir "$AMLT_OUTPUT_DIR/eval/t6_full" \
  --state-token-alias '<|vision_pad|>' --max-length 1536 --max-new-tokens 128 \
  --state-batch-size 2 --candidate-row-batch-size 2 \
  --tasks predict_error_persistence --seed 42 --save-unredacted-predictions

cp -r "$RUN_DIR/final_adapter/." "$AMLT_OUTPUT_DIR/model/"
cp "$RUN_DIR/recurrent_state.pt" "$AMLT_OUTPUT_DIR/model/recurrent_state.pt"
cp "$RUN_DIR"/*.json "$AMLT_OUTPUT_DIR/metrics/"
cp -r "$RUN_DIR/final_adapter/." "$STABLE_DIR/model/"
cp "$RUN_DIR/recurrent_state.pt" "$STABLE_DIR/model/recurrent_state.pt"
cp "$RUN_DIR/completion.json" "$STABLE_DIR/metrics/completion.json"
cp "$RUN_DIR/state8_natural_curriculum_view.json" "$STABLE_DIR/metrics/state8_natural_curriculum_view.json"
python - "$STABLE_DIR" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); c=json.load(open(root/'metrics/completion.json'))
def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
assert c['sha256']['adapter_model']==digest(root/'model/adapter_model.safetensors'),c
assert c['sha256']['recurrent_state']==digest(root/'model/recurrent_state.pt'),c
(root/'metrics/immutable_receipt.json').write_text(json.dumps({'status':'complete','source':'single_A100_round3250_finalizer','sha256':c['sha256']},indent=2,sort_keys=True)+'\n')
PY
