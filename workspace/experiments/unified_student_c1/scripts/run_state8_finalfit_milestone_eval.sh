#!/usr/bin/env bash
set -euo pipefail

export VENV_ROOT="${VENV_ROOT:-/tmp/studentmodeling-state8-finalfit-milestone-venv}"
export PATH="$VENV_ROOT/bin:$PATH"
python(){ "$VENV_ROOT/bin/python" "$@"; }
export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/hf_cache
export HF_HUB_CACHE=/tmp/hf_cache/hub
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error
export LOCAL_MODEL=/tmp/Qwen3-32B
: "${MILESTONE_VARIANT:?MILESTONE_VARIANT is required}"
: "${MILESTONE_ROUND:?MILESTONE_ROUND is required}"
case "$MILESTONE_ROUND" in 3249|6497|9745|12993) ;; *) exit 2 ;; esac
case "$MILESTONE_VARIANT" in
  t3_heavy)
    export RESULT_ROOT=/mnt/amulet/studentmodeling/diagnostics/state8_qualified_teacher_curriculum_v1_immutable
    export EXPECTED_IMMUTABLE_STAGE=state8_qualified_teacher_curriculum_immutable_v1
    export EXPECTED_ONLINE_KD_WEIGHT=0.5 ;;
  balanced)
    export RESULT_ROOT=/mnt/amulet/studentmodeling/diagnostics/state8_qualified_teacher_balanced_curriculum_v1_immutable
    export EXPECTED_IMMUTABLE_STAGE=state8_qualified_teacher_balanced_curriculum_immutable_v1
    export EXPECTED_ONLINE_KD_WEIGHT=0.5 ;;
  highkd)
    export RESULT_ROOT=/mnt/amulet/studentmodeling/diagnostics/state8_qualified_teacher_curriculum_highkd_v1_immutable
    export EXPECTED_IMMUTABLE_STAGE=state8_qualified_teacher_curriculum_highkd_immutable_v1
    export EXPECTED_ONLINE_KD_WEIGHT=1.0 ;;
  *) exit 2 ;;
esac
export SOURCE_RUN=/mnt/amulet/studentmodeling/checkpoints/state8_qualified_unified_full259845_seed42_${MILESTONE_VARIANT}_v1_20260805
export CHECKPOINT="$SOURCE_RUN/checkpoint-round-$MILESTONE_ROUND"
export INITIAL_MODEL="$RESULT_ROOT/model"
export INITIAL_RECEIPT="$RESULT_ROOT/metrics/immutable_receipt.json"
export RAW_FULL=/mnt/amulet/studentmodeling/full_trainplusvalid_no_t4_strict_t6/train.jsonl
export RAW_SPLIT=/tmp/state8_finalfit_milestone_raw
export RECURRENT_SPLIT=/tmp/state8_finalfit_milestone_recurrent
export FINAL_JSONL="$RECURRENT_SPLIT/final_fit_trainvalid.jsonl"
export RAW_EVAL=/mnt/amulet/studentmodeling/formal_evaluation_v1/test.jsonl
export RECURRENT_EVAL=/tmp/state8_finalfit_milestone_eval/test.jsonl
export C4_DATA=/mnt/amulet/studentmodeling/c4_fullopen_20260724/malrule_c1_clean
export CAUSAL_BASE=/tmp/state8_finalfit_milestone_causal_base
export CAUSAL_DATA=/tmp/state8_finalfit_milestone_causal_same_rule
export FINALIZED=/tmp/state8_finalfit_milestone_finalized
export FORMAL_OUT="$AMLT_OUTPUT_DIR/eval/formal_screen"
export C4_OUT="$AMLT_OUTPUT_DIR/eval/c4"
export C4_PROGRESS_DIR="/mnt/amulet/studentmodeling/eval_progress/state8_finalfit_milestones/${MILESTONE_VARIANT}/round${MILESTONE_ROUND}/c4_seed42_v1"

mkdir -p "$HF_HUB_CACHE" "$RAW_SPLIT" "$RECURRENT_SPLIT" "$(dirname "$RECURRENT_EVAL")" \
  "$CAUSAL_BASE" "$CAUSAL_DATA" "$FINALIZED" "$FORMAL_OUT" "$C4_OUT" "$C4_PROGRESS_DIR" \
  "$AMLT_OUTPUT_DIR/metrics"
test "$(df --output=avail /tmp | tail -1)" -ge 125829120
test -f "$CHECKPOINT/adapter/adapter_model.safetensors"
test -f "$CHECKPOINT/recurrent_state.pt"
test -f "$CHECKPOINT/checkpoint_completion.json"
test -f "$INITIAL_MODEL/adapter_model.safetensors"
test -f "$INITIAL_MODEL/recurrent_state.pt"
test -f "$INITIAL_RECEIPT"
test -f "$RAW_FULL"
test -f "$RAW_EVAL"
test -f "$C4_DATA/train.jsonl"
test -f "$C4_DATA/val.jsonl"
test -f "$C4_DATA/test_same_template.jsonl"
test -f /mnt/amulet/studentmodeling/models/Qwen3-32B/model.safetensors.index.json
test "$(sha256sum scripts/train_recurrent_student_m4.py | cut -d' ' -f1)" = ae502968872ff6597e73bcc12cb3ff9a5bbb0fe9d7a763383f6e76bc392de42c
test "$(sha256sum scripts/evaluate_recurrent_formal_protocols.py | cut -d' ' -f1)" = 262370592e018641b99d6282858d8990b38314c05bb6e3bbc645823834cf73ed
test "$(sha256sum scripts/evaluate_recurrent_causal_gate.py | cut -d' ' -f1)" = e0c9fe1361dfda5eddc14abcd275fbe8e2b92f472f9eaf27468b71f800fafccf

python - "$CHECKPOINT" "$INITIAL_RECEIPT" "$INITIAL_MODEL" "$MILESTONE_ROUND" \
  "$EXPECTED_IMMUTABLE_STAGE" "$EXPECTED_ONLINE_KD_WEIGHT" <<'PY'
import hashlib,json,pathlib,sys,torch
checkpoint=pathlib.Path(sys.argv[1]); immutable=json.load(open(sys.argv[2])); initial=pathlib.Path(sys.argv[3])
expected_round=int(sys.argv[4]); expected_stage=sys.argv[5]; expected_kd=float(sys.argv[6])
def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
 return h.hexdigest()
observed={'adapter_model':digest(checkpoint/'adapter/adapter_model.safetensors'),
          'recurrent_state':digest(checkpoint/'recurrent_state.pt')}
receipt=json.load(open(checkpoint/'checkpoint_completion.json'))
assert receipt=={'status':'complete','round':expected_round,'world_size':16,'sha256':observed},receipt
payload=torch.load(checkpoint/'recurrent_state.pt',map_location='cpu',weights_only=False)
assert payload.get('distributed') is True and payload.get('round')==expected_round,payload
assert payload.get('world_size')==16,payload
history=list(payload.get('history') or []); expected_rows=min(259845,expected_round*16)
assert len(history)==expected_rows,(len(history),expected_rows)
assert [int(row['global_position']) for row in history]==list(range(expected_rows)),history[:2]
config=payload.get('config') or {}
assert config['state_architecture']=='layerwise' and config['state_token_count']==8,config
assert config['history_dropout_curriculum'] is True and config['dual_path_curriculum'] is True,config
assert config['output_schema']=='label_diagnosis_evidence_v1',config
assert config['final_inference_route']=='strict_state_only',config
assert config['causal_objective_mode']=='specificity_triplet',config
assert float(config['online_kd_weight'])==expected_kd,config
initial_hash={'adapter_model':digest(initial/'adapter_model.safetensors'),
              'recurrent_state':digest(initial/'recurrent_state.pt')}
assert immutable['stage']==expected_stage and immutable['sha256']==initial_hash,(immutable,initial_hash)
assert str(config['init_adapter'])==str(initial),config
assert str(config['init_recurrent_state'])==str(initial/'recurrent_state.pt'),config
PY

rm -rf "$RAW_SPLIT" "$RECURRENT_SPLIT" "$CAUSAL_BASE" "$CAUSAL_DATA" "$FINALIZED" "$(dirname "$RECURRENT_EVAL")"
mkdir -p "$RAW_SPLIT" "$RECURRENT_SPLIT" "$CAUSAL_BASE" "$CAUSAL_DATA" "$FINALIZED" "$(dirname "$RECURRENT_EVAL")"
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
python scripts/build_recurrent_evaluation_manifest.py \
  --input-jsonl "$RAW_EVAL" --output-jsonl "$RECURRENT_EVAL" \
  --audit-json /tmp/state8_finalfit_milestone_eval/audit.json --max-updates 4 --seed 20260730
test "$(sha256sum "$FINAL_JSONL" | cut -d' ' -f1)" = 9961f746cb17e6c314e182d3453b6157022fb48b8de44d2e68bc8e87a1131ccd
test "$(sha256sum "$RECURRENT_SPLIT/valid.jsonl" | cut -d' ' -f1)" = 78bfce75576dc3873cbb6459669a952adeedb8b77bda15f281af9519e2506d14
test "$(sha256sum "$RECURRENT_EVAL" | cut -d' ' -f1)" = a368cefb0c49cb6a305ad1a892808361fba2d44d555a0a28ea64e56babe32525
test "$(sha256sum "$CAUSAL_DATA/train.jsonl" | cut -d' ' -f1)" = a170b085858b656432725cd6e835703cb195c469d8374c3abfd9f0a51f909f24
test "$(sha256sum "$CAUSAL_DATA/valid.jsonl" | cut -d' ' -f1)" = d1f97a387b6be30467918813475edac6ac0d38e5923767f27f3aca792b130696

rm -rf "$LOCAL_MODEL"; mkdir -p "$LOCAL_MODEL"
python scripts/copy_model_parallel.py --source /mnt/amulet/studentmodeling/models/Qwen3-32B \
  --destination "$LOCAL_MODEL" --workers 16
test -f "$LOCAL_MODEL/model.safetensors.index.json"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

python -u scripts/train_recurrent_student_m4.py \
  --model "$LOCAL_MODEL" --init-adapter "$CHECKPOINT/adapter" \
  --init-recurrent-state "$CHECKPOINT/recurrent_state.pt" \
  --state-architecture layerwise --layerwise-operation-checkpointing \
  --layerwise-transition-rank 256 --state-token-count 8 \
  --train-jsonl "$FINAL_JSONL" --valid-jsonl "$RECURRENT_SPLIT/valid.jsonl" \
  --causal-train-jsonl "$CAUSAL_DATA/train.jsonl" --causal-valid-jsonl "$CAUSAL_DATA/valid.jsonl" \
  --output-dir "$FINALIZED" --state-token-alias '<|vision_pad|>' \
  --history-dropout-curriculum --dual-path-curriculum \
  --online-kd-weight "$EXPECTED_ONLINE_KD_WEIGHT" --online-kd-temperature 1.0 --online-kd-topk 256 \
  --max-length 1536 --learning-rate 1e-6 --lr-scheduler cosine --warmup-ratio 0.03 \
  --generation-weight 1.0 --behavior-kd-weight 0.0 --consistency-weight 0.0 --direct-replay-weight 0.0 \
  --causal-weight 2.0 --causal-margin 0.5 --causal-preference-reduction sum \
  --causal-objective-mode specificity_triplet --causal-repeat-fraction 0.05 --normalize-causal-exposure \
  --gradient-accumulation-steps 1 --eval-limit 300 --causal-eval-limit 165 \
  --save-steps 25 --checkpoint-retention 0 --distributed-world-size 16 \
  --diagnostic-stop-after-round "$MILESTONE_ROUND" \
  --resume-from-checkpoint "$CHECKPOINT/recurrent_state.pt" \
  --validation-only-from-distributed-checkpoint --seed 42
test -f "$FINALIZED/completion.json"
test -f "$FINALIZED/final_adapter/adapter_model.safetensors"
test -f "$FINALIZED/recurrent_state.pt"

python -u scripts/evaluate_recurrent_formal_protocols.py \
  --model "$LOCAL_MODEL" --adapter "$FINALIZED/final_adapter" \
  --recurrent-state "$FINALIZED/recurrent_state.pt" --state-architecture layerwise \
  --eval-jsonl "$RECURRENT_EVAL" --output-dir "$FORMAL_OUT" \
  --state-token-alias '<|vision_pad|>' --max-samples-per-group 25 \
  --max-length 1536 --max-new-tokens 128 --state-batch-size 2 \
  --candidate-row-batch-size 2 --seed 42 --save-unredacted-predictions --t3-natural-schema

export C4_PROGRESS_KEY="$(python - "$CHECKPOINT/checkpoint_completion.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); print(f"adapter={x['sha256']['adapter_model']}:state={x['sha256']['recurrent_state']}")
PY
)"
python -u scripts/evaluate_recurrent_causal_gate.py \
  --model "$LOCAL_MODEL" --adapter "$FINALIZED/final_adapter" \
  --recurrent-state "$FINALIZED/recurrent_state.pt" --state-architecture layerwise \
  --jsonl "$C4_DATA/test_same_template.jsonl" \
  --pair-manifest manifests/c4_stage_a_gate_pairs_seed42_n150_v1.json \
  --output-json "$C4_OUT/recurrent_causal_gate.json" \
  --state-token-alias '<|vision_pad|>' --max-length 1536 --num-pairs 150 --seed 42 \
  --progress-dir "$C4_PROGRESS_DIR" --progress-key "$C4_PROGRESS_KEY"

python - "$CHECKPOINT" "$FINALIZED" "$FORMAL_OUT/metrics.json" \
  "$C4_OUT/recurrent_causal_gate.json" "$AMLT_OUTPUT_DIR/metrics/milestone_receipt.json" <<'PY'
import hashlib,json,pathlib,sys
checkpoint=pathlib.Path(sys.argv[1]); finalized=pathlib.Path(sys.argv[2]); formal=json.load(open(sys.argv[3])); c4=json.load(open(sys.argv[4])); out=pathlib.Path(sys.argv[5])
receipt=json.load(open(checkpoint/'checkpoint_completion.json')); completion=json.load(open(finalized/'completion.json'))
config=json.load(open(finalized/'run_config.json'))
source_artifacts=receipt['sha256']
evaluated_artifacts={key:completion['sha256'][key] for key in ('adapter_model','recurrent_state')}
assert completion['status']=='diagnostic_complete' and completion['diagnostic'] is True,completion
assert config['validation_only_from_distributed_checkpoint'] is True,config
assert config['validation_only_checkpoint']==str(checkpoint/'recurrent_state.pt'),config
assert formal['artifact_sha256']==evaluated_artifacts,(formal,evaluated_artifacts)
assert c4['artifact_sha256']==evaluated_artifacts,(c4,evaluated_artifacts)
assert formal['state_shape']==[64,8,5120] and c4['state_shape']==[64,8,5120],(formal,c4)
assert formal['raw_history_visible_to_decoder'] is False and formal['persistent_history_kv_cache'] is False,formal
assert c4['raw_history_visible_to_decoder'] is False and c4['persistent_history_kv_cache'] is False,c4
value={'stage':'state8_finalfit_milestone_eval_v1','status':'complete',
       'variant':__import__('os').environ['MILESTONE_VARIANT'],'round':int(__import__('os').environ['MILESTONE_ROUND']),
       'source_checkpoint_sha256':source_artifacts,'evaluated_artifact_sha256':evaluated_artifacts,
       'formal_rows':formal['sampled_rows'],
       'c4_clean_n':c4['summary']['clean_n'],'c4_D_specific':c4['summary']['D_specific'],
       'c4_scientific_gate_pass':c4['summary']['scientific_gate_pass']}
out.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
PY
cp "$CHECKPOINT/checkpoint_completion.json" "$AMLT_OUTPUT_DIR/metrics/checkpoint_completion.json"
cp "$FINALIZED/completion.json" "$AMLT_OUTPUT_DIR/metrics/finalizer_completion.json"
cp /tmp/state8_finalfit_milestone_eval/audit.json "$AMLT_OUTPUT_DIR/metrics/formal_manifest_audit.json"
