#!/usr/bin/env bash
set -euo pipefail

export VENV_ROOT="${VENV_ROOT:-/tmp/studentmodeling-state8-repair-venv}"
export PATH="$VENV_ROOT/bin:$PATH"
python(){ "$VENV_ROOT/bin/python" "$@"; }
export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/hf_cache
export HF_HUB_CACHE=/tmp/hf_cache/hub
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error
export LOCAL_MODEL=/tmp/Qwen3-32B
export TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:-scripts/train_state8_stateonly_teacher_repair.py}"
export RUN_TAG="${RUN_TAG:-v1}"
export TRAIN_SEED="${TRAIN_SEED:-44}"
export REQUESTED_WORLD_SIZE="${REQUESTED_WORLD_SIZE:-4}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export STATE8_REPAIR_SAVE_STEPS="${STATE8_REPAIR_SAVE_STEPS:-1}"
export STATE8_REPAIR_RECOVERY_SAVE_STEPS="${STATE8_REPAIR_RECOVERY_SAVE_STEPS:-$STATE8_REPAIR_SAVE_STEPS}"
[[ "$STATE8_REPAIR_SAVE_STEPS" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ "$STATE8_REPAIR_RECOVERY_SAVE_STEPS" =~ ^[1-9][0-9]*$ ]] || exit 2
export STATE8_CAUSAL_DIAGNOSIS_ONLY="${STATE8_CAUSAL_DIAGNOSIS_ONLY:-0}"
export STATE8_CAUSAL_REPEAT_FRACTION="${STATE8_CAUSAL_REPEAT_FRACTION:-0.50}"
export STATE8_CAUSAL_WEIGHT="${STATE8_CAUSAL_WEIGHT:-2.0}"
export STATE8_CAUSAL_MARGIN="${STATE8_CAUSAL_MARGIN:-0.5}"
export STATE8_LAYERWISE_TRANSITION_ONLY="${STATE8_LAYERWISE_TRANSITION_ONLY:-0}"
export STATE8_UPDATER_ONLY="${STATE8_UPDATER_ONLY:-0}"
export STATE8_FORMAL_SCOPE="${STATE8_FORMAL_SCOPE:-t3}"
case "$STATE8_FORMAL_SCOPE" in t3|unified) ;; *) exit 2 ;; esac
: "${REPAIR_ARM:?REPAIR_ARM is required}"
case "$REPAIR_ARM" in
  logits)
    export STATE8_REPAIR_TRAJECTORY_WEIGHT=0.0 ;;
  trajectory01)
    export STATE8_REPAIR_TRAJECTORY_WEIGHT=0.1 ;;
  taskanchor)
    export STATE8_REPAIR_TRAJECTORY_WEIGHT=1.0 ;;
  *) echo "Unknown REPAIR_ARM=$REPAIR_ARM" >&2; exit 2 ;;
esac
export STATE8_REPAIR_LABEL_WEIGHT="${STATE8_REPAIR_LABEL_WEIGHT:-4.0}"

export RAW_FULL=/mnt/amulet/studentmodeling/full_trainplusvalid_no_t4_strict_t6/train.jsonl
export RAW_SPLIT=/tmp/state8_repair_raw_split
export RECURRENT_SPLIT=/tmp/state8_repair_recurrent_split
export FULL_JSONL="$RECURRENT_SPLIT/final_fit_trainvalid.jsonl"
export SUBSET=/tmp/state8_stateonly_repair_train1000.jsonl
export EEDI_PAIR_SUBSET=/tmp/state8_stateonly_repair_train1000_eedi_pairs.jsonl
export EEDI_PAIR_RECEIPT=/tmp/state8_stateonly_repair_train1000_eedi_pairs.receipt.json
export MANIFEST="${STATE8_REPAIR_MANIFEST:-manifests/state8_stateonly_repair_seed44_v1.json}"
export MANIFEST_SHA256="${STATE8_REPAIR_MANIFEST_SHA256:-5aee46644236170613142c688ec0c05d14e5b965656167c3e1d059ef493d4cf3}"
export RAW_EVAL=/mnt/amulet/studentmodeling/formal_evaluation_v1/test.jsonl
export RECURRENT_EVAL=/tmp/state8_repair_formal_u4/test.jsonl
export C4_DATA=/mnt/amulet/studentmodeling/c4_fullopen_20260724/malrule_c1_clean
export CAUSAL_BASE=/tmp/state8_repair_c4_base
export CAUSAL_SAME_RULE=/tmp/state8_repair_c4_same_rule
export CAUSAL_SUBSET=/tmp/state8_repair_c4_subset250
export CAUSAL_DATA=/tmp/state8_repair_c4_diagnosis
export CAUSAL_TRAIN_JSONL="$CAUSAL_DATA/train.jsonl"
export CAUSAL_VALID_JSONL="$CAUSAL_DATA/valid.jsonl"
export PAIRS=manifests/c4_stage_a_gate_pairs_seed42_n150_v1.json
export STUDENT_ROOT="${STUDENT_ROOT:-/mnt/amulet/studentmodeling/diagnostics/state8_qualified_teacher_curriculum_v1_immutable}"
export STUDENT_MODEL_DIR="${STUDENT_MODEL_DIR:-$STUDENT_ROOT/model}"
export STUDENT_STATE_PATH="${STUDENT_STATE_PATH:-$STUDENT_MODEL_DIR/recurrent_state.pt}"
if [[ -z "${STUDENT_RECEIPT:-}" ]]; then
  if [[ -f "$STUDENT_ROOT/metrics/immutable_receipt.json" ]]; then
    export STUDENT_RECEIPT="$STUDENT_ROOT/metrics/immutable_receipt.json"
  elif [[ -f "$STUDENT_ROOT/metrics/publish_receipt.json" ]]; then
    export STUDENT_RECEIPT="$STUDENT_ROOT/metrics/publish_receipt.json"
  else
    export STUDENT_RECEIPT="$STUDENT_ROOT/completion.json"
  fi
fi
export EXPECTED_STUDENT_ADAPTER_SHA256="${EXPECTED_STUDENT_ADAPTER_SHA256:-}"
export EXPECTED_STUDENT_STATE_SHA256="${EXPECTED_STUDENT_STATE_SHA256:-}"
export TEACHER_ROOT=/mnt/amulet/studentmodeling/diagnostics/state8_full_history_natural_teacher_v1
export RUN_DIR="/mnt/amulet/studentmodeling/checkpoints/state8_stateonly_repair_seed${TRAIN_SEED}_${REPAIR_ARM}_${RUN_TAG}"
export STABLE_ROOT="/mnt/amulet/studentmodeling/diagnostics/state8_stateonly_repair_seed${TRAIN_SEED}_${REPAIR_ARM}_${RUN_TAG}"
export FORMAL_OUT="$AMLT_OUTPUT_DIR/eval/formal_t3_300"
export DIAGNOSIS_OUT="$AMLT_OUTPUT_DIR/eval/diagnosis_c4_50"

mkdir -p "$HF_HUB_CACHE" "$RAW_SPLIT" "$RECURRENT_SPLIT" \
  "$(dirname "$RECURRENT_EVAL")" "$CAUSAL_BASE" "$CAUSAL_SAME_RULE" \
  "$CAUSAL_SUBSET" "$CAUSAL_DATA" "$RUN_DIR" "$STABLE_ROOT" \
  "$AMLT_OUTPUT_DIR/model" "$AMLT_OUTPUT_DIR/metrics" "$FORMAL_OUT" "$DIAGNOSIS_OUT"
test "$(df --output=avail /tmp | tail -1)" -ge 125829120
for path in "$RAW_FULL" "$RAW_EVAL" "$C4_DATA/train.jsonl" \
  "$C4_DATA/val.jsonl" "$C4_DATA/test_same_template.jsonl" \
  "$STUDENT_MODEL_DIR/adapter_model.safetensors" \
  "$STUDENT_STATE_PATH" "$STUDENT_RECEIPT" \
  "$TEACHER_ROOT/model/adapter_model.safetensors" \
  "$TEACHER_ROOT/metrics/teacher_gate.json" "$MANIFEST" "$PAIRS"; do
  test -f "$path"
done
test "$(sha256sum "$RAW_FULL" | cut -d' ' -f1)" = aeabd3b4dfef3c4d92ac68583879c91448ef90f58304cedf32e470b586ca4542
test "$(sha256sum "$RAW_EVAL" | cut -d' ' -f1)" = bff7726953ae048bedf6dacdbb7500a54cb8de305e406f2497ef1f11c04167f8
test "$(sha256sum "$MANIFEST" | cut -d' ' -f1)" = "$MANIFEST_SHA256"
test "$(sha256sum "$PAIRS" | cut -d' ' -f1)" = edcdb934a65252557f6c569eea9ef7a85e5445549deac9fb4829c3f579e1a9b8
case "$TRAIN_ENTRYPOINT" in
  scripts/train_state8_stateonly_teacher_repair.py)
    test "$(sha256sum "$TRAIN_ENTRYPOINT" | cut -d' ' -f1)" = 3f8e05a6b35515c52b53482f894afbaedca6de4ce163a53b0737aa8971637452 ;;
  scripts/train_state8_semantic_causal_repair.py)
    test "$(sha256sum "$TRAIN_ENTRYPOINT" | cut -d' ' -f1)" = 14db1f4c430c61537a33c0b4617f1734a0bd51501249dfa8f61bc8c0ec01759a ;;
  scripts/train_state8_taxonomy_causal_repair.py)
    test "$(sha256sum "$TRAIN_ENTRYPOINT" | cut -d' ' -f1)" = 26e62969a03aab89047dd42d5bbe6d144eae9bb77d0fa0b962b0da89b9b75347
    test "$(sha256sum scripts/state8_taxonomy_contract.py | cut -d' ' -f1)" = 1ca32112e76c795502f3cb1edbedcc7220f093c133fae8d20ff5cab789e512ed ;;
  scripts/train_state8_task_anchored_causal_repair.py)
    test "$(sha256sum "$TRAIN_ENTRYPOINT" | cut -d' ' -f1)" = b622a62fffef45a19b3b02b771cee4412d5f9d17f1a3fe99c6ba046e6111e66b
    test "$(sha256sum scripts/state8_taxonomy_contract.py | cut -d' ' -f1)" = 1ca32112e76c795502f3cb1edbedcc7220f093c133fae8d20ff5cab789e512ed ;;
  *) echo "Unapproved TRAIN_ENTRYPOINT=$TRAIN_ENTRYPOINT" >&2; exit 2 ;;
esac
test "$(sha256sum scripts/train_recurrent_student_m4.py | cut -d' ' -f1)" = ae502968872ff6597e73bcc12cb3ff9a5bbb0fe9d7a763383f6e76bc392de42c
test "$(sha256sum scripts/evaluate_recurrent_formal_protocols.py | cut -d' ' -f1)" = 262370592e018641b99d6282858d8990b38314c05bb6e3bbc645823834cf73ed
test "$(sha256sum scripts/evaluate_recurrent_diagnosis_c4.py | cut -d' ' -f1)" = e7421a850c2327d8df1436073742e71f796bec41c87f68403b7b00b60fac4664
if [[ "${STATE8_EEDI_GEOMETRY_WEIGHT:-0}" != 0 ]]; then
  test "$(sha256sum scripts/augment_state8_eedi_semantic_pairs.py | cut -d' ' -f1)" = c9006ac322d075ea1fa35ee9ace3a6561f799bdfea767786a6e459825b0ad46c
fi

export STUDENTMODELING_EXTERNAL_STATE_TEACHER_ADAPTER="$TEACHER_ROOT/model"
export STUDENTMODELING_EXTERNAL_STATE_TEACHER_SHA256="$(python - \
  "$STUDENT_MODEL_DIR" "$STUDENT_STATE_PATH" "$STUDENT_RECEIPT" \
  "$TEACHER_ROOT" "$EXPECTED_STUDENT_ADAPTER_SHA256" \
  "$EXPECTED_STUDENT_STATE_SHA256" <<'PY'
import hashlib,json,pathlib,sys,torch
student=pathlib.Path(sys.argv[1]); state_path=pathlib.Path(sys.argv[2])
receipt_path=pathlib.Path(sys.argv[3]); teacher=pathlib.Path(sys.argv[4])
expected_adapter=sys.argv[5]; expected_state=sys.argv[6]
def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
 return h.hexdigest()
si=json.load(open(receipt_path))
tg=json.load(open(teacher/'metrics/teacher_gate.json'))
observed={'adapter_model':digest(student/'adapter_model.safetensors'),
          'recurrent_state':digest(state_path)}
if si.get('stage') in {
 'state8_qualified_teacher_curriculum_immutable_v1',
 'state8_qualified_teacher_curriculum_highkd_immutable_v1_taxonomy_semantic_v1',
 'state8_qualified_teacher_balanced_curriculum_immutable_v1_taxonomy_semantic_v1',
}:
 assert si['status']=='complete' and si['state_shape']==[64,8,5120],si
 expected=si['sha256']
elif si.get('stage')=='state8_stateonly_repair_pilot_publish_v1':
 assert si['status']=='complete',si
 expected=si['artifact_sha256']
elif si.get('status')=='complete' and si.get('sha256'):
 expected={key:si['sha256'][key] for key in observed}
 assert si['micro_steps']>=1000 and si['real_examples_seen_once'] is True,si
else: raise AssertionError(si)
assert expected==observed,(si,observed)
if expected_adapter: assert observed['adapter_model']==expected_adapter,observed
if expected_state: assert observed['recurrent_state']==expected_state,observed
state=torch.load(state_path,map_location='cpu',weights_only=False)
config=state['config']
assert config['state_architecture']=='layerwise' and config['state_token_count']==8,config
assert config['final_inference_route']=='strict_state_only',config
assert config['output_schema']=='label_diagnosis_evidence_v1',config
if 'state_shape' in si: assert si['state_shape']==[64,8,5120],si
if 'raw_history_visible_to_decoder' in si:
 assert si['raw_history_visible_to_decoder'] is False,si
assert tg['stage']=='state8_independent_full_history_teacher_gate_v1' and tg['pass'] is True,tg
assert tg['support']==1494 and tg['exact_match_is_authoritative'] is False,tg
print(digest(teacher/'model/adapter_model.safetensors'))
PY
)"
test "${#STUDENTMODELING_EXTERNAL_STATE_TEACHER_SHA256}" = 64

python -m pytest -q tests/test_train_state8_stateonly_teacher_repair.py \
  tests/test_state8_external_teacher_curriculum.py \
  tests/test_augment_recurrent_causal_diagnosis.py \
  tests/test_evaluate_recurrent_diagnosis_c4.py \
  tests/test_train_recurrent_student_m4.py
if [[ "$TRAIN_ENTRYPOINT" == scripts/train_state8_semantic_causal_repair.py \
      || "$TRAIN_ENTRYPOINT" == scripts/train_state8_taxonomy_causal_repair.py \
      || "$TRAIN_ENTRYPOINT" == scripts/train_state8_task_anchored_causal_repair.py ]]; then
  python -m pytest -q tests/test_train_state8_semantic_causal_repair.py \
    tests/test_augment_recurrent_causal_same_rule.py
fi
if [[ "$TRAIN_ENTRYPOINT" == scripts/train_state8_taxonomy_causal_repair.py \
      || "$TRAIN_ENTRYPOINT" == scripts/train_state8_task_anchored_causal_repair.py ]]; then
  python -m pytest -q tests/test_state8_taxonomy_contract.py
fi
if [[ "$TRAIN_ENTRYPOINT" == scripts/train_state8_task_anchored_causal_repair.py ]]; then
  python -m pytest -q tests/test_train_state8_task_anchored_causal_repair.py
fi

exec 8>/tmp/state8-stateonly-repair-data.lock; flock -x 8
DATA_COMPLETE="/tmp/state8-stateonly-repair-data-${MANIFEST_SHA256}.complete"
if [[ ! -f "$DATA_COMPLETE" ]]; then
  rm -rf "$RAW_SPLIT" "$RECURRENT_SPLIT" "$(dirname "$RECURRENT_EVAL")" \
    "$CAUSAL_BASE" "$CAUSAL_SAME_RULE" "$CAUSAL_SUBSET" "$CAUSAL_DATA"
  mkdir -p "$RAW_SPLIT" "$RECURRENT_SPLIT" "$(dirname "$RECURRENT_EVAL")" \
    "$CAUSAL_BASE" "$CAUSAL_SAME_RULE" "$CAUSAL_SUBSET" "$CAUSAL_DATA"
  python scripts/split_full_mix_by_original_split.py --input-jsonl "$RAW_FULL" --output-dir "$RAW_SPLIT"
  python scripts/build_unified_recurrent_manifest.py \
    --train-jsonl "$RAW_SPLIT/train.jsonl" --valid-jsonl "$RAW_SPLIT/valid.jsonl" \
    --output-dir "$RECURRENT_SPLIT" --max-updates 4 --seed 20260730
  python scripts/merge_recurrent_full_corpus.py \
    --train-jsonl "$RECURRENT_SPLIT/train.jsonl" --valid-jsonl "$RECURRENT_SPLIT/valid.jsonl" \
    --output-jsonl "$FULL_JSONL" --manifest "$RECURRENT_SPLIT/final_fit_manifest.json"
  python scripts/build_recurrent_evaluation_manifest.py \
    --input-jsonl "$RAW_EVAL" --output-jsonl "$RECURRENT_EVAL" \
    --audit-json /tmp/state8_repair_formal_u4/audit.json --max-updates 4 --seed 20260730
  python scripts/materialize_recurrent_index_manifest.py \
    --source-jsonl "$FULL_JSONL" --manifest "$MANIFEST" --output-jsonl "$SUBSET" \
    --formal-eval-jsonl "$RECURRENT_EVAL"
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
    --output-jsonl "$CAUSAL_SUBSET/train.jsonl" --twins 125 --seed 44
  cp "$CAUSAL_SAME_RULE/valid.jsonl" "$CAUSAL_SUBSET/valid.jsonl"
  python scripts/augment_recurrent_causal_diagnosis.py \
    --train-jsonl "$CAUSAL_SUBSET/train.jsonl" \
    --valid-jsonl "$CAUSAL_SUBSET/valid.jsonl" --output-dir "$CAUSAL_DATA"
  touch "$DATA_COMPLETE"
fi
flock -u 8
test "$(sha256sum "$FULL_JSONL" | cut -d' ' -f1)" = 9961f746cb17e6c314e182d3453b6157022fb48b8de44d2e68bc8e87a1131ccd
test "$(sha256sum "$RECURRENT_EVAL" | cut -d' ' -f1)" = a368cefb0c49cb6a305ad1a892808361fba2d44d555a0a28ea64e56babe32525
python - "$SUBSET.receipt.json" "$CAUSAL_DATA/augmentation_receipt.json" <<'PY'
import json,sys
s=json.load(open(sys.argv[1])); c=json.load(open(sys.argv[2]))
assert s['rows']==1000 and s['formal_example_id_overlap']==0,s
assert c['train']['input_pairs']==250 and c['train']['output_pairs']==500,c
assert c['train']['canonical_label_auxiliary_supervision'] is True,c
assert c['train']['raw_history_visible_to_decoder'] is False,c
PY

if [[ "$STATE8_CAUSAL_DIAGNOSIS_ONLY" == 1 ]]; then
  export CAUSAL_DIAGNOSIS_ONLY=/tmp/state8_repair_c4_diagnosis_only
  exec 4>/tmp/state8-repair-c4-diagnosis-only.lock; flock -x 4
  if [[ ! -f "$CAUSAL_DIAGNOSIS_ONLY/receipt.json" ]]; then
    mkdir -p "$CAUSAL_DIAGNOSIS_ONLY"
    python scripts/filter_state8_causal_diagnosis.py \
      --train-jsonl "$CAUSAL_DATA/train.jsonl" \
      --valid-jsonl "$CAUSAL_DATA/valid.jsonl" \
      --output-dir "$CAUSAL_DIAGNOSIS_ONLY"
  fi
  flock -u 4
  test -f "$CAUSAL_DIAGNOSIS_ONLY/train.jsonl"
  test -f "$CAUSAL_DIAGNOSIS_ONLY/valid.jsonl"
  test -f "$CAUSAL_DIAGNOSIS_ONLY/receipt.json"
  export CAUSAL_TRAIN_JSONL="$CAUSAL_DIAGNOSIS_ONLY/train.jsonl"
  export CAUSAL_VALID_JSONL="$CAUSAL_DIAGNOSIS_ONLY/valid.jsonl"
elif [[ "$STATE8_CAUSAL_DIAGNOSIS_ONLY" != 0 ]]; then
  echo "STATE8_CAUSAL_DIAGNOSIS_ONLY must be 0 or 1" >&2
  exit 2
fi

export TRAIN_JSONL="$SUBSET"
if [[ "${STATE8_EEDI_GEOMETRY_WEIGHT:-0}" != 0 ]]; then
  exec 5>/tmp/state8-eedi-semantic-pair.lock; flock -x 5
  if [[ ! -f "$EEDI_PAIR_SUBSET" || ! -f "$EEDI_PAIR_RECEIPT" ]]; then
    rm -f "$EEDI_PAIR_SUBSET" "$EEDI_PAIR_RECEIPT"
    python scripts/augment_state8_eedi_semantic_pairs.py \
      --train-jsonl "$SUBSET" --corpus-jsonl "$FULL_JSONL" \
      --output-jsonl "$EEDI_PAIR_SUBSET" --receipt "$EEDI_PAIR_RECEIPT" \
      --seed 20260806
  fi
  flock -u 5
  python - "$EEDI_PAIR_RECEIPT" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['stage']=='state8_eedi_semantic_pair_augmentation_v1',x
assert x['status']=='passed' and x['rows']==1000,x
assert x['eligible_eedi_t3']>0 and x['paired_eedi_t3']>0,x
assert x['pair_coverage']>=0.70,x
assert x['canonical_labels_visible_to_decoder'] is False,x
assert x['pair_supervision_scope']=='state_geometry_train_only',x
PY
  export TRAIN_JSONL="$EEDI_PAIR_SUBSET"
fi

exec 7>/tmp/state8-stateonly-repair-model.lock; flock -x 7
if [[ ! -f /tmp/state8-stateonly-repair-model.complete ]]; then
  rm -rf "$LOCAL_MODEL"; mkdir -p "$LOCAL_MODEL"
  python scripts/copy_model_parallel.py \
    --source /mnt/amulet/studentmodeling/models/Qwen3-32B \
    --destination "$LOCAL_MODEL" --workers 16
  touch /tmp/state8-stateonly-repair-model.complete
fi
flock -u 7
test -f "$LOCAL_MODEL/model.safetensors.index.json"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

resume_checkpoint=""
TOPOLOGY_ARGS=()
if [[ "$REQUESTED_WORLD_SIZE" == 4 ]]; then
  export CUDA_VISIBLE_DEVICES="${OMPI_COMM_WORLD_LOCAL_RANK:?MPI local rank is required}"
  export RANK="${OMPI_COMM_WORLD_RANK:?MPI rank is required}"
  export WORLD_SIZE="${OMPI_COMM_WORLD_SIZE:?MPI world size is required}"
  export LOCAL_RANK=0
  test "$WORLD_SIZE" = 4
  TOPOLOGY_ARGS=(--distributed-mpi --distributed-world-size 4)
  selected="$(python scripts/select_latest_valid_distributed_checkpoint.py \
    --run-dir "$RUN_DIR" --expected-world-size 4 --max-round 250 \
    --expected-planned-optimizer-steps 250 \
    --save-steps "$STATE8_REPAIR_SAVE_STEPS" \
    --recovery-save-steps "$STATE8_REPAIR_RECOVERY_SAVE_STEPS" \
    --probe-expected-cadence)"
elif [[ "$REQUESTED_WORLD_SIZE" == 1 ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  export RANK=0 WORLD_SIZE=1 LOCAL_RANK=0
  selected=""
else
  echo "Unsupported REQUESTED_WORLD_SIZE=$REQUESTED_WORLD_SIZE" >&2
  exit 2
fi
if [[ "$REQUESTED_WORLD_SIZE" == 4 ]]; then
  test "$GRADIENT_ACCUMULATION_STEPS" = 1
else
  test "$GRADIENT_ACCUMULATION_STEPS" = 4
fi
if [[ -n "$selected" ]]; then
  resume_key="$(python - "$selected" <<'PY'
import hashlib,os,sys
p=os.path.realpath(sys.argv[1]); s=os.stat(p)
print(hashlib.sha256(f'{p}:{s.st_size}:{s.st_mtime_ns}'.encode()).hexdigest())
PY
)"
  local_resume="/tmp/state8-stateonly-repair-${REPAIR_ARM}-resume/$resume_key"
  mkdir -p "$local_resume"
  exec 6>"$local_resume/stage.lock"; flock -x 6
  python scripts/stage_distributed_resume_bundle.py --source-state "$selected" --destination "$local_resume"
  flock -u 6
  resume_checkpoint="$local_resume/recurrent_state.pt"
fi
if [[ -n "$resume_checkpoint" ]]; then
  INIT_ARGS=(--init-recurrent-state "$STUDENT_STATE_PATH" \
             --resume-from-checkpoint "$resume_checkpoint")
else
  INIT_ARGS=(--init-recurrent-state "$STUDENT_STATE_PATH")
fi

TRAIN_MODE_ARGS=()
if [[ "$STATE8_LAYERWISE_TRANSITION_ONLY" == 1 ]]; then
  TRAIN_MODE_ARGS+=(--layerwise-transition-only)
elif [[ "$STATE8_LAYERWISE_TRANSITION_ONLY" != 0 ]]; then
  echo "STATE8_LAYERWISE_TRANSITION_ONLY must be 0 or 1" >&2
  exit 2
fi

python -u "$TRAIN_ENTRYPOINT" \
  --model "$LOCAL_MODEL" --init-adapter "$STUDENT_MODEL_DIR" "${INIT_ARGS[@]}" \
  --state-architecture layerwise --layerwise-operation-checkpointing \
  --layerwise-transition-rank 256 --state-token-count 8 \
  --train-jsonl "$TRAIN_JSONL" --valid-jsonl "$RECURRENT_SPLIT/valid.jsonl" \
  --causal-train-jsonl "$CAUSAL_TRAIN_JSONL" \
  --causal-valid-jsonl "$CAUSAL_VALID_JSONL" \
  --output-dir "$RUN_DIR" --state-token-alias '<|vision_pad|>' \
  --history-dropout-curriculum --dual-path-curriculum \
  --online-kd-weight 1.0 --online-kd-temperature 1.0 --online-kd-topk 256 \
  --max-length 1536 --direct-offload-min-tokens 1 \
  --learning-rate 5e-7 --lr-scheduler cosine --warmup-ratio 0.03 \
  --generation-weight 1.0 --behavior-kd-weight 0.0 \
  --consistency-weight "$STATE8_REPAIR_TRAJECTORY_WEIGHT" --direct-replay-weight 0.0 \
  --causal-weight "$STATE8_CAUSAL_WEIGHT" --causal-margin "$STATE8_CAUSAL_MARGIN" \
  --causal-preference-reduction sum --causal-objective-mode specificity_triplet \
  --causal-repeat-fraction "$STATE8_CAUSAL_REPEAT_FRACTION" \
  --normalize-causal-exposure --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
  --eval-limit 300 --causal-eval-limit 165 --save-steps "$STATE8_REPAIR_SAVE_STEPS" \
  --checkpoint-retention 2 "${TOPOLOGY_ARGS[@]}" "${TRAIN_MODE_ARGS[@]}" \
  --seed "$TRAIN_SEED"

if [[ "${OMPI_COMM_WORLD_RANK:-0}" != 0 ]]; then exit 0; fi
test -f "$RUN_DIR/completion.json"
test -f "$RUN_DIR/final_adapter/adapter_model.safetensors"
test -f "$RUN_DIR/recurrent_state.pt"
test -f "$RUN_DIR/state8_stateonly_teacher_repair.json"
if [[ "$TRAIN_ENTRYPOINT" == scripts/train_state8_semantic_causal_repair.py \
      || "$TRAIN_ENTRYPOINT" == scripts/train_state8_taxonomy_causal_repair.py \
      || "$TRAIN_ENTRYPOINT" == scripts/train_state8_task_anchored_causal_repair.py ]]; then
  test -f "$RUN_DIR/semantic_causal_repair.json"
fi
if [[ "$TRAIN_ENTRYPOINT" == scripts/train_state8_taxonomy_causal_repair.py \
      || "$TRAIN_ENTRYPOINT" == scripts/train_state8_task_anchored_causal_repair.py ]]; then
  test -f "$RUN_DIR/taxonomy_causal_repair.json"
fi
if [[ "$TRAIN_ENTRYPOINT" == scripts/train_state8_task_anchored_causal_repair.py ]]; then
  test -f "$RUN_DIR/task_anchored_causal_repair.json"
fi
python - "$RUN_DIR" "$REPAIR_ARM" "$TRAIN_ENTRYPOINT" <<'PY'
import collections,json,os,pathlib,sys
root=pathlib.Path(sys.argv[1]); arm=sys.argv[2]; entrypoint=sys.argv[3]
c=json.load(open(root/'completion.json')); cfg=json.load(open(root/'run_config.json'))
h=json.load(open(root/'trainer_history.json')); r=json.load(open(root/'state8_stateonly_teacher_repair.json'))
assert c['status']=='complete' and c['micro_steps']==1000,c
assert c['optimizer_steps']==250,c
expected_pairs=250 if os.environ['STATE8_CAUSAL_DIAGNOSIS_ONLY']=='1' else 500
expected_exposures=max(expected_pairs,round(1000*float(os.environ['STATE8_CAUSAL_REPEAT_FRACTION'])))
assert c['all_causal_pairs_covered'] is True,c
assert c['unique_causal_pairs_seen']==expected_pairs,c
assert c['causal_pairs_seen']==expected_exposures,c
assert cfg['state_token_count']==8 and cfg['final_inference_route']=='strict_state_only',cfg
assert int(cfg['distributed_world_size'])==int(os.environ['REQUESTED_WORLD_SIZE']),cfg
assert collections.Counter(float(x['history_dropout']) for x in h)=={1.0:1000},h[:2]
assert all(bool(x['history_dropped']) for x in h),h[:2]
assert r['online_kd_phase_weight']==1.0,r
assert r['label_auxiliary_weight']==float(os.environ['STATE8_REPAIR_LABEL_WEIGHT']),r
expected_trajectory={'logits':0.0,'trajectory01':0.1,'taskanchor':1.0}[arm]
assert float(r['trajectory_weight'])==expected_trajectory,r
if entrypoint.endswith(('train_state8_semantic_causal_repair.py',
                        'train_state8_taxonomy_causal_repair.py',
                        'train_state8_task_anchored_causal_repair.py')):
 s=json.load(open(root/'semantic_causal_repair.json'))
 expected_random=('held_out_evaluation_only' if os.environ['STATE8_UPDATER_ONLY']=='1'
                  else 'insufficient_evidence')
 assert s['random_target']==expected_random and s['same_rule_geometry'] is True,s
 assert s['geometry_counts']['applied']>0,s
 assert s['geometry_counts']['applied']==s['geometry_counts']['eligible'],s
 assert s['geometry_counts']['audit_source']=='global_trainer_history',s
 assert s['final_inference_route']=='strict_state_only',s
 if float(s['eedi_geometry_weight'])>0:
  assert s['eedi_geometry_counts']['eligible']>=100,s
  assert s['eedi_geometry_counts']['applied']==s['eedi_geometry_counts']['eligible'],s
  assert s['eedi_pair_supervision_visible_at_inference'] is False,s
PY

FORMAL_EVALUATOR=scripts/evaluate_recurrent_formal_protocols.py
DIAGNOSIS_EVALUATOR=scripts/evaluate_recurrent_diagnosis_c4.py
if [[ "$TRAIN_ENTRYPOINT" == scripts/train_state8_taxonomy_causal_repair.py \
      || "$TRAIN_ENTRYPOINT" == scripts/train_state8_task_anchored_causal_repair.py ]]; then
  FORMAL_EVALUATOR=scripts/evaluate_recurrent_formal_protocols_taxonomy.py
  DIAGNOSIS_EVALUATOR=scripts/evaluate_recurrent_diagnosis_c4_taxonomy.py
  test "$(sha256sum "$FORMAL_EVALUATOR" | cut -d' ' -f1)" = 648baa998672f30652ad3e68f114c2648c5bf839dcddd5b1e0acae47400e5b22
  test "$(sha256sum "$DIAGNOSIS_EVALUATOR" | cut -d' ' -f1)" = c3d34960d0f6caff3265b772686bd84222637f628d5fce000215d076c5b2a355
fi

FORMAL_SCOPE_ARGS=(--tasks diagnose_misconception --sources eedi_mining malrule \
  --max-samples-per-group 100)
if [[ "$STATE8_FORMAL_SCOPE" == unified ]]; then
  FORMAL_SCOPE_ARGS=(--max-samples-per-group 25)
fi
python -u "$FORMAL_EVALUATOR" \
  --model "$LOCAL_MODEL" --adapter "$RUN_DIR/final_adapter" \
  --recurrent-state "$RUN_DIR/recurrent_state.pt" --state-architecture layerwise \
  --eval-jsonl "$RECURRENT_EVAL" --output-dir "$FORMAL_OUT" \
  --state-token-alias '<|vision_pad|>' "${FORMAL_SCOPE_ARGS[@]}" \
  --max-length 1536 --max-new-tokens 160 --state-batch-size 2 \
  --candidate-row-batch-size 2 --seed 44 --save-unredacted-predictions --t3-natural-schema
readarray -t ARTIFACT_SHA256 < <(python - "$RUN_DIR/completion.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); print(x['sha256']['adapter_model']); print(x['sha256']['recurrent_state'])
PY
)
python -u "$DIAGNOSIS_EVALUATOR" \
  --model "$LOCAL_MODEL" --adapter "$RUN_DIR/final_adapter" \
  --recurrent-state "$RUN_DIR/recurrent_state.pt" --state-architecture layerwise \
  --jsonl "$C4_DATA/test_same_template.jsonl" --pair-manifest "$PAIRS" \
  --output-dir "$DIAGNOSIS_OUT" --state-token-alias '<|vision_pad|>' \
  --max-length 1536 --max-new-tokens 160 --num-pairs 50 --seed 42 \
  --expected-adapter-sha256 "${ARTIFACT_SHA256[0]}" \
  --expected-recurrent-state-sha256 "${ARTIFACT_SHA256[1]}"

rm -rf "$STABLE_ROOT/model" "$STABLE_ROOT/metrics" "$STABLE_ROOT/eval"
mkdir -p "$STABLE_ROOT/model" "$STABLE_ROOT/metrics" "$STABLE_ROOT/eval"
cp -r "$RUN_DIR/final_adapter/." "$AMLT_OUTPUT_DIR/model/"
cp "$RUN_DIR/recurrent_state.pt" "$AMLT_OUTPUT_DIR/model/recurrent_state.pt"
cp -r "$AMLT_OUTPUT_DIR/model/." "$STABLE_ROOT/model/"
cp "$RUN_DIR"/*.json "$AMLT_OUTPUT_DIR/metrics/"
cp "$SUBSET.receipt.json" "$AMLT_OUTPUT_DIR/metrics/training_subset_receipt.json"
if [[ -f "$EEDI_PAIR_RECEIPT" ]]; then
  cp "$EEDI_PAIR_RECEIPT" "$AMLT_OUTPUT_DIR/metrics/eedi_semantic_pair_augmentation.json"
fi
cp "$CAUSAL_DATA/augmentation_receipt.json" "$AMLT_OUTPUT_DIR/metrics/causal_augmentation.json"
if [[ "$STATE8_CAUSAL_DIAGNOSIS_ONLY" == 1 ]]; then
  cp "$CAUSAL_DIAGNOSIS_ONLY/receipt.json" \
    "$AMLT_OUTPUT_DIR/metrics/causal_diagnosis_only.json"
fi
cp -r "$AMLT_OUTPUT_DIR/metrics/." "$STABLE_ROOT/metrics/"
cp -r "$AMLT_OUTPUT_DIR/eval/." "$STABLE_ROOT/eval/"
python - "$STABLE_ROOT" "$REPAIR_ARM" "$TRAIN_ENTRYPOINT" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); arm=sys.argv[2]; entrypoint=sys.argv[3]
def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
receipt={'stage':'state8_stateonly_repair_pilot_publish_v1','status':'complete','repair_arm':arm,
 'state_shape':[64,8,5120],'raw_history_visible_to_decoder':False,
 'history_dropout':1.0,'online_teacher_kd':True,
 'train_entrypoint':entrypoint,
 'artifact_sha256':{'adapter_model':digest(root/'model/adapter_model.safetensors'),
                    'recurrent_state':digest(root/'model/recurrent_state.pt')}}
(root/'metrics/publish_receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
PY
cp "$STABLE_ROOT/metrics/publish_receipt.json" "$AMLT_OUTPUT_DIR/metrics/publish_receipt.json"
