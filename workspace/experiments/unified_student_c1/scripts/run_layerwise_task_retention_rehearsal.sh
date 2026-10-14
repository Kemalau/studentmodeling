#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "TASK_RETENTION_FAILED line=%s rc=%s stage=%s command=%s\n" "${LINENO}" "$rc" "${TASK_RETENTION_STAGE:-initialization}" "${BASH_COMMAND}" >&2; exit "$rc"' ERR
stage(){ export TASK_RETENTION_STAGE="$1"; printf 'TASK_RETENTION_STAGE=%s time=%s\n' "$1" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"; }

stage argument_validation

: "${BRANCH:?BRANCH is required}"
: "${MANIFEST:?MANIFEST is required}"
: "${MANIFEST_SHA256:?MANIFEST_SHA256 is required}"
case "$BRANCH" in eedi_dense|eedi_full|eedi_full_open_t3|equal_t3_t6|equal_t3_t6_open_t3|t3_heavy|staged_t3_t6) ;; *) exit 2 ;; esac
SEGMENT_ROWS="${SEGMENT_ROWS:-1000}"
LEARNING_RATE="${LEARNING_RATE:-5e-7}"
DIAGNOSTIC_START_GLOBAL_POSITION="${DIAGNOSTIC_START_GLOBAL_POSITION:-12000}"
EXPECTED_CAUSAL_PAIRS="${EXPECTED_CAUSAL_PAIRS:-$((SEGMENT_ROWS / 20))}"
C3_WORLD_SIZE="${C3_WORLD_SIZE:-4}"
C3_GRADIENT_ACCUMULATION_STEPS="${C3_GRADIENT_ACCUMULATION_STEPS:-1}"
DIAGNOSTIC_STOP_AFTER_ROUND="${DIAGNOSTIC_STOP_AFTER_ROUND:-$(((DIAGNOSTIC_START_GLOBAL_POSITION + SEGMENT_ROWS) / C3_WORLD_SIZE))}"
[[ "$SEGMENT_ROWS" =~ ^[1-9][0-9]*$ ]] && (( SEGMENT_ROWS % 4 == 0 )) || exit 2
[[ "$EXPECTED_CAUSAL_PAIRS" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ "$DIAGNOSTIC_START_GLOBAL_POSITION" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ "$DIAGNOSTIC_STOP_AFTER_ROUND" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ "$C3_WORLD_SIZE" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ "$C3_GRADIENT_ACCUMULATION_STEPS" =~ ^[1-9][0-9]*$ ]] || exit 2
(( C3_WORLD_SIZE * C3_GRADIENT_ACCUMULATION_STEPS == 4 )) || exit 2
(( DIAGNOSTIC_STOP_AFTER_ROUND * C3_WORLD_SIZE == DIAGNOSTIC_START_GLOBAL_POSITION + SEGMENT_ROWS )) || exit 2
LOSS_PROFILE="${LOSS_PROFILE:-standard}"
CAUSAL_VARIANT="${CAUSAL_VARIANT:-base}"
case "$CAUSAL_VARIANT" in base|same_rule) ;; *) exit 2 ;; esac
SAME_RULE_AUGMENTER_SHA256="${SAME_RULE_AUGMENTER_SHA256:-d0f5eb225e771a745874ab9a6c829b8cd4ac2bd71754ae96f8c8007498bc1ff2}"
TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:-scripts/train_recurrent_student_m4.py}"
case "$TRAIN_ENTRYPOINT" in
  scripts/train_recurrent_student_m4.py|scripts/train_recurrent_student_m4_open_t3.py|scripts/train_recurrent_student_m4_open_t3_selective_kd.py|scripts/train_recurrent_student_m4_external_state_teacher.py|scripts/train_recurrent_student_m4_gold_semantic_teacher.py) ;;
  *) exit 2 ;;
esac
case "$TRAIN_ENTRYPOINT" in
  scripts/train_recurrent_student_m4.py)
    EXPECTED_ENTRYPOINT_SHA256=ae502968872ff6597e73bcc12cb3ff9a5bbb0fe9d7a763383f6e76bc392de42c
    ;;
  scripts/train_recurrent_student_m4_open_t3.py)
    EXPECTED_ENTRYPOINT_SHA256=b5e7a5c301280df64083dadf20273902747bcf8ea4e4cdd03de3e49d4e53187a
    ;;
  scripts/train_recurrent_student_m4_open_t3_selective_kd.py)
    EXPECTED_ENTRYPOINT_SHA256=2474b77459701fa36ec564c84b7f3b6b0d158c9ab128b5147111d252662b0931
    ;;
  scripts/train_recurrent_student_m4_external_state_teacher.py)
    EXPECTED_ENTRYPOINT_SHA256=585f4c2e8e7edb44e94a8b89e86d88b07d6cbdd48802de400f0a479cad13e800
    ;;
  scripts/train_recurrent_student_m4_gold_semantic_teacher.py)
    EXPECTED_ENTRYPOINT_SHA256=62f692c96159b55f2393ee33e6c759fc5bc64363a5a42034eb94d886a79a9541
    ;;
esac
CONSISTENCY_WEIGHT="${CONSISTENCY_WEIGHT:-0.25}"
if [[ "$TRAIN_ENTRYPOINT" == scripts/train_recurrent_student_m4_external_state_teacher.py \
    || "$TRAIN_ENTRYPOINT" == scripts/train_recurrent_student_m4_gold_semantic_teacher.py ]]; then
  : "${STUDENTMODELING_EXTERNAL_STATE_TEACHER_ADAPTER:?external state teacher adapter is required}"
  : "${STUDENTMODELING_EXTERNAL_STATE_TEACHER_SHA256:?external state teacher SHA256 is required}"
  [[ "$LOSS_PROFILE" == state_path ]] || exit 2
  [[ "$STUDENTMODELING_EXTERNAL_STATE_TEACHER_SHA256" =~ ^[0-9a-f]{64}$ ]] || exit 2
  test -f "$STUDENTMODELING_EXTERNAL_STATE_TEACHER_ADAPTER/adapter_model.safetensors"
  test "$(sha256sum "$STUDENTMODELING_EXTERNAL_STATE_TEACHER_ADAPTER/adapter_model.safetensors" | cut -d' ' -f1)" = \
    "$STUDENTMODELING_EXTERNAL_STATE_TEACHER_SHA256"
fi
case "$LOSS_PROFILE" in
  standard)
    BEHAVIOR_KD_WEIGHT=0.5
    DIRECT_REPLAY_WEIGHT=0.25
    ;;
  state_kd)
    # Preserve teacher supervision on the closed state -> answer route while
    # removing direct replay, which is the only loss term that can bypass the
    # recurrent state.  This separates route leakage from KD regularization.
    BEHAVIOR_KD_WEIGHT=0.5
    DIRECT_REPLAY_WEIGHT=0.0
    ;;
  state_path)
    # Keep state -> answer supervision, trajectory consistency, and the
    # specificity-triplet intervention loss.  Remove direct replay (a route
    # around recurrence) and KD regularization (which still acts on state-path
    # logits) as one bounded composite objective-conflict diagnostic.
    BEHAVIOR_KD_WEIGHT=0.0
    DIRECT_REPLAY_WEIGHT=0.0
    ;;
  *) exit 2 ;;
esac
if [[ "$BEHAVIOR_KD_WEIGHT" == "0.0" ]]; then
  # Keep manifest/index provenance checks, but skip multi-GB shard hashing for
  # an objective whose KD coefficient is exactly zero. TeacherCache rejects
  # any unexpected shard load while this declaration is active.
  export STUDENTMODELING_TEACHER_CACHE_UNUSED=1
else
  unset STUDENTMODELING_TEACHER_CACHE_UNUSED
fi

export VENV_ROOT="${VENV_ROOT:-/tmp/studentmodeling-layerwise-task-retention-venv}"
export PATH="$VENV_ROOT/bin:$PATH"
# AMLT may resolve/cache the image Python before entering the MPI command.  A
# PATH-only override can therefore run rank-local tests with /opt/conda even
# though dependencies were installed into the job venv.  Route every Python
# invocation through the selected interpreter explicitly and wait for the
# environment setup receipt before any rank imports training code.
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
export RAW_SPLIT=/tmp/layerwise_task_retention_full_trainvalid_v1
export RECURRENT_SPLIT=/tmp/layerwise_task_retention_recurrent_full259845_u4_v1
export FINAL_JSONL="$RECURRENT_SPLIT/final_fit_trainvalid.jsonl"
export C4_DATA=/mnt/amulet/studentmodeling/c4_fullopen_20260724/malrule_c1_clean
export CAUSAL_DATA_BASE=/tmp/layerwise_task_retention_cf4_u4_base_v1
export CAUSAL_DATA="/tmp/layerwise_task_retention_cf4_u4_${CAUSAL_VARIANT}_v1"
export CACHE_ROOT=/mnt/amulet/studentmodeling/world_model_v1/final_fit/cache_full259845_top256_seed42_v2_16way
export CHECKPOINT="${WARM_START_CHECKPOINT_DIR:-/mnt/amulet/studentmodeling/diagnostics/layerwise_unified_finalfit_seed42_20260801_snapshot/checkpoint-round-750}"
export WARM_START_EXPECTED_ROUND="${WARM_START_EXPECTED_ROUND:-750}"
export WARM_START_EXPECTED_WORLD_SIZE="${WARM_START_EXPECTED_WORLD_SIZE:-16}"
export RAW_EVAL=/mnt/amulet/studentmodeling/formal_evaluation_v1/test.jsonl
export RECURRENT_EVAL=/tmp/layerwise_task_retention_formal_eval_u4_v1/test.jsonl
export RUN_ID="${WANDB_RUN_ID:?WANDB_RUN_ID is required for an isolated diagnostic workspace}"
export RUN_DIR="${TMPDIR:-/tmp}/layerwise_task_retention_${BRANCH}_${RUN_ID}"
export C4_PROGRESS_DIR="$RUN_DIR/c4_eval_progress_v1"
stage source_provenance
TRAINER_SHA256="$(sha256sum scripts/train_recurrent_student_m4.py | cut -d' ' -f1)"
ENTRYPOINT_SHA256="$(sha256sum "$TRAIN_ENTRYPOINT" | cut -d' ' -f1)"
test "$ENTRYPOINT_SHA256" = "$EXPECTED_ENTRYPOINT_SHA256"
if [[ "$CAUSAL_VARIANT" == base ]]; then
  if [[ "$TRAIN_ENTRYPOINT" == scripts/train_recurrent_student_m4_external_state_teacher.py \
      || "$TRAIN_ENTRYPOINT" == scripts/train_recurrent_student_m4_gold_semantic_teacher.py ]]; then
    export CHECKPOINT_MIRROR_DIR="/mnt/amulet/studentmodeling/diagnostics/layerwise_task_retention_service_recovery/${BRANCH}/${LOSS_PROFILE}/world${C3_WORLD_SIZE}_ga${C3_GRADIENT_ACCUMULATION_STEPS}/external_state_teacher_${STUDENTMODELING_EXTERNAL_STATE_TEACHER_SHA256:0:12}_cw${CONSISTENCY_WEIGHT}/lr${LEARNING_RATE}_start${DIAGNOSTIC_START_GLOBAL_POSITION}_stop${DIAGNOSTIC_STOP_AFTER_ROUND}_${MANIFEST_SHA256}_${TRAINER_SHA256}_${ENTRYPOINT_SHA256}"
  else
    export CHECKPOINT_MIRROR_DIR="/mnt/amulet/studentmodeling/diagnostics/layerwise_task_retention_service_recovery/${BRANCH}/${LOSS_PROFILE}/world${C3_WORLD_SIZE}_ga${C3_GRADIENT_ACCUMULATION_STEPS}/lr${LEARNING_RATE}_start${DIAGNOSTIC_START_GLOBAL_POSITION}_stop${DIAGNOSTIC_STOP_AFTER_ROUND}_${MANIFEST_SHA256}_${TRAINER_SHA256}_${ENTRYPOINT_SHA256}"
  fi
else
  export CHECKPOINT_MIRROR_DIR="/mnt/amulet/studentmodeling/diagnostics/layerwise_task_retention_service_recovery/${BRANCH}/${LOSS_PROFILE}/${CAUSAL_VARIANT}/world${C3_WORLD_SIZE}_ga${C3_GRADIENT_ACCUMULATION_STEPS}/lr${LEARNING_RATE}_start${DIAGNOSTIC_START_GLOBAL_POSITION}_stop${DIAGNOSTIC_STOP_AFTER_ROUND}_${MANIFEST_SHA256}_${TRAINER_SHA256}_${ENTRYPOINT_SHA256}"
fi
mkdir -p "$HF_HUB_CACHE" "$RAW_SPLIT" "$RECURRENT_SPLIT" "$CAUSAL_DATA_BASE" "$CAUSAL_DATA" \
  "$(dirname "$RECURRENT_EVAL")" \
  "$AMLT_OUTPUT_DIR/model" "$AMLT_OUTPUT_DIR/metrics" \
  "$AMLT_OUTPUT_DIR/eval/c4" "$AMLT_OUTPUT_DIR/eval/formal_screen" \
  "$AMLT_OUTPUT_DIR/eval/t6_full"

stage remote_input_preflight
if [[ -n "${WARM_START_SOURCE_CHECKPOINT_DIR:-}" ]]; then
  : "${WARM_START_PRIOR_MANIFEST:?WARM_START_PRIOR_MANIFEST is required}"
  exec 5>/tmp/layerwise-task-retention-consolidation-warm-start.lock
  flock -x 5
  if [[ ! -f "$CHECKPOINT/checkpoint_completion.json" ]]; then
    python scripts/prepare_layerwise_consolidation_warm_start.py \
      --source "$WARM_START_SOURCE_CHECKPOINT_DIR" \
      --prior-manifest "$WARM_START_PRIOR_MANIFEST" \
      --output "$CHECKPOINT" \
      --expected-round "$WARM_START_EXPECTED_ROUND" \
      --expected-world-size "$WARM_START_EXPECTED_WORLD_SIZE"
  fi
  flock -u 5
fi

test "$(df --output=avail /tmp | tail -1)" -ge 125829120
test -f "$RAW_FULL"
test -f "$RAW_EVAL"
test -f "$C4_DATA/train.jsonl"
test -f "$C4_DATA/val.jsonl"
test -f "$C4_DATA/test_same_template.jsonl"
test -f "$CACHE_ROOT/integrity.json"
test -f "$CACHE_ROOT/manifest.json"
test -f "$CHECKPOINT/adapter/adapter_model.safetensors"
test -f "$CHECKPOINT/recurrent_state.pt"
test -f "$CHECKPOINT/checkpoint_completion.json"
mkdir -p "$CHECKPOINT_MIRROR_DIR"
if [[ "$BRANCH" != eedi_full && "$BRANCH" != eedi_full_open_t3 ]]; then
  test -f "$MANIFEST"
  test "$(sha256sum "$MANIFEST" | cut -d' ' -f1)" = "$MANIFEST_SHA256"
fi
test "$(sha256sum scripts/train_recurrent_student_m4.py | cut -d' ' -f1)" = a67662448408a587f891d32d4c5170fdca0435adac47554268edf5a3972330b5
test "$(sha256sum scripts/augment_recurrent_causal_same_rule.py | cut -d' ' -f1)" = "$SAME_RULE_AUGMENTER_SHA256"
test "$(sha256sum scripts/distributed_epoch.py | cut -d' ' -f1)" = 20f85e37512965d560761de26d99d3329b15c16de679d34ad460a1407bad2f9a
test "$(sha256sum scripts/evaluate_recurrent_causal_gate.py | cut -d' ' -f1)" = 69fd4f2f4943a78a0afe9a701533d9ab034f247f87c7a09327f7523e310b7a52
test "$(sha256sum scripts/evaluate_recurrent_formal_protocols.py | cut -d' ' -f1)" = 262370592e018641b99d6282858d8990b38314c05bb6e3bbc645823834cf73ed
test "$(sha256sum scripts/audit_layerwise_distributed_checkpoint.py | cut -d' ' -f1)" = 6690ece8935c6343bec7136ad34a9899b72a866c7bde6286ca26b2c82890007f
test "$(sha256sum scripts/frozen_layerwise_finalfit/layerwise_recurrent_student_model.py | cut -d' ' -f1)" = c80357282e600b31f68f55d8ab434ad349c0e9eeeeb222b52dbadae7caea11ca
test "$(sha256sum tests/frozen_layerwise_finalfit/test_layerwise_recurrent_student_model.py | cut -d' ' -f1)" = a995f82f364eb3247ff3e17873cc720bbce7a2a8f52813c9369421fa61d3c274
test "$(sha256sum scripts/materialize_legacy_layerwise_state_count_compat.py | cut -d' ' -f1)" = fd8272398d3278de9c9161f2b56c8d85949961845ccd69000b6ced7de9be4a51
python scripts/materialize_legacy_layerwise_state_count_compat.py \
  --source scripts/frozen_layerwise_finalfit/layerwise_recurrent_student_model.py \
  --output scripts/layerwise_recurrent_student_model.py \
  --receipt /tmp/legacy-layerwise-state-count-compat.json
cp tests/frozen_layerwise_finalfit/test_layerwise_recurrent_student_model.py tests/test_layerwise_recurrent_student_model.py
stage local_unit_tests
python -m pytest -q tests/test_materialize_legacy_layerwise_state_count_compat.py
python -m unittest discover -s tests -p 'test_train_recurrent_student_m4.py' -v
python -m unittest discover -s tests -p 'test_layerwise_recurrent_student_model.py' -v
python -m unittest discover -s tests -p 'test_distributed_epoch.py' -v
python -m unittest discover -s tests -p 'test_evaluate_recurrent_causal_gate.py' -v
python -m unittest discover -s tests -p 'test_evaluate_recurrent_formal_protocols.py' -v

exec 9>/tmp/layerwise-task-retention-checkpoint-audit.lock
flock -x 9
stage warm_start_audit
audit_key="${WARM_START_EXPECTED_WORLD_SIZE}x${WARM_START_EXPECTED_ROUND}"
audit_json="/tmp/layerwise-task-retention-checkpoint-audit-${audit_key}.json"
audit_complete="/tmp/layerwise-task-retention-checkpoint-audit-${audit_key}.complete"
if [[ ! -f "$audit_complete" ]]; then
  if [[ "$WARM_START_EXPECTED_WORLD_SIZE" == 16 ]]; then
    python scripts/audit_layerwise_distributed_checkpoint.py \
      --checkpoint "$CHECKPOINT/recurrent_state.pt" --expected-round "$WARM_START_EXPECTED_ROUND" \
      --output "$audit_json"
  else
    python scripts/audit_layerwise_diagnostic_warm_start.py \
      --checkpoint "$CHECKPOINT" \
      --expected-round "$WARM_START_EXPECTED_ROUND" \
      --expected-world-size "$WARM_START_EXPECTED_WORLD_SIZE" \
      --expected-manifest-sha256 "$(sha256sum "$WARM_START_PRIOR_MANIFEST" | cut -d' ' -f1)" \
      --output "$audit_json"
  fi
  python - "$audit_json" "$WARM_START_EXPECTED_ROUND" \
    "$WARM_START_EXPECTED_WORLD_SIZE" "$DIAGNOSTIC_START_GLOBAL_POSITION" <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
expected_round=int(sys.argv[2]); expected_world=int(sys.argv[3]); expected_start=int(sys.argv[4])
assert x['status']=='passed',x
assert x['checkpoint_round']==expected_round and x['world_size']==expected_world,x
assert expected_round*expected_world==expected_start,(x,expected_start)
PY
  touch "$audit_complete"
fi
flock -u 9

exec 8>"/tmp/layerwise-task-retention-data-${CAUSAL_VARIANT}.lock"
flock -x 8
stage data_materialization
data_complete="/tmp/layerwise-task-retention-data-${CAUSAL_VARIANT}.complete"
if [[ ! -f "$data_complete" ]]; then
  rm -rf "$RAW_SPLIT" "$RECURRENT_SPLIT" "$CAUSAL_DATA_BASE" "$CAUSAL_DATA" "$(dirname "$RECURRENT_EVAL")"
  mkdir -p "$RAW_SPLIT" "$RECURRENT_SPLIT" "$CAUSAL_DATA_BASE" "$CAUSAL_DATA" "$(dirname "$RECURRENT_EVAL")"
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
    --output-dir "$CAUSAL_DATA_BASE" --max-updates 4
  if [[ "$CAUSAL_VARIANT" == same_rule ]]; then
    python scripts/augment_recurrent_causal_same_rule.py \
      --train-source "$C4_DATA/train.jsonl" --valid-source "$C4_DATA/val.jsonl" \
      --causal-train "$CAUSAL_DATA_BASE/train.jsonl" \
      --causal-valid "$CAUSAL_DATA_BASE/valid.jsonl" \
      --output-dir "$CAUSAL_DATA" --seed 20260804 --max-updates 4
  else
    # In the base arm CAUSAL_DATA and CAUSAL_DATA_BASE intentionally resolve
    # to the same directory; the audited builder output is already final.
    :
  fi
  python scripts/build_recurrent_evaluation_manifest.py \
    --input-jsonl "$RAW_EVAL" --output-jsonl "$RECURRENT_EVAL" \
    --audit-json /tmp/layerwise_task_retention_formal_eval_u4_v1/audit.json \
    --max-updates 4 --seed 20260730
  touch "$data_complete"
fi
flock -u 8
test "$(sha256sum "$FINAL_JSONL" | cut -d' ' -f1)" = 9961f746cb17e6c314e182d3453b6157022fb48b8de44d2e68bc8e87a1131ccd
test "$(sha256sum "$RECURRENT_SPLIT/valid.jsonl" | cut -d' ' -f1)" = 78bfce75576dc3873cbb6459669a952adeedb8b77bda15f281af9519e2506d14
case "$CAUSAL_VARIANT" in
  base)
    EXPECTED_CAUSAL_TRAIN_SHA256=30f68f54120e5a9ada0a37b1dba2b6480de0f8eaeac98d0b7ed7c627429f9766
    EXPECTED_CAUSAL_VALID_SHA256=8e038c3a60c2e1b420779cd54eb203b12ae5d51544c87a805fe19d47fdbe294e
    ;;
  same_rule)
    EXPECTED_CAUSAL_TRAIN_SHA256=a170b085858b656432725cd6e835703cb195c469d8374c3abfd9f0a51f909f24
    EXPECTED_CAUSAL_VALID_SHA256=d1f97a387b6be30467918813475edac6ac0d38e5923767f27f3aca792b130696
    test -s "$CAUSAL_DATA/augmentation_receipt.json"
    ;;
esac
test "$(sha256sum "$CAUSAL_DATA/train.jsonl" | cut -d' ' -f1)" = "$EXPECTED_CAUSAL_TRAIN_SHA256"
test "$(sha256sum "$CAUSAL_DATA/valid.jsonl" | cut -d' ' -f1)" = "$EXPECTED_CAUSAL_VALID_SHA256"
test "$(sha256sum "$RECURRENT_EVAL" | cut -d' ' -f1)" = a368cefb0c49cb6a305ad1a892808361fba2d44d555a0a28ea64e56babe32525
if [[ "$BRANCH" == eedi_full || "$BRANCH" == eedi_full_open_t3 ]]; then
  mkdir -p "$(dirname "$MANIFEST")"
  python scripts/build_layerwise_task_retention_manifests.py \
    --train-jsonl "$FINAL_JSONL" --formal-eval-jsonl "$RECURRENT_EVAL" \
    --output-dir "$(dirname "$MANIFEST")" --seed 42
fi
test -f "$MANIFEST"
test "$(sha256sum "$MANIFEST" | cut -d' ' -f1)" = "$MANIFEST_SHA256"
manifest_branch="$(python - "$MANIFEST" <<'PY'
import json,sys
print(json.load(open(sys.argv[1],encoding='utf-8'))['branch'])
PY
)"
expected_manifest_branch="$BRANCH"
[[ "$BRANCH" != eedi_full_open_t3 ]] || expected_manifest_branch=eedi_full
[[ "$BRANCH" != equal_t3_t6_open_t3 ]] || expected_manifest_branch=equal_t3_t6
test "$manifest_branch" = "$expected_manifest_branch"

exec 7>/tmp/layerwise-task-retention-model.lock
flock -x 7
stage model_copy
if [[ ! -f /tmp/layerwise-task-retention-model.complete ]]; then
  rm -rf "$LOCAL_MODEL"; mkdir -p "$LOCAL_MODEL"
  python scripts/copy_model_parallel.py \
    --source /mnt/amulet/studentmodeling/models/Qwen3-32B \
    --destination "$LOCAL_MODEL" --workers 16
  test -f "$LOCAL_MODEL/model.safetensors.index.json"
  touch /tmp/layerwise-task-retention-model.complete
fi
flock -u 7
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

export CUDA_VISIBLE_DEVICES="${OMPI_COMM_WORLD_LOCAL_RANK:?MPI local rank is required}"
export RANK="${OMPI_COMM_WORLD_RANK:?MPI rank is required}"
export WORLD_SIZE="${OMPI_COMM_WORLD_SIZE:?MPI world size is required}"
export LOCAL_RANK=0
exec 6>"/tmp/layerwise-task-retention-${BRANCH}-${RUN_ID}-run.lock"
flock -x 6
stage training
if [[ ! -f "/tmp/layerwise-task-retention-${BRANCH}-${RUN_ID}-run.ready" ]]; then
  # A mounted Blob/FUSE directory can block indefinitely in rm(2) after an
  # interrupted run.  Diagnostic checkpoints need no cross-job persistence:
  # every AMLT job gets a unique local workspace and publishes only verified
  # final artifacts through AMLT_OUTPUT_DIR below.
  test ! -e "$RUN_DIR"
  mkdir -p "$RUN_DIR" "$C4_PROGRESS_DIR"
  touch "/tmp/layerwise-task-retention-${BRANCH}-${RUN_ID}-run.ready"
fi
test -d "$RUN_DIR"
test -d "$C4_PROGRESS_DIR"
flock -u 6

# A shared checkpoint is eligible only after its receipt and both payload
# hashes verify.  Interrupted Blob/FUSE copies are ignored.  Every MPI rank
# deterministically selects the same highest completed round.
RESUME_CHECKPOINT="$(python scripts/select_latest_distributed_checkpoint.py \
  --root "$CHECKPOINT_MIRROR_DIR" --max-round "$DIAGNOSTIC_STOP_AFTER_ROUND")"
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  TRAIN_RECOVERY_ARGS=(--resume-from-checkpoint "$RESUME_CHECKPOINT")
else
  TRAIN_RECOVERY_ARGS=(--diagnostic-warm-start-checkpoint "$CHECKPOINT/recurrent_state.pt")
fi

# The diagnostic segment starts at global round 3000.  Platform SIGTERM at
# rounds 3060--3170 exposed that a 250-round cadence could leave no atomic
# recovery point.  This changes only checkpoint plumbing, not optimization.
python -u "$TRAIN_ENTRYPOINT" \
  --model "$LOCAL_MODEL" --init-adapter "$CHECKPOINT/adapter" \
  --state-architecture layerwise --layerwise-operation-checkpointing \
  --layerwise-transition-rank 256 \
  --train-jsonl "$FINAL_JSONL" --valid-jsonl "$RECURRENT_SPLIT/valid.jsonl" \
  --causal-train-jsonl "$CAUSAL_DATA/train.jsonl" \
  --causal-valid-jsonl "$CAUSAL_DATA/valid.jsonl" \
  --teacher-cache "$CACHE_ROOT" --behavior-kd-weight "$BEHAVIOR_KD_WEIGHT" --behavior-kd-temperature 1.0 \
  --output-dir "$RUN_DIR" --state-token-alias '<|vision_pad|>' \
  --max-length 1536 --learning-rate "$LEARNING_RATE" --lr-scheduler constant \
  --generation-weight 1.0 --consistency-weight "$CONSISTENCY_WEIGHT" --direct-replay-weight "$DIRECT_REPLAY_WEIGHT" \
  --causal-weight 2.0 --causal-margin 0.5 \
  --causal-preference-reduction sum --causal-objective-mode specificity_triplet \
  --causal-repeat-fraction 0.05 --normalize-causal-exposure \
  --gradient-accumulation-steps "$C3_GRADIENT_ACCUMULATION_STEPS" --eval-limit 300 --causal-eval-limit 165 \
  --save-steps 50 --checkpoint-retention 2 \
  --checkpoint-mirror-dir "$CHECKPOINT_MIRROR_DIR" \
  "${TRAIN_RECOVERY_ARGS[@]}" \
  --diagnostic-start-global-position "$DIAGNOSTIC_START_GLOBAL_POSITION" \
  --diagnostic-index-manifest "$MANIFEST" \
  --diagnostic-stop-after-round "$DIAGNOSTIC_STOP_AFTER_ROUND" \
  --distributed-mpi --distributed-world-size "$C3_WORLD_SIZE" --seed 42

if [[ "${OMPI_COMM_WORLD_RANK:-0}" != 0 ]]; then exit 0; fi
test -f "$RUN_DIR/completion.json"
test -f "$RUN_DIR/final_adapter/adapter_model.safetensors"
test -f "$RUN_DIR/recurrent_state.pt"
stage posttrain_audit
python - "$RUN_DIR" "$MANIFEST" "$MANIFEST_SHA256" "$BRANCH" "$LOSS_PROFILE" \
  "$SEGMENT_ROWS" "$DIAGNOSTIC_START_GLOBAL_POSITION" "$EXPECTED_CAUSAL_PAIRS" \
  "$DIAGNOSTIC_STOP_AFTER_ROUND" "$LEARNING_RATE" "$CAUSAL_VARIANT" <<'PY'
import collections,json,os,pathlib,sys
root=pathlib.Path(sys.argv[1]); manifest=json.load(open(sys.argv[2],encoding='utf-8'))
completion=json.load(open(root/'completion.json',encoding='utf-8'))
config=json.load(open(root/'run_config.json',encoding='utf-8'))
history=json.load(open(root/'trainer_history.json',encoding='utf-8'))
assert completion['status']=='diagnostic_complete' and completion['diagnostic'] is True,completion
segment_rows=int(sys.argv[6]); start=int(sys.argv[7]); causal_pairs=int(sys.argv[8])
stop_round=int(sys.argv[9]); learning_rate=float(sys.argv[10])
assert completion['micro_steps']==segment_rows and completion['optimizer_steps']==segment_rows//4,completion
assert completion['history_rows']==segment_rows and completion['history_global_position_start']==start,completion
assert completion['history_global_position_end']==start+segment_rows-1 and completion['causal_pairs_seen']==causal_pairs,completion
assert completion['history_integrity_pass'] is True,completion
assert config['diagnostic_weight_only_warm_start'] is True and config['diagnostic_optimizer_reset'] is True,config
assert config['diagnostic_warm_start_source_world_size']==int(os.environ['WARM_START_EXPECTED_WORLD_SIZE']),config
assert config['diagnostic_warm_start_source_round']==int(os.environ['WARM_START_EXPECTED_ROUND']),config
assert config['diagnostic_segment_rounds']==segment_rows//int(os.environ['C3_WORLD_SIZE']),config
assert config['planned_optimizer_steps']==segment_rows//4,config
assert config['diagnostic_stop_after_round']==stop_round,config
assert config['distributed_world_size']==int(os.environ['C3_WORLD_SIZE']),config
assert config['gradient_accumulation_steps']==int(os.environ['C3_GRADIENT_ACCUMULATION_STEPS']),config
assert config['distributed_world_size']*config['gradient_accumulation_steps']==4,config
assert abs(float(config['learning_rate'])-learning_rate)<1e-15,config
assert abs(float(config['consistency_weight'])-float(os.environ.get('CONSISTENCY_WEIGHT','0.25')))<1e-15,config
assert config['diagnostic_index_manifest_sha256']==sys.argv[3],config
assert config['causal_weight']==2.0 and config['causal_repeat_fraction']==0.05,config
assert config['causal_objective_mode']=='specificity_triplet' and config['layerwise_transition_rank']==256,config
causal_variant=sys.argv[11]
expected_causal={
 'base':('30f68f54120e5a9ada0a37b1dba2b6480de0f8eaeac98d0b7ed7c627429f9766',1051),
 'same_rule':('a170b085858b656432725cd6e835703cb195c469d8374c3abfd9f0a51f909f24',2102),
}[causal_variant]
assert config['causal_train_sha256']==expected_causal[0],config
assert config['causal_train_pairs']==expected_causal[1],config
expected_weights={
 'standard':(0.5,0.25),
 'state_kd':(0.5,0.0),
 'state_path':(0.0,0.0),
}[sys.argv[5]]
assert (config['behavior_kd_weight'],config['direct_replay_weight'])==expected_weights,config
if os.environ.get('TRAIN_ENTRYPOINT') in {
 'scripts/train_recurrent_student_m4_external_state_teacher.py',
 'scripts/train_recurrent_student_m4_gold_semantic_teacher.py',
}:
 assert config['external_state_teacher_adapter']==os.environ['STUDENTMODELING_EXTERNAL_STATE_TEACHER_ADAPTER'],config
 assert config['external_state_teacher_sha256']==os.environ['STUDENTMODELING_EXTERNAL_STATE_TEACHER_SHA256'],config
 assert config['external_state_teacher_trainable'] is False,config
 if os.environ.get('TRAIN_ENTRYPOINT')=='scripts/train_recurrent_student_m4_external_state_teacher.py':
  assert config['external_state_teacher_scope']=='direct_layerwise_trajectory_only',config
 else:
  assert config['external_state_teacher_scope']=='gold_semantic_eedi_t3_layerwise_trajectory',config
  assert config['gold_semantic_label_visible_at_inference'] is False,config
assert collections.Counter(row['task'] for row in history)==collections.Counter(manifest['task_counts']),history[:2]
assert collections.Counter(row['source'] for row in history)==collections.Counter(manifest['source_counts']),history[:2]
receipt={
 'stage':'layerwise_task_retention_rehearsal_training','branch':sys.argv[4],
 'loss_profile':sys.argv[5],
 'causal_variant':causal_variant,
 'diagnostic_only':True,'exact_resume':False,'optimizer_reset':True,
 'source_checkpoint_round':int(os.environ['WARM_START_EXPECTED_ROUND']),
 'source_world_size':int(os.environ['WARM_START_EXPECTED_WORLD_SIZE']),
 'current_world_size':int(os.environ['C3_WORLD_SIZE']),
 'gradient_accumulation_steps':int(os.environ['C3_GRADIENT_ACCUMULATION_STEPS']),
 'global_batch_size':4,
 'start_global_position':start,'end_global_position':start+segment_rows-1,
 'micro_steps':segment_rows,'optimizer_steps':segment_rows//4,'causal_pairs_seen':causal_pairs,
 'learning_rate':learning_rate,
 'consistency_weight':float(config['consistency_weight']),
 'causal_weight':float(config['causal_weight']),
 'train_entrypoint':os.environ.get('TRAIN_ENTRYPOINT','scripts/train_recurrent_student_m4.py'),
 'manifest_sha256':sys.argv[3],'task_counts':manifest['task_counts'],
 'source_counts':manifest['source_counts'],'t6_label_counts':manifest['t6_label_counts'],
 'formal_example_id_overlap':manifest['formal_example_id_overlap'],
}
if sys.argv[4] in {'eedi_full_open_t3','equal_t3_t6_open_t3'}:
 receipt['training_view']=(
  'open_t3_selective_kd'
  if os.environ.get('TRAIN_ENTRYPOINT')=='scripts/train_recurrent_student_m4_open_t3_selective_kd.py'
  else 'open_t3_no_candidate_enumeration'
 )
(root/'task_retention_training_receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps(receipt,indent=2,sort_keys=True))
PY

if [[ "$TRAIN_ENTRYPOINT" == scripts/train_recurrent_student_m4_open_t3_selective_kd.py ]]; then
  test -f "$RUN_DIR/open_t3_selective_kd_training_view.json"
  cp "$RUN_DIR/open_t3_selective_kd_training_view.json" \
    "$AMLT_OUTPUT_DIR/metrics/open_t3_selective_kd_training_view.json"
fi
if [[ "$TRAIN_ENTRYPOINT" == scripts/train_recurrent_student_m4_external_state_teacher.py ]]; then
  test -f "$RUN_DIR/external_state_teacher_training_view.json"
  cp "$RUN_DIR/external_state_teacher_training_view.json" \
    "$AMLT_OUTPUT_DIR/metrics/external_state_teacher_training_view.json"
fi
if [[ "$TRAIN_ENTRYPOINT" == scripts/train_recurrent_student_m4_gold_semantic_teacher.py ]]; then
  test -f "$RUN_DIR/gold_semantic_state_teacher_training_view.json"
  cp "$RUN_DIR/gold_semantic_state_teacher_training_view.json" \
    "$AMLT_OUTPUT_DIR/metrics/gold_semantic_state_teacher_training_view.json"
fi

export C4_PROGRESS_KEY="$(python - "$RUN_DIR/completion.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
print(f"branch={x['sha256']['adapter_model']}:state={x['sha256']['recurrent_state']}")
PY
)"
printf '{"stage":"c4","branch":"%s","loss_profile":"%s","started_at":"%s"}\n' \
  "$BRANCH" "$LOSS_PROFILE" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
python -u scripts/evaluate_recurrent_causal_gate.py \
  --model "$LOCAL_MODEL" --adapter "$RUN_DIR/final_adapter" \
  --recurrent-state "$RUN_DIR/recurrent_state.pt" --state-architecture layerwise \
  --jsonl "$C4_DATA/test_same_template.jsonl" \
  --pair-manifest manifests/c4_stage_a_gate_pairs_seed42_n150_v1.json \
  --output-json "$AMLT_OUTPUT_DIR/eval/c4/recurrent_causal_gate.json" \
  --state-token-alias '<|vision_pad|>' --max-length 1536 --num-pairs 150 --seed 42 \
  --progress-dir "$C4_PROGRESS_DIR" --progress-key "$C4_PROGRESS_KEY"
printf '{"stage":"formal_screen","branch":"%s","loss_profile":"%s","started_at":"%s"}\n' \
  "$BRANCH" "$LOSS_PROFILE" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
python -u scripts/evaluate_recurrent_formal_protocols.py \
  --model "$LOCAL_MODEL" --adapter "$RUN_DIR/final_adapter" \
  --recurrent-state "$RUN_DIR/recurrent_state.pt" --state-architecture layerwise \
  --eval-jsonl "$RECURRENT_EVAL" --output-dir "$AMLT_OUTPUT_DIR/eval/formal_screen" \
  --state-token-alias '<|vision_pad|>' --max-samples-per-group 25 \
  --max-length 1536 --max-new-tokens 128 --state-batch-size 2 \
  --candidate-row-batch-size 2 --seed 42 --save-unredacted-predictions
printf '{"stage":"t6_full","branch":"%s","loss_profile":"%s","started_at":"%s"}\n' \
  "$BRANCH" "$LOSS_PROFILE" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
python -u scripts/evaluate_recurrent_formal_protocols.py \
  --model "$LOCAL_MODEL" --adapter "$RUN_DIR/final_adapter" \
  --recurrent-state "$RUN_DIR/recurrent_state.pt" --state-architecture layerwise \
  --eval-jsonl "$RECURRENT_EVAL" --output-dir "$AMLT_OUTPUT_DIR/eval/t6_full" \
  --state-token-alias '<|vision_pad|>' --max-length 1536 --max-new-tokens 128 \
  --state-batch-size 2 --candidate-row-batch-size 2 \
  --tasks predict_error_persistence --seed 42 --save-unredacted-predictions
printf '{"stage":"publish","branch":"%s","loss_profile":"%s","started_at":"%s"}\n' \
  "$BRANCH" "$LOSS_PROFILE" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

cp -r "$RUN_DIR/final_adapter/." "$AMLT_OUTPUT_DIR/model/"
cp "$RUN_DIR/recurrent_state.pt" "$AMLT_OUTPUT_DIR/model/recurrent_state.pt"
cp "$RUN_DIR"/*.json "$AMLT_OUTPUT_DIR/metrics/"
cp "$MANIFEST" "$AMLT_OUTPUT_DIR/metrics/task_retention_index_manifest.json"
cp "$audit_json" "$AMLT_OUTPUT_DIR/metrics/checkpoint_resume_audit.json"
cp /tmp/layerwise_task_retention_formal_eval_u4_v1/audit.json "$AMLT_OUTPUT_DIR/metrics/formal_manifest_audit.json"
if [[ "$CAUSAL_VARIANT" == same_rule ]]; then
  cp "$CAUSAL_DATA/augmentation_receipt.json" "$AMLT_OUTPUT_DIR/metrics/causal_same_rule_augmentation.json"
fi
