#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "STATE8_CURRICULUM_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR
stage(){ printf 'STATE8_CURRICULUM_STAGE=%s time=%s\n' "$1" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"; }

export VENV_ROOT="${VENV_ROOT:-/tmp/studentmodeling-state8-qualified-teacher-venv}"
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
export RAW_SPLIT=/tmp/state8_qualified_teacher_raw_split
export RECURRENT_SPLIT=/tmp/state8_qualified_teacher_recurrent_split
export FULL_JSONL="$RECURRENT_SPLIT/final_fit_trainvalid.jsonl"
export RAW_EVAL=/mnt/amulet/studentmodeling/formal_evaluation_v1/test.jsonl
export RECURRENT_EVAL=/tmp/state8_qualified_teacher_formal_u4/test.jsonl
export STATE8_REPLAY_VARIANT="${STATE8_REPLAY_VARIANT:-t3_heavy}"
export STATE8_KD_VARIANT="${STATE8_KD_VARIANT:-base}"
export TRAIN_SEED="${TRAIN_SEED:-42}"
export STATE8_FINALIZE_ONLY="${STATE8_FINALIZE_ONLY:-0}"
[[ "$TRAIN_SEED" =~ ^[1-9][0-9]*$ ]] || exit 2
case "$STATE8_FINALIZE_ONLY" in 0|1) ;; *) exit 2 ;; esac
export STATE8_CURRICULUM_NAMESPACE="${STATE8_CURRICULUM_NAMESPACE:-}"
export TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:-scripts/train_recurrent_student_m4_state8_external_teacher_curriculum.py}"
export FORMAL_EVALUATOR="${FORMAL_EVALUATOR:-scripts/evaluate_recurrent_formal_protocols.py}"
case "$STATE8_CURRICULUM_NAMESPACE" in ''|taxonomy_semantic_v1) ;; *) exit 2 ;; esac
case "$STATE8_REPLAY_VARIANT" in
  t3_heavy)
    export MANIFEST=manifests/layerwise_task_retention_eedi_full_seed42_v1.json
    export MANIFEST_SHA256=97bef6748b1fa8d75809e9e01c9b76d57b6534d336b7037217289932a31df393
    export SUBSET_SHA256=023e692afe932a6bbbd70547b698007c89b96ae7f7d79adae4622fdf34900540
    export EXPECTED_TASK_COUNTS='{"diagnose_misconception":4500,"predict_correctness":80,"predict_error_persistence":300,"predict_prospective_error":70,"predict_response":50}'
    case "$STATE8_KD_VARIANT" in
      base)
        export STATE8_ONLINE_KD_WEIGHT=0.5
        export STATE8_STAGE_B_RECIPE_ID=exact5000_rounds1250_v2
        export STABLE_ROOT=/mnt/amulet/studentmodeling/diagnostics/state8_qualified_teacher_curriculum_v1_immutable
        export IMMUTABLE_STAGE=state8_qualified_teacher_curriculum_immutable_v1
        ;;
      highkd)
        export STATE8_ONLINE_KD_WEIGHT=1.0
        # v1 was created while the materialized replay still contained 4,600
        # rows (1,150 distributed rounds).  Its otherwise-valid atomic
        # checkpoints must never be considered for the corrected 5,000-row,
        # 1,250-round recipe.  A new namespace is the fail-closed boundary:
        # recovery may resume only checkpoints produced by this exact recipe.
        export STATE8_STAGE_B_RECIPE_ID=exact5000_rounds1250_kd1_v2
        export STABLE_ROOT=/mnt/amulet/studentmodeling/diagnostics/state8_qualified_teacher_curriculum_highkd_v1_immutable
        export IMMUTABLE_STAGE=state8_qualified_teacher_curriculum_highkd_immutable_v1
        ;;
      *) echo "Unknown STATE8_KD_VARIANT=$STATE8_KD_VARIANT" >&2; exit 2 ;;
    esac
    export RUN_DIR=/mnt/amulet/studentmodeling/checkpoints/state8_qualified_teacher_curriculum_seed42_${STATE8_STAGE_B_RECIPE_ID}_20260805
    ;;
  balanced)
    test "$STATE8_KD_VARIANT" = base
    export STATE8_ONLINE_KD_WEIGHT=0.5
    # The earlier v2 namespace contains checkpoints from the superseded
    # 4,300-row/1,075-round materialization.  Exact resume must never mix that
    # optimizer history with the corrected 5,000-row/1,250-round recipe.
    export STATE8_STAGE_B_RECIPE_ID=exact5000_rounds1250_v3
    export MANIFEST=manifests/state8_qualified_balanced_replay_seed42_v1.json
    export MANIFEST_SHA256=baa0c9f7c062822cfffe11a5a4ec1e416bf726dfaf66fd8251f7dd6bb03bbadd
    export SUBSET_SHA256=338ca9e62da39e03e8e2a25dec56c57179a4206ebf1dfef9729042b1ea681deb
    export EXPECTED_TASK_COUNTS='{"diagnose_misconception":2500,"predict_correctness":800,"predict_error_persistence":500,"predict_prospective_error":600,"predict_response":600}'
    export RUN_DIR=/mnt/amulet/studentmodeling/checkpoints/state8_qualified_teacher_balanced_curriculum_seed42_${STATE8_STAGE_B_RECIPE_ID}_20260805
    export STABLE_ROOT=/mnt/amulet/studentmodeling/diagnostics/state8_qualified_teacher_balanced_curriculum_v1_immutable
    export IMMUTABLE_STAGE=state8_qualified_teacher_balanced_curriculum_immutable_v1
    ;;
  *) echo "Unknown STATE8_REPLAY_VARIANT=$STATE8_REPLAY_VARIANT" >&2; exit 2 ;;
esac
if [[ "$STATE8_FINALIZE_ONLY" == 1 ]]; then
  test "$STATE8_REPLAY_VARIANT" = t3_heavy
  test "$STATE8_KD_VARIANT" = highkd
  test "$STATE8_CURRICULUM_NAMESPACE" = taxonomy_semantic_v1
fi
if [[ -n "$STATE8_CURRICULUM_NAMESPACE" ]]; then
  export RUN_DIR="${RUN_DIR}_${STATE8_CURRICULUM_NAMESPACE}"
  export STABLE_ROOT="${STABLE_ROOT}_${STATE8_CURRICULUM_NAMESPACE}"
  export IMMUTABLE_STAGE="${IMMUTABLE_STAGE}_${STATE8_CURRICULUM_NAMESPACE}"
fi
export SUBSET="/tmp/state8_qualified_teacher_${STATE8_REPLAY_VARIANT}_train5000.jsonl"
export VALID_JSONL="$RECURRENT_SPLIT/valid.jsonl"
export C4_DATA=/mnt/amulet/studentmodeling/c4_fullopen_20260724/malrule_c1_clean
export CAUSAL_BASE=/tmp/state8_qualified_teacher_cf4_base
export CAUSAL_DATA=/tmp/state8_qualified_teacher_cf4_same_rule
export CAUSAL_SUBSET=/tmp/state8_qualified_teacher_cf4_subset250
export STUDENT_ROOT=/mnt/amulet/studentmodeling/diagnostics/state8_natural_dualpath_samerule_v5_immutable
export TEACHER_ROOT=/mnt/amulet/studentmodeling/diagnostics/state8_full_history_natural_teacher_v1
export TOTAL_ROUNDS=1250
export STATE8_SAVE_STEPS="${STATE8_SAVE_STEPS:-125}"
[[ "$STATE8_SAVE_STEPS" =~ ^[1-9][0-9]*$ ]] || exit 2
(( STATE8_SAVE_STEPS % 5 == 0 )) || exit 2
if [[ "$STATE8_CURRICULUM_NAMESPACE" == taxonomy_semantic_v1 ]]; then
  RESUME_CADENCE_ARGS=(--save-steps "$STATE8_SAVE_STEPS" --recovery-save-steps 5)
else
  RESUME_CADENCE_ARGS=(--save-steps "$STATE8_SAVE_STEPS" --recovery-save-steps 25)
fi
export DATA_CACHE_ROOT=/mnt/amulet/studentmodeling/cache/state8_taxonomy_curriculum_data_v1

stage remote_input_preflight
mkdir -p "$HF_HUB_CACHE" "$RAW_SPLIT" "$RECURRENT_SPLIT" "$(dirname "$RECURRENT_EVAL")" \
  "$CAUSAL_BASE" "$CAUSAL_DATA" "$CAUSAL_SUBSET" "$RUN_DIR" "$AMLT_OUTPUT_DIR/model" "$AMLT_OUTPUT_DIR/metrics" \
  "$AMLT_OUTPUT_DIR/eval/formal_screen" "$STABLE_ROOT/model" "$STABLE_ROOT/metrics" "$STABLE_ROOT/eval/formal_screen"
test "$(df --output=avail /tmp | tail -1)" -ge 125829120
test -f "$RAW_FULL"
test -f "$RAW_EVAL"
test -f "$STUDENT_ROOT/model/adapter_model.safetensors"
test -f "$STUDENT_ROOT/model/recurrent_state.pt"
test -f "$STUDENT_ROOT/metrics/immutable_receipt.json"
test -f "$TEACHER_ROOT/model/adapter_model.safetensors"
test -f "$TEACHER_ROOT/model/adapter_config.json"
test -f "$TEACHER_ROOT/metrics/teacher_gate.json"
test -f "$TEACHER_ROOT/metrics/teacher_provenance_gate.json"
test -f "$TEACHER_ROOT/metrics/publish_receipt.json"
test -f "$C4_DATA/train.jsonl"
test -f "$C4_DATA/val.jsonl"
test "$(sha256sum "$RAW_FULL" | cut -d' ' -f1)" = aeabd3b4dfef3c4d92ac68583879c91448ef90f58304cedf32e470b586ca4542
test "$(sha256sum "$RAW_EVAL" | cut -d' ' -f1)" = bff7726953ae048bedf6dacdbb7500a54cb8de305e406f2497ef1f11c04167f8
test "$(sha256sum "$MANIFEST" | cut -d' ' -f1)" = "$MANIFEST_SHA256"
test "$(sha256sum scripts/train_recurrent_student_m4.py | cut -d' ' -f1)" = ae502968872ff6597e73bcc12cb3ff9a5bbb0fe9d7a763383f6e76bc392de42c
test "$(sha256sum scripts/train_recurrent_student_m4_state8_external_teacher_curriculum.py | cut -d' ' -f1)" = af1da212bc9c457a7b5a9297e895bb56fffcc7d98c8b2c9ff21ed0ae7f671539
test -f "$TRAIN_ENTRYPOINT"
test -f "$FORMAL_EVALUATOR"
if [[ "$STATE8_CURRICULUM_NAMESPACE" == taxonomy_semantic_v1 ]]; then
  test "$TRAIN_ENTRYPOINT" = scripts/train_state8_taxonomy_teacher_curriculum.py
  test "$FORMAL_EVALUATOR" = scripts/evaluate_recurrent_formal_protocols_taxonomy.py
  test "${STATE8_CURRICULUM_LABEL_WEIGHT:-}" = 4.0
  test "$(sha256sum scripts/train_state8_taxonomy_teacher_curriculum.py | cut -d' ' -f1)" = 1ba1fb420190d9b6a3d49e3dcda97a0830fe0603376e79b8ddb7fa4c0de21ce3
  test "$(sha256sum scripts/state8_taxonomy_contract.py | cut -d' ' -f1)" = 1ca32112e76c795502f3cb1edbedcc7220f093c133fae8d20ff5cab789e512ed
  test "$(sha256sum scripts/evaluate_recurrent_formal_protocols_taxonomy.py | cut -d' ' -f1)" = 648baa998672f30652ad3e68f114c2648c5bf839dcddd5b1e0acae47400e5b22
fi
test "$(sha256sum scripts/materialize_recurrent_index_manifest.py | cut -d' ' -f1)" = 977ad3cf419f850b74979f430025eda3959ed3c7122aa5a2db5afff114f6b7df
test "$(sha256sum scripts/state8_natural_curriculum_view.py | cut -d' ' -f1)" = 98c961180d2428b2c68ef9399996ed991d5bd34120a3046b4f56fb9ba4e09682
test "$(sha256sum scripts/augment_recurrent_causal_same_rule.py | cut -d' ' -f1)" = 1ae82541e97353d23b674af80dfb0e2d50cfd92e3a6ee99c2b55f3b0caf1d785
test "$(sha256sum scripts/subset_state8_causal_twins.py | cut -d' ' -f1)" = fcca21bd73ad8ddd83d1cc9e308183b6b6c751b97537ed3d3d35838567ef87cb
test "$(sha256sum scripts/evaluate_recurrent_formal_protocols.py | cut -d' ' -f1)" = 262370592e018641b99d6282858d8990b38314c05bb6e3bbc645823834cf73ed
test "$(sha256sum scripts/quarantine_incompatible_checkpoint_receipts.py | cut -d' ' -f1)" = 3ce649ee6b507d12add27d15c979a62c403724438d245a41e46f3548153f946e

stage teacher_and_student_provenance
export STUDENTMODELING_EXTERNAL_STATE_TEACHER_ADAPTER="$TEACHER_ROOT/model"
export STUDENTMODELING_EXTERNAL_STATE_TEACHER_SHA256="$(python - "$TEACHER_ROOT" "$STUDENT_ROOT" <<'PY'
import hashlib,json,pathlib,sys
teacher=pathlib.Path(sys.argv[1]); student=pathlib.Path(sys.argv[2])
def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
 return h.hexdigest()
tp=json.load(open(teacher/'metrics/publish_receipt.json')); tg=json.load(open(teacher/'metrics/teacher_gate.json'))
pg=json.load(open(teacher/'metrics/teacher_provenance_gate.json'))
si=json.load(open(student/'metrics/immutable_receipt.json'))
teacher_hash=digest(teacher/'model/adapter_model.safetensors')
student_hash=digest(student/'model/adapter_model.safetensors')
state_hash=digest(student/'model/recurrent_state.pt')
assert tp['status']=='complete' and tp['adapter_sha256']==teacher_hash,tp
assert tg['stage']=='state8_independent_full_history_teacher_gate_v1' and tg['pass'] is True,tg
assert tg['support']==1494 and tg['exact_match_is_authoritative'] is False,tg
assert pg['stage']=='state8_clean_prompt_teacher_provenance_gate_v2' and pg['status']=='complete',pg
assert pg['teacher_gate_pass'] is True and pg['teacher_adapter_sha256']==teacher_hash,pg
assert pg['evaluation_input_sha256']=='78850c5434f24c3072b7ab89370400347955686e9aac5b99aadb69ae19c7cfa5',pg
assert pg['clean_prompt_sha256']=='c35fd46e0cfbccd9141eccb524992bd5452416d32720686f74ad9373f57de314',pg
assert pg['changed_prompt_rows']==600 and pg['legacy_answer_adjacent_contracts']==0,pg
# The qualified teacher was completed by the clean 16-way accelerator (the
# earlier four-way attempt was superseded).  Pin that provenance so Stage B
# cannot silently consume a partial or legacy teacher evaluation.
assert pg['evaluation_shards']==16 and pg['evaluation_rows']==1494,pg
assert si['status']=='complete' and si['state_shape']==[64,8,5120],si
assert si['sha256']=={'adapter_model':student_hash,'recurrent_state':state_hash},(si,student_hash,state_hash)
print(teacher_hash)
PY
)"
test "${#STUDENTMODELING_EXTERNAL_STATE_TEACHER_SHA256}" = 64

stage local_unit_tests
python -m pytest -q tests/test_state8_external_teacher_curriculum.py \
  tests/test_materialize_recurrent_index_manifest.py \
  tests/test_state8_natural_curriculum_view.py \
  tests/test_subset_state8_causal_twins.py \
  tests/test_train_recurrent_student_m4.py \
  tests/test_layerwise_recurrent_student_model.py \
  tests/test_distributed_epoch.py

stage data_materialization
exec 8>"/tmp/state8-qualified-teacher-${STATE8_REPLAY_VARIANT}-data.lock"; flock -x 8
cache_ready=0
if [[ "$STATE8_CURRICULUM_NAMESPACE" == taxonomy_semantic_v1 ]] && \
   python scripts/state8_curriculum_data_cache.py verify \
     --cache-root "$DATA_CACHE_ROOT" --variant "$STATE8_REPLAY_VARIANT" \
     > /tmp/state8-curriculum-cache-paths.json 2>/tmp/state8-curriculum-cache-verify.log; then
  cache_ready=1
  # Keep the historical command-line identities while reusing the immutable
  # cache.  Exact-resume validation intentionally includes input paths; passing
  # the cache's physical paths directly would make byte-identical data look
  # like scientific recipe drift.  Symlinks preserve the original logical
  # paths and all hashes below still bind the cache contents.
  rm -f "$VALID_JSONL" "$RECURRENT_EVAL" "$SUBSET" "$SUBSET.receipt.json"
  rm -rf "$CAUSAL_DATA" "$CAUSAL_SUBSET"
  ln -s "$DATA_CACHE_ROOT/common/valid.jsonl" "$VALID_JSONL"
  ln -s "$DATA_CACHE_ROOT/common/formal_eval.jsonl" "$RECURRENT_EVAL"
  ln -s "$DATA_CACHE_ROOT/$STATE8_REPLAY_VARIANT/train5000.jsonl" "$SUBSET"
  ln -s "$DATA_CACHE_ROOT/$STATE8_REPLAY_VARIANT/train5000.jsonl.receipt.json" \
    "$SUBSET.receipt.json"
  ln -s "$DATA_CACHE_ROOT/common/causal_same_rule" "$CAUSAL_DATA"
  ln -s "$DATA_CACHE_ROOT/common/causal_subset250" "$CAUSAL_SUBSET"
  stage data_cache_verified
elif [[ ! -f "/tmp/state8-qualified-teacher-${STATE8_REPLAY_VARIANT}-data.complete" ]]; then
  rm -rf "$RAW_SPLIT" "$RECURRENT_SPLIT" "$(dirname "$RECURRENT_EVAL")" "$CAUSAL_BASE" "$CAUSAL_DATA" "$CAUSAL_SUBSET"
  mkdir -p "$RAW_SPLIT" "$RECURRENT_SPLIT" "$(dirname "$RECURRENT_EVAL")" "$CAUSAL_BASE" "$CAUSAL_DATA" "$CAUSAL_SUBSET"
  python scripts/split_full_mix_by_original_split.py --input-jsonl "$RAW_FULL" --output-dir "$RAW_SPLIT"
  python scripts/build_unified_recurrent_manifest.py \
    --train-jsonl "$RAW_SPLIT/train.jsonl" --valid-jsonl "$RAW_SPLIT/valid.jsonl" \
    --output-dir "$RECURRENT_SPLIT" --max-updates 4 --seed 20260730
  python scripts/merge_recurrent_full_corpus.py \
    --train-jsonl "$RECURRENT_SPLIT/train.jsonl" --valid-jsonl "$RECURRENT_SPLIT/valid.jsonl" \
    --output-jsonl "$FULL_JSONL" --manifest "$RECURRENT_SPLIT/final_fit_manifest.json"
  python scripts/build_recurrent_evaluation_manifest.py \
    --input-jsonl "$RAW_EVAL" --output-jsonl "$RECURRENT_EVAL" \
    --audit-json /tmp/state8_qualified_teacher_formal_u4/audit.json --max-updates 4 --seed 20260730
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
    --output-dir "$CAUSAL_DATA" --seed 20260804 --max-updates 4
  python scripts/subset_state8_causal_twins.py \
    --input-jsonl "$CAUSAL_DATA/train.jsonl" --output-jsonl "$CAUSAL_SUBSET/train.jsonl" \
    --twins 125 --seed 42
  cp "$CAUSAL_DATA/valid.jsonl" "$CAUSAL_SUBSET/valid.jsonl"
  touch "/tmp/state8-qualified-teacher-${STATE8_REPLAY_VARIANT}-data.complete"
fi
flock -u 8
if (( cache_ready == 0 )); then
  test "$(sha256sum "$FULL_JSONL" | cut -d' ' -f1)" = 9961f746cb17e6c314e182d3453b6157022fb48b8de44d2e68bc8e87a1131ccd
fi
test "$(sha256sum "$VALID_JSONL" | cut -d' ' -f1)" = 78bfce75576dc3873cbb6459669a952adeedb8b77bda15f281af9519e2506d14
test "$(sha256sum "$RECURRENT_EVAL" | cut -d' ' -f1)" = a368cefb0c49cb6a305ad1a892808361fba2d44d555a0a28ea64e56babe32525
test "$(sha256sum "$SUBSET" | cut -d' ' -f1)" = "$SUBSET_SHA256"
# These hashes bind the augmented rows produced by the pinned augmenter above.
# The v2 rows retain explicit recipient/same-rule trajectories for the State8
# causal curriculum; accepting the legacy v1 hashes would silently drop that
# control while leaving the objective name unchanged.
test "$(sha256sum "$CAUSAL_DATA/train.jsonl" | cut -d' ' -f1)" = e2a57273bd4c49229efc8cec522fc221e460174c388c43e2b0d4888cc5079225
test "$(sha256sum "$CAUSAL_DATA/valid.jsonl" | cut -d' ' -f1)" = 8bd0dc3ec367ce30e5e5da3526681c21f852ba991394cac423682cd721f48c8e
python - "$CAUSAL_SUBSET/train.jsonl.receipt.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); assert x['pairs']==250 and x['twins']==125,x
assert x['role_counts']=={'recipient_native':125,'same_rule_native':125},x
PY
python - "$MANIFEST" "$SUBSET.receipt.json" <<'PY'
import json,os,sys
m=json.load(open(sys.argv[1])); r=json.load(open(sys.argv[2]))
expected=json.loads(os.environ['EXPECTED_TASK_COUNTS'])
assert m['segment_rows']==5000 and m['task_counts']==expected,(m,expected)
assert m['formal_example_id_overlap']==0 and r['formal_example_id_overlap']==0,(m,r)
assert r['manifest_sha256']==os.environ['MANIFEST_SHA256'],r
assert r['output_sha256']==os.environ['SUBSET_SHA256'],r
PY

stage model_copy
exec 7>/tmp/state8-qualified-teacher-model.lock; flock -x 7
if [[ ! -f /tmp/state8-qualified-teacher-model.complete ]]; then
  rm -rf "$LOCAL_MODEL"; mkdir -p "$LOCAL_MODEL"
  python scripts/copy_model_parallel.py --source /mnt/amulet/studentmodeling/models/Qwen3-32B --destination "$LOCAL_MODEL" --workers 16
  touch /tmp/state8-qualified-teacher-model.complete
fi
flock -u 7
test -f "$LOCAL_MODEL/model.safetensors.index.json"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

if [[ "$STATE8_FINALIZE_ONLY" == 1 ]]; then
  export CUDA_VISIBLE_DEVICES=0 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0
else
  export CUDA_VISIBLE_DEVICES="${OMPI_COMM_WORLD_LOCAL_RANK:?MPI local rank is required}"
  export RANK="${OMPI_COMM_WORLD_RANK:?MPI rank is required}"
  export WORLD_SIZE="${OMPI_COMM_WORLD_SIZE:?MPI world size is required}"
  export LOCAL_RANK=0
  test "$WORLD_SIZE" = 4
fi

stage resume_selection
exec 5>"/tmp/state8-qualified-teacher-${STATE8_REPLAY_VARIANT}-checkpoint-quarantine.lock"
flock -x 5
if [[ ! -f "$RUN_DIR/incompatible_checkpoint_quarantine.json" ]]; then
  python scripts/quarantine_incompatible_checkpoint_receipts.py \
    --run-dir "$RUN_DIR" --expected-planned-optimizer-steps "$TOTAL_ROUNDS" \
    --audit "$RUN_DIR/incompatible_checkpoint_quarantine.json"
fi
flock -u 5
resume_checkpoint=""
selected="$(python scripts/select_latest_valid_distributed_checkpoint.py \
  --run-dir "$RUN_DIR" --expected-world-size 4 --max-round "$TOTAL_ROUNDS" \
  --expected-planned-optimizer-steps "$TOTAL_ROUNDS" \
  "${RESUME_CADENCE_ARGS[@]}" \
  --probe-expected-cadence)"
if [[ -n "$selected" ]]; then
  resume_key="$(python - "$selected" <<'PY'
import hashlib,os,sys
p=os.path.realpath(sys.argv[1]); s=os.stat(p)
print(hashlib.sha256(f'{p}:{s.st_size}:{s.st_mtime_ns}'.encode()).hexdigest())
PY
)"
  local_resume="/tmp/state8-qualified-teacher-${STATE8_REPLAY_VARIANT}-resume/$resume_key"
  mkdir -p "$local_resume"
  exec 6>"$local_resume/stage.lock"; flock -x 6
  python scripts/stage_distributed_resume_bundle.py --source-state "$selected" --destination "$local_resume"
  flock -u 6
  resume_checkpoint="$local_resume/recurrent_state.pt"
fi
if [[ -n "$resume_checkpoint" ]]; then
  # Preserve the original initialization provenance in the exact-resume
  # config.  The checkpoint still overwrites these initial weights/state and
  # remains the sole source of optimizer, scheduler, RNG, order, and history.
  INIT_ARGS=(--init-recurrent-state "$STUDENT_ROOT/model/recurrent_state.pt" \
             --resume-from-checkpoint "$resume_checkpoint")
else
  INIT_ARGS=(--init-recurrent-state "$STUDENT_ROOT/model/recurrent_state.pt")
fi
if [[ "$STATE8_FINALIZE_ONLY" == 1 ]]; then
  test -n "$resume_checkpoint"
  python - "$resume_checkpoint" "$TOTAL_ROUNDS" <<'PY'
import sys,torch
payload=torch.load(sys.argv[1],map_location='cpu',weights_only=False)
assert payload.get('distributed') is True,payload
assert int(payload.get('world_size',-1))==4,payload
assert int(payload.get('round',-1))==int(sys.argv[2]),payload
PY
  EXECUTION_ARGS=(--validation-only-from-distributed-checkpoint --finalize-complete-distributed-epoch)
else
  EXECUTION_ARGS=(--distributed-mpi)
fi

stage training
python -u "$TRAIN_ENTRYPOINT" \
  --model "$LOCAL_MODEL" --init-adapter "$STUDENT_ROOT/model" "${INIT_ARGS[@]}" \
  --state-architecture layerwise --layerwise-operation-checkpointing \
  --layerwise-transition-rank 256 --state-token-count 8 \
  --train-jsonl "$SUBSET" --valid-jsonl "$VALID_JSONL" \
  --causal-train-jsonl "$CAUSAL_SUBSET/train.jsonl" --causal-valid-jsonl "$CAUSAL_SUBSET/valid.jsonl" \
  --output-dir "$RUN_DIR" --state-token-alias '<|vision_pad|>' \
  --history-dropout-curriculum --dual-path-curriculum \
  --online-kd-weight "$STATE8_ONLINE_KD_WEIGHT" --online-kd-temperature 1.0 --online-kd-topk 256 \
  --max-length 1536 --direct-offload-min-tokens 1 \
  --learning-rate 1e-6 --lr-scheduler cosine --warmup-ratio 0.03 \
  --generation-weight 1.0 --behavior-kd-weight 0.0 --consistency-weight 0.0 --direct-replay-weight 0.0 \
  --causal-weight 2.0 --causal-margin 0.5 --causal-preference-reduction sum \
  --causal-objective-mode specificity_triplet --causal-repeat-fraction 0.05 --normalize-causal-exposure \
  --gradient-accumulation-steps 1 --eval-limit 300 --causal-eval-limit 165 \
  --save-steps "$STATE8_SAVE_STEPS" --checkpoint-retention 2 \
  "${EXECUTION_ARGS[@]}" --distributed-world-size 4 --seed "$TRAIN_SEED"

if [[ "${OMPI_COMM_WORLD_RANK:-0}" != 0 ]]; then exit 0; fi
stage posttrain_audit
test -f "$RUN_DIR/completion.json"
test -f "$RUN_DIR/final_adapter/adapter_model.safetensors"
test -f "$RUN_DIR/recurrent_state.pt"
test -f "$RUN_DIR/state8_external_teacher_curriculum.json"
if [[ "$STATE8_CURRICULUM_NAMESPACE" == taxonomy_semantic_v1 ]]; then
  test -f "$RUN_DIR/taxonomy_teacher_curriculum.json"
fi
python - "$RUN_DIR" "$STUDENTMODELING_EXTERNAL_STATE_TEACHER_SHA256" "$STATE8_REPLAY_VARIANT" <<'PY'
import collections,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); teacher_hash=sys.argv[2]; variant=sys.argv[3]
c=json.load(open(root/'completion.json')); cfg=json.load(open(root/'run_config.json'))
h=json.load(open(root/'trainer_history.json')); r=json.load(open(root/'state8_external_teacher_curriculum.json'))
assert c['status']=='complete' and c['diagnostic'] is False,c
assert c['micro_steps']==5000 and c['real_examples_seen_once'] is True,c
assert c['all_causal_pairs_covered'] is True and c['causal_exposure_count_matches_plan'] is True,c
assert cfg['state_token_count']==8 and cfg['state_architecture']=='layerwise',cfg
assert cfg['direct_offload_min_tokens']==1,cfg
assert cfg['external_state_teacher_sha256']==teacher_hash,cfg
assert float(cfg['online_kd_weight'])==float(__import__('os').environ['STATE8_ONLINE_KD_WEIGHT']),cfg
assert cfg['consistency_weight']==0.0 and cfg['direct_replay_weight']==0.0,cfg
assert cfg['final_inference_route']=='strict_state_only',cfg
assert cfg['state8_natural_view_sha256']=='98c961180d2428b2c68ef9399996ed991d5bd34120a3046b4f56fb9ba4e09682',cfg
assert cfg['teacher_prompt_provenance_stage']=='state8_clean_prompt_teacher_provenance_gate_v2',cfg
assert cfg['teacher_evaluation_input_sha256']=='78850c5434f24c3072b7ab89370400347955686e9aac5b99aadb69ae19c7cfa5',cfg
assert cfg['teacher_clean_prompt_sha256']=='c35fd46e0cfbccd9141eccb524992bd5452416d32720686f74ad9373f57de314',cfg
assert cfg['legacy_answer_adjacent_contracts']==0,cfg
assert cfg['trajectory_teacher_scope']=='none_unverified_carrier_trajectory_excluded',cfg
assert len(h)==5000 and [row['global_position'] for row in h]==list(range(5000)),h[:2]
counts=collections.Counter(float(row['history_dropout']) for row in h)
assert counts=={0.0:1000,0.25:1000,0.5:1000,0.75:1000,1.0:1000},counts
assert r['teacher_gate_required'] is True and r['teacher_trainable'] is False,r
taxonomy=(root/'taxonomy_teacher_curriculum.json')
taxonomy_receipt=json.load(open(taxonomy)) if taxonomy.is_file() else None
if taxonomy_receipt is not None:
 assert cfg['taxonomy_label_contract']=='source_aware_no_answer_labels_v1',cfg
 assert cfg['taxonomy_candidates_visible_to_decoder'] is False,cfg
 assert cfg['semantic_judge_authoritative'] is True,cfg
 assert cfg['exact_label_metric_authoritative'] is False,cfg
 assert float(cfg['label_auxiliary_weight'])==4.0,cfg
 assert taxonomy_receipt['semantic_judge_authoritative'] is True,taxonomy_receipt
 assert taxonomy_receipt['exact_label_metric_authoritative'] is False,taxonomy_receipt
receipt={'status':'complete','stage':'state8_qualified_teacher_curriculum_v1',
 'replay_variant':variant,'micro_steps':5000,'state_shape':[64,8,5120],'history_dropout_counts':dict(counts),
 'online_kd_weight':float(cfg['online_kd_weight']),
 'direct_offload_min_tokens':int(cfg['direct_offload_min_tokens']),
 'teacher_adapter_sha256':teacher_hash,'raw_history_visible_to_decoder':False,
 'teacher_prompt_provenance_stage':'state8_clean_prompt_teacher_provenance_gate_v2',
 'teacher_evaluation_input_sha256':'78850c5434f24c3072b7ab89370400347955686e9aac5b99aadb69ae19c7cfa5',
 'teacher_clean_prompt_sha256':'c35fd46e0cfbccd9141eccb524992bd5452416d32720686f74ad9373f57de314',
 'legacy_answer_adjacent_contracts':0,
 'label_auxiliary_weight':float(cfg.get('label_auxiliary_weight',1.0)),
 'taxonomy_label_contract':cfg.get('taxonomy_label_contract'),
 'semantic_judge_authoritative':bool(cfg.get('semantic_judge_authoritative',False)),
 'exact_label_metric_authoritative':bool(cfg.get('exact_label_metric_authoritative',True)),
 'persistent_history_kv_cache':False,'adapter_sha256':c['sha256']['adapter_model'],
 'recurrent_state_sha256':c['sha256']['recurrent_state']}
(root/'qualified_teacher_receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
PY
cp -r "$RUN_DIR/final_adapter/." "$AMLT_OUTPUT_DIR/model/"
cp "$RUN_DIR/recurrent_state.pt" "$AMLT_OUTPUT_DIR/model/recurrent_state.pt"
for name in completion.json run_config.json training_metrics.json validation_metrics.json causal_validation_metrics.json state8_external_teacher_curriculum.json qualified_teacher_receipt.json; do
  cp "$RUN_DIR/$name" "$AMLT_OUTPUT_DIR/metrics/$name"
done
if [[ -f "$RUN_DIR/taxonomy_teacher_curriculum.json" ]]; then
  cp "$RUN_DIR/taxonomy_teacher_curriculum.json" "$AMLT_OUTPUT_DIR/metrics/taxonomy_teacher_curriculum.json"
fi
cp "$SUBSET.receipt.json" "$AMLT_OUTPUT_DIR/metrics/training_subset_receipt.json"
cp "$CAUSAL_DATA/augmentation_receipt.json" "$AMLT_OUTPUT_DIR/metrics/causal_same_rule_augmentation.json"
cp "$CAUSAL_SUBSET/train.jsonl.receipt.json" "$AMLT_OUTPUT_DIR/metrics/causal_subset_receipt.json"

stage formal_screen
python -u "$FORMAL_EVALUATOR" \
  --model "$LOCAL_MODEL" --adapter "$RUN_DIR/final_adapter" \
  --recurrent-state "$RUN_DIR/recurrent_state.pt" --state-architecture layerwise \
  --eval-jsonl "$RECURRENT_EVAL" --output-dir "$AMLT_OUTPUT_DIR/eval/formal_screen" \
  --state-token-alias '<|vision_pad|>' --max-samples-per-group 25 \
  --max-length 1536 --max-new-tokens 128 --state-batch-size 2 \
  --candidate-row-batch-size 2 --seed 42 --save-unredacted-predictions --t3-natural-schema
python - "$AMLT_OUTPUT_DIR/eval/formal_screen/metrics.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['state_shape']==[64,8,5120],x
assert x['raw_history_visible_to_decoder'] is False,x
assert x['persistent_history_kv_cache'] is False,x
assert x['unsharded_filtered_rows']==21676,x
assert x['sampled_rows']==487,x
PY

stage immutable_publish
rm -rf "$STABLE_ROOT/model" "$STABLE_ROOT/metrics" "$STABLE_ROOT/eval/formal_screen"
mkdir -p "$STABLE_ROOT/model" "$STABLE_ROOT/metrics" "$STABLE_ROOT/eval/formal_screen"
cp -r "$RUN_DIR/final_adapter/." "$STABLE_ROOT/model/"
cp "$RUN_DIR/recurrent_state.pt" "$STABLE_ROOT/model/recurrent_state.pt"
for name in completion.json run_config.json training_metrics.json validation_metrics.json causal_validation_metrics.json state8_external_teacher_curriculum.json qualified_teacher_receipt.json; do
  cp "$RUN_DIR/$name" "$STABLE_ROOT/metrics/$name"
done
if [[ -f "$RUN_DIR/taxonomy_teacher_curriculum.json" ]]; then
  cp "$RUN_DIR/taxonomy_teacher_curriculum.json" "$STABLE_ROOT/metrics/taxonomy_teacher_curriculum.json"
fi
cp "$AMLT_OUTPUT_DIR/eval/formal_screen/metrics.json" "$STABLE_ROOT/eval/formal_screen/metrics.json"
cp "$AMLT_OUTPUT_DIR/eval/formal_screen/predictions.jsonl" "$STABLE_ROOT/eval/formal_screen/predictions.jsonl"
python - "$STABLE_ROOT" "$STATE8_REPLAY_VARIANT" "$IMMUTABLE_STAGE" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); variant=sys.argv[2]; immutable_stage=sys.argv[3]
c=json.load(open(root/'metrics/completion.json'))
q=json.load(open(root/'metrics/qualified_teacher_receipt.json'))
def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
 return h.hexdigest()
observed={'adapter_model':digest(root/'model/adapter_model.safetensors'),
          'recurrent_state':digest(root/'model/recurrent_state.pt')}
assert c['status']=='complete' and c['diagnostic'] is False,c
assert c['micro_steps']==5000 and c['real_examples_seen_once'] is True,c
assert q['stage']=='state8_qualified_teacher_curriculum_v1' and q['replay_variant']==variant,q
assert q['online_kd_weight'] in {0.5,1.0},q
assert q['direct_offload_min_tokens']==1,q
assert q['teacher_adapter_sha256'] and len(q['teacher_adapter_sha256'])==64,q
assert q['teacher_prompt_provenance_stage']=='state8_clean_prompt_teacher_provenance_gate_v2',q
assert q['teacher_evaluation_input_sha256']=='78850c5434f24c3072b7ab89370400347955686e9aac5b99aadb69ae19c7cfa5',q
assert q['teacher_clean_prompt_sha256']=='c35fd46e0cfbccd9141eccb524992bd5452416d32720686f74ad9373f57de314',q
assert q['legacy_answer_adjacent_contracts']==0,q
if q.get('taxonomy_label_contract') is not None:
 assert q['taxonomy_label_contract']=='source_aware_no_answer_labels_v1',q
 assert q['semantic_judge_authoritative'] is True,q
 assert q['exact_label_metric_authoritative'] is False,q
 assert float(q['label_auxiliary_weight'])==4.0,q
assert observed=={'adapter_model':q['adapter_sha256'],'recurrent_state':q['recurrent_state_sha256']},(observed,q)
receipt={'status':'complete','stage':immutable_stage,'replay_variant':variant,
 'state_shape':[64,8,5120],'micro_steps':5000,
 'remote_root':str(root),
 'teacher_adapter_sha256':q['teacher_adapter_sha256'],'sha256':observed,
 'teacher_prompt_provenance_stage':q['teacher_prompt_provenance_stage'],
 'teacher_evaluation_input_sha256':q['teacher_evaluation_input_sha256'],
 'teacher_clean_prompt_sha256':q['teacher_clean_prompt_sha256'],
 'legacy_answer_adjacent_contracts':q['legacy_answer_adjacent_contracts'],
 'label_auxiliary_weight':q.get('label_auxiliary_weight',1.0),
 'taxonomy_label_contract':q.get('taxonomy_label_contract'),
 'semantic_judge_authoritative':q.get('semantic_judge_authoritative',False),
 'exact_label_metric_authoritative':q.get('exact_label_metric_authoritative',True),
 'raw_history_visible_to_decoder':False,'persistent_history_kv_cache':False,
 'formal_screen_rows':487}
(root/'metrics/immutable_receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
PY
cp "$STABLE_ROOT/metrics/immutable_receipt.json" \
  "$AMLT_OUTPUT_DIR/metrics/immutable_receipt.json"
