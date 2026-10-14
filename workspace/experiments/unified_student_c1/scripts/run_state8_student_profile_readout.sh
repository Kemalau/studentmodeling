#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "STATE8_STUDENT_PROFILE_READOUT_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

export VENV_ROOT="${VENV_ROOT:-/tmp/studentmodeling-state8-profile-readout-venv}"
export PATH="$VENV_ROOT/bin:$PATH"
python(){ "$VENV_ROOT/bin/python" "$@"; }
export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/hf_cache HF_HUB_CACHE=/tmp/hf_cache/hub HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRANSFORMERS_VERBOSITY=error

test "$(sha256sum scripts/build_state8_student_profile_readout.py | cut -d' ' -f1)" = 19b8fc2e5875b3ca55cd7d553da597996081f530c745ecdfdac44a7882a8022b
test "$(sha256sum scripts/train_state8_student_profile_readout.py | cut -d' ' -f1)" = 77dbccded1f900f70c5a1e7531bdc6f90f68723ff148fe99f40468eea10c30f0
test "$(sha256sum scripts/train_state8_profile_ownership_repair.py | cut -d' ' -f1)" = 2ef6e546dc3666ef52cbb016ffddba6e6f079f11bbbebba127cd14d8c8e87f49
test "$(sha256sum scripts/train_recurrent_student_m4.py | cut -d' ' -f1)" = 18fdc7fd9c8fef46801cebda0ac8b656d7820d9bf970eee97e27a3173117f9d7
test "$(sha256sum scripts/evaluate_state8_student_profile_readout.py | cut -d' ' -f1)" = 0fcbb63cfbf023c5025753cc95b826fe0ac78d9a8335f39239ac59446adfca42
test "$(sha256sum scripts/evaluate_state8_reusable_interface.py | cut -d' ' -f1)" = d74c6859912e00b0d2ce116c557206e3e8c000a0e890cbc1eb6f7e6ae35d78aa
test "$(sha256sum scripts/evaluate_dbe_student_profile_transport_v3.py | cut -d' ' -f1)" = 2406e41ba64273cb186d1e0af3d41c767bb931f1d631bf30caaaf7792010bbda
test "$(sha256sum tests/test_build_state8_student_profile_readout.py | cut -d' ' -f1)" = 861a88fba36c7307e348e303e15e81eda3b3a83ed6519a3f348787b3c2572b98

MODEL_REMOTE=/mnt/amulet/studentmodeling/models/Qwen3-32B
MODEL_LOCAL=/tmp/Qwen3-32B
SOURCE_RUN=/mnt/amulet/studentmodeling/checkpoints/state8_label_first_repair_seed8501_v1
SOURCE_ADAPTER="$SOURCE_RUN/final_adapter"
SOURCE_STATE="$SOURCE_RUN/recurrent_state.pt"
EXPECTED_ADAPTER=ff247c06229c78c8b873f1717d100bfc7f99b2e2b953c7c428228961c8db80df
EXPECTED_STATE=7648c37fc508c0a594b18bfddd848a591756efecf1eef73f125f49497d45890b
PROFILE_SOURCE=manifests/dbe_student_profile_v1
RETENTION_SOURCE=/mnt/amulet/studentmodeling/full_trainplusvalid_no_t4_strict_t6/train.jsonl
RETENTION_RAW_SPLIT=/tmp/state8_profile_retention_raw
RETENTION_TRAIN_RECURRENT=/tmp/state8_profile_retention_recurrent
DATA=/tmp/state8_student_profile_readout_v1
RUN_DIR="${PROFILE_READOUT_RUN_DIR:-/mnt/amulet/studentmodeling/checkpoints/state8_future_profile_ownership_seed8801_v3}"
RESUME_FROM="${PROFILE_READOUT_RESUME_FROM:-}"
EVAL_ONLY="${PROFILE_READOUT_EVAL_ONLY:-0}"
STABLE="${PROFILE_READOUT_STABLE_DIR:-/mnt/amulet/studentmodeling/diagnostics/state8_future_profile_ownership_seed8801_v3}"
WORLD_SIZE="${PROFILE_READOUT_DISTRIBUTED_WORLD_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${PROFILE_READOUT_GRADIENT_ACCUMULATION_STEPS:-4}"
DIAGNOSTIC_STOP_ROUND="${PROFILE_READOUT_DIAGNOSTIC_STOP_ROUND:-}"
DIAGNOSTIC_MAX_MICRO_STEPS="${PROFILE_READOUT_DIAGNOSTIC_MAX_MICRO_STEPS:-}"
EVAL_LIMIT="${PROFILE_READOUT_EVAL_LIMIT:-184}"
CAUSAL_EVAL_LIMIT="${PROFILE_READOUT_CAUSAL_EVAL_LIMIT:-50}"
MPI_RANK="${OMPI_COMM_WORLD_RANK:-0}"
OUT="$AMLT_OUTPUT_DIR/profile_readout"
RETENTION_EVAL=/mnt/amulet/studentmodeling/formal_evaluation_v1/test.jsonl
RETENTION_RECURRENT=/tmp/state8_profile_retention_eval/test.jsonl
RETENTION_OUT="$AMLT_OUTPUT_DIR/retention_eval"
INTERFACE_FEATURES="$AMLT_OUTPUT_DIR/reusable_interface_features"
INTERFACE_OUT="$AMLT_OUTPUT_DIR/reusable_interface"

for path in "$SOURCE_ADAPTER/adapter_model.safetensors" "$SOURCE_STATE" \
  "$SOURCE_RUN/completion.json" "$RETENTION_SOURCE" "$RETENTION_EVAL" \
  "$PROFILE_SOURCE/train.jsonl" "$PROFILE_SOURCE/valid.jsonl" "$PROFILE_SOURCE/test.jsonl"; do
  test -f "$path"
done
test "$(sha256sum "$SOURCE_ADAPTER/adapter_model.safetensors" | cut -d' ' -f1)" = "$EXPECTED_ADAPTER"
test "$(sha256sum "$SOURCE_STATE" | cut -d' ' -f1)" = "$EXPECTED_STATE"
test "$(df --output=avail /tmp | tail -1)" -ge 104857600

# All MPI ranks on an 80G4 node share /tmp.  Prepare immutable inputs and the
# 61-GiB model exactly once under a node-local lock, then let every rank reuse
# them.  The receipt is written last so interrupted preparation cannot pass.
PREPARED=/tmp/state8-profile-ownership-prepared-v3
exec 9>/tmp/state8-profile-ownership-prepared-v3.lock
flock -x 9
if [[ ! -f "$PREPARED" ]]; then
  python -m pytest -q \
    tests/test_build_state8_student_profile_readout.py \
    tests/test_evaluate_state8_student_profile_readout.py \
    tests/test_train_state8_profile_ownership_repair.py \
    tests/test_train_state8_semantic_causal_repair.py \
    tests/test_train_recurrent_student_m4.py \
    tests/test_distributed_epoch.py

  rm -rf "$DATA" "$RETENTION_RAW_SPLIT" "$RETENTION_TRAIN_RECURRENT"
  mkdir -p "$RETENTION_RAW_SPLIT" "$RETENTION_TRAIN_RECURRENT"
  python scripts/split_full_mix_by_original_split.py \
    --input-jsonl "$RETENTION_SOURCE" --output-dir "$RETENTION_RAW_SPLIT"
  python scripts/build_unified_recurrent_manifest.py \
    --train-jsonl "$RETENTION_RAW_SPLIT/train.jsonl" \
    --valid-jsonl "$RETENTION_RAW_SPLIT/valid.jsonl" \
    --output-dir "$RETENTION_TRAIN_RECURRENT" --max-updates 4 --seed 20260730
  python scripts/build_state8_student_profile_readout.py \
    --profile-manifest-dir "$PROFILE_SOURCE" \
    --retention-train-jsonl "$RETENTION_TRAIN_RECURRENT/train.jsonl" \
    --retention-valid-jsonl "$RETENTION_TRAIN_RECURRENT/valid.jsonl" \
    --output-dir "$DATA" --retention-train 337 --retention-valid 100 \
    --causal-train-pairs 300 --causal-valid-pairs 50 \
    --ownership-min-train-kc-support "${STATE8_PROFILE_OWNERSHIP_MIN_KC_TRAIN_SUPPORT:-1}" \
    --ownership-negative-count "${PROFILE_READOUT_OWNERSHIP_NEGATIVE_COUNT:-1}" \
    --ownership-scope "${PROFILE_READOUT_OWNERSHIP_SCOPE:-all}" \
    --seed 20260817

  rm -rf "$MODEL_LOCAL"; mkdir -p "$MODEL_LOCAL"
  python scripts/copy_model_parallel.py --source "$MODEL_REMOTE" \
    --destination "$MODEL_LOCAL" --workers 16
  test -f "$MODEL_LOCAL/model.safetensors.index.json"
  touch "$PREPARED.partial"
  mv "$PREPARED.partial" "$PREPARED"
fi
flock -u 9

python - "$DATA/receipt.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['status']=='complete' and x['student_disjoint'] is True,x
assert x['future_targets_used_for_training'] is True,x
assert x['future_targets_visible_to_decoder'] is False,x
assert x['rows']=={'causal_train.jsonl':300,'causal_valid.jsonl':50,'test_profiles.jsonl':88,'train.jsonl':1000,'valid.jsonl':184},x
assert x['raw_history_visible_to_decoder'] is False and x['student_id_visible_to_decoder'] is False,x
PY
test -f "$MODEL_LOCAL/model.safetensors.index.json"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

TRAIN_RECOVERY_ARGS=()
TRAIN_DISTRIBUTED_ARGS=()
TRAIN_DIAGNOSTIC_ARGS=()
if (( WORLD_SIZE > 1 )); then
  test "$GRADIENT_ACCUMULATION_STEPS" -eq 1
  test "${OMPI_COMM_WORLD_SIZE:?MPI world size is required}" -eq "$WORLD_SIZE"
  test -n "${OMPI_COMM_WORLD_LOCAL_RANK:?MPI local rank is required}"
  TRAIN_DISTRIBUTED_ARGS=(--distributed-mpi --distributed-world-size "$WORLD_SIZE")
else
  test "$WORLD_SIZE" -eq 1
  test "$GRADIENT_ACCUMULATION_STEPS" -eq 4
fi
if [[ "$EVAL_ONLY" == 1 ]]; then
  test "$WORLD_SIZE" -eq 1
  test -z "$RESUME_FROM"
  test -f "$RUN_DIR/final_adapter/adapter_model.safetensors"
  test -f "$RUN_DIR/recurrent_state.pt"
  test -f "$RUN_DIR/completion.json"
  test -f "$RUN_DIR/student_profile_readout_repair.json"
  rm -rf "$OUT"
elif [[ -n "$RESUME_FROM" ]]; then
  test "$RESUME_FROM" = "$RUN_DIR/checkpoint-1000/recurrent_state.pt"
  test -f "$RESUME_FROM"
  test -f "$RUN_DIR/checkpoint-1000/adapter/adapter_model.safetensors"
  TRAIN_RECOVERY_ARGS=(--resume-from-checkpoint "$RESUME_FROM")
  rm -rf "$OUT"
else
  if (( WORLD_SIZE > 1 )); then
    PREPARED="$RUN_DIR/.distributed_prepared"
    if [[ "$MPI_RANK" == 0 ]]; then
      rm -rf "$RUN_DIR" "$OUT"
      mkdir -p "$RUN_DIR" "$OUT"
      touch "$PREPARED.partial"
      mv "$PREPARED.partial" "$PREPARED"
    else
      for _ in $(seq 1 600); do
        [[ -f "$PREPARED" ]] && break
        sleep 1
      done
      test -f "$PREPARED"
    fi
  else
    rm -rf "$RUN_DIR" "$OUT"
  fi
fi
if [[ -n "$DIAGNOSTIC_STOP_ROUND" ]]; then
  test "$WORLD_SIZE" -gt 1
  test "$DIAGNOSTIC_STOP_ROUND" -gt 0
  TRAIN_DIAGNOSTIC_ARGS=(--diagnostic-stop-after-round "$DIAGNOSTIC_STOP_ROUND")
fi
if [[ -n "$DIAGNOSTIC_MAX_MICRO_STEPS" ]]; then
  test "$WORLD_SIZE" -eq 1
  test -z "$DIAGNOSTIC_STOP_ROUND"
  test "$DIAGNOSTIC_MAX_MICRO_STEPS" -gt 0
  TRAIN_DIAGNOSTIC_ARGS=(--diagnostic-max-micro-steps "$DIAGNOSTIC_MAX_MICRO_STEPS")
fi
mkdir -p "$RUN_DIR" "$OUT"
if [[ "$EVAL_ONLY" != 1 ]]; then
  TRAIN_ENTRYPOINT="${PROFILE_READOUT_TRAIN_ENTRYPOINT:-scripts/train_state8_profile_ownership_repair.py}"
  case "$TRAIN_ENTRYPOINT" in
    scripts/train_state8_profile_ownership_repair.py|scripts/train_state8_profile_ownership_v5.py) ;;
    *) echo "unapproved profile training entrypoint: $TRAIN_ENTRYPOINT" >&2; exit 2 ;;
  esac
  test -f "$TRAIN_ENTRYPOINT"
  python -u "$TRAIN_ENTRYPOINT" \
    --model "$MODEL_LOCAL" --init-adapter "$SOURCE_ADAPTER" \
    --init-recurrent-state "$SOURCE_STATE" --state-architecture layerwise \
    --layerwise-operation-checkpointing --layerwise-transition-rank 256 \
    --state-token-count 8 --train-jsonl "$DATA/train.jsonl" \
    --valid-jsonl "$DATA/valid.jsonl" --causal-train-jsonl "$DATA/causal_train.jsonl" \
    --causal-valid-jsonl "$DATA/causal_valid.jsonl" --output-dir "$RUN_DIR" \
    --state-token-alias '<|vision_pad|>' --online-kd-weight 0 \
    --max-length 1536 --direct-offload-min-tokens 1 \
    --learning-rate 1e-6 --lr-scheduler cosine --warmup-ratio 0.03 \
    --generation-weight 1.0 --behavior-kd-weight 0 --consistency-weight 0 \
    --direct-replay-weight 0 --causal-weight 0.1 --causal-margin 0.5 \
    --causal-preference-reduction sum --causal-objective-mode specificity_triplet \
    --causal-repeat-fraction 0.3 --normalize-causal-exposure \
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
    --eval-limit "$EVAL_LIMIT" --causal-eval-limit "$CAUSAL_EVAL_LIMIT" \
    --save-steps 100 --checkpoint-retention 2 --seed 8801 \
    "${TRAIN_DISTRIBUTED_ARGS[@]}" \
    "${TRAIN_DIAGNOSTIC_ARGS[@]}" \
    "${TRAIN_RECOVERY_ARGS[@]}"
fi

# Distributed workers have already contributed gradients and history.  The
# trainer deliberately releases NCCL before rank zero performs long validation;
# only rank zero may publish and evaluate the shared checkpoint afterwards.
if [[ "${OMPI_COMM_WORLD_RANK:-0}" != 0 ]]; then
  exit 0
fi

test -f "$RUN_DIR/final_adapter/adapter_model.safetensors"
test -f "$RUN_DIR/recurrent_state.pt"
test -f "$RUN_DIR/student_profile_readout_repair.json"
test -f "$RUN_DIR/profile_ownership_repair.json"
if [[ -n "$DIAGNOSTIC_STOP_ROUND" ]]; then
  python - "$RUN_DIR/completion.json" "$DIAGNOSTIC_STOP_ROUND" "$WORLD_SIZE" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]))
rounds=int(sys.argv[2]); world=int(sys.argv[3])
assert c['status']=='diagnostic_complete',c
assert c['micro_steps']==rounds*world,c
assert c['history_integrity_pass'] is True,c
PY
  cp "$RUN_DIR/completion.json" "$AMLT_OUTPUT_DIR/distributed_diagnostic_completion.json"
  exit 0
fi
if [[ -n "$DIAGNOSTIC_MAX_MICRO_STEPS" ]]; then
  python - "$RUN_DIR/completion.json" "$DIAGNOSTIC_MAX_MICRO_STEPS" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]))
steps=int(sys.argv[2])
assert c['status']=='diagnostic_complete',c
assert c['micro_steps']==steps,c
PY
  mkdir -p "$AMLT_OUTPUT_DIR/checkpoint_early_audit"
  python -u scripts/early_audit_state8_profile_ownership_checkpoint.py \
    --model "$MODEL_LOCAL" --adapter "$RUN_DIR/final_adapter" \
    --checkpoint "$RUN_DIR/recurrent_state.pt" \
    --rows "$PROFILE_SOURCE/test.jsonl" \
    --output "$AMLT_OUTPUT_DIR/checkpoint_early_audit/metrics.json" \
    --state-token-alias '<|vision_pad|>' --max-length 1536 --batch-size 4 \
    --ownership-scope "${PROFILE_READOUT_OWNERSHIP_SCOPE:-all}" \
    --seed 20260820 --permutation-replicates 2000
  cp "$RUN_DIR/completion.json" "$AMLT_OUTPUT_DIR/diagnostic_completion.json"
  cp "$RUN_DIR/profile_ownership_repair.json" "$AMLT_OUTPUT_DIR/"
  if [[ -f "$RUN_DIR/profile_ownership_v5.json" ]]; then
    cp "$RUN_DIR/profile_ownership_v5.json" "$AMLT_OUTPUT_DIR/"
  fi
  exit 0
fi
python - "$RUN_DIR/completion.json" "$RUN_DIR/run_config.json" "$RUN_DIR/student_profile_readout_repair.json" "$RUN_DIR/profile_ownership_repair.json" "$RUN_DIR/profile_ownership_v5.json" <<'PY'
import json,sys
c,r,p,o=(json.load(open(x)) for x in sys.argv[1:5])
v5_path=sys.argv[5]
assert c['status']=='complete' and c['micro_steps']==1000 and c['optimizer_steps']==250,c
assert c['all_causal_pairs_covered'] is True and c['unique_causal_pairs_seen']==300,c
assert r['state_token_count']==8 and r['final_inference_route']=='strict_state_only',r
assert p['profile_rows']==663 and p['profile_causal_exposures']==300,p
assert p['future_labels_visible_to_training'] is True,p
assert p['future_labels_visible_to_decoder'] is False,p
assert p['profile_label_query_mode']=='state_only',p
assert p['profile_label_visible_at_inference'] is False,p
assert o['status']=='complete' and o['applied_rows']>0,o
assert o['weight']>0 and o['margin']>=0,o
assert o['head_input']=='layerwise_mean_slots_rp',o
assert o['projection_dim']==64 and o['projection_seed']==20260818,o
assert o['ownership_supervision_visible_at_inference'] is False,o
assert o['final_inference_route']=='strict_state_only',o
if r.get('state8_v5_loss_fix'):
    v=json.load(open(v5_path))
    assert v['status']=='complete' and v['diagnostic'] is False,v
    assert v['profile_rows']==663 and v['multi_negative_profile_rows']>0,v
    assert v['profile_numeric_head_frozen'] is True,v
    assert v['profile_generation_weight']==0.1,v
    assert v['profile_legacy_text_transport_weight']==0.0,v
    assert v['profile_legacy_semantic_geometry_weight']==0.0,v
    assert v['profile_legacy_early_label_weight']==0.0,v
    assert v['negative_policy']=='deterministic_topk_panel_mean',v
    assert v['raw_history_visible_to_decoder'] is False,v
    assert v['student_id_visible_to_decoder'] is False,v
    assert v['final_inference_route']=='strict_state_only',v
    assert p['legacy_profile_causal_enabled'] is False,p
    assert p['ownership_geometry_exposures']==0,p
    assert p['prefix_free_profile_label_exposures']==0,p
else:
    assert p['ownership_geometry_exposures']==300,p
    assert p['prefix_free_profile_label_exposures']==300,p
PY

python -u scripts/evaluate_state8_student_profile_readout.py \
  --model "$MODEL_LOCAL" --adapter "$RUN_DIR/final_adapter" \
  --recurrent-state "$RUN_DIR/recurrent_state.pt" --test-jsonl "$DATA/test_profiles.jsonl" \
  --causal-jsonl "$DATA/causal_valid.jsonl" --output-dir "$OUT" \
  --state-token-alias '<|vision_pad|>' --max-length 1536 --max-new-tokens 192 \
  --num-profiles 88 --num-pairs 50 --seed 20260817

rm -rf "$(dirname "$RETENTION_RECURRENT")" "$RETENTION_OUT"
mkdir -p "$(dirname "$RETENTION_RECURRENT")" "$RETENTION_OUT"
python scripts/build_recurrent_evaluation_manifest.py \
  --input-jsonl "$RETENTION_EVAL" --output-jsonl "$RETENTION_RECURRENT" \
  --audit-json "$(dirname "$RETENTION_RECURRENT")/audit.json" --max-updates 4 --seed 20260730
python -u scripts/evaluate_recurrent_formal_protocols.py \
  --model "$MODEL_LOCAL" --adapter "$RUN_DIR/final_adapter" \
  --recurrent-state "$RUN_DIR/recurrent_state.pt" --state-architecture layerwise \
  --eval-jsonl "$RETENTION_RECURRENT" --output-dir "$RETENTION_OUT" \
  --state-token-alias '<|vision_pad|>' --tasks predict_correctness predict_response \
  diagnose_misconception predict_prospective_error predict_error_persistence \
  --max-samples-per-group 10 --max-length 1536 --max-new-tokens 160 \
  --state-batch-size 2 --candidate-row-batch-size 2 --seed 8801

rm -rf "$INTERFACE_FEATURES" "$INTERFACE_OUT"
mkdir -p "$INTERFACE_FEATURES" "$INTERFACE_OUT"
for split in train valid test; do
  python -u scripts/evaluate_state8_reusable_interface.py extract \
    --model "$MODEL_LOCAL" --adapter "$RUN_DIR/final_adapter" \
    --recurrent-state "$RUN_DIR/recurrent_state.pt" \
    --rows-jsonl "$PROFILE_SOURCE/$split.jsonl" --split "$split" \
    --output "$INTERFACE_FEATURES/$split.npz" \
    --state-token-alias '<|vision_pad|>' --max-length 1536 --batch-size 4 \
    --projection-dim 64 --projection-seed 20260818
done
python -u scripts/evaluate_state8_reusable_interface.py analyze \
  --manifest-dir "$PROFILE_SOURCE" --feature-dir "$INTERFACE_FEATURES" \
  --output-dir "$INTERFACE_OUT" --bootstrap-replicates 5000 \
  --permutation-replicates 2000 --seed 20260818
test -s "$INTERFACE_OUT/metrics.json"

rm -rf "$STABLE"
mkdir -p "$STABLE/model" "$STABLE/eval" "$STABLE/metrics"
cp -r "$RUN_DIR/final_adapter/." "$STABLE/model/"
cp "$RUN_DIR/recurrent_state.pt" "$STABLE/model/recurrent_state.pt"
cp -r "$OUT/." "$STABLE/eval/"
mkdir -p "$STABLE/eval/retention"
cp -r "$RETENTION_OUT/." "$STABLE/eval/retention/"
mkdir -p "$STABLE/eval/reusable_interface"
cp -r "$INTERFACE_OUT/." "$STABLE/eval/reusable_interface/"
cp "$DATA/receipt.json" "$STABLE/metrics/data_receipt.json"
cp "$RUN_DIR"/*.json "$STABLE/metrics/"
cp -r "$OUT/." "$AMLT_OUTPUT_DIR/"
cp -r "$RUN_DIR/final_adapter" "$AMLT_OUTPUT_DIR/"
cp "$RUN_DIR/recurrent_state.pt" "$AMLT_OUTPUT_DIR/recurrent_state.pt"
cp "$RUN_DIR"/*.json "$AMLT_OUTPUT_DIR/"
