#!/usr/bin/env bash
set -euo pipefail

export VENV_ROOT="${VENV_ROOT:-/tmp/studentmodeling-state8-finalfit-venv}"
export PATH="$VENV_ROOT/bin:$PATH"
hash -r
python(){ "$VENV_ROOT/bin/python" "$@"; }
for _probe in $(seq 1 120); do
  if python -c 'import accelerate,bitsandbytes,peft,safetensors,transformers' >/dev/null 2>&1; then break; fi
  sleep 1
done
python -c 'import accelerate,bitsandbytes,peft,safetensors,transformers'
export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/hf_cache
export HF_HUB_CACHE=/tmp/hf_cache/hub
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error
export LOCAL_MODEL=/tmp/Qwen3-32B
export RAW_FULL=/mnt/amulet/studentmodeling/full_trainplusvalid_no_t4_strict_t6/train.jsonl
export RAW_SPLIT=/tmp/state8_finalfit_full_trainvalid_v1
export RECURRENT_SPLIT=/tmp/state8_finalfit_recurrent_full259845_u4_v1
export FINAL_JSONL="$RECURRENT_SPLIT/final_fit_trainvalid.jsonl"
export FINAL_TRAIN_JSONL="$FINAL_JSONL"
export EEDI_PAIR_JSONL=/tmp/state8_finalfit_recurrent_full259845_u4_v1/final_fit_trainvalid_eedi_pairs.jsonl
export EEDI_PAIR_RECEIPT=/tmp/state8_finalfit_recurrent_full259845_u4_v1/eedi_semantic_pair_augmentation.json
export C4_DATA=/mnt/amulet/studentmodeling/c4_fullopen_20260724/malrule_c1_clean
export CAUSAL_BASE=/tmp/state8_finalfit_cf4_base_v1
export CAUSAL_SAME_RULE=/tmp/state8_finalfit_cf4_same_rule_v1
export CAUSAL_DATA=/tmp/state8_finalfit_cf4_answer_diagnosis_v1
export FINAL_CAUSAL_TRAIN_JSONL="$CAUSAL_DATA/train.jsonl"
export FINAL_CAUSAL_VALID_JSONL="$CAUSAL_DATA/valid.jsonl"
export FINALFIT_VARIANT="${FINALFIT_VARIANT:?FINALFIT_VARIANT is required}"
export STATE8_FINALFIT_ENTRYPOINT=scripts/train_recurrent_student_m4_state8_external_teacher_curriculum.py
export STATE8_FINALFIT_LEARNING_RATE=1e-6
export STATE8_REPAIR_LABEL_WEIGHT=4.0
export STATE8_REPAIR_TRAJECTORY_WEIGHT=0.0
export STATE8_CAUSAL_LABEL_WEIGHT=4.0
export STATE8_RANDOM_NEUTRAL=0
export STATE8_UPDATER_ONLY=0
export STATE8_GEOMETRY_WEIGHT=0.0
export STATE8_EEDI_GEOMETRY_WEIGHT=0.0
export STATE8_DIFFERENT_MAX_COSINE=0.90
export STATE8_CAUSAL_DIAGNOSIS_ONLY=0
export STATE8_CAUSAL_REPEAT_FRACTION=0.05
export STATE8_CAUSAL_WEIGHT=2.0
export STATE8_CAUSAL_MARGIN=0.5
export STATE8_LAYERWISE_TRANSITION_ONLY=0
export INIT_HISTORY_MODE=five_stage_curriculum
export EXPECTED_HISTORY_MODE=five_stage_curriculum
case "$FINALFIT_VARIANT" in
  t3_heavy)
    export RESULT_ROOT=/mnt/amulet/studentmodeling/diagnostics/state8_qualified_teacher_curriculum_v1_immutable
    export EXPECTED_IMMUTABLE_STAGE=state8_qualified_teacher_curriculum_immutable_v1
    export STATE8_FINALFIT_ONLINE_KD_WEIGHT=0.5 ;;
  balanced)
    export RESULT_ROOT=/mnt/amulet/studentmodeling/diagnostics/state8_qualified_teacher_balanced_curriculum_v1_immutable
    export EXPECTED_IMMUTABLE_STAGE=state8_qualified_teacher_balanced_curriculum_immutable_v1
    export STATE8_FINALFIT_ONLINE_KD_WEIGHT=0.5 ;;
  highkd)
    export RESULT_ROOT=/mnt/amulet/studentmodeling/diagnostics/state8_qualified_teacher_curriculum_highkd_v1_immutable
    export EXPECTED_IMMUTABLE_STAGE=state8_qualified_teacher_curriculum_highkd_immutable_v1
    export STATE8_FINALFIT_ONLINE_KD_WEIGHT=1.0 ;;
  repair_logits)
    export RESULT_ROOT=/mnt/amulet/studentmodeling/diagnostics/state8_stateonly_repair_seed44_logits_v1
    export EXPECTED_IMMUTABLE_STAGE=state8_stateonly_repair_pilot_publish_v1
    export STATE8_FINALFIT_ONLINE_KD_WEIGHT=1.0
    export STATE8_FINALFIT_ENTRYPOINT=scripts/train_state8_stateonly_teacher_repair.py
    export STATE8_FINALFIT_LEARNING_RATE=5e-7
    export INIT_HISTORY_MODE=strict_state_only
    export EXPECTED_HISTORY_MODE=strict_state_only
    export STATE8_REPAIR_TRAJECTORY_WEIGHT=0.0 ;;
  repair_trajectory01)
    export RESULT_ROOT=/mnt/amulet/studentmodeling/diagnostics/state8_stateonly_repair_seed44_trajectory01_v1
    export EXPECTED_IMMUTABLE_STAGE=state8_stateonly_repair_pilot_publish_v1
    export STATE8_FINALFIT_ONLINE_KD_WEIGHT=1.0
    export STATE8_FINALFIT_ENTRYPOINT=scripts/train_state8_stateonly_teacher_repair.py
    export STATE8_FINALFIT_LEARNING_RATE=5e-7
    export INIT_HISTORY_MODE=strict_state_only
    export EXPECTED_HISTORY_MODE=strict_state_only
    export STATE8_REPAIR_TRAJECTORY_WEIGHT=0.1 ;;
  semantic_c4_g005|semantic_c4_g010|semantic_c4_g025|semantic_eedi_g005|semantic_eedi_g010|semantic_eedi_g025|semantic_eedi_g050|semantic_eedi_g025_single_seed46)
    source_seed=45
    case "$FINALFIT_VARIANT" in
      semantic_c4_g005)
        tag=semantic_causal_strongbase_g005_v1; export STATE8_GEOMETRY_WEIGHT=0.05 ;;
      semantic_c4_g010)
        tag=semantic_causal_strongbase_v1; export STATE8_GEOMETRY_WEIGHT=0.10 ;;
      semantic_c4_g025)
        tag=semantic_causal_strongbase_g025_v1; export STATE8_GEOMETRY_WEIGHT=0.25 ;;
      semantic_eedi_g005)
        tag=semantic_causal_strongbase_eedi_g005_v1
        export STATE8_GEOMETRY_WEIGHT=0.10
        export STATE8_EEDI_GEOMETRY_WEIGHT=0.05 ;;
      semantic_eedi_g010)
        tag=semantic_causal_strongbase_eedi_g010_v1
        export STATE8_GEOMETRY_WEIGHT=0.10
        export STATE8_EEDI_GEOMETRY_WEIGHT=0.10 ;;
      semantic_eedi_g025)
        tag=semantic_causal_strongbase_eedi_g025_v1
        export STATE8_GEOMETRY_WEIGHT=0.10
        export STATE8_EEDI_GEOMETRY_WEIGHT=0.25 ;;
      semantic_eedi_g050)
        tag=semantic_causal_strongbase_eedi_g050_v1
        export STATE8_GEOMETRY_WEIGHT=0.10
        export STATE8_EEDI_GEOMETRY_WEIGHT=0.50 ;;
      semantic_eedi_g025_single_seed46)
        source_seed=46
        tag=semantic_causal_strongbase_eedi_g025_single_v1
        export STATE8_GEOMETRY_WEIGHT=0.10
        export STATE8_EEDI_GEOMETRY_WEIGHT=0.25 ;;
    esac
    export RESULT_ROOT="/mnt/amulet/studentmodeling/diagnostics/state8_stateonly_repair_seed${source_seed}_trajectory01_${tag}"
    export EXPECTED_IMMUTABLE_STAGE=state8_stateonly_repair_pilot_publish_v1
    export STATE8_FINALFIT_ONLINE_KD_WEIGHT=1.0
    export STATE8_FINALFIT_ENTRYPOINT=scripts/train_state8_semantic_causal_repair.py
    export STATE8_FINALFIT_LEARNING_RATE=5e-7
    export INIT_HISTORY_MODE=strict_state_only
    export EXPECTED_HISTORY_MODE=strict_state_only
    export STATE8_REPAIR_TRAJECTORY_WEIGHT=0.1
    export STATE8_RANDOM_NEUTRAL=1 ;;
  highcausal_joint|highcausal_updater|highcausal_mid|highcausal_eedi|highcausal_taxonomy)
    case "$FINALFIT_VARIANT" in
      highcausal_joint)
        source_seed=47
        tag=highcausal_joint_v1 ;;
      highcausal_updater)
        source_seed=48
        tag=highcausal_updater_only_v1
        export STATE8_LAYERWISE_TRANSITION_ONLY=1
        export STATE8_UPDATER_ONLY=1 ;;
      highcausal_mid)
        source_seed=49
        tag=highcausal_mid_v1 ;;
      highcausal_eedi)
        source_seed=50
        tag=highcausal_eedi025_v1
        export STATE8_EEDI_GEOMETRY_WEIGHT=0.25 ;;
      highcausal_taxonomy)
        source_seed=51
        tag=taxonomy_eedi_highcausal_v1
        export STATE8_EEDI_GEOMETRY_WEIGHT=0.25 ;;
    esac
    export RESULT_ROOT="/mnt/amulet/studentmodeling/diagnostics/state8_stateonly_repair_seed${source_seed}_trajectory01_${tag}"
    export EXPECTED_IMMUTABLE_STAGE=state8_stateonly_repair_pilot_publish_v1
    export STATE8_FINALFIT_ONLINE_KD_WEIGHT=1.0
    export STATE8_FINALFIT_ENTRYPOINT=scripts/train_state8_semantic_causal_repair.py
    if [[ "$FINALFIT_VARIANT" == highcausal_taxonomy ]]; then
      export STATE8_FINALFIT_ENTRYPOINT=scripts/train_state8_taxonomy_causal_repair.py
    fi
    export STATE8_FINALFIT_LEARNING_RATE=5e-7
    export INIT_HISTORY_MODE=strict_state_only
    export EXPECTED_HISTORY_MODE=strict_state_only
    export STATE8_REPAIR_TRAJECTORY_WEIGHT=0.1
    export STATE8_CAUSAL_LABEL_WEIGHT=8.0
    export STATE8_RANDOM_NEUTRAL=1
    export STATE8_GEOMETRY_WEIGHT=1.0
    export STATE8_DIFFERENT_MAX_COSINE=0.75
    # Screening isolates diagnosis causality, but the promoted unified fit must
    # retain both prospective-answer and natural-diagnosis C4 twins.
    export STATE8_CAUSAL_DIAGNOSIS_ONLY=0
    export STATE8_CAUSAL_WEIGHT=8.0
    export STATE8_CAUSAL_MARGIN=1.0
    if [[ "$FINALFIT_VARIANT" == highcausal_mid \
          || "$FINALFIT_VARIANT" == highcausal_taxonomy ]]; then
      export STATE8_CAUSAL_WEIGHT=4.0
      export STATE8_GEOMETRY_WEIGHT=0.5
    fi
    export PROMOTION_GATE=/mnt/amulet/studentmodeling/gates/state8_highcausal_binding_v1/selection.json ;;
  taxonomy_curriculum_t3heavy|taxonomy_curriculum_balanced|taxonomy_curriculum_t3heavy_diag|taxonomy_curriculum_balanced_diag)
    case "$FINALFIT_VARIANT" in
    taxonomy_curriculum_t3heavy)
      export RESULT_ROOT=/mnt/amulet/studentmodeling/diagnostics/state8_stateonly_repair_seed52_trajectory01_taxonomy_curriculum_consolidation_t3heavy_v1
      ;;
    taxonomy_curriculum_balanced)
      export RESULT_ROOT=/mnt/amulet/studentmodeling/diagnostics/state8_stateonly_repair_seed53_trajectory01_taxonomy_curriculum_consolidation_balanced_v1
      ;;
    taxonomy_curriculum_t3heavy_diag)
      export RESULT_ROOT=/mnt/amulet/studentmodeling/diagnostics/state8_stateonly_repair_seed54_trajectory01_taxonomy_curriculum_consolidation_t3heavy_diag_v1
      ;;
    taxonomy_curriculum_balanced_diag)
      export RESULT_ROOT=/mnt/amulet/studentmodeling/diagnostics/state8_stateonly_repair_seed55_trajectory01_taxonomy_curriculum_consolidation_balanced_diag_v1
      ;;
    esac
    export EXPECTED_IMMUTABLE_STAGE=state8_stateonly_repair_pilot_publish_v1
    # The selected five-stage checkpoint is consolidated for one exact epoch
    # with a strictly closed decoder.  The full-history teacher remains only
    # as a detached answer-logit/trajectory target.
    export STATE8_FINALFIT_ONLINE_KD_WEIGHT=1.0
    export STATE8_FINALFIT_ENTRYPOINT=scripts/train_state8_taxonomy_causal_repair.py
    export STATE8_FINALFIT_LEARNING_RATE=5e-7
    export INIT_HISTORY_MODE=strict_state_only
    export EXPECTED_HISTORY_MODE=strict_state_only
    export STATE8_REPAIR_TRAJECTORY_WEIGHT=0.1
    export STATE8_CAUSAL_LABEL_WEIGHT=4.0
    export STATE8_RANDOM_NEUTRAL=1
    export STATE8_GEOMETRY_WEIGHT=0.10
    export STATE8_EEDI_GEOMETRY_WEIGHT=0.05
    export STATE8_DIFFERENT_MAX_COSINE=0.90
    export STATE8_CAUSAL_WEIGHT=2.0
    export STATE8_CAUSAL_MARGIN=0.5
    export STATE8_CAUSAL_REPEAT_FRACTION=0.05
    export PROMOTION_GATE=/mnt/amulet/studentmodeling/gates/state8_unified_candidate_v1/selection.json ;;
  task_anchor_base|task_anchor_c4strong)
    case "$FINALFIT_VARIANT" in
    task_anchor_base)
      export RESULT_ROOT=/mnt/amulet/studentmodeling/diagnostics/state8_stateonly_repair_seed56_taskanchor_task_anchor_base_v1
      export STATE8_CAUSAL_WEIGHT=2.0
      ;;
    task_anchor_c4strong)
      export RESULT_ROOT=/mnt/amulet/studentmodeling/diagnostics/state8_stateonly_repair_seed57_taskanchor_task_anchor_c4strong_v1
      export STATE8_CAUSAL_WEIGHT=4.0
      ;;
    esac
    export EXPECTED_IMMUTABLE_STAGE=state8_stateonly_repair_pilot_publish_v1
    export STATE8_FINALFIT_ONLINE_KD_WEIGHT=1.0
    export STATE8_FINALFIT_ENTRYPOINT=scripts/train_state8_task_anchored_causal_repair.py
    export STATE8_FINALFIT_LEARNING_RATE=5e-7
    export INIT_HISTORY_MODE=strict_state_only
    export EXPECTED_HISTORY_MODE=strict_state_only
    export STATE8_REPAIR_TRAJECTORY_WEIGHT=1.0
    export STATE8_T3_ANCHOR_SCHEDULE=1.0,0.5,0.25
    export STATE8_OTHER_ANCHOR_WEIGHT=0.1
    export STATE8_CAUSAL_LABEL_WEIGHT=4.0
    export STATE8_RANDOM_NEUTRAL=1
    export STATE8_GEOMETRY_WEIGHT=0.10
    export STATE8_EEDI_GEOMETRY_WEIGHT=0.05
    export STATE8_DIFFERENT_MAX_COSINE=0.90
    export STATE8_CAUSAL_MARGIN=0.5
    # 5% of the 259,845-row epoch is already >12k causal exposures.  This is
    # denser than the 500/1000 bounded screen while preserving task rehearsal.
    export STATE8_CAUSAL_REPEAT_FRACTION=0.05
    export PROMOTION_GATE=/mnt/amulet/studentmodeling/gates/state8_unified_candidate_v1/selection.json ;;
  strong1_widen8_lr5e7_seed58|strong1_widen8_lr1e6_seed59|strong1_widen8_single_lr5e7_seed60)
    case "$FINALFIT_VARIANT" in
    strong1_widen8_lr5e7_seed58)
      suffix=strong1_widen8_lr5e7_seed58 ;;
    strong1_widen8_lr1e6_seed59)
      suffix=strong1_widen8_lr1e6_seed59 ;;
    strong1_widen8_single_lr5e7_seed60)
      suffix=strong1_widen8_single_lr5e7_seed60_single_ga4 ;;
    esac
    export RESULT_ROOT="/mnt/amulet/studentmodeling/diagnostics/state8_qualified_teacher_curriculum_highkd_v1_immutable_taxonomy_semantic_v1_${suffix}"
    export EXPECTED_IMMUTABLE_STAGE="state8_qualified_teacher_curriculum_highkd_immutable_v1_taxonomy_semantic_v1_${suffix}"
    # The bridge provenance is the audited five-stage teacher curriculum.  The
    # full epoch is nevertheless closed from its first micro-step: the
    # state-only wrapper fixes history dropout to one while retaining detached
    # full-history teacher logits and trajectory targets.
    export STATE8_FINALFIT_ONLINE_KD_WEIGHT=1.0
    export STATE8_FINALFIT_ENTRYPOINT=scripts/train_state8_taxonomy_causal_repair.py
    export STATE8_FINALFIT_LEARNING_RATE=5e-7
    export INIT_HISTORY_MODE=five_stage_curriculum
    export EXPECTED_HISTORY_MODE=strict_state_only
    export STATE8_REPAIR_TRAJECTORY_WEIGHT=0.1
    export STATE8_CAUSAL_LABEL_WEIGHT=4.0
    export STATE8_RANDOM_NEUTRAL=1
    export STATE8_GEOMETRY_WEIGHT=0.10
    export STATE8_EEDI_GEOMETRY_WEIGHT=0.05
    export STATE8_DIFFERENT_MAX_COSINE=0.90
    export STATE8_CAUSAL_WEIGHT=2.0
    export STATE8_CAUSAL_MARGIN=0.5
    export STATE8_CAUSAL_REPEAT_FRACTION=0.05
    export PROMOTION_GATE=/mnt/amulet/studentmodeling/gates/state8_strong1_widen8_bridge_v1/selection.json ;;
  strong1_rescue_taxonomy_joint_seed61|strong1_rescue_taskanchor_c4strong_seed62)
    case "$FINALFIT_VARIANT" in
    strong1_rescue_taxonomy_joint_seed61)
      export RESULT_ROOT=/mnt/amulet/studentmodeling/diagnostics/state8_stateonly_repair_seed61_trajectory01_strong1_rescue_taxonomy_joint_v1
      # Preserve the selected rescue's semantic-first readout all the way
      # through the exact full epoch.  Reverting to label-first here would be
      # a scientific recipe change after the short bridge gate.
      export STATE8_FINALFIT_ENTRYPOINT=scripts/train_state8_diagnosis_first_causal_repair.py
      export STATE8_REPAIR_LABEL_WEIGHT=1.0
      export STATE8_CAUSAL_LABEL_WEIGHT=1.0
      export STATE8_REPAIR_TRAJECTORY_WEIGHT=0.1
      export STATE8_CAUSAL_WEIGHT=2.0 ;;
    strong1_rescue_taskanchor_c4strong_seed62)
      export RESULT_ROOT=/mnt/amulet/studentmodeling/diagnostics/state8_stateonly_repair_seed62_taskanchor_strong1_rescue_taskanchor_c4strong_v1
      export STATE8_FINALFIT_ENTRYPOINT=scripts/train_state8_task_anchored_causal_repair.py
      export STATE8_REPAIR_TRAJECTORY_WEIGHT=1.0
      export STATE8_CAUSAL_WEIGHT=4.0 ;;
    esac
    export EXPECTED_IMMUTABLE_STAGE=state8_stateonly_repair_pilot_publish_v1
    export STATE8_FINALFIT_ONLINE_KD_WEIGHT=1.0
    export STATE8_FINALFIT_LEARNING_RATE=5e-7
    export INIT_HISTORY_MODE=strict_state_only
    export EXPECTED_HISTORY_MODE=strict_state_only
    export STATE8_T3_ANCHOR_SCHEDULE=1.0,0.5,0.25
    export STATE8_OTHER_ANCHOR_WEIGHT=0.1
    export STATE8_RANDOM_NEUTRAL=1
    export STATE8_GEOMETRY_WEIGHT=0.10
    export STATE8_EEDI_GEOMETRY_WEIGHT=0.05
    export STATE8_DIFFERENT_MAX_COSINE=0.90
    export STATE8_CAUSAL_MARGIN=0.5
    export STATE8_CAUSAL_REPEAT_FRACTION=0.05
    export PROMOTION_GATE=/mnt/amulet/studentmodeling/gates/state8_strong1_rescue_v1/selection.json ;;
  strong1_midpoint_diagnosis_first_seed63)
    export RESULT_ROOT=/mnt/amulet/studentmodeling/diagnostics/state8_stateonly_repair_seed63_trajectory01_midpoint_step2000_diagnosis_first_v1
    export EXPECTED_IMMUTABLE_STAGE=state8_stateonly_repair_pilot_publish_v1
    export STATE8_FINALFIT_ONLINE_KD_WEIGHT=1.0
    export STATE8_FINALFIT_ENTRYPOINT=scripts/train_state8_diagnosis_first_causal_repair.py
    export STATE8_FINALFIT_LEARNING_RATE=5e-7
    export INIT_HISTORY_MODE=strict_state_only
    export EXPECTED_HISTORY_MODE=strict_state_only
    export STATE8_REPAIR_LABEL_WEIGHT=1.0
    export STATE8_CAUSAL_LABEL_WEIGHT=1.0
    export STATE8_REPAIR_TRAJECTORY_WEIGHT=0.1
    export STATE8_RANDOM_NEUTRAL=1
    export STATE8_GEOMETRY_WEIGHT=0.10
    export STATE8_EEDI_GEOMETRY_WEIGHT=0.05
    export STATE8_DIFFERENT_MAX_COSINE=0.90
    export STATE8_CAUSAL_WEIGHT=2.0
    export STATE8_CAUSAL_MARGIN=0.5
    export STATE8_CAUSAL_REPEAT_FRACTION=0.05
    export PROMOTION_GATE=/mnt/amulet/studentmodeling/gates/state8_midpoint_diagnosis_first_v1/selection.json ;;
  strong1_exact_diagnosis_curriculum_seed65)
    export RESULT_ROOT=/mnt/amulet/studentmodeling/diagnostics/state8_stateonly_repair_seed65_trajectory01_exact_teacher_curriculum_c4_seed65_v1
    export EXPECTED_IMMUTABLE_STAGE=state8_stateonly_repair_pilot_publish_v1
    export STATE8_FINALFIT_ONLINE_KD_WEIGHT=1.0
    export STATE8_FINALFIT_ENTRYPOINT=scripts/train_state8_diagnosis_first_causal_repair.py
    export STATE8_FINALFIT_LEARNING_RATE=5e-7
    export INIT_HISTORY_MODE=strict_state_only
    export EXPECTED_HISTORY_MODE=strict_state_only
    export STATE8_REPAIR_LABEL_WEIGHT=1.0
    export STATE8_CAUSAL_LABEL_WEIGHT=1.0
    export STATE8_REPAIR_TRAJECTORY_WEIGHT=0.1
    export STATE8_RANDOM_NEUTRAL=1
    export STATE8_GEOMETRY_WEIGHT=0.10
    export STATE8_EEDI_GEOMETRY_WEIGHT=0.05
    export STATE8_DIFFERENT_MAX_COSINE=0.90
    export STATE8_CAUSAL_WEIGHT=2.0
    export STATE8_CAUSAL_MARGIN=0.5
    export STATE8_CAUSAL_REPEAT_FRACTION=0.05
    export PROMOTION_GATE=/mnt/amulet/studentmodeling/gates/state8_exact_diagnosis_curriculum_v1/selection.json ;;
  *) exit 2 ;;
esac
export INIT_DIR="$RESULT_ROOT/model"
export INIT_COMPLETION="$RESULT_ROOT/metrics/completion.json"
if [[ "$INIT_HISTORY_MODE" == strict_state_only ]]; then
  export INIT_IMMUTABLE="$RESULT_ROOT/metrics/publish_receipt.json"
else
  export INIT_IMMUTABLE="$RESULT_ROOT/metrics/immutable_receipt.json"
fi
export TEACHER_ROOT=/mnt/amulet/studentmodeling/diagnostics/state8_full_history_natural_teacher_v1
export RUN_DIR=/mnt/amulet/studentmodeling/checkpoints/state8_qualified_unified_full259845_seed42_${FINALFIT_VARIANT}_v1_20260805
export CANONICAL_RUN_DIR=/mnt/amulet/studentmodeling/checkpoints/state8_qualified_unified_full259845_seed42_v1_20260805
if [[ -n "${STATE8_CANONICAL_RUN_DIR:-}" ]]; then
  export CANONICAL_RUN_DIR="$STATE8_CANONICAL_RUN_DIR"
fi
export TOTAL_ROUNDS=16241
export MILESTONE_ROUNDS="3249 6497 9745 12993"

mkdir -p "$HF_HUB_CACHE" "$RAW_SPLIT" "$RECURRENT_SPLIT" "$CAUSAL_BASE" \
  "$CAUSAL_SAME_RULE" "$CAUSAL_DATA" \
  "$RUN_DIR" "$AMLT_OUTPUT_DIR/model" "$AMLT_OUTPUT_DIR/metrics"
test "$(df --output=avail /tmp | tail -1)" -ge 125829120
test -f "$RAW_FULL"
test -f "$C4_DATA/train.jsonl"
test -f "$C4_DATA/val.jsonl"
test -f "$INIT_DIR/adapter_model.safetensors"
test -f "$INIT_DIR/recurrent_state.pt"
test -f "$INIT_COMPLETION"
test -f "$INIT_IMMUTABLE"
test -f "$TEACHER_ROOT/model/adapter_model.safetensors"
test -f "$TEACHER_ROOT/model/adapter_config.json"
test -f "$TEACHER_ROOT/metrics/teacher_gate.json"
test -f "$TEACHER_ROOT/metrics/teacher_provenance_gate.json"
test -f "$TEACHER_ROOT/metrics/publish_receipt.json"
test -f /mnt/amulet/studentmodeling/models/Qwen3-32B/model.safetensors.index.json
if [[ "$FINALFIT_VARIANT" == highcausal_* ]]; then
  test -f "$PROMOTION_GATE"
  python - "$PROMOTION_GATE" "$FINALFIT_VARIANT" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); variant=sys.argv[2]
expected={
 'highcausal_joint':'joint',
 'highcausal_updater':'updater',
 'highcausal_mid':'mid',
 'highcausal_eedi':'eedi',
 'highcausal_taxonomy':'taxonomy',
}[variant]
assert x['stage']=='state8_high_causal_candidate_selection_v1' and x['pass'] is True,x
assert x['selected']==expected,x
assert x['arms'][expected]['eligible'] is True,x
PY
fi
if [[ "$FINALFIT_VARIANT" == taxonomy_curriculum_* ]]; then
  test -f "$PROMOTION_GATE"
  python - "$PROMOTION_GATE" "$FINALFIT_VARIANT" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); variant=sys.argv[2]
expected={
 'taxonomy_curriculum_t3heavy':'taxonomy_t3heavy',
 'taxonomy_curriculum_balanced':'taxonomy_balanced',
 'taxonomy_curriculum_t3heavy_diag':'taxonomy_t3heavy_diag',
 'taxonomy_curriculum_balanced_diag':'taxonomy_balanced_diag',
}[variant]
assert x['stage']=='state8_unified_candidate_selection_v1' and x['pass'] is True,x
assert x['selected']['slug']==expected and x['selected']['qualified'] is True,x
assert x['selected']['checks']['C4_pass'] is True,x
PY
fi
if [[ "$FINALFIT_VARIANT" == task_anchor_* ]]; then
  test -f "$PROMOTION_GATE"
  python - "$PROMOTION_GATE" "$FINALFIT_VARIANT" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); variant=sys.argv[2]
expected={'task_anchor_base':'base','task_anchor_c4strong':'c4strong'}[variant]
assert x['stage']=='state8_unified_candidate_selection_v1' and x['pass'] is True,x
assert x['selected']['slug']==expected and x['selected']['qualified'] is True,x
assert x['selected']['finalfit_variant']==variant,x
assert x['selected']['checks']['C4_pass'] is True,x
PY
fi
if [[ "$FINALFIT_VARIANT" == strong1_widen8_* ]]; then
  test -f "$PROMOTION_GATE"
  python - "$PROMOTION_GATE" "$FINALFIT_VARIANT" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); variant=sys.argv[2]
expected={
 'strong1_widen8_lr5e7_seed58':'lr5e7_seed58',
 'strong1_widen8_lr1e6_seed59':'lr1e6_seed59',
 'strong1_widen8_single_lr5e7_seed60':'single_lr5e7_seed60',
}[variant]
assert x['stage']=='state8_strong1_widen8_joint_promotion_v1' and x['pass'] is True,x
assert x['selected']['slug']==expected and x['selected']['qualified'] is True,x
assert x['selected']['checks']['answer_C4_pass'] is True,x
assert x['selected']['checks']['diagnosis_C4_pass'] is True,x
PY
fi
if [[ "$FINALFIT_VARIANT" == strong1_rescue_* ]]; then
  test -f "$PROMOTION_GATE"
  python - "$PROMOTION_GATE" "$FINALFIT_VARIANT" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); variant=sys.argv[2]
expected={
 'strong1_rescue_taxonomy_joint_seed61':'taxonomy_joint',
 'strong1_rescue_taskanchor_c4strong_seed62':'taskanchor_c4strong',
}[variant]
assert x['stage']=='state8_strong1_rescue_joint_promotion_v1' and x['pass'] is True,x
assert x['selected']['slug']==expected and x['selected']['qualified'] is True,x
assert x['selected']['checks']['answer_C4_pass'] is True,x
assert x['selected']['checks']['diagnosis_C4_pass'] is True,x
PY
fi
if [[ "$FINALFIT_VARIANT" == strong1_midpoint_diagnosis_first_seed63 ]]; then
  test -f "$PROMOTION_GATE"
  python - "$PROMOTION_GATE" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['stage']=='state8_midpoint_diagnosis_first_joint_promotion_v1' and x['pass'] is True,x
assert x['selected']['slug']=='midpoint_diagnosis_first_seed63',x
assert x['selected']['qualified'] is True,x
assert x['selected']['checks']['answer_C4_pass'] is True,x
assert x['selected']['checks']['diagnosis_C4_pass'] is True,x
PY
fi
if [[ "$FINALFIT_VARIANT" == strong1_exact_diagnosis_curriculum_seed65 ]]; then
  test -f "$PROMOTION_GATE"
  python - "$PROMOTION_GATE" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['stage']=='state8_exact_diagnosis_curriculum_joint_promotion_v1' and x['pass'] is True,x
assert x['selected']['slug']=='exact_diagnosis_curriculum_seed65',x
assert x['selected']['qualified'] is True,x
assert x['selected']['checks']['answer_C4_pass'] is True,x
assert x['selected']['checks']['diagnosis_C4_pass'] is True,x
PY
fi

test "$(sha256sum scripts/train_recurrent_student_m4.py | cut -d' ' -f1)" = a67662448408a587f891d32d4c5170fdca0435adac47554268edf5a3972330b5
test "$(sha256sum scripts/train_recurrent_student_m4_state8_external_teacher_curriculum.py | cut -d' ' -f1)" = af1da212bc9c457a7b5a9297e895bb56fffcc7d98c8b2c9ff21ed0ae7f671539
test "$(sha256sum scripts/train_state8_stateonly_teacher_repair.py | cut -d' ' -f1)" = 3f8e05a6b35515c52b53482f894afbaedca6de4ce163a53b0737aa8971637452
test "$(sha256sum scripts/train_state8_semantic_causal_repair.py | cut -d' ' -f1)" = 14db1f4c430c61537a33c0b4617f1734a0bd51501249dfa8f61bc8c0ec01759a
test "$(sha256sum scripts/train_state8_taxonomy_causal_repair.py | cut -d' ' -f1)" = 26e62969a03aab89047dd42d5bbe6d144eae9bb77d0fa0b962b0da89b9b75347
test "$(sha256sum scripts/train_state8_task_anchored_causal_repair.py | cut -d' ' -f1)" = b622a62fffef45a19b3b02b771cee4412d5f9d17f1a3fe99c6ba046e6111e66b
test "$(sha256sum scripts/train_state8_diagnosis_first_causal_repair.py | cut -d' ' -f1)" = 7d638013f932f29f52cffeea709b96872df8ba0960b9bb4e56ed23cc3475b216
test "$(sha256sum scripts/state8_taxonomy_contract.py | cut -d' ' -f1)" = 1ca32112e76c795502f3cb1edbedcc7220f093c133fae8d20ff5cab789e512ed
test "$(sha256sum scripts/filter_state8_causal_diagnosis.py | cut -d' ' -f1)" = ba04c7d3cd1e74843678d2b3e66a0dea7c0aac647e7b149f18fc1f7b3bebbc58
test "$(sha256sum scripts/augment_state8_eedi_semantic_pairs.py | cut -d' ' -f1)" = c9006ac322d075ea1fa35ee9ace3a6561f799bdfea767786a6e459825b0ad46c
test "$(sha256sum scripts/state8_natural_curriculum_view.py | cut -d' ' -f1)" = 98c961180d2428b2c68ef9399996ed991d5bd34120a3046b4f56fb9ba4e09682
test "$(sha256sum scripts/augment_recurrent_causal_same_rule.py | cut -d' ' -f1)" = 1ae82541e97353d23b674af80dfb0e2d50cfd92e3a6ee99c2b55f3b0caf1d785
test "$(sha256sum scripts/augment_recurrent_causal_diagnosis.py | cut -d' ' -f1)" = b74684001e54bf0c9b023540c76a269708f1e8fa529cfec703a4439288590926
test "$(sha256sum scripts/distributed_epoch.py | cut -d' ' -f1)" = 20f85e37512965d560761de26d99d3329b15c16de679d34ad460a1407bad2f9a
test "$(sha256sum scripts/layerwise_recurrent_student_model.py | cut -d' ' -f1)" = b41fe1f0cfa055b784ccc5f6b2be2c07cfb424d3be6e63e6d1eb2ed061851152
test "$(sha256sum scripts/recurrent_student_model.py | cut -d' ' -f1)" = b0a33b50fe5631a9865358664b3c71d45dcbd9dad466e131107dbe3783eedb12
test "$(sha256sum scripts/select_latest_valid_distributed_checkpoint.py | cut -d' ' -f1)" = f038885f5bc054e4c326b3819dfd9e20a3f6b13f1145c1005f79a97fdf41d54b
test "$(sha256sum scripts/stage_distributed_resume_bundle.py | cut -d' ' -f1)" = c89b26b4e5b69d60774e2fceed57e7c7a635d4b5d403e7e0f1e952127ada6708
test "$(sha256sum scripts/publish_state8_final_checkpoint.py | cut -d' ' -f1)" = b34da8bc3c94488d1e4a1563bf40e606a877b780fecbb2d81956dd9a8fef238e

export STUDENTMODELING_EXTERNAL_STATE_TEACHER_ADAPTER="$TEACHER_ROOT/model"
export INITIAL_ADAPTER_SHA256="$(sha256sum "$INIT_DIR/adapter_model.safetensors" | cut -d' ' -f1)"
export INITIAL_RECURRENT_STATE_SHA256="$(sha256sum "$INIT_DIR/recurrent_state.pt" | cut -d' ' -f1)"
if [[ "$FINALFIT_VARIANT" == taxonomy_curriculum_* \
      || "$FINALFIT_VARIANT" == task_anchor_* \
      || "$FINALFIT_VARIANT" == strong1_widen8_* \
      || "$FINALFIT_VARIANT" == strong1_rescue_* \
      || "$FINALFIT_VARIANT" == strong1_midpoint_diagnosis_first_seed63 \
      || "$FINALFIT_VARIANT" == strong1_exact_diagnosis_curriculum_seed65 ]]; then
  python - "$PROMOTION_GATE" "$INITIAL_ADAPTER_SHA256" "$INITIAL_RECURRENT_STATE_SHA256" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
observed={'adapter_model':sys.argv[2],'recurrent_state':sys.argv[3]}
assert x['selected']['artifact_sha256']==observed,(x['selected'],observed)
PY
fi
export STUDENTMODELING_EXTERNAL_STATE_TEACHER_SHA256="$(python - "$INIT_DIR" "$INIT_COMPLETION" "$INIT_IMMUTABLE" "$TEACHER_ROOT" "$FINALFIT_VARIANT" "$EXPECTED_IMMUTABLE_STAGE" "$INIT_HISTORY_MODE" <<'PY'
import hashlib,json,pathlib,sys,torch
root=pathlib.Path(sys.argv[1]); receipt=json.load(open(sys.argv[2])); immutable=json.load(open(sys.argv[3])); teacher=pathlib.Path(sys.argv[4]); variant=sys.argv[5]; expected_stage=sys.argv[6]; history_mode=sys.argv[7]
def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()
observed={'adapter_model':digest(root/'adapter_model.safetensors'),
          'recurrent_state':digest(root/'recurrent_state.pt')}
teacher_hash=digest(teacher/'model/adapter_model.safetensors')
teacher_gate=json.load(open(teacher/'metrics/teacher_gate.json'))
teacher_provenance=json.load(open(teacher/'metrics/teacher_provenance_gate.json'))
teacher_publish=json.load(open(teacher/'metrics/publish_receipt.json'))
assert receipt['status']=='complete' and receipt['diagnostic'] is False,receipt
expected_steps=1000 if history_mode=='strict_state_only' else 5000
assert receipt['micro_steps']==expected_steps and receipt['real_examples_seen_once'] is True,receipt
assert {key:receipt['sha256'][key] for key in observed}==observed,(receipt,observed)
if history_mode=='strict_state_only':
 assert immutable['stage']==expected_stage and immutable['status']=='complete',immutable
 expected_arm=(variant.removeprefix('repair_') if variant.startswith('repair_') else
               ('taskanchor' if variant.startswith('task_anchor_') or
                variant=='strong1_rescue_taskanchor_c4strong_seed62'
                else 'trajectory01'))
 assert immutable['repair_arm']==expected_arm,immutable
 assert immutable['artifact_sha256']==observed,immutable
 assert immutable['state_shape']==[64,8,5120],immutable
 assert immutable['raw_history_visible_to_decoder'] is False,immutable
 assert immutable['history_dropout']==1.0 and immutable['online_teacher_kd'] is True,immutable
 if variant.startswith('semantic_'):
  assert immutable['train_entrypoint']=='scripts/train_state8_semantic_causal_repair.py',immutable
 if variant.startswith('taxonomy_curriculum_'):
  assert immutable['train_entrypoint']=='scripts/train_state8_taxonomy_causal_repair.py',immutable
 if variant.startswith('task_anchor_'):
  assert immutable['train_entrypoint']=='scripts/train_state8_task_anchored_causal_repair.py',immutable
 if variant.startswith('strong1_rescue_'):
  expected_entrypoint=(
   'scripts/train_state8_task_anchored_causal_repair.py'
   if 'taskanchor' in variant
   else 'scripts/train_state8_diagnosis_first_causal_repair.py'
  )
  assert immutable['train_entrypoint']==expected_entrypoint,immutable
 if variant=='strong1_midpoint_diagnosis_first_seed63':
  assert immutable['train_entrypoint']=='scripts/train_state8_diagnosis_first_causal_repair.py',immutable
 if variant=='strong1_exact_diagnosis_curriculum_seed65':
  assert immutable['train_entrypoint']=='scripts/train_state8_diagnosis_first_causal_repair.py',immutable
else:
 expected_replay={
  'highkd':'t3_heavy',
  'taxonomy_curriculum_t3heavy':'t3_heavy',
  'taxonomy_curriculum_balanced':'balanced',
  'strong1_widen8_lr5e7_seed58':'t3_heavy',
  'strong1_widen8_lr1e6_seed59':'t3_heavy',
  'strong1_widen8_single_lr5e7_seed60':'t3_heavy',
 }.get(variant,variant)
 assert immutable['stage']==expected_stage and immutable['replay_variant']==expected_replay,immutable
 assert immutable['teacher_adapter_sha256']==teacher_hash and immutable['sha256']==observed,immutable
 assert immutable['teacher_prompt_provenance_stage']=='state8_clean_prompt_teacher_provenance_gate_v2',immutable
 assert immutable['teacher_evaluation_input_sha256']=='78850c5434f24c3072b7ab89370400347955686e9aac5b99aadb69ae19c7cfa5',immutable
 assert immutable['teacher_clean_prompt_sha256']=='c35fd46e0cfbccd9141eccb524992bd5452416d32720686f74ad9373f57de314',immutable
 assert immutable['legacy_answer_adjacent_contracts']==0,immutable
assert teacher_gate['stage']=='state8_independent_full_history_teacher_gate_v1' and teacher_gate['pass'] is True,teacher_gate
assert teacher_gate['support']==1494 and teacher_gate['exact_match_is_authoritative'] is False,teacher_gate
assert teacher_publish['status']=='complete' and teacher_publish['adapter_sha256']==teacher_hash,teacher_publish
assert teacher_provenance['stage']=='state8_clean_prompt_teacher_provenance_gate_v2',teacher_provenance
assert teacher_provenance['teacher_gate_pass'] is True and teacher_provenance['teacher_adapter_sha256']==teacher_hash,teacher_provenance
assert teacher_provenance['evaluation_input_sha256']=='78850c5434f24c3072b7ab89370400347955686e9aac5b99aadb69ae19c7cfa5',teacher_provenance
assert teacher_provenance['clean_prompt_sha256']=='c35fd46e0cfbccd9141eccb524992bd5452416d32720686f74ad9373f57de314',teacher_provenance
assert teacher_provenance['legacy_answer_adjacent_contracts']==0,teacher_provenance
state=torch.load(root/'recurrent_state.pt',map_location='cpu',weights_only=False)
config=state['config']
assert config['state_architecture']=='layerwise' and config['state_token_count']==8,config
assert config['output_schema']=='label_diagnosis_evidence_v1',config
assert config['final_inference_route']=='strict_state_only',config
assert config['state8_natural_view_sha256']=='98c961180d2428b2c68ef9399996ed991d5bd34120a3046b4f56fb9ba4e09682',config
assert config['external_state_teacher_sha256']==teacher_hash,config
if history_mode=='strict_state_only':
 assert config['history_dropout_schedule']==[1.0],config
 assert config['final_inference_route']=='strict_state_only',config
 expected_label_weight=1.0 if variant in {'strong1_rescue_taxonomy_joint_seed61','strong1_midpoint_diagnosis_first_seed63','strong1_exact_diagnosis_curriculum_seed65'} else 4.0
 assert float(config['label_auxiliary_weight'])==expected_label_weight,config
 if variant in {'strong1_rescue_taxonomy_joint_seed61','strong1_midpoint_diagnosis_first_seed63','strong1_exact_diagnosis_curriculum_seed65'}:
  assert config['diagnosis_field_order']==['diagnosis','evidence','label'],config
  assert config['canonical_label_role']=='auxiliary_ordinary_weight',config
else:
 assert config['teacher_prompt_provenance_stage']=='state8_clean_prompt_teacher_provenance_gate_v2',config
 assert config['teacher_evaluation_input_sha256']=='78850c5434f24c3072b7ab89370400347955686e9aac5b99aadb69ae19c7cfa5',config
 assert config['teacher_clean_prompt_sha256']=='c35fd46e0cfbccd9141eccb524992bd5452416d32720686f74ad9373f57de314',config
 assert config['legacy_answer_adjacent_contracts']==0,config
print(teacher_hash)
PY
)"
test "${#STUDENTMODELING_EXTERNAL_STATE_TEACHER_SHA256}" = 64
test "${#INITIAL_ADAPTER_SHA256}" = 64
test "${#INITIAL_RECURRENT_STATE_SHA256}" = 64

python -m pytest -q tests/test_train_recurrent_student_m4.py \
  tests/test_state8_natural_curriculum_view.py \
  tests/test_train_state8_semantic_causal_repair.py \
  tests/test_train_state8_diagnosis_first_causal_repair.py \
  tests/test_augment_state8_eedi_semantic_pairs.py \
  tests/test_layerwise_recurrent_student_model.py \
  tests/test_distributed_epoch.py

exec 8>/tmp/state8-finalfit-data.lock
flock -x 8
if [[ ! -f /tmp/state8-finalfit-answer-diagnosis-data-v2.complete ]]; then
  rm -rf "$RAW_SPLIT" "$RECURRENT_SPLIT" "$CAUSAL_BASE" \
    "$CAUSAL_SAME_RULE" "$CAUSAL_DATA"
  mkdir -p "$RAW_SPLIT" "$RECURRENT_SPLIT" "$CAUSAL_BASE" \
    "$CAUSAL_SAME_RULE" "$CAUSAL_DATA"
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
    --output-dir "$CAUSAL_BASE" --max-updates 4
  python scripts/augment_recurrent_causal_same_rule.py \
    --train-source "$C4_DATA/train.jsonl" --valid-source "$C4_DATA/val.jsonl" \
    --causal-train "$CAUSAL_BASE/train.jsonl" --causal-valid "$CAUSAL_BASE/valid.jsonl" \
    --output-dir "$CAUSAL_SAME_RULE" --seed 20260804 --max-updates 4
  python scripts/augment_recurrent_causal_diagnosis.py \
    --train-jsonl "$CAUSAL_SAME_RULE/train.jsonl" \
    --valid-jsonl "$CAUSAL_SAME_RULE/valid.jsonl" \
    --output-dir "$CAUSAL_DATA"
  touch /tmp/state8-finalfit-answer-diagnosis-data-v2.complete
fi
flock -u 8
test "$(sha256sum "$FINAL_JSONL" | cut -d' ' -f1)" = 9961f746cb17e6c314e182d3453b6157022fb48b8de44d2e68bc8e87a1131ccd
test "$(sha256sum "$RECURRENT_SPLIT/valid.jsonl" | cut -d' ' -f1)" = 78bfce75576dc3873cbb6459669a952adeedb8b77bda15f281af9519e2506d14
test "$(sha256sum "$CAUSAL_SAME_RULE/train.jsonl" | cut -d' ' -f1)" = e2a57273bd4c49229efc8cec522fc221e460174c388c43e2b0d4888cc5079225
test "$(sha256sum "$CAUSAL_SAME_RULE/valid.jsonl" | cut -d' ' -f1)" = 8bd0dc3ec367ce30e5e5da3526681c21f852ba991394cac423682cd721f48c8e
test "$(sha256sum "$CAUSAL_DATA/train.jsonl" | cut -d' ' -f1)" = 1c8916da482c316dee92982aa3afaa0d2a123f1574140789b14ba5e1e92aee95
test "$(sha256sum "$CAUSAL_DATA/valid.jsonl" | cut -d' ' -f1)" = 05ae83c0c5df365988fedc1d9a6ef291bb8ebc3669135b0430a031f59989e12f
python - "$CAUSAL_DATA/augmentation_receipt.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['stage']=='recurrent_causal_answer_diagnosis_twins_v1' and x['status']=='passed',x
for split,answer_pairs in (('train',2102),('valid',990)):
 row=x[split]
 assert row['input_pairs']==answer_pairs and row['output_pairs']==2*answer_pairs,row
 assert row['target_kind_counts']=={
  'diagnosis_natural_schema':answer_pairs,'prospective_answer':answer_pairs},row
 assert row['canonical_label_auxiliary_supervision'] is True,row
 assert row['raw_history_visible_to_decoder'] is False,row
PY

if [[ "$FINALFIT_VARIANT" == semantic_eedi_* \
      || "$FINALFIT_VARIANT" == highcausal_eedi \
      || "$FINALFIT_VARIANT" == highcausal_taxonomy \
      || "$FINALFIT_VARIANT" == taxonomy_curriculum_* \
      || "$FINALFIT_VARIANT" == task_anchor_* \
      || "$FINALFIT_VARIANT" == strong1_widen8_* \
      || "$FINALFIT_VARIANT" == strong1_rescue_* \
      || "$FINALFIT_VARIANT" == strong1_midpoint_diagnosis_first_seed63 \
      || "$FINALFIT_VARIANT" == strong1_exact_diagnosis_curriculum_seed65 ]]; then
  exec 5>/tmp/state8-finalfit-eedi-pairs.lock; flock -x 5
  if [[ ! -f "$EEDI_PAIR_JSONL" || ! -f "$EEDI_PAIR_RECEIPT" ]]; then
    rm -f "$EEDI_PAIR_JSONL" "$EEDI_PAIR_RECEIPT"
    python scripts/augment_state8_eedi_semantic_pairs.py \
      --train-jsonl "$FINAL_JSONL" --corpus-jsonl "$FINAL_JSONL" \
      --output-jsonl "$EEDI_PAIR_JSONL" --receipt "$EEDI_PAIR_RECEIPT" \
      --seed 20260806
  fi
  flock -u 5
  python - "$EEDI_PAIR_RECEIPT" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['stage']=='state8_eedi_semantic_pair_augmentation_v1',x
assert x['status']=='passed' and x['rows']==259845,x
assert x['eligible_eedi_t3']==3476 and x['paired_eedi_t3']==2846,x
assert x['pair_coverage']>=0.81,x
assert x['hard_negative']==2846 and x['fallback_negative']==0,x
assert x['canonical_labels_visible_to_decoder'] is False,x
PY
  export FINAL_TRAIN_JSONL="$EEDI_PAIR_JSONL"
fi

if [[ "$STATE8_CAUSAL_DIAGNOSIS_ONLY" == 1 ]]; then
  export FINAL_CAUSAL_DIAGNOSIS_ONLY=/tmp/state8_finalfit_cf4_diagnosis_only_v1
  exec 4>/tmp/state8-finalfit-c4-diagnosis-only.lock; flock -x 4
  if [[ ! -f "$FINAL_CAUSAL_DIAGNOSIS_ONLY/receipt.json" ]]; then
    mkdir -p "$FINAL_CAUSAL_DIAGNOSIS_ONLY"
    python scripts/filter_state8_causal_diagnosis.py \
      --train-jsonl "$CAUSAL_DATA/train.jsonl" \
      --valid-jsonl "$CAUSAL_DATA/valid.jsonl" \
      --output-dir "$FINAL_CAUSAL_DIAGNOSIS_ONLY"
  fi
  flock -u 4
  export FINAL_CAUSAL_TRAIN_JSONL="$FINAL_CAUSAL_DIAGNOSIS_ONLY/train.jsonl"
  export FINAL_CAUSAL_VALID_JSONL="$FINAL_CAUSAL_DIAGNOSIS_ONLY/valid.jsonl"
fi

exec 7>/tmp/state8-finalfit-model.lock
flock -x 7
if [[ ! -f /tmp/state8-finalfit-model.complete ]]; then
  rm -rf "$LOCAL_MODEL"; mkdir -p "$LOCAL_MODEL"
  python scripts/copy_model_parallel.py \
    --source /mnt/amulet/studentmodeling/models/Qwen3-32B \
    --destination "$LOCAL_MODEL" --workers 16
  test -f "$LOCAL_MODEL/model.safetensors.index.json"
  touch /tmp/state8-finalfit-model.complete
fi
flock -u 7
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

export CUDA_VISIBLE_DEVICES="${OMPI_COMM_WORLD_LOCAL_RANK:?MPI local rank is required}"
export RANK="${OMPI_COMM_WORLD_RANK:?MPI rank is required}"
export WORLD_SIZE="${OMPI_COMM_WORLD_SIZE:?MPI world size is required}"
export LOCAL_RANK=0
test "$WORLD_SIZE" = 16

resume_checkpoint=""
selected="$(python scripts/select_latest_valid_distributed_checkpoint.py \
  --run-dir "$RUN_DIR" --expected-world-size 16 --max-round "$TOTAL_ROUNDS" \
  --expected-planned-optimizer-steps "$TOTAL_ROUNDS" \
  --save-steps 50 --recovery-save-steps 25)"
if [[ -n "$selected" ]]; then
  resume_key="$(python - "$selected" <<'PY'
import hashlib,os,sys
p=os.path.realpath(sys.argv[1]); s=os.stat(p)
print(hashlib.sha256(f'{p}:{s.st_size}:{s.st_mtime_ns}'.encode()).hexdigest())
PY
)"
  local_resume_dir="/tmp/state8-finalfit-resume/$resume_key"
  mkdir -p "$local_resume_dir"
  exec 6>"$local_resume_dir/stage.lock"; flock -x 6
  python scripts/stage_distributed_resume_bundle.py \
    --source-state "$selected" --destination "$local_resume_dir"
  flock -u 6
  resume_checkpoint="$local_resume_dir/recurrent_state.pt"
fi

if [[ -n "$resume_checkpoint" ]]; then
  INIT_ARGS=(--init-recurrent-state "$INIT_DIR/recurrent_state.pt" \
             --resume-from-checkpoint "$resume_checkpoint")
else
  INIT_ARGS=(--init-recurrent-state "$INIT_DIR/recurrent_state.pt")
fi

FINAL_TRAIN_MODE_ARGS=()
if [[ "$STATE8_LAYERWISE_TRANSITION_ONLY" == 1 ]]; then
  FINAL_TRAIN_MODE_ARGS+=(--layerwise-transition-only)
fi

python -u "$STATE8_FINALFIT_ENTRYPOINT" \
  --model "$LOCAL_MODEL" --init-adapter "$INIT_DIR" "${INIT_ARGS[@]}" \
  --state-architecture layerwise --layerwise-operation-checkpointing \
  --layerwise-transition-rank 256 --state-token-count 8 \
  --train-jsonl "$FINAL_TRAIN_JSONL" --valid-jsonl "$RECURRENT_SPLIT/valid.jsonl" \
  --causal-train-jsonl "$FINAL_CAUSAL_TRAIN_JSONL" \
  --causal-valid-jsonl "$FINAL_CAUSAL_VALID_JSONL" \
  --output-dir "$RUN_DIR" --state-token-alias '<|vision_pad|>' \
  --history-dropout-curriculum --dual-path-curriculum \
  --online-kd-weight "$STATE8_FINALFIT_ONLINE_KD_WEIGHT" --online-kd-temperature 1.0 --online-kd-topk 256 \
  --max-length 1536 --direct-offload-min-tokens 1 \
  --learning-rate "$STATE8_FINALFIT_LEARNING_RATE" --lr-scheduler cosine --warmup-ratio 0.03 \
  --generation-weight 1.0 --behavior-kd-weight 0.0 \
  --consistency-weight "$STATE8_REPAIR_TRAJECTORY_WEIGHT" --direct-replay-weight 0.0 \
  --causal-weight "$STATE8_CAUSAL_WEIGHT" --causal-margin "$STATE8_CAUSAL_MARGIN" \
  --causal-preference-reduction sum --causal-objective-mode specificity_triplet \
  --causal-repeat-fraction "$STATE8_CAUSAL_REPEAT_FRACTION" \
  --normalize-causal-exposure --gradient-accumulation-steps 1 \
  --eval-limit 300 --causal-eval-limit 165 \
  --save-steps 50 --checkpoint-retention 2 \
  --checkpoint-milestone-rounds $MILESTONE_ROUNDS \
  "${FINAL_TRAIN_MODE_ARGS[@]}" --distributed-mpi --distributed-world-size 16 --seed 42

if [[ "${OMPI_COMM_WORLD_RANK:-0}" != 0 ]]; then exit 0; fi
test -f "$RUN_DIR/completion.json"
test -f "$RUN_DIR/final_adapter/adapter_model.safetensors"
test -f "$RUN_DIR/recurrent_state.pt"
if [[ "$EXPECTED_HISTORY_MODE" == strict_state_only ]]; then
  test -f "$RUN_DIR/state8_stateonly_teacher_repair.json"
  export FINALFIT_VIEW_RECEIPT=state8_stateonly_teacher_repair.json
  if [[ "$FINALFIT_VARIANT" == semantic_* || "$FINALFIT_VARIANT" == highcausal_* \
        || "$FINALFIT_VARIANT" == taxonomy_curriculum_* \
        || "$FINALFIT_VARIANT" == task_anchor_* \
        || "$FINALFIT_VARIANT" == strong1_widen8_* \
        || "$FINALFIT_VARIANT" == strong1_rescue_* \
        || "$FINALFIT_VARIANT" == strong1_midpoint_diagnosis_first_seed63 \
        || "$FINALFIT_VARIANT" == strong1_exact_diagnosis_curriculum_seed65 ]]; then
    test -f "$RUN_DIR/semantic_causal_repair.json"
  fi
  if [[ "$FINALFIT_VARIANT" == highcausal_taxonomy \
        || "$FINALFIT_VARIANT" == taxonomy_curriculum_* \
        || "$FINALFIT_VARIANT" == task_anchor_* \
        || "$FINALFIT_VARIANT" == strong1_widen8_* \
        || "$FINALFIT_VARIANT" == strong1_rescue_* \
        || "$FINALFIT_VARIANT" == strong1_midpoint_diagnosis_first_seed63 \
        || "$FINALFIT_VARIANT" == strong1_exact_diagnosis_curriculum_seed65 ]]; then
    test -f "$RUN_DIR/taxonomy_causal_repair.json"
  fi
  if [[ "$FINALFIT_VARIANT" == task_anchor_* \
        || "$FINALFIT_VARIANT" == strong1_rescue_taskanchor_* ]]; then
    test -f "$RUN_DIR/task_anchored_causal_repair.json"
  fi
  if [[ "$FINALFIT_VARIANT" == strong1_rescue_taxonomy_joint_seed61 \
        || "$FINALFIT_VARIANT" == strong1_midpoint_diagnosis_first_seed63 \
        || "$FINALFIT_VARIANT" == strong1_exact_diagnosis_curriculum_seed65 ]]; then
    test -f "$RUN_DIR/diagnosis_first_causal_repair.json"
  fi
else
  test -f "$RUN_DIR/state8_external_teacher_curriculum.json"
  export FINALFIT_VIEW_RECEIPT=state8_external_teacher_curriculum.json
fi
python - "$RUN_DIR" "$STUDENTMODELING_EXTERNAL_STATE_TEACHER_SHA256" "$FINALFIT_VARIANT" "$EXPECTED_HISTORY_MODE" <<'PY'
import collections,json,os,pathlib,sys
root=pathlib.Path(sys.argv[1]); teacher_hash=sys.argv[2]; variant=sys.argv[3]; history_mode=sys.argv[4]; completion=json.load(open(root/'completion.json'))
config=json.load(open(root/'run_config.json')); history=json.load(open(root/'trainer_history.json'))
view=json.load(open(root/os.environ['FINALFIT_VIEW_RECEIPT']))
causal_augmentation=json.load(open(pathlib.Path(os.environ['CAUSAL_DATA'])/'augmentation_receipt.json'))
causal_counts=dict(causal_augmentation['train']['target_kind_counts'])
assert causal_counts=={
 'diagnosis_natural_schema':2102,
 'prospective_answer':2102,
},causal_counts
causal_supervision=[
 key for key in ('prospective_answer','diagnosis_natural_schema')
 if int(causal_counts.get(key,0))>0
]
init_artifacts={'adapter_model':os.environ['INITIAL_ADAPTER_SHA256'],
                'recurrent_state':os.environ['INITIAL_RECURRENT_STATE_SHA256']}
assert completion['status']=='complete' and completion['diagnostic'] is False,completion
assert completion['micro_steps']==259845 and completion['real_examples_seen']==259845,completion
assert completion['real_examples_seen_once'] is True and completion['exact_history_support'] is True,completion
assert completion['causal_exposure_count_matches_plan'] is True,completion
assert completion['all_causal_pairs_covered'] is True,completion
assert config['state_architecture']=='layerwise' and config['state_token_count']==8,config
assert config['history_dropout_curriculum'] is True and config['dual_path_curriculum'] is True,config
assert config['direct_offload_min_tokens']==1,config
assert config['output_schema']=='label_diagnosis_evidence_v1',config
assert config['final_inference_route']=='strict_state_only',config
assert config['external_state_teacher_sha256']==teacher_hash,config
assert float(config['online_kd_weight'])==float(os.environ['STATE8_FINALFIT_ONLINE_KD_WEIGHT']),config
assert config['teacher_gate_required'] is True and config['teacher_trainable'] is False,config
assert config['direct_replay_weight']==0.0,config
assert config['distributed_world_size']==16 and config['distributed_rounds']==16241,config
assert len(history)==259845 and [x['global_position'] for x in history]==list(range(259845)),history[:2]
counts=collections.Counter(float(x['history_dropout']) for x in history)
if history_mode=='strict_state_only':
 assert counts=={1.0:259845},counts
 assert all(bool(x['history_dropped']) for x in history),history[:2]
 assert all(float(x['online_kd_phase_weight'])==1.0 for x in history),history[:2]
 expected_label_weight=1.0 if variant in {'strong1_rescue_taxonomy_joint_seed61','strong1_midpoint_diagnosis_first_seed63','strong1_exact_diagnosis_curriculum_seed65'} else 4.0
 assert float(config['label_auxiliary_weight'])==expected_label_weight,config
 expected_trajectory=0.1 if variant=='repair_trajectory01' else 0.0
 if variant.startswith(('semantic_','highcausal_','taxonomy_curriculum_')): expected_trajectory=0.1
 if variant.startswith('strong1_widen8_'): expected_trajectory=0.1
 if variant in {'strong1_rescue_taxonomy_joint_seed61','strong1_midpoint_diagnosis_first_seed63','strong1_exact_diagnosis_curriculum_seed65'}: expected_trajectory=0.1
 if variant=='strong1_rescue_taskanchor_c4strong_seed62': expected_trajectory=1.0
 if variant.startswith('task_anchor_'): expected_trajectory=1.0
 assert float(config['consistency_weight'])==expected_trajectory,config
 assert float(view['online_kd_phase_weight'])==1.0,view
 assert float(view['label_auxiliary_weight'])==expected_label_weight,view
 assert float(view['trajectory_weight'])==expected_trajectory,view
 if variant.startswith(('semantic_','highcausal_','taxonomy_curriculum_','task_anchor_','strong1_widen8_','strong1_rescue_','strong1_exact_')):
  semantic=json.load(open(root/'semantic_causal_repair.json'))
  expected_random=('held_out_evaluation_only' if variant=='highcausal_updater'
                   else 'insufficient_evidence')
  assert semantic['random_target']==expected_random,semantic
  assert float(semantic['geometry_weight'])==float(os.environ['STATE8_GEOMETRY_WEIGHT']),semantic
  assert float(semantic['eedi_geometry_weight'])==float(os.environ['STATE8_EEDI_GEOMETRY_WEIGHT']),semantic
  assert semantic['eedi_pair_supervision_visible_at_inference'] is False,semantic
 if variant.startswith('task_anchor_'):
  anchor=json.load(open(root/'task_anchored_causal_repair.json'))
  assert anchor['stage']=='state8_task_anchored_taxonomy_causal_repair_v1',anchor
  assert anchor['diagnosis_anchor_schedule']==[1.0,0.5,0.25],anchor
  assert float(anchor['other_task_anchor_weight'])==0.1,anchor
  assert anchor['raw_history_visible_to_decoder'] is False,anchor
 if variant.startswith('strong1_rescue_taskanchor_'):
  anchor=json.load(open(root/'task_anchored_causal_repair.json'))
  assert anchor['stage']=='state8_task_anchored_taxonomy_causal_repair_v1',anchor
  assert anchor['diagnosis_anchor_schedule']==[1.0,0.5,0.25],anchor
  assert float(anchor['other_task_anchor_weight'])==0.1,anchor
  assert anchor['raw_history_visible_to_decoder'] is False,anchor
 if variant in {'strong1_rescue_taxonomy_joint_seed61','strong1_midpoint_diagnosis_first_seed63','strong1_exact_diagnosis_curriculum_seed65'}:
  diagnosis_first=json.load(open(root/'diagnosis_first_causal_repair.json'))
  assert diagnosis_first['stage']=='state8_diagnosis_first_causal_repair_v1',diagnosis_first
  assert diagnosis_first['field_order']==['diagnosis','evidence','label'],diagnosis_first
  assert diagnosis_first['semantic_fields_primary'] is True,diagnosis_first
  assert diagnosis_first['canonical_label_role']=='auxiliary_ordinary_weight',diagnosis_first
  assert float(diagnosis_first['label_auxiliary_weight'])==1.0,diagnosis_first
else:
 assert counts=={0.0:51969,0.25:51969,0.5:51969,0.75:51969,1.0:51969},counts
 assert config['trajectory_teacher_scope']=='none_unverified_carrier_trajectory_excluded',config
 assert config['consistency_weight']==0.0,config
audits={pathlib.Path(x['path']).name:x for x in view['dataset_audits']}
assert len(audits)==2 and all(int(x['changed_t3_rows'])>0 for x in audits.values()),audits
assert view['teacher_adapter_sha256']==teacher_hash,view
if history_mode=='strict_state_only':
 assert view['teacher_trainable'] is False,view
 assert view['history_dropout_schedule']==[1.0],view
 assert view['final_inference_route']=='strict_state_only',view
else:
 assert view['teacher_gate_required'] is True,view
receipt={'status':'final_fit_complete','stage':'state8_qualified_unified_final_fit_v1',
 'selected_replay_variant':variant,'one_epoch_exact':True,'real_examples':259845,'state_architecture':'layerwise',
 'state_shape':[64,8,5120],'history_dropout_counts':dict(sorted(counts.items())),
 'causal_supervision':causal_supervision,
 'causal_target_kind_counts':causal_counts,
 'causal_output_schema':'label_diagnosis_evidence_v1',
 'canonical_label_auxiliary_supervision':True,
 'initial_artifact_sha256':init_artifacts,
 'teacher_adapter_sha256':teacher_hash,'teacher_gate_required':True,
 'online_kd_weight':float(config['online_kd_weight']),
 'history_training_mode':history_mode,
 'label_auxiliary_weight':float(config.get('label_auxiliary_weight',1.0)),
 'diagnosis_field_order':config.get('diagnosis_field_order',['label','diagnosis','evidence']),
 'canonical_label_role':config.get('canonical_label_role','auxiliary_weighted'),
 'trajectory_weight':float(config['consistency_weight']),
 'semantic_c4_geometry_weight':float(config.get('semantic_geometry_weight',0.0)),
 'semantic_eedi_geometry_weight':float(config.get('eedi_semantic_geometry_weight',0.0)),
 'semantic_random_target':config.get('semantic_random_target','legacy_recipient'),
 'eedi_pair_supervision_visible_at_inference':False,
 'direct_offload_min_tokens':int(config['direct_offload_min_tokens']),
 'teacher_prompt_provenance_stage':'state8_clean_prompt_teacher_provenance_gate_v2',
 'teacher_evaluation_input_sha256':'78850c5434f24c3072b7ab89370400347955686e9aac5b99aadb69ae19c7cfa5',
 'teacher_clean_prompt_sha256':'c35fd46e0cfbccd9141eccb524992bd5452416d32720686f74ad9373f57de314',
 'legacy_answer_adjacent_contracts':0,
 'raw_history_visible_to_decoder':False,'persistent_history_kv_cache':False,
 'adapter_sha256':completion['sha256']['adapter_model'],
 'recurrent_state_sha256':completion['sha256']['recurrent_state']}
(root/'state8_final_fit_receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
PY
python scripts/publish_state8_final_checkpoint.py \
  --source "$RUN_DIR" --destination "$CANONICAL_RUN_DIR"
test -f "$CANONICAL_RUN_DIR/canonical_publish_receipt.json"
cp -r "$RUN_DIR/final_adapter/." "$AMLT_OUTPUT_DIR/model/"
cp "$RUN_DIR/recurrent_state.pt" "$AMLT_OUTPUT_DIR/model/recurrent_state.pt"
cp "$RUN_DIR/completion.json" "$AMLT_OUTPUT_DIR/metrics/completion.json"
cp "$RUN_DIR/run_config.json" "$AMLT_OUTPUT_DIR/metrics/run_config.json"
cp "$RUN_DIR/training_metrics.json" "$AMLT_OUTPUT_DIR/metrics/training_metrics.json"
cp "$RUN_DIR/validation_metrics.json" "$AMLT_OUTPUT_DIR/metrics/validation_metrics.json"
cp "$RUN_DIR/causal_validation_metrics.json" "$AMLT_OUTPUT_DIR/metrics/causal_validation_metrics.json"
cp "$RUN_DIR/$FINALFIT_VIEW_RECEIPT" "$AMLT_OUTPUT_DIR/metrics/$FINALFIT_VIEW_RECEIPT"
if [[ -f "$RUN_DIR/diagnosis_first_causal_repair.json" ]]; then
  cp "$RUN_DIR/diagnosis_first_causal_repair.json" \
    "$AMLT_OUTPUT_DIR/metrics/diagnosis_first_causal_repair.json"
fi
cp "$RUN_DIR/state8_final_fit_receipt.json" "$AMLT_OUTPUT_DIR/metrics/state8_final_fit_receipt.json"
cp "$CANONICAL_RUN_DIR/canonical_publish_receipt.json" "$AMLT_OUTPUT_DIR/metrics/canonical_publish_receipt.json"
cp "$RECURRENT_SPLIT/audit.json" "$AMLT_OUTPUT_DIR/metrics/data_audit.json"
cp "$RECURRENT_SPLIT/final_fit_manifest.json" "$AMLT_OUTPUT_DIR/metrics/final_fit_manifest.json"
cp "$CAUSAL_SAME_RULE/augmentation_receipt.json" "$AMLT_OUTPUT_DIR/metrics/causal_same_rule_augmentation.json"
cp "$CAUSAL_DATA/augmentation_receipt.json" "$AMLT_OUTPUT_DIR/metrics/causal_answer_diagnosis_augmentation.json"
if [[ "$STATE8_CAUSAL_DIAGNOSIS_ONLY" == 1 ]]; then
  cp "$FINAL_CAUSAL_DIAGNOSIS_ONLY/receipt.json" \
    "$AMLT_OUTPUT_DIR/metrics/causal_diagnosis_only.json"
fi
if [[ -f "$EEDI_PAIR_RECEIPT" ]]; then
  cp "$EEDI_PAIR_RECEIPT" "$AMLT_OUTPUT_DIR/metrics/eedi_semantic_pair_augmentation.json"
fi
