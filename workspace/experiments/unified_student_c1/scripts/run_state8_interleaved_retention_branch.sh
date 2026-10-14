#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "STATE8_INTERLEAVED_RETENTION_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

: "${REPAIR_SEED:?}" "${SOURCE_CHECKPOINT_ROOT:?}"
: "${EXPECTED_STUDENT_ADAPTER_SHA256:?}" "${EXPECTED_STUDENT_STATE_SHA256:?}"
: "${REPAIR_REMOTE_ROOT:?}" "${STATE8_REPAIR_MANIFEST:?}"
: "${STATE8_REPAIR_MANIFEST_SHA256:?}" "${STATE8_T6_ANCHOR_SCHEDULE:?}"

test "${REPAIR_BRANCH:?}" = early_label_high
test "$REPAIR_SEED" = 8401
test "${SOURCE_CHECKPOINT_ARM:?}" = curriculum_current_query
test "${SOURCE_CHECKPOINT_STEP:?}" = 3000
test "${SOURCE_ANCHOR_MODE:?}" = full_mse1
test "$SOURCE_CHECKPOINT_ROOT" = /mnt/amulet/studentmodeling/checkpoint_snapshots/state8_diagnosis_first_eedi_current_query_curriculum_20260812/checkpoint-3000
test "$REPAIR_REMOTE_ROOT" = /mnt/amulet/studentmodeling/diagnostics/state8_stateonly_repair_seed8401_taskanchor_source_anchor_full_mse1_eedi_same_question_v1_early_label_high_seed8401
test "$STATE8_REPAIR_MANIFEST" = manifests/state8_interleaved_retention_seed8401_v1.json
test "$STATE8_T6_ANCHOR_SCHEDULE" = 1.0,1.0,1.0
test "${STATE8_EARLY_LABEL_BINDING_WEIGHT:?}" = 0.5
test "$(sha256sum "$STATE8_REPAIR_MANIFEST" | cut -d' ' -f1)" = "$STATE8_REPAIR_MANIFEST_SHA256"

source_runner=scripts/run_state8_eedi_same_question_repair_branch.sh
test "$(sha256sum "$source_runner" | cut -d' ' -f1)" = 68086c2e3d73ecd0d9f8a18e726cc1a1eeeecd1e2e6c225f9ae31ab07e148554
entrypoint=scripts/train_state8_interleaved_retention_repair.py
entrypoint_sha="$(sha256sum "$entrypoint" | cut -d' ' -f1)"

export STATE8_RENDER_ONLY=1
bash "$source_runner"
unset STATE8_RENDER_ONLY
rendered=/tmp/run_state8_interleaved_retention.final.sh
python - /tmp/run_state8_eedi_same_question_causal_repair.final.sh "$rendered" \
  "$entrypoint_sha" <<'PY'
from pathlib import Path
import os, sys

source, output = map(Path, sys.argv[1:3])
entrypoint_sha = sys.argv[3]
value = source.read_text(encoding="utf-8")

def replace(old: str, new: str) -> None:
    global value
    observed = value.count(old)
    if observed != 1:
        raise ValueError((old, observed))
    value = value.replace(old, new)

source_case = '''  scripts/train_state8_diagnosis_first_source_anchored_repair.py)
    test "$(sha256sum "$TRAIN_ENTRYPOINT" | cut -d' ' -f1)" = "${STATE8_SOURCE_ANCHOR_ENTRYPOINT_SHA256:?}"
    test "$(sha256sum scripts/state8_taxonomy_contract.py | cut -d' ' -f1)" = 1ca32112e76c795502f3cb1edbedcc7220f093c133fae8d20ff5cab789e512ed
    test "${STATE8_REPAIR_LABEL_WEIGHT:-}" = 1.0
    test "${STATE8_CAUSAL_LABEL_WEIGHT:-}" = 1.0
    test "${STATE8_TRAJECTORY_RELATIVE_MSE_WEIGHT:-}" != 0 ;;'''
interleaved_case = source_case + f'''
  scripts/train_state8_interleaved_retention_repair.py)
    test "$(sha256sum "$TRAIN_ENTRYPOINT" | cut -d' ' -f1)" = "{entrypoint_sha}"
    test "$(sha256sum scripts/state8_taxonomy_contract.py | cut -d' ' -f1)" = 1ca32112e76c795502f3cb1edbedcc7220f093c133fae8d20ff5cab789e512ed
    test "${{STATE8_REPAIR_LABEL_WEIGHT:-}}" = 1.0
    test "${{STATE8_CAUSAL_LABEL_WEIGHT:-}}" = 1.0
    test "${{STATE8_TRAJECTORY_RELATIVE_MSE_WEIGHT:-}}" != 0 ;;'''
replace(source_case, interleaved_case)

source_or = '|| "$TRAIN_ENTRYPOINT" == scripts/train_state8_diagnosis_first_source_anchored_repair.py'
value = value.replace(
    source_or,
    source_or + ' \\\n      || "$TRAIN_ENTRYPOINT" == scripts/train_state8_interleaved_retention_repair.py',
)
if value.count("train_state8_interleaved_retention_repair.py") < 8:
    raise ValueError("Interleaved entrypoint was not composed into every source-aware gate")

source_only = 'if [[ "$TRAIN_ENTRYPOINT" == scripts/train_state8_diagnosis_first_source_anchored_repair.py ]]; then'
if value.count(source_only) != 2:
    raise ValueError((source_only, value.count(source_only)))
value = value.replace(
    source_only,
    'if [[ "$TRAIN_ENTRYPOINT" == scripts/train_state8_diagnosis_first_source_anchored_repair.py \\\n'
    '      || "$TRAIN_ENTRYPOINT" == scripts/train_state8_interleaved_retention_repair.py ]]; then',
)

replace(
    "                        'train_state8_diagnosis_first_source_anchored_repair.py')):",
    "                        'train_state8_diagnosis_first_source_anchored_repair.py',\n"
    "                        'train_state8_interleaved_retention_repair.py')):",
)
replace(
    '''if [[ "$TRAIN_ENTRYPOINT" == scripts/train_state8_diagnosis_first_source_anchored_repair.py \\
      || "$TRAIN_ENTRYPOINT" == scripts/train_state8_interleaved_retention_repair.py ]]; then
  test -f "$RUN_DIR/diagnosis_first_source_anchored_repair.json"
fi''',
    '''if [[ "$TRAIN_ENTRYPOINT" == scripts/train_state8_diagnosis_first_source_anchored_repair.py \\
      || "$TRAIN_ENTRYPOINT" == scripts/train_state8_interleaved_retention_repair.py ]]; then
  test -f "$RUN_DIR/diagnosis_first_source_anchored_repair.json"
fi
if [[ "$TRAIN_ENTRYPOINT" == scripts/train_state8_interleaved_retention_repair.py ]]; then
  test -f "$RUN_DIR/interleaved_retention_repair.json"
fi''',
)

# The immutable index manifest is already materialized in the requested order.
# Bind that order to the optimizer and prove every recorded example follows it.
replace(
    'export TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:-scripts/train_state8_stateonly_teacher_repair.py}"',
    'export TRAIN_ENTRYPOINT=scripts/train_state8_interleaved_retention_repair.py',
)
replace(
    '''cp "$SUBSET.receipt.json" "$AMLT_OUTPUT_DIR/metrics/training_subset_receipt.json"''',
    '''cp "$SUBSET.receipt.json" "$AMLT_OUTPUT_DIR/metrics/training_subset_receipt.json"
cp "$RUN_DIR/interleaved_retention_repair.json" \
  "$AMLT_OUTPUT_DIR/metrics/interleaved_retention_repair.json"''',
)
output.write_text(value, encoding="utf-8")
PY
bash -n "$rendered"
if [[ "${STATE8_INTERLEAVED_RENDER_ONLY:-0}" == 1 ]]; then
  exit 0
fi

# The repair renderer normally executes its generated launcher as a descendant
# shell, so these bindings are inherited.  This wrapper intentionally stops the
# renderer in render-only mode and later executes the generated launcher from
# this parent shell; re-establish the exact source-anchor contract explicitly.
# Without this block the generated launcher falls back to the base defaults
# (for example STATE8_REPAIR_LABEL_WEIGHT=4.0) and fails closed before loading
# the checkpoint.
export TRAIN_SEED="$REPAIR_SEED"
export RUN_TAG="source_anchor_full_mse1_eedi_same_question_v1_${REPAIR_BRANCH}_seed${REPAIR_SEED}"
export STATE8_LAYERWISE_TRANSITION_ONLY=0
export STATE8_UPDATER_ONLY=0
export STATE8_TRAJECTORY_RELATIVE_MSE_WEIGHT=1.0
export REQUESTED_WORLD_SIZE=1
export GRADIENT_ACCUMULATION_STEPS=4
export STATE8_REPAIR_SAVE_STEPS=20
export STATE8_REPAIR_RECOVERY_SAVE_STEPS=20
export STATE8_REPAIR_LABEL_WEIGHT=1.0
export STATE8_CAUSAL_LABEL_WEIGHT=1.0
export STATE8_RANDOM_NEUTRAL=1
export STATE8_GEOMETRY_WEIGHT=0.10
export STATE8_DIFFERENT_MAX_COSINE=0.90
export STATE8_CAUSAL_DIAGNOSIS_ONLY=0
export STATE8_FORMAL_SCOPE=unified
export STATE8_EEDI_ANCHOR_SCHEDULE=1.0,1.0,1.0
export STATE8_MALRULE_ANCHOR_SCHEDULE=1.0,1.0,1.0
export STATE8_OTHER_DIAGNOSIS_ANCHOR_SCHEDULE=1.0,0.5,0.25
export STATE8_OTHER_ANCHOR_WEIGHT=0.1
export STATE8_TRAJECTORY_COSINE_WEIGHT=1.0
export STATE8_DIAGNOSIS_FIRST_ENTRYPOINT_SHA256=e38c93ac1f44779716fb9f82ad9ced47a88d1cd91c1319354f47168b1088911e
export STATE8_SOURCE_ANCHOR_ENTRYPOINT_SHA256=f3f2954cb93df4d312078f727648a3563c85f71ce95a9799a4c9a75bdeb25265
export STATE8_DIAGNOSIS_FIRST_FORMAL_SHA256=967711843054673eab67e43a8d5392ac7329a31bf5005cc1ec5be8a60507244f
export STATE8_DIAGNOSIS_FIRST_C4_SHA256=660df80664c6012f6eac1d379cc1d611c9f085c91496d1c9cc8e7f7bcb30d7ca
export REPAIR_ARM=taskanchor
# STATE8_RENDER_ONLY stops the source renderer before its child shell exports
# the selected checkpoint paths.  The final launcher runs from this parent,
# so restore those source bindings here along with the training hyperparameters.
export STUDENT_ROOT="$SOURCE_CHECKPOINT_ROOT"
export STUDENT_MODEL_DIR="$STUDENT_ROOT/adapter"
export STUDENT_STATE_PATH="$STUDENT_ROOT/recurrent_state.pt"
export STUDENT_RECEIPT="$STUDENT_ROOT/checkpoint_completion.json"
bash "$rendered"
