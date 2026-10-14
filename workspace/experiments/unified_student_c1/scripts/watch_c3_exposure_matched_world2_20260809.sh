#!/usr/bin/env bash
set -euo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
export C3_STATE_DIR="$project/results/c3_exposure_matched_world2_20260809"
export C3_CONFIG="$project/amlt/feeds_c3_exposure_matched_round1500_world2_20260809.yaml"
export C3_BASE_EXPERIMENT=c3-exposure-matched-world2-promoted-order-v1-20260809
export C3_RECOVERY_PREFIX=c3-world2-promoted-order-v1
export C3_EXPERIMENT_MAP="$C3_STATE_DIR/experiments.promoted-order-v1.tsv"
export C3_REQUIRED_GPUS=2
export C3_EXPECTED_TOPOLOGY=world2
export C3_MAX_RECOVERY_WAVES=24
export C3_MAX_ACTIVE_ARMS=2
# C3 is independent of the read-only checkpoint-5000 T3/C4 screen.  Waiting
# for both evaluators to finish serialized six-GPU C3 continuations behind
# hours of generation even when the controller's fail-closed quota audit had
# enough room for both workloads.  Submission is the safe dependency: it
# proves the priority final evaluators were accepted first, while
# ``wait_quota`` still rechecks both the 32-GPU user cap and live Feeds quota
# immediately before every C3 arm submission.
export C3_PRIORITY_READY_MARKER="$project/results/state8_exact_diagnosis_checkpoint5000_eval_20260809/submitted.state8-exact-diagnosis-checkpoint5000-eval-20260809"

exec bash "$project/scripts/watch_c3_exposure_matched_round1500.sh"
