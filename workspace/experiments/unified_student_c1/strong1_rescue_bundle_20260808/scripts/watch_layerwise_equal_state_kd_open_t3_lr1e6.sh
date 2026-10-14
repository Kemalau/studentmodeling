#!/usr/bin/env bash
set -euo pipefail

project_dir=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
cd "$project_dir"
export STATE_SLUG=equal_state_kd_open_t3_lr1e6
export CONFIG_PATH="${CONFIG_PATH:-amlt/feeds_layerwise_equal_state_kd_open_t3_lr1e6_20260803.yaml}"
# The Feeds target accepted and allocated the 80G4-A100 form.  A previous
# controller migration to 4x80G1-A100 was rejected at submission time, so all
# recoveries stay on the validated single-node four-rank topology.
export BASE_EXPERIMENT="${BASE_EXPERIMENT:-layerwise-equal-t3-t6-open-t3-statekd-lr1e6-recovery1-seed42-20260803}"
export RECOVERY_PREFIX="${RECOVERY_PREFIX:-layerwise-equal-t3-t6-open-t3-statekd-lr1e6}"
export JOB_NAME=layerwise-equal-t3-t6-open-t3-statekd-lr1e6-seed42
export LOSS_PROFILE_EXPECTED=state_kd
export SUMMARY_BRANCH_EXPECTED=equal_t3_t6_open_t3
export SUMMARY_LEARNING_RATE=1e-6
export VALIDATOR=scripts/validate_layerwise_equal_state_kd_open_t3_lr1e6_amlt.py
export EXPERIMENT_DESCRIPTION='Equal-curriculum open-T3 selective-KD lr1e-6 midpoint'
export EXPERIMENT_DATE=20260803

exec bash scripts/watch_layerwise_equal_state_path.sh
