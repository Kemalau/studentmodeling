#!/usr/bin/env bash
set -euo pipefail

project_dir=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
cd "$project_dir"
export STATE_SLUG=equal_state_kd_open_t3_lr_scan
export CONFIG_PATH=amlt/feeds_layerwise_equal_state_kd_open_t3_lr_scan_20260803.yaml
export BASE_EXPERIMENT=layerwise-equal-t3-t6-open-t3-statekd-lr-scan
export JOB_PREFIX=layerwise-equal-t3-t6-open-t3-statekd
export LOSS_PROFILE_EXPECTED=state_kd
export SUMMARY_BRANCH_EXPECTED=equal_t3_t6_open_t3
export VALIDATOR=scripts/validate_layerwise_equal_state_kd_open_t3_lr_scan_amlt.py
export EXPERIMENT_DESCRIPTION='Equal-curriculum open-T3 selective-KD matched LR scan'
export EXPERIMENT_DATE=20260803

exec bash scripts/watch_layerwise_eedi_full_lr_scan.sh
