#!/usr/bin/env bash
set -euo pipefail

project_dir=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
state_dir="$project_dir/results/layerwise_unified_final_pipeline_20260731/equal_state_path_open_t3"
mkdir -p "$state_dir"
cd "$project_dir"
export AZURE_CONFIG_DIR="${AZURE_CONFIG_DIR:-/home/v-yuejiali/.azure-studentmodeling-feeds}"

# Reuse the audited, idempotent one-arm monitor. All scientific knobs are
# supplied explicitly; recovery waves retain the same AMLT config and seed.
export STATE_SLUG=equal_state_path_open_t3
export CONFIG_PATH=amlt/feeds_layerwise_equal_state_path_open_t3_20260803.yaml
export BASE_EXPERIMENT=layerwise-equal-t3-t6-open-t3-state-path-seed42-20260803
export RECOVERY_PREFIX=layerwise-equal-t3-t6-open-t3-state-path
export JOB_NAME=layerwise-equal-t3-t6-open-t3-state-path-seed42
export LOSS_PROFILE_EXPECTED=state_path
export SUMMARY_BRANCH_EXPECTED=equal_t3_t6_open_t3
export VALIDATOR=scripts/validate_layerwise_equal_state_path_open_t3_amlt.py
export EXPERIMENT_DESCRIPTION='Matched equal-T3/T6 open-T3 state-path diagnostic'

exec bash scripts/watch_layerwise_equal_state_path.sh
