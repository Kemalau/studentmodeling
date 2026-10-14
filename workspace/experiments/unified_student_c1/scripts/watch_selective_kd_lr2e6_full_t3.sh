#!/usr/bin/env bash
set -euo pipefail
project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
export CONFIG="$project/amlt/feeds_selective_kd_lr2e6_full_t3_20260804.yaml"
export VALIDATOR="$project/scripts/validate_selective_kd_lr2e6_full_t3_amlt.py"
export BASE_EXPERIMENT=selective-kd-lr2e6-full-t3-seed42-20260804
export JOB=selective-kd-lr2e6-full-t3-seed42
export STATE_DIR="$project/results/layerwise_unified_final_pipeline_20260731/selective_kd_lr2e6_full_t3_20260804"
export DESCRIPTION="selective-KD lr2e-6 strict full-T3"
exec "$project/scripts/watch_full_t3_experiment.sh"

