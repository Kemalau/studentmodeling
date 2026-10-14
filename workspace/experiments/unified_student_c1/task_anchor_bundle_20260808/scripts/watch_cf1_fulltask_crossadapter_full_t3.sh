#!/usr/bin/env bash
set -euo pipefail
project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
export CONFIG="$project/amlt/feeds_cf1_fulltask_crossadapter_full_t3_20260804.yaml"
export VALIDATOR="$project/scripts/validate_cf1_fulltask_crossadapter_full_t3_amlt.py"
export BASE_EXPERIMENT=cf1-fulltask-crossadapter-full-t3-seed42-20260804
export JOB=cf1-fulltask-crossadapter-full-t3-seed42
export STATE_DIR="$project/results/layerwise_unified_final_pipeline_20260731/cf1_fulltask_crossadapter_full_t3_20260804"
export DESCRIPTION="CF1/full-task cross-adapter strict full-T3"
exec "$project/scripts/watch_full_t3_experiment.sh"

