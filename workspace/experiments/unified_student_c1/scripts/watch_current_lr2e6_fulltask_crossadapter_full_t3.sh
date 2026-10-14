#!/usr/bin/env bash
set -euo pipefail
project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
export CONFIG="$project/amlt/feeds_t3_encoder_decoder_crossmatrix_20260804.yaml"
export VALIDATOR="$project/scripts/validate_t3_encoder_decoder_crossmatrix_amlt.py"
export BASE_EXPERIMENT=current-lr2e6-fulltask-crossadapter-full-t3-seed42-20260804
export JOB=current-lr2e6-fulltask-crossadapter-full-t3-seed42
export STATE_DIR="$project/results/layerwise_unified_final_pipeline_20260731/current_lr2e6_fulltask_crossadapter_full_t3_20260804"
export DESCRIPTION="current-lr2e6/full-task cross-adapter strict full-T3"
exec "$project/scripts/watch_full_t3_experiment.sh"
