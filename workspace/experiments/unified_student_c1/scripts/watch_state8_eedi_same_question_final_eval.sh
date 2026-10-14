#!/usr/bin/env bash
set -euo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
export FIT_DIR="$project/results/state8_eedi_same_question_finalfit_world2_20260811"
export STATE_DIR="$project/results/state8_eedi_same_question_complete_final_eval_20260811"
export FINAL_EVAL_CONFIG="$project/amlt/feeds_state8_eedi_same_question_final_eval_20260811.yaml"
export BASE_EXPERIMENT=state8-eedi-same-question-complete-final-eval-20260811
export RECOVERY_EXPERIMENT_PREFIX=state8-eedi-same-question-final-eval-recovery
export RECOVERY_EXPERIMENT_SUFFIX=-20260811
export SAME_QUESTION_FINAL_EVAL=1
export MAX_RECOVERY_WAVES=32

exec bash "$project/scripts/watch_state8_final_eval.sh"
