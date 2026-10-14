#!/usr/bin/env bash
set -euo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
export STATE8_FINALFIT_PROMOTION_DIR="$project/results/state8_curriculum_promotion_20260812"
export STATE8_FINALFIT_STATE_DIR="$project/results/state8_curriculum_finalfit_world2_20260812"
export STATE8_FINALFIT_TEMPLATE="$project/amlt/feeds_state8_eedi_same_question_final_fit_world2_20260811.yaml"
export STATE8_FINALFIT_BASE_EXPERIMENT=state8-curriculum-unified-finalfit-world2-20260812
exec bash "$project/scripts/watch_state8_eedi_same_question_finalfit_world2.sh"
