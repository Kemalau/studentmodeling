#!/usr/bin/env bash
set -euo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
export STATE8_CURRICULUM_REPAIR=1
export STATE8_REPAIR_SELECTION_DIR="$project/results/state8_curriculum_selection_20260812"
export STATE8_REPAIR_STATE_DIR="$project/results/state8_curriculum_repair_branches_20260812"
export STATE8_REPAIR_BASE_EXPERIMENT=state8-curriculum-repair-branches-20260812
exec bash "$project/scripts/watch_state8_eedi_same_question_repair_branches.sh"
