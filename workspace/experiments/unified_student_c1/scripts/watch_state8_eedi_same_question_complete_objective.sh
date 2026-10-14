#!/usr/bin/env bash
set -euo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
export OBJECTIVE_MODEL_DIR="$project/results/state8_eedi_same_question_objective_20260811"
export OBJECTIVE_C3_DIR="$project/results/c3_exposure_matched_world2_20260809"
export OBJECTIVE_STATE_DIR="$project/results/state8_eedi_same_question_complete_objective_20260811"
export OBJECTIVE_FINAL_FIT_DIR="$project/results/state8_eedi_same_question_finalfit_world2_20260811"
export OBJECTIVE_FINAL_EVAL_DIR="$project/results/state8_eedi_same_question_complete_final_eval_20260811"
export OBJECTIVE_SELECTION="$project/results/state8_eedi_same_question_promotion_20260811/promotion.json"
export OBJECTIVE_FINAL_GATE="$OBJECTIVE_FINAL_EVAL_DIR/final_gate.json"
export OBJECTIVE_C3_SUMMARY="$OBJECTIVE_C3_DIR/summary.json"
export OBJECTIVE_MODEL_ATTENTION_DIR="$OBJECTIVE_FINAL_FIT_DIR"

exec bash "$project/scripts/watch_complete_unified_objective.sh"
