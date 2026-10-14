#!/usr/bin/env bash
set -euo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
source "$project/scripts/use_studentmodeling_feeds_auth.sh"

subscription=b6dc87f3-c479-49c8-8cb5-7896da3ff895
az account get-access-token --subscription "$subscription" \
  --query expiresOn -o tsv >/dev/null

restart() {
  local session="$1" state="$2" command="$3"
  tmux kill-session -t "$session" 2>/dev/null || true
  rm -f "$project/results/$state/controller.active.pid"
  mkdir -p "$project/results/$state"
  tmux new-session -d -s "$session" \
    "cd '$project' && source scripts/use_studentmodeling_feeds_auth.sh && exec $command 2>&1 | tee -a 'results/$state/controller.tmux.log'"
}

restart state8_eedi_same_question_causal_repair_20260811 \
  state8_eedi_same_question_causal_repair_20260811 \
  "bash scripts/watch_state8_eedi_same_question_causal_repair.sh"
restart state8_eedi_same_question_full_mse1_20260811 \
  state8_eedi_same_question_full_mse1_20260811 \
  "bash scripts/watch_state8_eedi_same_question_full_mse1.sh"

for arm in updater full; do
  for step in 20 100; do
    restart "state8_eedi_same_question_${arm}_checkpoint${step}_screen_20260811" \
      "state8_eedi_same_question_${arm}_checkpoint${step}_screen_20260811" \
      "env SCREEN_ARM=$arm CHECKPOINT_STEP=$step bash scripts/watch_state8_eedi_same_question_checkpoint_screen.sh"
  done
done

restart state8_eedi_same_question_early_selection_20260811 \
  state8_eedi_same_question_early_selection_20260811 \
  "bash scripts/watch_state8_eedi_same_question_early_selection.sh"
restart state8_eedi_same_question_repair_branches \
  state8_eedi_same_question_repair_branches_20260811 \
  "bash scripts/watch_state8_eedi_same_question_repair_branches.sh"
restart state8_eedi_same_question_promotion_20260811 \
  state8_eedi_same_question_promotion_20260811 \
  "bash scripts/watch_state8_eedi_same_question_promotion.sh"
restart state8_eedi_same_question_finalfit_world2 \
  state8_eedi_same_question_finalfit_world2_20260811 \
  "bash scripts/watch_state8_eedi_same_question_finalfit_world2.sh"
restart state8_eedi_same_question_complete_final_eval_20260811 \
  state8_eedi_same_question_complete_final_eval_20260811 \
  "bash scripts/watch_state8_eedi_same_question_final_eval.sh"
restart state8_eedi_same_question_complete_objective_20260811 \
  state8_eedi_same_question_complete_objective_20260811 \
  "bash scripts/watch_state8_eedi_same_question_complete_objective.sh"

date -u +%Y-%m-%dT%H:%M:%SZ \
  > "$project/results/state8_eedi_same_question_complete_objective_20260811/controllers_restarted_after_auth"

