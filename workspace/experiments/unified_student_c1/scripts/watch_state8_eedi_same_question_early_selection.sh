#!/usr/bin/env bash
set -euo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
state="$project/results/state8_eedi_same_question_early_selection_20260811"
mkdir -p "$state"; cd "$project"
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"; trap 'rm -f "$state/controller.active.pid"' EXIT
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
while true; do
  heartbeat
  candidates=()
  ready=1
  for arm in updater full; do
    for step in 20 100; do
      path="$project/results/state8_eedi_same_question_${arm}_checkpoint${step}_screen_20260811/summary.json"
      [[ -f "$path" ]] || ready=0
      candidates+=(--candidate "$path")
    done
  done
  if (( ready )); then
    if /home/v-yuejiali/.venvs/studentmodeling/bin/python \
      scripts/select_state8_eedi_same_question_checkpoint.py "${candidates[@]}" \
      --output "$state/selection.json" > "$state/selection.log" 2>&1; then
      /home/v-yuejiali/.venvs/studentmodeling/bin/python \
        scripts/plan_state8_eedi_same_question_repair.py \
        --early-selection "$state/selection.json" \
        --output "$state/extension_plan.json" \
        > "$state/extension_plan.log" 2>&1 || {
          touch "$state/needs_attention"; heartbeat; exit 1;
        }
      touch "$state/viable_early_checkpoint" "$state/extension_required" \
        "$state/complete"
      heartbeat; exit 0
    fi
    /home/v-yuejiali/.venvs/studentmodeling/bin/python \
      scripts/plan_state8_eedi_same_question_repair.py "${candidates[@]}" \
      --output "$state/repair_plan.json" > "$state/repair_plan.log" 2>&1 || {
        touch "$state/needs_attention"; heartbeat; exit 1;
      }
    touch "$state/no_viable_early_checkpoint" "$state/repair_required" \
      "$state/complete"
    heartbeat
    exit 0
  fi
  sleep 60 9>&-
done
