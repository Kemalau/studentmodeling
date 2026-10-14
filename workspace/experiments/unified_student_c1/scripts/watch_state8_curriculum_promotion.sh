#!/usr/bin/env bash
set -euo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
selection="$project/results/state8_curriculum_selection_20260812"
while [[ ! -f "$selection/complete" ]]; do sleep 60; done
if [[ ! -f "$selection/scientific_gate_pass" ]]; then
  recovery="$project/results/state8_curriculum_repair_branches_20260812"
  while [[ ! -f "$recovery/complete" ]]; do sleep 60; done
  if [[ ! -f "$recovery/scientific_gate_pass" ]]; then
    state="$project/results/state8_curriculum_promotion_20260812"
    mkdir -p "$state"; touch "$state/diagnostic_gate_failed" "$state/complete"
    exit 0
  fi
  selection_file="$recovery/selection.json"
else
  selection_file="$selection/selection.json"
fi
export STATE8_PROMOTION_STATE_DIR="$project/results/state8_curriculum_promotion_20260812"
export STATE8_PROMOTION_SELECTION_FILE="$selection_file"
export STATE8_PROMOTION_BASE_EXPERIMENT=state8-curriculum-answer-c4-20260812
variant="$("/home/v-yuejiali/.venvs/studentmodeling/bin/python" - "$selection_file" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); print((x.get('selected') or {})['finalfit_variant'])
PY
)"
if [[ "$variant" == same_question_curriculum_promoted ]]; then
  export STATE8_PROMOTION_GATE_BLOB=studentmodeling/gates/state8_curriculum_v1/promotion.json
elif [[ "$variant" == same_question_repair_promoted ]]; then
  export STATE8_PROMOTION_GATE_BLOB=studentmodeling/gates/state8_eedi_same_question_v1/promotion.json
else
  echo "unsupported curriculum promotion finalfit variant: $variant" >&2
  exit 2
fi
exec bash "$project/scripts/watch_state8_eedi_same_question_promotion.sh"
