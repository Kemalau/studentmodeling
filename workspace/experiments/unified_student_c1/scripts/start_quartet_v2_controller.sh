#!/usr/bin/env bash
set -euo pipefail
project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
state_dir="$project_dir/results/paired_state_quartet_v2_maskfix_submit_20260730"
mkdir -p "$state_dir"
cd "$project_dir"
if [[ -f "$state_dir/controller.active.pid" ]]; then
  pid="$(cat "$state_dir/controller.active.pid")"
  kill -0 "$pid" 2>/dev/null && { echo "controller_already_active=$pid"; exit 0; }
  rm -f "$state_dir/controller.active.pid"
fi
session="paired_quartet_v2_controller"
tmux kill-session -t "$session" 2>/dev/null || true
tmux new-session -d -s "$session" \
  "cd '$project_dir' && exec bash scripts/resume_quartet_v2_after_auth.sh"
launcher="$(tmux list-panes -t "$session" -F '#{pane_pid}' | head -1)"
echo "$launcher" > "$state_dir/controller.pid"
echo "controller_launcher=$launcher"
