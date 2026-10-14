#!/usr/bin/env bash
set -euo pipefail
project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
state_dir="$project_dir/results/recurrent_m4_pipeline_20260730/causal_bootstrap_rehearsal"
mkdir -p "$state_dir"
if [[ -f "$state_dir/controller.active.pid" ]]; then
  active="$(cat "$state_dir/controller.active.pid")"
  if kill -0 "$active" 2>/dev/null; then
    echo "controller_already_active=$active"
    exit 0
  fi
  rm -f "$state_dir/controller.active.pid"
fi
session="m4_causal_bootstrap_rehearsal"
tmux kill-session -t "$session" 2>/dev/null || true
tmux new-session -d -s "$session" \
  "cd '$project_dir' && exec bash scripts/watch_m4_causal_bootstrap_rehearsal.sh 2>&1 | tee -a '$state_dir/controller.tmux.log'"
launcher="$(tmux list-panes -t "$session" -F '#{pane_pid}' | head -1)"
echo "$launcher" > "$state_dir/controller.pid"
echo "controller_started=$launcher"
