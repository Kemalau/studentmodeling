#!/usr/bin/env bash
set -euo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
state_dir="$project_dir/results/recurrent_m3_submit_20260730"
mkdir -p "$state_dir"
cd "$project_dir"

if [[ -f "$state_dir/controller.active.pid" ]]; then
  active="$(cat "$state_dir/controller.active.pid")"
  if kill -0 "$active" 2>/dev/null; then
    echo "controller_already_active=$active"
    exit 0
  fi
  rm -f "$state_dir/controller.active.pid"
fi

session="recurrent_m3_controller"
tmux kill-session -t "$session" 2>/dev/null || true
tmux new-session -d -s "$session" \
  "cd '$project_dir' && exec bash scripts/resume_recurrent_m3_after_m2.sh"
launcher="$(tmux list-panes -t "$session" -F '#{pane_pid}' | head -1)"
echo "$launcher" > "$state_dir/controller.pid"
for _ in $(seq 1 15); do
  if [[ -f "$state_dir/controller.active.pid" ]]; then
    active="$(cat "$state_dir/controller.active.pid")"
    if kill -0 "$active" 2>/dev/null; then
      echo "controller_started=$active"
      exit 0
    fi
  fi
  tmux has-session -t "$session" 2>/dev/null || break
  sleep 1
done
echo "controller_failed_to_acquire_lock launcher=$launcher" >&2
exit 1
