#!/usr/bin/env bash
set -euo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
state_dir="$project_dir/results/recurrent_final_pipeline_20260730/cache_v2_16way"
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
session="finalfit_cache_v2_16way_monitor"
tmux kill-session -t "$session" 2>/dev/null || true
tmux new-session -d -s "$session" \
  "cd '$project_dir' && exec bash scripts/watch_finalfit_cache_v2_16way.sh"
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
  sleep 1
done
echo "controller_failed_to_start launcher=$launcher" >&2
exit 1
