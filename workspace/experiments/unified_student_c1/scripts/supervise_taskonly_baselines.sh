#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
state="$project/results/taskonly_baseline_supervisor_20260813"
poll_seconds="${TASKONLY_BASELINE_POLL_SECONDS:-60}"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
mkdir -p "$state"; cd "$project" || exit 1
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"
trap 'rm -f "$state/controller.active.pid"' EXIT

log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
alive(){ local f="$1" pid; [[ -s "$f" ]] || return 1; pid="$(<"$f")"; [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; }

ensure_task_controller(){
  local task="$1" root="$project/results/taskonly_qwen_formal_eval_20260813/$task"
  [[ -f "$root/complete" || -f "$root/needs_attention" ]] && return 0
  alive "$root/controller.active.pid" && return 0
  local session="taskonly_qwen_formal_${task,,}_20260813"
  tmux has-session -t "$session" 2>/dev/null && tmux kill-session -t "$session" 2>/dev/null || true
  rm -f "$root/controller.active.pid"
  tmux new-session -d -s "$session" \
    "cd '$project' && exec env TASKONLY_PUBLIC_TASK=$task bash scripts/watch_taskonly_qwen_formal_eval.sh >> '$root/watcher.tmux.log' 2>&1"
  log "restarted task-only formal controller task=$task"
}

ensure_summary(){
  local root="$project/results/taskonly_qwen_formal_eval_20260813"
  [[ -f "$root/complete" || -f "$root/needs_attention" ]] && return 0
  alive "$root/summary.active.pid" && return 0
  local session=taskonly_qwen_formal_summary_20260813
  tmux has-session -t "$session" 2>/dev/null && tmux kill-session -t "$session" 2>/dev/null || true
  rm -f "$root/summary.active.pid"
  tmux new-session -d -s "$session" \
    "cd '$project' && exec bash scripts/watch_taskonly_qwen_formal_summary.sh >> '$root/summary.tmux.log' 2>&1"
  log "restarted task-only formal summary controller"
}

ensure_t3_eedi(){
  local root="$project/results/t3_eedi_taskonly_qwen_formal_20260813"
  [[ -f "$root/complete" || -f "$root/needs_attention" ]] && return 0
  alive "$root/controller.active.pid" && return 0
  local session=t3_eedi_taskonly_qwen_formal_20260813
  tmux has-session -t "$session" 2>/dev/null && tmux kill-session -t "$session" 2>/dev/null || true
  rm -f "$root/controller.active.pid"
  tmux new-session -d -s "$session" \
    "cd '$project' && exec bash scripts/watch_t3_eedi_taskonly_qwen_sharded_recovery.sh >> '$root/watcher.tmux.log' 2>&1"
  log "restarted submitted T3-Eedi formal monitor"
}

ensure_unified_t3_v2(){
  local root="$project/results/formal_specialist_v2/unrestricted_qwen_t3_v2"
  [[ -f "$root/complete" || -f "$root/needs_attention" ]] && return 0
  alive "$root/controller.active.pid" && return 0
  local session=unrestricted_qwen_t3_v2_rejudge_20260813
  mkdir -p "$root"
  tmux has-session -t "$session" 2>/dev/null && tmux kill-session -t "$session" 2>/dev/null || true
  rm -f "$root/controller.active.pid"
  tmux new-session -d -s "$session" \
    "cd '$project' && exec bash scripts/watch_unrestricted_qwen_t3_v2_rejudge.sh >> '$root/controller.tmux.log' 2>&1"
  log "restarted unrestricted Base/Unified T3 v2 rejudge"
}

ensure_resumable_recovery(){
  local task="$1"
  local root="$project/results/taskonly_qwen_resumable_recovery_20260814/$task"
  [[ -f "$root/complete" || -f "$root/needs_attention" ]] && return 0
  alive "$root/controller.active.pid" && return 0
  local session="taskonly_qwen_resumable_${task,,}_20260814"
  mkdir -p "$root"
  tmux has-session -t "$session" 2>/dev/null \
    && tmux kill-session -t "$session" 2>/dev/null || true
  rm -f "$root/controller.active.pid"
  tmux new-session -d -s "$session" \
    "cd '$project' && exec env TASKONLY_PUBLIC_TASK=$task bash scripts/watch_taskonly_qwen_resumable_recovery.sh >> '$root/controller.tmux.log' 2>&1"
  log "restarted resumable task-only controller task=$task"
}

snapshot_remote(){
  local experiment="$1" output="$2" tmp
  # AMLT uses terminal rendering internally.  Keep it in the foreground process
  # group so a tmux-owned supervisor cannot leave it stopped by job control.
  # `mktemp` also prevents a stale snapshot from an older supervisor instance
  # colliding with the current atomic replace.
  tmp="$(mktemp "$output.partial.XXXXXX")"
  if timeout --foreground 90 amlt --json-tables status "$experiment" \
      --hide-urls --force-status-update > "$tmp" 2>> "$state/controller.log"; then
    mv "$tmp" "$output"
  else
    rm -f "$tmp"
  fi
}

while true; do
  date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"
  source scripts/use_studentmodeling_feeds_auth.sh
  if [[ "$STUDENTMODELING_FEEDS_AUTH_HEALTH" == unhealthy ]]; then
    rm -f "$state/auth_ready"
    date -u +%Y-%m-%dT%H:%M:%SZ > "$state/auth_required"
  else
    rm -f "$state/auth_required"
    date -u +%Y-%m-%dT%H:%M:%SZ > "$state/auth_ready"
    snapshot_remote taskonly-qwen-t1-t2-t4-oneepoch-20260813 "$state/train_long_status.json"
    snapshot_remote taskonly-qwen-t3-t5-oneepoch-20260813 "$state/train_short_status.json"
    snapshot_remote t3-eedi-taskonly-qwen-formal-recovery1-20260813 "$state/t3_eedi_status.json"
  fi
  for task in T1 T2 T3 T4 T5; do ensure_task_controller "$task"; done
  for task in T1 T2 T4; do ensure_resumable_recovery "$task"; done
  ensure_summary
  ensure_t3_eedi
  ensure_unified_t3_v2
  "$python_bin" scripts/summarize_taskonly_baseline_live_status.py \
    --project "$project" --output "$state/live_status.json" \
    > "$state/live_status.log" 2>&1 || true
  if "$python_bin" - "$state/live_status.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); raise SystemExit(0 if x.get('overall_complete') else 1)
PY
  then
    touch "$state/complete"; log "task-only baseline pipeline complete"; exit 0
  fi
  sleep "$poll_seconds"
done
