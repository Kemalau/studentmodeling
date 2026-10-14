#!/usr/bin/env bash
set -euo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
state="$project/results/state8_curriculum_priority_queue_20260812"
mkdir -p "$state"; cd "$project"
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"
trap 'rm -f "$state/controller.active.pid"' EXIT
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
sleep_unlocked(){ sleep "$1" 9>&-; }

wait_complete(){
  local directory="$1" label="$2"
  while [[ ! -f "$directory/complete" ]]; do
    heartbeat
    if [[ -f "$directory/needs_attention" ]]; then
      log "ERROR: prioritized stage needs attention: $label"
      touch "$state/needs_attention"
      return 1
    fi
    sleep_unlocked 60
  done
  log "prioritized stage complete: $label"
}

launch(){
  local session="$1" command="$2"
  if ! tmux has-session -t "$session" 2>/dev/null; then
    tmux new-session -d -s "$session" "$command"
    log "launched prioritized stage: $session"
  fi
}

current="$project/results/state8_diagnosis_first_eedi_current_query_curriculum_20260812"
generic="$project/results/state8_diagnosis_first_eedi_curriculum_20260812"
current_screen="$project/results/state8_curriculum_screens_20260812/current_query/checkpoint-5000"
generic_screen="$project/results/state8_curriculum_screens_20260812/generic/checkpoint-5000"

wait_complete "$current" current_query_tail_c4

launch state8_eedi_curriculum_main_20260812 \
  "cd '$project' && STATE_DIR='$generic' CONFIG_PATH='$project/amlt/feeds_state8_diagnosis_first_eedi_curriculum_20260812.yaml' EXPERIMENT_NAME=state8-diagnosis-first-eedi-curriculum-recovery2-20260812 EXPERIMENT_BASE=state8-diagnosis-first-eedi-curriculum INITIAL_RECOVERY_WAVE=2 JOB_NAME=state8-diagnosis-first-eedi-curriculum-seed83 VALIDATOR=scripts/validate_state8_diagnosis_first_eedi_curriculum_amlt.py exec bash scripts/watch_state8_diagnosis_first_eedi_curriculum.sh"
wait_complete "$generic" generic_tail_c4

launch state8_curr_current_query_cp5000_screen \
  "cd '$project' && CURRICULUM_ARM=current_query CHECKPOINT_STEP=5000 exec bash scripts/watch_state8_curriculum_checkpoint_screen.sh"
wait_complete "$current_screen" current_query_cp5000_screen

launch state8_curr_generic_cp5000_screen \
  "cd '$project' && CURRICULUM_ARM=generic CHECKPOINT_STEP=5000 exec bash scripts/watch_state8_curriculum_checkpoint_screen.sh"
wait_complete "$generic_screen" generic_cp5000_screen

heartbeat
touch "$state/complete"
log "prioritized State8 curriculum tail and checkpoint screens complete"
