#!/usr/bin/env bash
set -euo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
state="$project/results/state8_curriculum_pipeline_supervisor_20260812"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
mkdir -p "$state"; cd "$project"
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"
trap 'rm -f "$state/controller.active.pid" "$state/snapshot.partial.tsv"' EXIT
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
sleep_unlocked(){ sleep "$1" 9>&-; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }

alive_pid(){
  local directory="$1" pid
  [[ -s "$directory/controller.active.pid" ]] || return 1
  pid="$(<"$directory/controller.active.pid")"
  [[ "$pid" =~ ^[1-9][0-9]*$ && -d "/proc/$pid" ]]
}

launch_stage(){
  local key="$1" session="$2"
  case "$key" in
    main_generic)
      tmux new-session -d -s "$session" \
        "cd '$project' && STATE_DIR='$project/results/state8_diagnosis_first_eedi_curriculum_20260812' CONFIG_PATH='$project/amlt/feeds_state8_diagnosis_first_eedi_curriculum_20260812.yaml' EXPERIMENT_NAME=state8-diagnosis-first-eedi-curriculum-recovery1-20260812 EXPERIMENT_BASE=state8-diagnosis-first-eedi-curriculum INITIAL_RECOVERY_WAVE=1 JOB_NAME=state8-diagnosis-first-eedi-curriculum-seed83 VALIDATOR=scripts/validate_state8_diagnosis_first_eedi_curriculum_amlt.py exec bash scripts/watch_state8_diagnosis_first_eedi_curriculum.sh"
      ;;
    main_current_query)
      tmux new-session -d -s "$session" \
        "cd '$project' && STATE_DIR='$project/results/state8_diagnosis_first_eedi_current_query_curriculum_20260812' CONFIG_PATH='$project/amlt/feeds_state8_diagnosis_first_eedi_current_query_curriculum_20260812.yaml' EXPERIMENT_NAME=state8-diagnosis-first-eedi-current-query-curriculum-20260812 EXPERIMENT_BASE=state8-diagnosis-first-eedi-current-query-curriculum INITIAL_RECOVERY_WAVE=0 JOB_NAME=state8-diagnosis-first-eedi-current-query-curriculum-seed83 VALIDATOR=scripts/validate_state8_diagnosis_first_eedi_current_query_curriculum_amlt.py exec bash scripts/watch_state8_diagnosis_first_eedi_curriculum.sh"
      ;;
    milestones_generic)
      tmux new-session -d -s "$session" \
        "cd '$project' && source scripts/use_studentmodeling_feeds_auth.sh && exec bash scripts/watch_state8_eedi_curriculum_milestones.sh 2>&1 | tee -a results/state8_diagnosis_first_eedi_curriculum_20260812/milestones/controller.tmux.log"
      ;;
    milestones_current_query)
      tmux new-session -d -s "$session" \
        "cd '$project' && source scripts/use_studentmodeling_feeds_auth.sh && exec bash scripts/watch_state8_eedi_current_query_curriculum_milestones.sh 2>&1 | tee -a results/state8_diagnosis_first_eedi_current_query_curriculum_20260812/milestones/controller.tmux.log"
      ;;
    screen_*)
      local arm step
      arm="${key#screen_}"; step="${arm##*_}"; arm="${arm%_*}"
      tmux new-session -d -s "$session" \
        "cd '$project' && export CURRICULUM_ARM='$arm' CHECKPOINT_STEP='$step'; exec bash scripts/watch_state8_curriculum_checkpoint_screen.sh 2>&1 | tee -a results/state8_curriculum_screens_20260812/$arm/checkpoint-$step/controller.tmux.log"
      ;;
    selection)
      tmux new-session -d -s "$session" \
        "cd '$project' && exec bash scripts/watch_state8_curriculum_selection.sh 2>&1 | tee -a results/state8_curriculum_selection_20260812/controller.tmux.log"
      ;;
    proactive_repairs_cp3000)
      tmux new-session -d -s "$session" \
        "cd '$project' && source scripts/use_studentmodeling_feeds_auth.sh && CHECKPOINT_STEP=3000 SEED_BASE=6201 SELECTION_DIR='$project/results/state8_curriculum_cp3000_proactive_selection_20260812' REPAIR_STATE_DIR='$project/results/state8_curriculum_cp3000_proactive_repairs_20260812' REPAIR_BASE_EXPERIMENT=state8-curriculum-cp3000-proactive-repairs-20260812 exec bash scripts/watch_state8_curriculum_cp4000_proactive_repairs.sh 2>&1 | tee -a results/state8_curriculum_cp3000_proactive_selection_20260812/controller.tmux.log"
      ;;
    proactive_repairs_cp4000)
      tmux new-session -d -s "$session" \
        "cd '$project' && source scripts/use_studentmodeling_feeds_auth.sh && exec bash scripts/watch_state8_curriculum_cp4000_proactive_repairs.sh 2>&1 | tee -a results/state8_curriculum_cp4000_proactive_selection_20260812/controller.tmux.log"
      ;;
    repair_milestones)
      tmux new-session -d -s "$session" \
        "cd '$project' && source scripts/use_studentmodeling_feeds_auth.sh && exec bash scripts/watch_state8_curriculum_repair_milestones.sh 2>&1 | tee -a results/state8_curriculum_repair_milestones_20260812/controller.tmux.log"
      ;;
    repair_branches)
      tmux new-session -d -s "$session" \
        "cd '$project' && source scripts/use_studentmodeling_feeds_auth.sh && exec bash scripts/watch_state8_curriculum_repair_branches.sh 2>&1 | tee -a results/state8_curriculum_repair_branches_20260812/controller.tmux.log"
      ;;
    repair_checkpoint_selection)
      tmux new-session -d -s "$session" \
        "cd '$project' && source scripts/use_studentmodeling_feeds_auth.sh && exec bash scripts/watch_state8_curriculum_repair_checkpoint_selection.sh 2>&1 | tee -a results/state8_curriculum_repair_checkpoint_selection_20260812/controller.tmux.log"
      ;;
    causal_teacher_repairs)
      tmux new-session -d -s "$session" \
        "cd '$project' && STATE8_REPAIR_SELECTION_DIR='$project/results/state8_causal_teacher_binding_selection_20260812' STATE8_REPAIR_STATE_DIR='$project/results/state8_causal_teacher_binding_repairs_20260812' STATE8_REPAIR_BASE_EXPERIMENT=state8-causal-teacher-binding-repairs-20260812 exec bash scripts/watch_state8_eedi_same_question_repair_branches.sh 2>&1 | tee -a results/state8_causal_teacher_binding_repairs_20260812/controller.tmux.log"
      ;;
    causal_teacher_milestones)
      tmux new-session -d -s "$session" \
        "cd '$project' && exec bash scripts/watch_state8_causal_teacher_binding_milestones.sh 2>&1 | tee -a results/state8_causal_teacher_binding_milestones_20260812/controller.tmux.log"
      ;;
    causal_teacher_checkpoint_selection)
      tmux new-session -d -s "$session" \
        "cd '$project' && exec bash scripts/watch_state8_causal_teacher_checkpoint_selection.sh 2>&1 | tee -a results/state8_causal_teacher_checkpoint_selection_20260812/controller.tmux.log"
      ;;
    early_label_repairs)
      tmux new-session -d -s "$session" \
        "cd '$project' && source scripts/use_studentmodeling_feeds_auth.sh && STATE8_REPAIR_SELECTION_DIR='$project/results/state8_early_label_binding_selection_20260813' STATE8_REPAIR_STATE_DIR='$project/results/state8_early_label_binding_repairs_20260813' STATE8_REPAIR_BASE_EXPERIMENT=state8-early-label-binding-repairs-20260813 exec bash scripts/watch_state8_eedi_same_question_repair_branches.sh 2>&1 | tee -a results/state8_early_label_binding_repairs_20260813/controller.tmux.log"
      ;;
    early_label_milestones)
      tmux new-session -d -s "$session" \
        "cd '$project' && source scripts/use_studentmodeling_feeds_auth.sh && exec bash scripts/watch_state8_early_label_binding_milestones.sh 2>&1 | tee -a results/state8_early_label_binding_milestones_20260813/controller.tmux.log"
      ;;
    early_label_checkpoint_selection)
      tmux new-session -d -s "$session" \
        "cd '$project' && source scripts/use_studentmodeling_feeds_auth.sh && exec bash scripts/watch_state8_early_label_checkpoint_selection.sh 2>&1 | tee -a results/state8_early_label_checkpoint_selection_20260813/controller.tmux.log"
      ;;
    global_selection|promotion)
      tmux new-session -d -s "$session" \
        "cd '$project' && source scripts/use_studentmodeling_feeds_auth.sh && exec bash scripts/watch_state8_global_selection_and_promotion.sh 2>&1 | tee -a results/state8_curriculum_global_selection_20260812/controller.tmux.log"
      ;;
    finalfit)
      tmux new-session -d -s "$session" \
        "cd '$project' && source scripts/use_studentmodeling_feeds_auth.sh && exec bash scripts/watch_state8_curriculum_finalfit_world2.sh 2>&1 | tee -a results/state8_curriculum_finalfit_world2_20260812/controller.tmux.log"
      ;;
    finaleval)
      tmux new-session -d -s "$session" \
        "cd '$project' && source scripts/use_studentmodeling_feeds_auth.sh && exec bash scripts/watch_state8_curriculum_final_eval.sh 2>&1 | tee -a results/state8_curriculum_complete_final_eval_20260812/controller.tmux.log"
      ;;
    objective)
      tmux new-session -d -s "$session" \
        "cd '$project' && source scripts/use_studentmodeling_feeds_auth.sh && exec bash scripts/watch_state8_curriculum_complete_objective.sh 2>&1 | tee -a results/state8_curriculum_complete_objective_20260812/controller.tmux.log"
      ;;
    *) return 2 ;;
  esac
}

ensure_stage(){
  local key="$1" session="$2" directory="$3" status
  if [[ -f "$directory/complete" ]]; then
    status=complete
  elif [[ -f "$directory/needs_attention" ]]; then
    status=needs_attention
  elif alive_pid "$directory" || tmux has-session -t "$session" 2>/dev/null; then
    status=alive
  else
    launch_stage "$key" "$session"
    status=restarted
    log "restarted local orchestration stage=$key session=$session"
  fi
  printf '%s\t%s\t%s\t%s\n' "$key" "$session" "$directory" "$status" \
    >> "$state/snapshot.partial.tsv"
}

write_snapshot(){
  "$python_bin" - "$state/snapshot.partial.tsv" "$state/latest_status.json" <<'PY'
import datetime,json,pathlib,sys
source,target=map(pathlib.Path,sys.argv[1:])
rows=[]
for line in source.read_text().splitlines():
    key,session,directory,status=line.split("\t")
    rows.append({"stage":key,"session":session,"directory":directory,"status":status})
value={
    "stage":"state8_curriculum_local_orchestration_supervisor_v1",
    "observed_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "scientific_gates_unchanged":True,
    "remote_submission_authority":False,
    "rows":rows,
    "needs_attention":[row["stage"] for row in rows if row["status"]=="needs_attention"],
    "restarted":[row["stage"] for row in rows if row["status"]=="restarted"],
}
temporary=target.with_suffix(target.suffix+".partial")
temporary.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
temporary.replace(target)
PY
}

while [[ ! -f "$project/results/state8_curriculum_complete_objective_20260812/complete" ]]; do
  heartbeat
  : > "$state/snapshot.partial.tsv"
  ensure_stage main_generic state8_eedi_curriculum_main_20260812 \
    "$project/results/state8_diagnosis_first_eedi_curriculum_20260812"
  ensure_stage main_current_query state8_eedi_current_query_curriculum_main_20260812 \
    "$project/results/state8_diagnosis_first_eedi_current_query_curriculum_20260812"
  ensure_stage milestones_generic state8_eedi_curriculum_milestones \
    "$project/results/state8_diagnosis_first_eedi_curriculum_20260812/milestones"
  ensure_stage milestones_current_query state8_eedi_current_query_curriculum_milestones \
    "$project/results/state8_diagnosis_first_eedi_current_query_curriculum_20260812/milestones"
  for arm in generic current_query; do
    for step in 2000 3000 4000 5000; do
      ensure_stage "screen_${arm}_${step}" "state8_curr_${arm}_cp${step}_screen" \
        "$project/results/state8_curriculum_screens_20260812/$arm/checkpoint-$step"
    done
  done
  ensure_stage selection state8_curriculum_selection_20260812 \
    "$project/results/state8_curriculum_selection_20260812"
  # The old cp3000/cp4000 full-tail repair jobs are no longer orchestration
  # prerequisites once their receipt-last checkpoint-500 snapshots exist.
  # Their read-only checkpoint screens remain covered below; do not relaunch
  # expensive completed-tail recoveries merely to publish duplicate outputs.
  # Legacy repair checkpoints and their semantic screens are preserved as
  # optional evidence.  They are no longer relaunched after the donor-binding
  # failure mode was isolated and the causal-teacher lineage became required.
  ensure_stage causal_teacher_repairs state8_causal_teacher_binding_repairs_20260812 \
    "$project/results/state8_causal_teacher_binding_repairs_20260812"
  ensure_stage causal_teacher_milestones state8_causal_teacher_binding_milestones_20260812 \
    "$project/results/state8_causal_teacher_binding_milestones_20260812"
  ensure_stage causal_teacher_checkpoint_selection state8_causal_teacher_checkpoint_selection_20260812 \
    "$project/results/state8_causal_teacher_checkpoint_selection_20260812"
  ensure_stage early_label_repairs state8_early_label_binding_repairs_20260813 \
    "$project/results/state8_early_label_binding_repairs_20260813"
  ensure_stage early_label_milestones state8_early_label_binding_milestones_20260813 \
    "$project/results/state8_early_label_binding_milestones_20260813"
  ensure_stage early_label_checkpoint_selection state8_early_label_checkpoint_selection_20260813 \
    "$project/results/state8_early_label_checkpoint_selection_20260813"
  # Global selection owns promotion.  Both rows share one session so a live
  # selector satisfies the downstream row until it execs the answer-C4 gate;
  # if that process later disappears, relaunching the global entrypoint is
  # resumable and reconstructs the same hash-bound winner before promotion.
  ensure_stage global_selection state8_curriculum_global_selection_and_promotion_20260812 \
    "$project/results/state8_curriculum_global_selection_20260812"
  ensure_stage promotion state8_curriculum_global_selection_and_promotion_20260812 \
    "$project/results/state8_curriculum_promotion_20260812"
  ensure_stage finalfit state8_curriculum_finalfit \
    "$project/results/state8_curriculum_finalfit_world2_20260812"
  ensure_stage finaleval state8_curriculum_finaleval \
    "$project/results/state8_curriculum_complete_final_eval_20260812"
  ensure_stage objective state8_curriculum_objective \
    "$project/results/state8_curriculum_complete_objective_20260812"
  write_snapshot
  mv "$state/snapshot.partial.tsv" "$state/latest_status.tsv"
  sleep_unlocked 30
done
heartbeat
touch "$state/complete"
log "top-level curriculum objective reached terminal audited state"
