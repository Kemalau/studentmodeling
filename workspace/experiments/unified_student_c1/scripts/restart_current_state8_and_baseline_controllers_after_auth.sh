#!/usr/bin/env bash
set -euo pipefail

# Restart only local, idempotent controllers that were started before the
# current studentmodeling Azure cache was selected.  Remote experiments are
# never cancelled or resubmitted here; each controller resumes from its
# active_experiment/receipt/marker files and applies its own duplicate guard.

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
cd "$project"
source scripts/use_studentmodeling_feeds_auth.sh
subscription=b6dc87f3-c479-49c8-8cb5-7896da3ff895
az account get-access-token --subscription "$subscription" \
  --query expiresOn -o tsv >/dev/null

restart_pending() {
  local session="$1" state="$2" command="$3"
  local root="$project/results/$state"
  [[ ! -f "$root/complete" && ! -f "$root/needs_attention" ]] || return 0
  tmux kill-session -t "$session" 2>/dev/null || true
  rm -f "$root/controller.active.pid" "$root/summary.active.pid"
  mkdir -p "$root"
  tmux new-session -d -s "$session" \
    "cd '$project' && source scripts/use_studentmodeling_feeds_auth.sh && exec $command >> '$root/restart_after_auth.log' 2>&1"
}

restart_remote_watcher() {
  local session="$1" state="$2" experiment="$3" job_id="$4"
  local root="$project/results/$state"
  [[ ! -f "$root/remote_passed" && ! -f "$root/remote_failed" ]] || return 0
  tmux kill-session -t "$session" 2>/dev/null || true
  rm -f "$root/watcher.active.pid" "$root/watcher.pid"
  mkdir -p "$root"
  tmux new-session -d -s "$session" \
    "cd '$project' && source scripts/use_studentmodeling_feeds_auth.sh && exec env WATCH_EXPERIMENT='$experiment' WATCH_JOB_ID='$job_id' WATCH_STATE_DIR='results/$state' bash scripts/watch_amlt_job_resilient.sh >> '$root/restart_after_auth.log' 2>&1"
}

# Recover the narrow case where a checkpoint-screen GPU job passed but its
# local collector exited because the Azure token expired at download time.
# No remote state is changed: the resumed watcher reuses active_experiment.txt
# and remote_passed, then downloads and judges the existing artifacts.
recover_passed_screen_download() {
  local root="$1"
  [[ -f "$root/remote_passed" && -f "$root/needs_attention" \
      && ! -f "$root/complete" ]] || return 0
  grep -q "screen result download failed" "$root/controller.log" 2>/dev/null || return 0
  rm -f "$root/needs_attention" "$root/controller.active.pid"
}

# Checkpoint-screen parents pin the Azure environment present at process
# creation.  Stop only incomplete screens; their selectors below recreate the
# exact hash-bound coordinates from active_experiment.txt and immutable
# milestone receipts.
restart_causal_selector=0
restart_early_selector=0
for spec in \
  8201:causal_teacher_low:20:state8_ct_low_cp20_screen \
  8201:causal_teacher_low:100:state8_ct_low_cp100_screen \
  8201:causal_teacher_low:240:state8_ct_low_cp240_screen \
  8201:causal_teacher_low:500:state8_ct_low_cp500_screen \
  8202:causal_teacher_high:20:state8_ct_high_cp20_screen \
  8202:causal_teacher_high:100:state8_ct_high_cp100_screen \
  8202:causal_teacher_high:240:state8_ct_high_cp240_screen \
  8202:causal_teacher_high:500:state8_ct_high_cp500_screen \
  8301:early_label_low:20:state8_el_8301_cp20_screen \
  8301:early_label_low:100:state8_el_8301_cp100_screen \
  8301:early_label_low:240:state8_el_8301_cp240_screen \
  8301:early_label_low:500:state8_el_8301_cp500_screen \
  8302:early_label_high:20:state8_el_8302_cp20_screen \
  8302:early_label_high:100:state8_el_8302_cp100_screen \
  8302:early_label_high:240:state8_el_8302_cp240_screen \
  8302:early_label_high:500:state8_el_8302_cp500_screen; do
  IFS=: read -r seed branch step session <<< "$spec"
  root="$project/results/state8_curriculum_repair_screens_20260812/seed${seed}_${branch}/checkpoint-$step"
  recover_passed_screen_download "$root"
  if [[ ! -f "$root/complete" && ! -f "$root/needs_attention" ]]; then
    tmux kill-session -t "$session" 2>/dev/null || true
    rm -f "$root/controller.active.pid"
    case "$seed" in
      8201|8202) restart_causal_selector=1 ;;
      8301|8302) restart_early_selector=1 ;;
    esac
  fi
done


# A selector may have propagated the child's download-only attention marker.
# Clear that derived marker only after every child marker has been recovered;
# all other selector failures remain fail-closed.
early_selector="$project/results/state8_early_label_checkpoint_selection_20260813"
if [[ -f "$early_selector/needs_attention" && ! -f "$early_selector/complete" ]] \
    && grep -q "screen needs attention:" "$early_selector/controller.log" 2>/dev/null \
    && ! find "$project/results/state8_curriculum_repair_screens_20260812" \
      -path '*/seed830[12]_early_label_*/checkpoint-*/needs_attention' \
      -print -quit | grep -q .; then
  rm -f "$early_selector/needs_attention" "$early_selector/controller.active.pid"
fi

# A selector launches checkpoint-screen children before entering its wait loop.
# If authentication recovery stopped any child, the already-running parent
# will not revisit that launch block.  Restart that parent as well so every
# incomplete child is recreated from its immutable receipt/active experiment.
if (( restart_causal_selector )); then
  # The original controller used a 20260812 tmux session name while the later
  # recovery helper standardized on 20260813.  Kill either alias so a stale
  # parent cannot keep the selector lock after its children were stopped.
  tmux kill-session -t state8_causal_teacher_checkpoint_selection_20260812 \
    2>/dev/null || true
  tmux kill-session -t state8_causal_teacher_checkpoint_selection_20260813 \
    2>/dev/null || true
  rm -f "$project/results/state8_causal_teacher_checkpoint_selection_20260812/controller.active.pid"
fi
if (( restart_early_selector )); then
  tmux kill-session -t state8_early_label_checkpoint_selection_20260813 \
    2>/dev/null || true
  rm -f "$project/results/state8_early_label_checkpoint_selection_20260813/controller.active.pid"
fi

restart_pending state8_causal_teacher_checkpoint_selection_20260813 \
  state8_causal_teacher_checkpoint_selection_20260812 \
  "bash scripts/watch_state8_causal_teacher_checkpoint_selection.sh"
restart_pending state8_early_label_checkpoint_selection_20260813 \
  state8_early_label_checkpoint_selection_20260813 \
  "bash scripts/watch_state8_early_label_checkpoint_selection.sh"
restart_pending state8_early_label_binding_repairs_20260813 \
  state8_early_label_binding_repairs_20260813 \
  "env STATE8_REPAIR_SELECTION_DIR='$project/results/state8_early_label_binding_selection_20260813' STATE8_REPAIR_STATE_DIR='$project/results/state8_early_label_binding_repairs_20260813' STATE8_REPAIR_BASE_EXPERIMENT='state8-early-label-binding-repairs-20260813' bash scripts/watch_state8_eedi_same_question_repair_branches.sh"

restart_remote_watcher state8_interleaved_retention_recovery3_watcher \
  state8_interleaved_retention_recovery3_20260813 \
  state8-interleaved-retention-recovery3-20260813 good_juice_nzlc8vp27w
restart_pending state8_interleaved_retention_selection_20260813 \
  state8_interleaved_retention_selection_20260813 \
  "bash scripts/watch_state8_interleaved_retention_selection.sh"
restart_pending state8_curriculum_global_selection_20260813 \
  state8_curriculum_global_selection_20260812 \
  "bash scripts/watch_state8_global_selection_and_promotion.sh"
restart_pending state8_curriculum_finalfit \
  state8_curriculum_finalfit_world2_20260812 \
  "bash scripts/watch_state8_curriculum_finalfit_world2.sh"
restart_pending state8_curriculum_finaleval \
  state8_curriculum_complete_final_eval_20260812 \
  "bash scripts/watch_state8_curriculum_final_eval.sh"
restart_pending state8_curriculum_objective \
  state8_curriculum_complete_objective_20260812 \
  "bash scripts/watch_state8_curriculum_complete_objective.sh"
restart_pending state8_large_eval_status_20260813 \
  state8_large_evaluation_live_status_20260813 \
  "bash scripts/watch_state8_large_evaluation_live_status.sh"

# Same-backbone task-only baselines.  The formal controllers first validate
# remote adapter completion receipts, then submit only the frozen evaluator if
# its guarded experiment is absent.  Restarting them does not retrain adapters.
restart_remote_watcher taskonly_qwen_t1_train_watcher_20260813 \
  taskonly_qwen_long_train_20260813/taskonly_t1 \
  taskonly-qwen-t1-t2-t4-oneepoch-20260813 careful_toe_p87sfl2prk
restart_remote_watcher taskonly_qwen_t2_train_watcher_20260813 \
  taskonly_qwen_long_train_20260813/taskonly_t2 \
  taskonly-qwen-t1-t2-t4-oneepoch-20260813 strong_pepper_8ft9yhwng1
restart_remote_watcher taskonly_qwen_t3_train_watcher_20260813 \
  taskonly_qwen_short_train_20260813/taskonly_t3 \
  taskonly-qwen-t3-t5-oneepoch-20260813 lemon_button_fyw8ljlmqx
restart_remote_watcher taskonly_qwen_t4_train_watcher_20260813 \
  taskonly_qwen_long_train_20260813/taskonly_t4 \
  taskonly-qwen-t1-t2-t4-oneepoch-20260813 bubbly_carpet_7wbmd8wxx1
restart_remote_watcher taskonly_qwen_t5_train_watcher_20260813 \
  taskonly_qwen_short_train_20260813/taskonly_t5 \
  taskonly-qwen-t3-t5-oneepoch-20260813 bright_yam_cpt3f61t59
restart_pending taskonly_baseline_supervisor_20260813 \
  taskonly_baseline_supervisor_20260813 \
  "env TASKONLY_BASELINE_POLL_SECONDS=60 bash scripts/supervise_taskonly_baselines.sh"
for task in T1 T2 T3 T4 T5; do
  lower="${task,,}"
  restart_pending "taskonly_qwen_formal_${lower}_20260813" \
    "taskonly_qwen_formal_eval_20260813/$task" \
    "env TASKONLY_PUBLIC_TASK=$task bash scripts/watch_taskonly_qwen_formal_eval.sh"
done
restart_pending taskonly_qwen_formal_summary_20260813 \
  taskonly_qwen_formal_eval_20260813 \
  "bash scripts/watch_taskonly_qwen_formal_summary.sh"
restart_pending t3_eedi_taskonly_qwen_formal_20260813 \
  t3_eedi_taskonly_qwen_formal_20260813 \
  "bash scripts/watch_t3_eedi_taskonly_qwen_formal.sh"
restart_pending unrestricted_qwen_t3_v2_rejudge_20260813 \
  formal_specialist_v2/unrestricted_qwen_t3_v2 \
  "bash scripts/watch_unrestricted_qwen_t3_v2_rejudge.sh"

date -u +%Y-%m-%dT%H:%M:%SZ \
  > "$project/results/recurrent_unified_supervisor_20260731/current_controllers_restarted_after_auth"
