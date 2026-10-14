#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
subscription=b6dc87f3-c479-49c8-8cb5-7896da3ff895
# Use the caller/default Azure CLI cache only when it is healthy.
# Trust an inherited Azure cache only after it proves it can read this exact
# subscription.  Long-lived tmux owners have previously inherited another
# project's cache (for example .azure-keyflip-feeds), shadowing a healthy
# default login and leaving every unified-model monitor falsely auth-blocked.
if [[ -n "${AZURE_CONFIG_DIR:-}" ]] \
    && az account get-access-token --subscription "$subscription" \
      --query expiresOn -o tsv >/dev/null 2>&1; then
  export AZURE_CONFIG_DIR
elif env -u AZURE_CONFIG_DIR az account get-access-token \
    --subscription "$subscription" --query expiresOn -o tsv \
    >/dev/null 2>&1; then
  unset AZURE_CONFIG_DIR
else
  export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure-studentmodeling-feeds
fi
state_dir="$project_dir/results/recurrent_unified_supervisor_20260731"
poll_seconds="${POLL_SECONDS:-60}"
mkdir -p "$state_dir"
cd "$project_dir"

exec 9>"$state_dir/controller.lock"
flock -n 9 || exit 0
echo "$$" > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" \
    | tee -a "$state_dir/supervisor.log"
}

controller_alive() {
  local root="$1" pid
  [[ -s "$root/controller.active.pid" ]] || return 1
  pid="$(cat "$root/controller.active.pid")"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

restart_controller() {
  local session="$1" root="$2" command="$3"
  if tmux has-session -t "$session" 2>/dev/null; then
    tmux kill-session -t "$session" 2>/dev/null || true
  fi
  rm -f "$root/controller.active.pid"
  tmux new-session -d -s "$session" \
    "cd '$project_dir' && $command 2>&1 | tee -a '$root/controller.tmux.log'"
  log "restarted session=$session root=$root"
}

check_controller() {
  local session="$1" root="$2" command="$3"
  if [[ -f "$root/deferred" ]]; then
    printf '%s\tdeferred\n' "$session" >> "$state_dir/snapshot.partial"
    return 0
  fi
  if [[ -f "$root/superseded" ]]; then
    printf '%s\tsuperseded\n' "$session" >> "$state_dir/snapshot.partial"
    return 0
  fi
  if [[ -f "$root/complete" || -f "$root/overall_complete" ]]; then
    printf '%s\tcomplete\n' "$session" >> "$state_dir/snapshot.partial"
    return 0
  fi
  if [[ -f "$root/needs_attention" ]]; then
    printf '%s\tneeds_attention\n' "$session" >> "$state_dir/snapshot.partial"
    return 0
  fi
  if controller_alive "$root"; then
    printf '%s\talive\n' "$session" >> "$state_dir/snapshot.partial"
    return 0
  fi
  printf '%s\trestarting\n' "$session" >> "$state_dir/snapshot.partial"
  restart_controller "$session" "$root" "$command"
}

clear_stale_dependency_attention() {
  local owner_root="$1" dependency_root="$2" expected_error="$3" last_error
  # A downstream owner may have stopped because an upstream recovery briefly
  # needed attention.  Once that upstream controller is live again, its old
  # downstream marker is no longer authoritative and must not permanently
  # suppress the owner.
  last_error="$(grep 'ERROR:' "$owner_root/controller.log" 2>/dev/null | tail -n 1 || true)"
  if [[ -f "$owner_root/needs_attention" ]] \
      && [[ ! -f "$dependency_root/needs_attention" ]] \
      && { [[ -f "$dependency_root/complete" ]] || controller_alive "$dependency_root"; } \
      && grep -Fq "$expected_error" <<< "$last_error"; then
    rm -f "$owner_root/needs_attention"
    log "cleared stale dependency attention owner=$owner_root dependency=$dependency_root"
  fi
}

clear_stale_candidate_aggregation_attention() {
  local owner_root="$1" dependency_root last_error pending=0
  shift
  [[ -f "$owner_root/needs_attention" ]] || return 0
  last_error="$(grep 'ERROR:' "$owner_root/controller.log" 2>/dev/null | tail -n 1 || true)"
  case "$last_error" in
    *"task-retention rehearsal needs attention"*|*"candidate diagnostic needs attention"*) ;;
    *"no task-retention branch passed every promotion gate"*)
      # This result is final only after every newly added candidate diagnostic
      # has completed.  While any healthy arm is still running, retrying the
      # aggregator is required so that later candidates are considered.
      ;;
    *) return 0 ;;
  esac
  for dependency_root in "$@"; do
    [[ ! -f "$dependency_root/needs_attention" ]] || return 0
    if [[ ! -f "$dependency_root/complete" ]]; then
      controller_alive "$dependency_root" || return 0
      pending=1
    fi
  done
  if [[ "$last_error" == *"no task-retention branch passed every promotion gate"* ]] \
      && (( pending == 0 )); then
    return 0
  fi
  rm -f "$owner_root/needs_attention"
  log "cleared stale candidate-aggregation attention owner=$owner_root"
}

while true; do
  date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"
  : > "$state_dir/snapshot.partial"
  if az account get-access-token \
      --subscription "$subscription" \
      --query expiresOn -o tsv >/dev/null 2>&1; then
    rm -f "$state_dir/auth_required"
    printf 'feeds_auth\tready\n' >> "$state_dir/snapshot.partial"
  else
    date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/auth_required"
    printf 'feeds_auth\tauth_required\n' >> "$state_dir/snapshot.partial"
    # Authentication only controls local monitoring. Keep exactly one
    # device-code owner alive; it never cancels or submits remote jobs.
    if ! tmux has-session -t layerwise_unified_reauth 2>/dev/null; then
      mkdir -p "$state_dir/reauth"
      tmux new-session -d -s layerwise_unified_reauth \
        "cd '$project_dir' && exec bash scripts/watch_layerwise_unified_reauth.sh >> '$project_dir/results/recurrent_unified_supervisor_20260731/reauth/controller.tmux.log' 2>&1"
      log "started persistent Feeds reauthentication owner"
    fi
    printf 'feeds_reauth\tpending\n' >> "$state_dir/snapshot.partial"
  fi
  check_controller \
    m4_donor_round250_c4_devicefix \
    "$project_dir/results/recurrent_m4_pipeline_20260730/donor_round250_c4_devicefix" \
    "exec bash scripts/watch_m4_donor_round250_c4_devicefix.sh"
  check_controller \
    m4_donor_round500_c4_devicefix \
    "$project_dir/results/recurrent_m4_pipeline_20260730/donor_round500_c4_devicefix" \
    "exec bash scripts/watch_m4_donor_round500_c4_devicefix.sh"
  check_controller \
    m4_cf4_diagnostic \
    "$project_dir/results/recurrent_m4_pipeline_20260730/cf4_diagnostic" \
    "exec bash scripts/watch_m4_cf4_diagnostic.sh"
  check_controller \
    m4_cf4_single_hedge \
    "$project_dir/results/recurrent_m4_pipeline_20260730/cf4_single_hedge" \
    "exec bash scripts/watch_m4_cf4_single_hedge.sh"
  check_controller \
    m2_seeded_cf4_screen \
    "$project_dir/results/recurrent_m4_pipeline_20260730/m2_seeded_cf4_screen" \
    "exec bash scripts/watch_m2_seeded_cf4_screen.sh"
  check_controller \
    m2_seeded_specificity_triplet_v2 \
    "$project_dir/results/recurrent_m4_pipeline_20260730/m2_seeded_specificity_triplet_v2" \
    "STATE_SLUG=m2_seeded_specificity_triplet_v2 CONFIG_PATH=amlt/feeds_m2_seeded_specificity_triplet_screen_20260731.yaml BASE_EXPERIMENT=recurrent-m2-seeded-cf4-specificity-triplet-v2-single-gb4-seed42-20260731 EXPECTED_BRANCH=m2_seeded_cf4_specificity_triplet_v2_w1_single_gb4 EXPECTED_OBJECTIVE=specificity_triplet VALIDATOR=scripts/validate_m2_seeded_specificity_triplet_amlt.py exec bash scripts/watch_m2_seeded_cf4_screen.sh"
  check_controller \
    m2_m3_c4_transition \
    "$project_dir/results/recurrent_m4_pipeline_20260730/m2_m3_c4_transition" \
    "exec bash scripts/watch_m2_m3_c4_transition.sh"
  check_controller \
    one_shot_layer_transport \
    "$project_dir/results/recurrent_m4_pipeline_20260730/one_shot_layer_transport" \
    "exec bash scripts/watch_one_shot_layer_transport_c4.sh"
  check_controller \
    recurrent_m4_causal_rescue \
    "$project_dir/results/recurrent_m4_pipeline_20260730/causal_rescue" \
    "exec bash scripts/watch_m4_causal_rescue_pipeline.sh"
  check_controller \
    recurrent_final_pipeline \
    "$project_dir/results/recurrent_final_pipeline_20260730" \
    "exec bash scripts/resume_recurrent_final_after_m4.sh"
  check_controller \
    m4_donor_checkpoint50_c4 \
    "$project_dir/results/recurrent_m4_pipeline_20260730/donor_checkpoint50_c4" \
    "CHECKPOINT_ROUND=50 exec bash scripts/watch_m4_donor_checkpoint50_c4.sh"
  check_controller \
    m4_donor_checkpoint100_c4 \
    "$project_dir/results/recurrent_m4_pipeline_20260730/donor_checkpoint100_c4" \
    "CHECKPOINT_ROUND=100 exec bash scripts/watch_m4_donor_checkpoint50_c4.sh"
  check_controller \
    m4_donor_checkpoint400_c4 \
    "$project_dir/results/recurrent_m4_pipeline_20260730/donor_checkpoint400_c4" \
    "CHECKPOINT_ROUND=400 exec bash scripts/watch_m4_donor_checkpoint50_c4.sh"
  check_controller \
    layerwise_formal_screen \
    "$project_dir/results/recurrent_m4_pipeline_20260730/layerwise_formal_screen" \
    "exec bash scripts/watch_layerwise_formal_screen.sh"
  check_controller \
    layerwise_state_bridge \
    "$project_dir/results/recurrent_m4_pipeline_20260730/layerwise_state_bridge" \
    "exec bash scripts/watch_layerwise_state_bridge_c4.sh"
  check_controller \
    cross_adapter_layerwise_bridge \
    "$project_dir/results/recurrent_m4_pipeline_20260730/cross_adapter_layerwise_bridge" \
    "exec bash scripts/watch_cross_adapter_layerwise_bridge_c4.sh"
  check_controller \
    layerwise_recurrent_zero_train \
    "$project_dir/results/recurrent_m4_pipeline_20260730/layerwise_recurrent_zero_train" \
    "exec bash scripts/watch_layerwise_recurrent_c4.sh"
  check_controller \
    single_layer_state_bridge \
    "$project_dir/results/recurrent_m4_pipeline_20260730/single_layer_state_bridge" \
    "exec bash scripts/watch_single_layer_state_bridge_c4.sh"
  check_controller \
    layerwise_training_gate \
    "$project_dir/results/recurrent_m4_pipeline_20260730/layerwise_training_gate" \
    "exec bash scripts/watch_layerwise_training_gate.sh"
  check_controller \
    layerwise_recurrent_pilot \
    "$project_dir/results/recurrent_m4_pipeline_20260730/layerwise_recurrent_pilot" \
    "exec bash scripts/watch_layerwise_recurrent_pilot.sh"
  check_controller \
    layerwise_trajectory_curriculum \
    "$project_dir/results/layerwise_trajectory_curriculum_20260731" \
    "exec bash scripts/watch_layerwise_trajectory_curriculum.sh"
  check_controller \
    layerwise_expanded_head_backward_gate \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/expanded_head_backward_gate" \
    "exec bash scripts/watch_layerwise_expanded_head_backward_gate.sh"
  check_controller \
    layerwise_recompute_mask_release \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/recompute_mask_release" \
    "exec bash scripts/watch_layerwise_recompute_mask_release.sh"
  check_controller \
    layerwise_unified_final_fit \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/final_fit" \
    "exec bash scripts/watch_layerwise_unified_final_fit.sh"
  check_controller \
    layerwise_unified_training_progress \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/progress" \
    "exec bash scripts/watch_layerwise_unified_training_progress.sh"
  check_controller \
    layerwise_promoted_training_progress \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/promoted_progress" \
    "FIT_DIR='$project_dir/results/layerwise_unified_final_pipeline_20260731/task_retention_promoted_final_fit' STATE_DIR='$project_dir/results/layerwise_unified_final_pipeline_20260731/promoted_progress' TOTAL_ROUNDS=16241 MAX_RUN_SECONDS=345600 VALIDATION_RESERVE_SECONDS=21600 exec bash scripts/watch_layerwise_unified_training_progress.sh"
  check_controller \
    layerwise_unified_final_eval \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/final_eval" \
    "exec bash scripts/watch_layerwise_unified_final_eval.sh"
  check_controller \
    layerwise_t3_semantic_judge \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/t3_semantic_judge" \
    "exec bash scripts/watch_layerwise_t3_semantic_judge.sh"
  check_controller \
    layerwise_finalfit_checkpoint500_c4 \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/checkpoint500_c4" \
    "CHECKPOINT_ROUND=500 exec bash scripts/watch_layerwise_finalfit_checkpoint250_c4.sh"
  check_controller \
    layerwise_finalfit_checkpoint1000_c4 \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/checkpoint1000_c4" \
    "CHECKPOINT_ROUND=1000 exec bash scripts/watch_layerwise_finalfit_checkpoint250_c4.sh"
  check_controller \
    layerwise_finalfit_checkpoint1000_formal_screen \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/checkpoint1000_formal_screen" \
    "CHECKPOINT_ROUND=1000 exec bash scripts/watch_layerwise_checkpoint250_formal_screen.sh"
  check_controller \
    layerwise_finalfit_checkpoint1000_t3_semantic_judge \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/checkpoint1000_t3_semantic_judge" \
    "exec bash scripts/watch_layerwise_checkpoint1000_t3_semantic_judge.sh"
  check_controller \
    layerwise_finalfit_checkpoint1000_decision \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/checkpoint1000_decision" \
    "exec bash scripts/watch_layerwise_checkpoint1000_decision.sh"
  check_controller \
    layerwise_c500_causal_rescue_recovery \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/checkpoint500_causal_rescue_recovery" \
    "exec bash scripts/watch_layerwise_checkpoint500_causal_rescue_recovery.sh"
  check_controller \
    layerwise_known_oom_row_alignedmask \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/known_oom_row_alignedmask" \
    "exec bash scripts/watch_layerwise_known_oom_row_alignedmask.sh"
  check_controller \
    layerwise_task_retention_rehearsal \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/task_retention_rehearsal" \
    "exec bash scripts/watch_layerwise_task_retention_rehearsal.sh"
  check_controller \
    layerwise_eedi_state_path_ab \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/eedi_state_path_ab" \
    "exec bash scripts/watch_layerwise_eedi_state_path_ab.sh"
  check_controller \
    layerwise_state_kd \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/state_kd" \
    "exec bash scripts/watch_layerwise_state_kd.sh"
  check_controller \
    layerwise_equal_state_path \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/equal_state_path" \
    "exec bash scripts/watch_layerwise_equal_state_path.sh"
  check_controller \
    layerwise_equal_state_kd \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/equal_state_kd" \
    "STATE_SLUG=equal_state_kd CONFIG_PATH=amlt/feeds_layerwise_equal_state_kd_20260802.yaml BASE_EXPERIMENT=layerwise-equal-t3-t6-state-kd-seed42-20260802 RECOVERY_PREFIX=layerwise-equal-t3-t6-state-kd JOB_NAME=layerwise-equal-t3-t6-state-kd-seed42 LOSS_PROFILE_EXPECTED=state_kd VALIDATOR=scripts/validate_layerwise_equal_state_kd_amlt.py EXPERIMENT_DESCRIPTION='Matched equal-T3/T6 state-KD diagnostic' exec bash scripts/watch_layerwise_equal_state_path.sh"
  check_controller \
    layerwise_equal_state_kd_local_preflight \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/equal_state_kd/local_round3250_preflight" \
    "STATE_SLUG=equal_state_kd LOSS_PROFILE_EXPECTED=state_kd ARTIFACT_LABEL=equal-state-kd exec bash scripts/watch_layerwise_equal_state_path_local_preflight.sh"
  check_controller \
    layerwise_equal_state_kd_open_t3 \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/equal_state_kd_open_t3" \
    "STATE_SLUG=equal_state_kd_open_t3 CONFIG_PATH=amlt/feeds_layerwise_equal_state_kd_open_t3_20260802.yaml BASE_EXPERIMENT=layerwise-equal-t3-t6-open-t3-state-kd-seed42-20260802 RECOVERY_PREFIX=layerwise-equal-t3-t6-open-t3-state-kd JOB_NAME=layerwise-equal-t3-t6-open-t3-state-kd-seed42 LOSS_PROFILE_EXPECTED=state_kd SUMMARY_BRANCH_EXPECTED=equal_t3_t6_open_t3 VALIDATOR=scripts/validate_layerwise_equal_state_kd_open_t3_amlt.py EXPERIMENT_DESCRIPTION='Matched equal-T3/T6 open-T3 selective-KD diagnostic' exec bash scripts/watch_layerwise_equal_state_path.sh"
  check_controller \
    layerwise_equal_state_path_open_t3 \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/equal_state_path_open_t3" \
    "exec bash scripts/watch_layerwise_equal_state_path_open_t3.sh"
  check_controller \
    layerwise_equal_state_kd_open_t3_lr_scan \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/equal_state_kd_open_t3_lr_scan" \
    "exec bash scripts/watch_layerwise_equal_state_kd_open_t3_lr_scan.sh"
  check_controller \
    layerwise_equal_state_kd_open_t3_lr1e6 \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/equal_state_kd_open_t3_lr1e6" \
    "exec bash scripts/watch_layerwise_equal_state_kd_open_t3_lr1e6.sh"
  check_controller \
    layerwise_open_t3_consolidation \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/open_t3_consolidation" \
    "exec bash scripts/watch_layerwise_open_t3_consolidation.sh"
  check_controller \
    layerwise_equal_state_kd_open_t3_local_preflight \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/equal_state_kd_open_t3/local_round3250_preflight" \
    "STATE_SLUG=equal_state_kd_open_t3 LOSS_PROFILE_EXPECTED=state_kd ARTIFACT_LABEL=equal-open-t3-selective-kd CHECKPOINT_ROUND=3250 LOCAL_PREFLIGHT_SUBDIR=local_round3250_preflight REMOTE_PREFIX=studentmodeling/diagnostics/layerwise_task_retention_service_recovery/equal_t3_t6_open_t3/state_kd/lr5e-7_start12000_stop3250_32adfcc7bd1d78f541ec8685ec8f96f9934b35471011bb896c402b1e77a7bf6f_b926ee1acc6bf8a098693cef8705f88c739be214629dcea723daed6a0827c440_2474b77459701fa36ec564c84b7f3b6b0d158c9ab128b5147111d252662b0931/checkpoint-round-3250 exec bash scripts/watch_layerwise_equal_state_path_local_preflight.sh"
  check_controller \
    layerwise_staged_task_retention \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/staged_task_retention" \
    "exec bash scripts/watch_layerwise_staged_task_retention.sh"
  check_controller \
    layerwise_staged_state_path \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/staged_state_path" \
    "STATE_SLUG=staged_state_path CONFIG_PATH=amlt/feeds_layerwise_staged_state_path_20260802.yaml BASE_EXPERIMENT=layerwise-staged-task-retention-state-path-seed42-20260802 JOB_NAME=layerwise-staged-t3-t6-state-path-seed42 LOSS_PROFILE_EXPECTED=state_path VALIDATOR=scripts/validate_layerwise_staged_state_path_amlt.py exec bash scripts/watch_layerwise_staged_task_retention.sh"
  check_controller \
    layerwise_eedi_full_lr_scan \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/eedi_full_lr_scan" \
    "exec bash scripts/watch_layerwise_eedi_full_lr_scan.sh"
  check_controller \
    layerwise_eedi_full_state_kd_lr_scan \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/eedi_full_state_kd_lr_scan" \
    "STATE_SLUG=eedi_full_state_kd_lr_scan CONFIG_PATH=amlt/feeds_layerwise_eedi_full_state_kd_lr_scan_20260802.yaml BASE_EXPERIMENT=layerwise-eedi-full-statekd-lr-scan JOB_PREFIX=layerwise-eedi-full-statekd LOSS_PROFILE_EXPECTED=state_kd VALIDATOR=scripts/validate_layerwise_eedi_full_state_kd_lr_scan_amlt.py exec bash scripts/watch_layerwise_eedi_full_lr_scan.sh"
  check_controller \
    layerwise_eedi_full_state_kd_round3850_local_preflight \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/eedi_full_state_kd_lr_scan/local_round3850_preflight" \
    "STATE_SLUG=eedi_full_state_kd_lr_scan LOSS_PROFILE_EXPECTED=state_kd ARTIFACT_LABEL=state-kd-lr5e6 CHECKPOINT_ROUND=3850 LOCAL_PREFLIGHT_SUBDIR=local_round3850_preflight REMOTE_PREFIX=studentmodeling/diagnostics/layerwise_task_retention_service_recovery/eedi_full/state_kd/lr5e-6_start12000_stop4250_97bef6748b1fa8d75809e9e01c9b76d57b6534d336b7037217289932a31df393_b926ee1acc6bf8a098693cef8705f88c739be214629dcea723daed6a0827c440_b926ee1acc6bf8a098693cef8705f88c739be214629dcea723daed6a0827c440/checkpoint-round-3850 exec bash scripts/watch_layerwise_equal_state_path_local_preflight.sh"
  check_controller \
    layerwise_eedi_full_open_t3_lr_scan \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/eedi_full_open_t3_lr_scan" \
    "STATE_SLUG=eedi_full_open_t3_lr_scan CONFIG_PATH=amlt/feeds_layerwise_eedi_full_open_t3_lr_scan_20260802.yaml BASE_EXPERIMENT=layerwise-eedi-full-open-t3-lr-scan JOB_PREFIX=layerwise-eedi-full-open-t3 LOSS_PROFILE_EXPECTED=state_path SUMMARY_BRANCH_EXPECTED=eedi_full_open_t3 VALIDATOR=scripts/validate_layerwise_eedi_full_open_t3_lr_scan_amlt.py exec bash scripts/watch_layerwise_eedi_full_lr_scan.sh"
  check_controller \
    layerwise_eedi_full_open_t3_selective_kd_lr_scan \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/eedi_full_open_t3_selective_kd_lr_scan" \
    "STATE_SLUG=eedi_full_open_t3_selective_kd_lr_scan CONFIG_PATH=amlt/feeds_layerwise_eedi_full_open_t3_selective_kd_lr_scan_20260802.yaml BASE_EXPERIMENT=layerwise-eedi-full-open-t3-selective-kd-lr-scan JOB_PREFIX=layerwise-eedi-full-open-t3-selective-kd LOSS_PROFILE_EXPECTED=state_kd SUMMARY_BRANCH_EXPECTED=eedi_full_open_t3 VALIDATOR=scripts/validate_layerwise_eedi_full_open_t3_selective_kd_lr_scan_amlt.py exec bash scripts/watch_layerwise_eedi_full_lr_scan.sh"
  check_controller \
    layerwise_eedi_full_open_t3_selective_kd_lr2e6_round3800_local_preflight \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/eedi_full_open_t3_selective_kd_lr_scan/local_lr2e6_round3800_preflight" \
    "exec bash scripts/watch_layerwise_eedi_full_open_t3_selective_kd_lr2e6_round3800_local_preflight.sh"
  check_controller \
    layerwise_eedi_full_open_t3_selective_kd_lr2e6_round3950_local_c4 \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/eedi_full_open_t3_selective_kd_lr_scan/local_lr2e6_round3950_c4" \
    "exec bash scripts/watch_layerwise_eedi_full_open_t3_selective_kd_lr2e6_round3950_local_c4.sh"
  check_controller \
    layerwise_eedi_open_t3_t6_repair \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/eedi_open_t3_t6_repair" \
    "exec bash scripts/watch_layerwise_eedi_open_t3_t6_repair.sh"
  clear_stale_candidate_aggregation_attention \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/task_retention_promoted_final_fit" \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/task_retention_rehearsal" \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/eedi_state_path_ab" \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/state_kd" \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/staged_task_retention" \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/staged_state_path" \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/equal_state_path" \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/equal_state_kd" \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/equal_state_kd_open_t3" \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/equal_state_path_open_t3" \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/equal_state_kd_open_t3_lr_scan" \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/equal_state_kd_open_t3_lr1e6" \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/open_t3_consolidation" \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/eedi_full_lr_scan" \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/eedi_full_state_kd_lr_scan" \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/eedi_full_open_t3_lr_scan" \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/eedi_full_open_t3_selective_kd_lr_scan"
  check_controller \
    layerwise_task_retention_promoted_final_fit \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/task_retention_promoted_final_fit" \
    "exec bash scripts/watch_layerwise_task_retention_promoted_final_fit.sh"
  check_controller \
    layerwise_promoted_round1000_local_c4 \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/task_retention_promoted_final_fit/local_round1000_c4" \
    "exec bash scripts/watch_layerwise_promoted_round1000_local_c4.sh"
  check_controller \
    layerwise_promoted_round1000_c4_n150_multiseed \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/task_retention_promoted_final_fit/round1000_c4_n150_multiseed" \
    "exec bash scripts/watch_promoted_round1000_c4_n150_multiseed.sh"
  check_controller \
    layerwise_promoted_round1250_c4_n150_multiseed \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/task_retention_promoted_final_fit/round1250_c4_n150_multiseed" \
    "exec bash scripts/watch_promoted_round1250_c4_n150_multiseed.sh"
  check_controller \
    layerwise_promoted_round1250_same_rule_repair_ab \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/task_retention_promoted_final_fit/round1250_same_rule_repair_ab" \
    "exec bash scripts/watch_promoted_round1250_same_rule_repair_ab.sh"
  check_controller \
    layerwise_promoted_round1750_c4_n150_multiseed \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/task_retention_promoted_final_fit/round1750_c4_n150_multiseed" \
    "exec bash scripts/watch_promoted_round1750_c4_n150_multiseed.sh"
  check_controller \
    layerwise_promoted_round2500_local_c4 \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/task_retention_promoted_final_fit/local_round2500_c4" \
    "exec bash scripts/watch_promoted_round2500_local_c4.sh"
  check_controller \
    layerwise_promoted_round2500_trajectory \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/task_retention_promoted_final_fit/local_round2500_trajectory" \
    "exec bash scripts/watch_promoted_checkpoint2500_trajectory.sh"
  check_controller \
    layerwise_c3_exposure_matched_round1500 \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/c3_exposure_matched_round1500_hashfix_v2" \
    "exec bash scripts/watch_c3_exposure_matched_round1500.sh"
  check_controller \
    layerwise_candidate_lr5e6_statepath_full_t3 \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/t3_full_candidate_screen_20260804/lr5e6_statepath" \
    "exec bash scripts/watch_candidate_lr5e6_statepath_full_t3.sh"
  check_controller \
    layerwise_cf1_fulltask_crossadapter_full_t3 \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/cf1_fulltask_crossadapter_full_t3_20260804" \
    "exec bash scripts/watch_cf1_fulltask_crossadapter_full_t3.sh"
  check_controller \
    layerwise_selective_kd_lr2e6_full_t3 \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/selective_kd_lr2e6_full_t3_20260804" \
    "exec bash scripts/watch_selective_kd_lr2e6_full_t3.sh"
  check_controller \
    layerwise_current_lr2e6_fulltask_crossadapter_full_t3 \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/current_lr2e6_fulltask_crossadapter_full_t3_20260804" \
    "exec bash scripts/watch_current_lr2e6_fulltask_crossadapter_full_t3.sh"
  check_controller \
    layerwise_t3_direct_teacher_crossmatrix_eedi75 \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/t3_direct_teacher_crossmatrix_eedi75_20260804" \
    "exec bash scripts/watch_t3_direct_teacher_crossmatrix_eedi75.sh"
  check_controller \
    layerwise_external_state_teacher_eedi_full \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/external_state_teacher_eedi_full_20260804" \
    "exec bash scripts/watch_external_state_teacher_eedi_full.sh"
  check_controller \
    layerwise_gold_semantic_teacher_eedi_full \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/gold_semantic_teacher_eedi_full_20260804" \
    "CONFIG_PATH=amlt/feeds_gold_semantic_teacher_eedi_full_diagnostic_20260804.yaml VALIDATOR=scripts/validate_gold_semantic_teacher_eedi_full_amlt.py BASE_EXPERIMENT=gold-semantic-teacher-eedi-full-repair-seed42-20260804 STATE_SLUG=gold_semantic_teacher_eedi_full_20260804 JOB_PREFIX=gold-semantic-teacher RECOVERY_PREFIX=gold-semantic-teacher exec bash scripts/watch_external_state_teacher_eedi_full.sh"
  check_controller \
    layerwise_current_lr2e6_direct_teacher_t3 \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/current_lr2e6_direct_teacher_t3_20260804" \
    "exec bash scripts/watch_current_lr2e6_direct_teacher_t3.sh"
  check_controller \
    layerwise_promoted_round1000_t3_semantic \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/task_retention_promoted_final_fit/local_round1000_t3_semantic" \
    "exec bash scripts/watch_promoted_round1000_t3_semantic.sh"
  clear_stale_dependency_attention \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/task_retention_promoted_final_eval" \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/task_retention_promoted_final_fit" \
    "promoted final fit needs attention"
  check_controller \
    layerwise_task_retention_promoted_final_eval \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/task_retention_promoted_final_eval" \
    "exec bash scripts/watch_layerwise_task_retention_promoted_final_eval.sh"
  check_controller \
    layerwise_checkpoint750_local_c4 \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/checkpoint750_local_audit/c4_full" \
    "exec bash scripts/watch_layerwise_checkpoint750_local_c4.sh"
  check_controller \
    layerwise_checkpoint750_local_formal_full \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/checkpoint750_local_audit/formal_full" \
    "exec bash scripts/watch_layerwise_checkpoint750_local_formal_full.sh"
  check_controller \
    layerwise_checkpoint750_formal_full_feeds \
    "$project_dir/results/layerwise_unified_final_pipeline_20260731/checkpoint750_formal_full_feeds" \
    "exec bash scripts/watch_layerwise_checkpoint750_formal_full_feeds.sh"
  # Qualified eight-token teacher -> student curriculum and exact unified
  # model pipeline.  Every owner below remains responsible for its own AMLT
  # preflight, idempotent experiment lookup, recovery waves, artifact checks,
  # and scientific gates.  The supervisor only restores a dead local watcher;
  # it never submits a State8 job directly.
  check_controller \
    state8_qualified_primary_teacher \
    "$project_dir/results/state8_qualified_teacher_pipeline_20260805/primary_curriculum" \
    "export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure-keyflip-feeds; exec bash scripts/watch_state8_primary_teacher_curriculum.sh"
  check_controller \
    state8_semantic_c4_g005 \
    "$project_dir/results/state8_semantic_candidates_20260806/c4_g005" \
    "EXPERIMENT=state8-semantic-causal-strongbase-grid-seed45-20260806 JOB=state8-semantic-causal-g005-seed45 CANDIDATE_SLUG=c4_g005 CONFIG_PATH=amlt/feeds_state8_semantic_causal_grid_20260806.yaml exec bash scripts/watch_state8_semantic_candidate.sh"
  check_controller \
    state8_semantic_c4_g010 \
    "$project_dir/results/state8_semantic_candidates_20260806/c4_g010" \
    "EXPERIMENT=state8-semantic-causal-strongbase-seed45-20260806 JOB=state8-semantic-causal-seed45 CANDIDATE_SLUG=c4_g010 CONFIG_PATH=amlt/feeds_state8_semantic_causal_screen_20260806.yaml exec bash scripts/watch_state8_semantic_candidate.sh"
  check_controller \
    state8_semantic_c4_g025 \
    "$project_dir/results/state8_semantic_candidates_20260806/c4_g025" \
    "EXPERIMENT=state8-semantic-causal-strongbase-grid-seed45-20260806 JOB=state8-semantic-causal-g025-seed45 CANDIDATE_SLUG=c4_g025 CONFIG_PATH=amlt/feeds_state8_semantic_causal_grid_20260806.yaml exec bash scripts/watch_state8_semantic_candidate.sh"
  check_controller \
    state8_semantic_eedi_g005 \
    "$project_dir/results/state8_semantic_candidates_20260806/eedi_g005" \
    "EXPERIMENT=state8-eedi-semantic-geometry-grid-v2-seed45-20260806 JOB=state8-eedi-geometry-g005-seed45 CANDIDATE_SLUG=eedi_g005 CONFIG_PATH=amlt/feeds_state8_eedi_semantic_geometry_grid_20260806.yaml exec bash scripts/watch_state8_semantic_candidate.sh"
  check_controller \
    state8_semantic_eedi_g010 \
    "$project_dir/results/state8_semantic_candidates_20260806/eedi_g010" \
    "EXPERIMENT=state8-eedi-semantic-geometry-grid-v2-seed45-20260806 JOB=state8-eedi-geometry-g010-seed45 CANDIDATE_SLUG=eedi_g010 CONFIG_PATH=amlt/feeds_state8_eedi_semantic_geometry_grid_20260806.yaml exec bash scripts/watch_state8_semantic_candidate.sh"
  check_controller \
    state8_semantic_eedi_g025 \
    "$project_dir/results/state8_semantic_candidates_20260806/eedi_g025" \
    "EXPERIMENT=state8-eedi-semantic-geometry-high-grid-seed45-20260806 JOB=state8-eedi-geometry-g025-seed45 CANDIDATE_SLUG=eedi_g025 CONFIG_PATH=amlt/feeds_state8_eedi_semantic_geometry_high_grid_20260806.yaml exec bash scripts/watch_state8_semantic_candidate.sh"
  check_controller \
    state8_semantic_eedi_g050 \
    "$project_dir/results/state8_semantic_candidates_20260806/eedi_g050" \
    "EXPERIMENT=state8-eedi-semantic-geometry-high-grid-seed45-20260806 JOB=state8-eedi-geometry-g050-seed45 CANDIDATE_SLUG=eedi_g050 CONFIG_PATH=amlt/feeds_state8_eedi_semantic_geometry_high_grid_20260806.yaml exec bash scripts/watch_state8_semantic_candidate.sh"
  check_controller \
    state8_semantic_eedi_g025_single \
    "$project_dir/results/state8_semantic_candidates_20260806/eedi_g025_single" \
    "EXPERIMENT=state8-eedi-g025-single-seed46-20260806 JOB=state8-eedi-g025-single-seed46 CANDIDATE_SLUG=eedi_g025_single CONFIG_PATH=amlt/feeds_state8_eedi_g025_single_a100_20260806.yaml VALIDATOR=scripts/validate_state8_eedi_g025_single_amlt.py exec bash scripts/watch_state8_semantic_candidate.sh"
  check_controller \
    state8_semantic_training_progress \
    "$project_dir/results/state8_semantic_training_progress_20260806" \
    "export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure-keyflip-feeds; exec bash scripts/watch_state8_semantic_training_progress.sh"
  check_controller \
    state8_qualified_balanced_teacher \
    "$project_dir/results/state8_qualified_balanced_teacher_pipeline_20260805" \
    "export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure-keyflip-feeds; exec bash scripts/watch_state8_balanced_teacher_curriculum.sh"
  check_controller \
    state8_qualified_highkd_teacher \
    "$project_dir/results/state8_qualified_highkd_teacher_pipeline_20260805" \
    "export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure-keyflip-feeds; exec bash scripts/watch_state8_highkd_teacher_curriculum.sh"
  check_controller \
    state8_qualified_primary_promotion \
    "$project_dir/results/state8_qualified_promotion_eval_20260805" \
    "export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure-keyflip-feeds PROMOTION_VARIANT=t3_heavy; exec bash scripts/watch_state8_promotion_eval.sh"
  check_controller \
    state8_qualified_balanced_promotion \
    "$project_dir/results/state8_qualified_balanced_promotion_eval_20260805" \
    "export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure-keyflip-feeds PROMOTION_VARIANT=balanced; exec bash scripts/watch_state8_promotion_eval.sh"
  check_controller \
    state8_qualified_highkd_promotion \
    "$project_dir/results/state8_qualified_highkd_promotion_eval_20260805" \
    "export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure-keyflip-feeds PROMOTION_VARIANT=highkd; exec bash scripts/watch_state8_promotion_eval.sh"
  check_controller \
    state8_qualified_promotion_selection \
    "$project_dir/results/state8_qualified_promotion_selection_20260805" \
    "exec bash scripts/watch_state8_qualified_promotion_selection.sh"
  check_controller \
    state8_qualified_unified_final_fit \
    "$project_dir/results/state8_unified_final_fit_20260805" \
    "export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure-keyflip-feeds; exec bash scripts/watch_state8_unified_final_fit.sh"
  check_controller \
    state8_qualified_complete_final_eval \
    "$project_dir/results/state8_complete_final_eval_20260805" \
    "export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure-keyflip-feeds; exec bash scripts/watch_state8_final_eval.sh"
  check_controller \
    state8_qualified_live_status \
    "$project_dir/results/state8_qualified_pipeline_status" \
    "exec bash scripts/watch_state8_qualified_live_status.sh"
  check_controller \
    state8_training_progress_watchdog \
    "$project_dir/results/state8_training_progress_watchdog_20260805" \
    "export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure-keyflip-feeds; exec bash scripts/watch_state8_training_progress.sh"
  python scripts/summarize_layerwise_unified_live_status.py \
    --root "$project_dir/results/layerwise_unified_final_pipeline_20260731" \
    --output "$project_dir/results/layerwise_unified_final_pipeline_20260731/live_status.json" \
    > "$state_dir/live_status.log" 2>&1 || true
  mkdir -p "$project_dir/results/state8_large_evaluation_live_status_20260813"
  python scripts/summarize_state8_large_evaluation_live_status.py \
    --project "$project_dir" \
    --output "$project_dir/results/state8_large_evaluation_live_status_20260813/status.json" \
    > "$project_dir/results/state8_large_evaluation_live_status_20260813/status.log" \
    2>&1 || true
  mv "$state_dir/snapshot.partial" "$state_dir/snapshot.tsv"
  sleep "$poll_seconds"
done
