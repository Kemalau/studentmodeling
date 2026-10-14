#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
curricula="$project/results/state8_taxonomy_curriculum_candidates_20260807"
state="$project/results/state8_taxonomy_unified_pipeline_20260807"
config="$project/amlt/feeds_state8_taxonomy_curriculum_consolidation_20260807.yaml"
final_config="$project/amlt/feeds_state8_highcausal_final_fit_20260806.yaml"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
eval_jsonl="$project/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
posttrain_stall_seconds="${STATE8_PIPELINE_POSTTRAIN_STALL_SECONDS:-21600}"
if [[ -z "${AZURE_CONFIG_DIR:-}" ]]; then
  if AZURE_CONFIG_DIR=/home/v-yuejiali/.azure az account get-access-token \
      --subscription b6dc87f3-c479-49c8-8cb5-7896da3ff895 \
      --query expiresOn -o tsv >/dev/null 2>&1; then
    export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure
  else
    export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure-studentmodeling-feeds
  fi
fi

slugs=(taxonomy_t3heavy taxonomy_balanced taxonomy_t3heavy_diag taxonomy_balanced_diag)
jobs=(state8-taxonomy-curriculum-consolidation-t3heavy-seed52 \
      state8-taxonomy-curriculum-consolidation-balanced-seed53 \
      state8-taxonomy-curriculum-consolidation-t3heavy-diag-seed54 \
      state8-taxonomy-curriculum-consolidation-balanced-diag-seed55)
sources=(taxonomy_t3heavy taxonomy_balanced taxonomy_t3heavy taxonomy_balanced)

mkdir -p "$state"
cd "$project" || exit 1
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"
trap 'rm -f "$state/controller.active.pid"' EXIT
rm -f "$state/needs_attention"
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }

observe_progress(){
  local experiment="$1" job="$2" root="$3" total_rounds="$4" stall_seconds="$5" max_run_seconds="$6"
  local progress="$root/progress" listing="" log_file=""
  mkdir -p "$progress"
  listing="$(timeout 90 amlt logs list "$experiment" ":$job" 2>/dev/null)" || return 0
  log_file="$(LISTING="$listing" "$python_bin" - <<'PY'
import os,re
paths=[]
for line in os.environ['LISTING'].splitlines():
 value=line.strip()
 if re.fullmatch(r'user_logs/(?:retry_\d+/)?std_log_process_(?:0|00)\.txt',value):
  retry=re.search(r'retry_(\d+)',value); paths.append((int(retry.group(1)) if retry else -1,value))
print(max(paths)[1] if paths else '')
PY
)" || return 0
  [[ -n "$log_file" ]] || return 0
  timeout 90 amlt logs view -F "$log_file" -n 800 "$experiment" ":$job" \
    > "$progress/latest_rank0.log" 2> "$progress/latest_rank0.stderr" || return 0
  "$python_bin" scripts/monitor_layerwise_unified_training_progress.py \
    --log "$progress/latest_rank0.log" --experiment "$experiment" \
    --log-file "$log_file" --status running --output "$progress/progress.json" \
    --total-rounds "$total_rounds" --max-run-seconds "$max_run_seconds" \
    >/dev/null 2>&1 || return 0
  if "$python_bin" - "$progress/progress.json" "$stall_seconds" "$posttrain_stall_seconds" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); unhealthy=(x.get('health') or {}).get('healthy') is False
training_complete=(float(x.get('progress_fraction') or 0)>=1.0
                   or int(x.get('optimizer_step') or -1)>=int(x.get('total_rounds') or -1)>0)
threshold=float(sys.argv[3] if training_complete else sys.argv[2])
stalled=float(x.get('seconds_since_progress') or 0)>=threshold
raise SystemExit(0 if stalled or unhealthy else 1)
PY
  then
    cp "$progress/progress.json" "$root/$experiment.stalled.json"
    timeout 90 amlt show "$experiment" > "$root/$experiment.stalled.show.txt" 2>&1 || true
    timeout 180 amlt logs view -n 2500 "$experiment" ":$job" \
      > "$root/$experiment.stalled.log" 2>&1 || true
    if timeout 90 amlt cancel "$experiment" ":$job" -y \
      > "$root/$experiment.stalled.cancel.txt" 2>&1; then
      log "cancelled stalled/unhealthy job for exact recovery experiment=$experiment job=$job"
    fi
  fi
}

status_phase(){
  STATUS="$1" "$python_bin" - <<'PY'
import json,os
try: rows=json.loads(os.environ['STATUS'])
except Exception: print('unknown'); raise SystemExit
print(rows[0]['status'] if len(rows)==1 else 'invalid')
PY
}

remote_experiment_class(){
  local experiment="$1" output rc
  output="$(timeout 90 amlt status --no-update-db "$experiment" 2>&1)"; rc=$?
  STATUS_RC="$rc" STATUS_OUTPUT="$output" "$python_bin" - <<'PY'
import os
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(os.environ['STATUS_RC']), os.environ['STATUS_OUTPUT']))
PY
}

wait_source(){
  local source="$1" root="$curricula/$source"
  while [[ ! -f "$root/complete" ]]; do
    heartbeat
    if [[ -f "$root/needs_attention" ]]; then
      recover_source "$source"
    fi
    sleep 120
  done
  [[ -f "$root/remote_passed" ]] || fail "source curriculum completed without remote pass: $source"
}

recover_source(){
  local source="$1" root="$curricula/$source" job config validator current wave next session recover_fd
  exec {recover_fd}>"$root/recovery.lock"; flock "$recover_fd"
  if [[ ! -f "$root/needs_attention" || -f "$root/complete" ]]; then
    flock -u "$recover_fd"; return 0
  fi
  case "$source" in
    taxonomy_t3heavy)
      # The T3-heavy optimizer trajectory is already complete at the exact
      # round-1250 checkpoint.  Recover only the failed provenance/finalization
      # phase on one GPU; four-GPU exact recovery would reload the same final
      # checkpoint without performing another optimizer step.
      job=state8-taxonomy-t3heavy-finalizer-seed42
      config=amlt/feeds_state8_taxonomy_t3heavy_finalizer_20260807.yaml
      validator=scripts/validate_state8_taxonomy_t3heavy_finalizer_amlt.py ;;
    taxonomy_balanced)
      job=state8-taxonomy-balanced-curriculum-seed43
      config=amlt/feeds_state8_taxonomy_balanced_curriculum_20260807.yaml
      validator=scripts/validate_state8_taxonomy_teacher_curriculum_amlt.py ;;
    *) flock -u "$recover_fd"; fail "unknown source curriculum=$source" ;;
  esac
  current="$("$python_bin" - "$root/latest_status.json" <<'PY'
import json,sys
try: rows=json.load(open(sys.argv[1])); print(rows[0]['experiment'] if len(rows)==1 else '')
except Exception: print('')
PY
)"
  if [[ -f "$root/remote_passed" && -n "$current" ]]; then
    # The expensive remote phase already succeeded.  A later download or
    # semantic-judge failure is local post-processing and must never trigger
    # another four-GPU training job.
    rm -f "$root/needs_attention" "$root/complete"
    session="state8-${source}-postprocess-$(date -u +%s)"
    tmux new-session -d -s "$session" \
      "cd $project && STATE8_CURRICULUM_EXPERIMENT=$current STATE8_CURRICULUM_JOB=$job STATE8_CURRICULUM_SLUG=$source exec bash scripts/watch_state8_taxonomy_teacher_curriculum.sh 2>&1 | tee -a $root/controller.tmux.log" \
      || true
    log "restarted local postprocess without retraining source=$source experiment=$current"
    flock -u "$recover_fd"
    return 0
  fi
  wave=2
  [[ "$current" =~ recovery([0-9]+)-20260807 ]] && wave="${BASH_REMATCH[1]}"
  wave=$((wave+1))
  next="state8-${source#taxonomy_}-curriculum-recovery${wave}-20260807"
  exec 8>"$project/results/feeds_submission_global.lock"; flock 8
  amlt run "$config" ":$job" --dump -t Feeds > "$root/$next.preflight.yaml" \
    2> "$root/$next.preflight.stderr" || { flock -u 8; flock -u "$recover_fd"; return 0; }
  "$python_bin" "$validator" \
    --config "$root/$next.preflight.yaml" >> "$root/controller.log" 2>&1 \
    || { flock -u 8; flock -u "$recover_fd"; fail "source recovery preflight failed: $source"; }
  amlt --json-tables ti sing -t Feeds > "$root/$next.quota.json" 2>> "$root/controller.log" \
    || { flock -u 8; flock -u "$recover_fd"; return 0; }
  class="$(remote_experiment_class "$next")" \
    || { flock -u 8; flock -u "$recover_fd"; return 0; }
  if [[ "$class" == missing ]]; then
    amlt run "$config" ":$job" "$next" -t Feeds -y \
      --description "Automatic exact-resume recovery for taxonomy curriculum source=$source wave=$wave" \
      > "$root/$next.submission.txt" 2>&1 \
      || { flock -u 8; flock -u "$recover_fd"; return 0; }
  elif [[ "$class" == exists ]]; then
    printf 'status=preexisting_not_resubmitted\n' > "$root/$next.submission.txt"
  else
    flock -u 8; flock -u "$recover_fd"; return 0
  fi
  flock -u 8
  rm -f "$root/needs_attention" "$root/complete" "$root/remote_passed"
  session="state8-${source}-recovery${wave}"
  tmux new-session -d -s "$session" \
    "cd $project && STATE8_CURRICULUM_EXPERIMENT=$next STATE8_CURRICULUM_JOB=$job STATE8_CURRICULUM_SLUG=$source exec bash scripts/watch_state8_taxonomy_teacher_curriculum.sh 2>&1 | tee -a $root/controller.tmux.log" \
    || true
  log "recovered source curriculum=$source experiment=$next"
  flock -u "$recover_fd"
}

submit_arm(){
  local index="$1" slug="${slugs[$index]}" job="${jobs[$index]}" source="${sources[$index]}"
  local arm="$state/$slug" wave=0 experiment status phase
  mkdir -p "$arm"; wait_source "$source"
  if [[ -s "$arm/active_experiment.txt" ]]; then
    experiment="$(<"$arm/active_experiment.txt")"
    [[ "$experiment" =~ recovery([0-9]+)-20260807 ]] && wave="${BASH_REMATCH[1]}"
  else
    experiment="state8-taxonomy-consolidation-${slug}-20260807"
    printf '%s\n' "$experiment" > "$arm/active_experiment.txt"
  fi
  while true; do
    if [[ ! -f "$arm/submitted.$experiment" ]]; then
      exec 8>"$project/results/feeds_submission_global.lock"; flock 8
      amlt run "$config" ":$job" --dump -t Feeds > "$arm/$experiment.preflight.yaml" \
        2> "$arm/$experiment.preflight.stderr" || { flock -u 8; sleep 120; continue; }
      "$python_bin" scripts/validate_state8_taxonomy_curriculum_consolidation_amlt.py \
        --config "$arm/$experiment.preflight.yaml" --jobs "$job" \
        >> "$arm/controller.log" 2>&1 || { flock -u 8; fail "preflight failed arm=$slug"; }
      amlt --json-tables ti sing -t Feeds > "$arm/$experiment.quota.json" 2>> "$arm/controller.log" \
        || { flock -u 8; sleep 120; continue; }
      class="$(remote_experiment_class "$experiment")" || { flock -u 8; sleep 120; continue; }
      if [[ "$class" == missing ]]; then
        if ! amlt run "$config" ":$job" "$experiment" -t Feeds -y \
          --description "State8 taxonomy curriculum strict-state-only diagnosis consolidation arm=$slug recovery=$wave" \
          > "$arm/$experiment.submission.txt" 2>&1; then
          flock -u 8; sleep 120; continue
        fi
      elif [[ "$class" == exists ]]; then
        printf 'status=preexisting_not_resubmitted\n' > "$arm/$experiment.submission.txt"
      else
        flock -u 8; sleep 120; continue
      fi
      flock -u 8
      touch "$arm/submitted.$experiment"; printf '%s\n' "$experiment" > "$arm/active_experiment.txt"
      log "submitted consolidation arm=$slug experiment=$experiment"
    fi
    heartbeat
    status="$(timeout 90 amlt --json-tables status "$experiment" --hide-urls 2>/dev/null)" \
      || { sleep 120; continue; }
    printf '%s\n' "$status" > "$arm/latest_status.json"
    phase="$(status_phase "$status")"
    if [[ "$phase" == running ]]; then
      observe_progress "$experiment" "$job" "$arm" 250 2700 172800
    fi
    case "$phase" in
      preparing|queued|running|paused|resuming|pre-queued) sleep 120; continue ;;
      pass) touch "$arm/remote_passed"; break ;;
      *)
        timeout 90 amlt show "$experiment" > "$arm/$experiment.show.txt" 2>&1 || true
        timeout 180 amlt logs view -n 2500 "$experiment" ":$job" > "$arm/$experiment.failure.log" 2>&1 || true
        (( wave < 32 )) || fail "consolidation recovery exhausted arm=$slug"
        wave=$((wave+1)); experiment="state8-taxonomy-consolidation-${slug}-recovery${wave}-20260807"
        printf '%s\n' "$experiment" > "$arm/active_experiment.txt"
        ;;
    esac
  done
  rm -rf "$arm/download"; mkdir -p "$arm/download"
  amlt results download "$experiment" --status pass --include '*.json' --include '*.jsonl' --include '*.txt' \
    --output "$arm/download" >> "$arm/controller.log" 2>&1 || fail "download failed arm=$slug"
  mapfile -t formal < <(find "$arm/download" -type f -path "*/$job/eval/formal_t3_300/predictions.jsonl")
  mapfile -t diag_pred < <(find "$arm/download" -type f -path "*/$job/eval/diagnosis_c4_50/judge_predictions.jsonl")
  mapfile -t diag_eval < <(find "$arm/download" -type f -path "*/$job/eval/diagnosis_c4_50/judge_eval.jsonl")
  mapfile -t diag_metrics < <(find "$arm/download" -type f -path "*/$job/eval/diagnosis_c4_50/metrics.json")
  [[ ${#formal[@]} -eq 1 && ${#diag_pred[@]} -eq 1 && ${#diag_eval[@]} -eq 1 && ${#diag_metrics[@]} -eq 1 ]] \
    || fail "artifact cardinality drift arm=$slug"
  mkdir -p "$arm/t3_semantic" "$arm/c4_semantic"
  PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
    --predictions-jsonl "${formal[0]}" --eval-jsonl "$eval_jsonl" --output-dir "$arm/t3_semantic" \
    --judge-backend trapi --judge-workers 12 --judge-max-retries 12 --judge-timeout-seconds 60 \
    --judge-cache "$arm/t3_semantic/cache.jsonl" --judge-audit-samples 100 --seed 42 \
    >> "$arm/t3_semantic/judge.log" 2>&1 || fail "T3 judge failed arm=$slug"
  PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
    --predictions-jsonl "${diag_pred[0]}" --eval-jsonl "${diag_eval[0]}" --output-dir "$arm/c4_semantic" \
    --judge-backend trapi --judge-workers 12 --judge-max-retries 12 --judge-timeout-seconds 60 \
    --judge-cache "$arm/c4_semantic/cache.jsonl" --judge-audit-samples 100 --seed 42 \
    >> "$arm/c4_semantic/judge.log" 2>&1 || fail "C4 judge failed arm=$slug"
  "$python_bin" scripts/summarize_state8_diagnosis_c4.py \
    --judge-redacted "$arm/c4_semantic/predictions_redacted.jsonl" \
    --generation-metrics "${diag_metrics[0]}" --output "$arm/c4_semantic/summary.json" \
    >> "$arm/c4_semantic/controller.log" 2>&1 || fail "C4 summary failed arm=$slug"
  touch "$arm/complete"; log "consolidation and semantic screen complete arm=$slug"
}

# Start an arm as soon as its own source curriculum completes.  Each worker has
# an independent lock/status directory and only the parent performs selection.
worker_pids=()
for index in 0 1 2 3; do
  (submit_arm "$index") > "$state/${slugs[$index]}.worker.log" 2>&1 &
  worker_pids+=("$!")
done
for pid in "${worker_pids[@]}"; do wait "$pid" || fail "consolidation worker failed pid=$pid"; done

selection="$state/selection.json"
args=()
for slug in "${slugs[@]}"; do args+=(--candidate "$slug" "$state/$slug"); done
"$python_bin" scripts/select_state8_taxonomy_consolidation.py "${args[@]}" --output "$selection" \
  > "$state/selection.log" 2>&1
rc=$?
if (( rc != 0 )); then touch "$state/scientific_gate_failed"; fail "no consolidation arm passed unified semantic gate"; fi
az storage blob upload --auth-mode login --account-name csnewsandfeeds4150361735 --container-name amulet \
  --name studentmodeling/gates/state8_taxonomy_curriculum_v1/selection.json \
  --file "$selection" --overwrite >> "$state/controller.log" 2>&1 || fail "promotion gate upload failed"

readarray -t chosen < <("$python_bin" - "$selection" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); s=x['selected']; assert x['pass'] and s['qualified'],x
print(s['finalfit_variant']); print(s['finalfit_job']); print(s['artifact_sha256']['adapter_model']); print(s['artifact_sha256']['recurrent_state'])
PY
)
[[ ${#chosen[@]} -eq 4 ]] || fail "selection receipt invalid"
variant="${chosen[0]}"; final_job="${chosen[1]}"
final_state="$state/final_fit"; mkdir -p "$final_state"
wave=0; experiment="state8-taxonomy-unified-finalfit-20260807"
if [[ -s "$final_state/active_experiment.txt" ]]; then
  experiment="$(<"$final_state/active_experiment.txt")"
  [[ "$experiment" =~ recovery([0-9]+)-20260807 ]] && wave="${BASH_REMATCH[1]}"
else
  # Persist submission intent before the cost-bearing call.  If the local
  # controller is restarted in the narrow submit/receipt window, it resumes
  # this exact experiment name instead of falling back to the base run.
  printf '%s\n' "$experiment" > "$final_state/active_experiment.txt"
fi
while true; do
  if [[ ! -f "$final_state/submitted.$experiment" ]]; then
    exec 8>"$project/results/feeds_submission_global.lock"; flock 8
    amlt run "$final_config" ":$final_job" --dump -t Feeds > "$final_state/$experiment.preflight.yaml" \
      2> "$final_state/$experiment.preflight.stderr" || { flock -u 8; sleep 120; continue; }
    "$python_bin" scripts/validate_state8_unified_final_fit_amlt.py --config "$final_state/$experiment.preflight.yaml" \
      >> "$final_state/controller.log" 2>&1 || { flock -u 8; fail "final fit preflight failed"; }
    amlt --json-tables ti sing -t Feeds > "$final_state/$experiment.quota.json" 2>> "$final_state/controller.log" \
      || { flock -u 8; sleep 120; continue; }
    class="$(remote_experiment_class "$experiment")" || { flock -u 8; sleep 120; continue; }
    if [[ "$class" == missing ]]; then
      if ! amlt run "$final_config" ":$final_job" "$experiment" -t Feeds -y \
        --description "Selected taxonomy consolidation=$variant; exact 259845-row strict-state-only unified epoch; recovery=$wave" \
        > "$final_state/$experiment.submission.txt" 2>&1; then
        flock -u 8; sleep 120; continue
      fi
    elif [[ "$class" == exists ]]; then
      printf 'status=preexisting_not_resubmitted\n' > "$final_state/$experiment.submission.txt"
    else
      flock -u 8; sleep 120; continue
    fi
    flock -u 8
    touch "$final_state/submitted.$experiment"
    printf '%s\n' "$experiment" > "$final_state/active_experiment.txt"
    log "submitted selected full fit job=$final_job experiment=$experiment"
  fi
  heartbeat
  status="$(timeout 90 amlt --json-tables status "$experiment" --hide-urls 2>/dev/null)" || { sleep 120; continue; }
  printf '%s\n' "$status" > "$final_state/latest_status.json"; phase="$(status_phase "$status")"
  if [[ "$phase" == running ]]; then
    observe_progress "$experiment" "$final_job" "$final_state" 16241 7200 345600
  fi
  case "$phase" in
    preparing|queued|running|paused|resuming|pre-queued) sleep 120 ;;
    pass) touch "$final_state/remote_passed"; break ;;
    *)
      timeout 180 amlt logs view -n 3000 "$experiment" ":$final_job" > "$final_state/$experiment.failure.log" 2>&1 || true
      (( wave < 32 )) || fail "selected final fit exhausted recovery waves"
      wave=$((wave+1)); experiment="state8-taxonomy-unified-finalfit-recovery${wave}-20260807"
      printf '%s\n' "$experiment" > "$final_state/active_experiment.txt"
      ;;
  esac
done
rm -rf "$final_state/download"; mkdir -p "$final_state/download"
amlt results download "$experiment" --status pass --include '*.json' --include '*.safetensors' --include '*.pt' \
  --output "$final_state/download" >> "$final_state/controller.log" 2>&1 || fail "final-fit download failed"
mapfile -t receipts < <(find "$final_state/download" -type f -name state8_final_fit_receipt.json)
mapfile -t published < <(find "$final_state/download" -type f -name canonical_publish_receipt.json)
[[ ${#receipts[@]} -eq 1 && ${#published[@]} -eq 1 ]] || fail "final-fit receipt cardinality drift"
"$python_bin" - "${receipts[0]}" "${published[0]}" "$variant" "${chosen[2]}" "${chosen[3]}" <<'PY' || fail "final-fit receipt invalid"
import json,sys
x=json.load(open(sys.argv[1])); p=json.load(open(sys.argv[2])); variant=sys.argv[3]
initial={'adapter_model':sys.argv[4],'recurrent_state':sys.argv[5]}
assert x['status']=='final_fit_complete' and x['one_epoch_exact'] is True,x
assert x['real_examples']==259845 and x['selected_replay_variant']==variant,x
assert x['initial_artifact_sha256']==initial,(x,initial)
assert x['state_shape']==[64,8,5120] and x['history_training_mode']=='strict_state_only',x
assert x['history_dropout_counts']=={'1.0':259845} and x['raw_history_visible_to_decoder'] is False,x
assert x['canonical_label_auxiliary_supervision'] is True,x
assert p['status']=='complete' and p['artifact_sha256']=={'adapter_model':x['adapter_sha256'],'recurrent_state':x['recurrent_state_sha256']},(p,x)
PY
touch "$final_state/scientific_gate_pass" "$final_state/complete"

# Reuse the fully audited 16-shard + three-seed final evaluator.  It reads the
# canonical checkpoint atomically published by the selected final fit.
FIT_DIR="$final_state" STATE_DIR="$state/final_eval" \
BASE_EXPERIMENT=state8-taxonomy-complete-final-eval-20260807 \
RECOVERY_EXPERIMENT_PREFIX=state8-taxonomy-complete-final-eval-recovery \
RECOVERY_EXPERIMENT_SUFFIX=-20260807 \
bash scripts/watch_state8_final_eval.sh || fail "complete final evaluation controller failed"
[[ -f "$state/final_eval/complete" ]] || fail "final evaluation incomplete"
if [[ -f "$state/final_eval/scientific_gate_pass" ]]; then touch "$state/scientific_gate_pass"; fi
if [[ -f "$state/final_eval/scientific_gate_failed" ]]; then touch "$state/scientific_gate_failed"; fi
touch "$state/complete"; heartbeat
log "taxonomy unified pipeline reached terminal evaluated state"
