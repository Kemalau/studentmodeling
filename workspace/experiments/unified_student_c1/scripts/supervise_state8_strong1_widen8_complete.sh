#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
bridge="$project/results/state8_strong1_widen8_bridge_20260808"
fallback="$bridge/single_fallback_monitor"
rescue="$bridge/complete_pipeline/rescue"
midpoint="$project/results/state8_midpoint_diagnosis_first_20260808"
exact_diagnosis="$project/results/state8_exact_diagnosis_pipeline_20260808"
exact_rescue="$project/results/state8_exact_scientific_rescue_20260808"
state="$bridge/complete_pipeline"
fit="$state/final_fit"
eval_state="$state/final_eval"
fit_config_world4="$project/amlt/feeds_state8_strong1_widen8_final_fit_20260808.yaml"
fit_config_world2="$project/amlt/feeds_state8_strong1_widen8_final_fit_world2_20260808.yaml"
eval_config="$project/amlt/feeds_state8_strong1_widen8_final_eval_generated_20260808.yaml"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
max_recovery_waves=32

if [[ -z "${AZURE_CONFIG_DIR:-}" ]]; then
  if AZURE_CONFIG_DIR=/home/v-yuejiali/.azure az account get-access-token \
      --subscription b6dc87f3-c479-49c8-8cb5-7896da3ff895 \
      --query expiresOn -o tsv >/dev/null 2>&1; then
    export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure
  else
    export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure-studentmodeling-feeds
  fi
fi

mkdir -p "$state" "$fit" "$eval_state"
cd "$project" || exit 1
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
# Child sleeps and the heartbeat must not inherit the controller lock.  This
# keeps an interrupted supervisor immediately restartable by its outer watcher.
sleep_unlocked(){ sleep "$1" 9>&-; }
echo $$ > "$state/controller.active.pid"
heartbeat_pid=""
cleanup(){
  if [[ -n "$heartbeat_pid" ]]; then
    kill "$heartbeat_pid" 2>/dev/null || true
    wait "$heartbeat_pid" 2>/dev/null || true
  fi
  rm -f "$state/controller.active.pid"
}
trap cleanup EXIT
rm -f "$state/needs_attention"
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }
heartbeat
(exec 9>&-; while true; do heartbeat; sleep 30; done) &
heartbeat_pid=$!

retire_queued_experiment(){
  local experiment_file="$1" receipt="$2" experiment status phase
  [[ -s "$experiment_file" ]] || return 0
  experiment="$(<"$experiment_file")"
  status="$(timeout 90 amlt --json-tables status "$experiment" --hide-urls 2>/dev/null)" \
    || { log "could not inspect redundant candidate experiment=$experiment"; return 0; }
  phase="$(STATUS="$status" "$python_bin" - <<'PY'
import json,os
try: rows=json.loads(os.environ['STATUS'])
except Exception: print('unknown'); raise SystemExit
states={str(row.get('status') or '').casefold() for row in rows}
print(next(iter(states)) if len(states)==1 else 'mixed')
PY
)"
  case "$phase" in
    queued|pre-queued)
      if timeout 90 amlt cancel --status "$phase" --yes "$experiment" \
          > "$receipt.cancel.txt" 2>&1; then
        STATUS="$status" EXPERIMENT="$experiment" PHASE="$phase" \
          RECEIPT="$receipt" "$python_bin" - <<'PY'
import json,os,pathlib
path=pathlib.Path(os.environ['RECEIPT'])
path.write_text(json.dumps({
  'stage':'state8_redundant_short_candidate_retirement_v1',
  'status':'cancelled_before_training',
  'experiment':os.environ['EXPERIMENT'],
  'observed_phase':os.environ['PHASE'],
  'jobs':json.loads(os.environ['STATUS']),
  'scientific_recipe_changed':False,
},indent=2,sort_keys=True)+'\n')
PY
        log "retired redundant not-yet-running candidate experiment=$experiment phase=$phase"
      else
        log "redundant candidate cancellation raced or failed; preserving experiment=$experiment"
      fi
      ;;
    *) log "preserving redundant candidate experiment=$experiment phase=$phase" ;;
  esac
}

selection_passes(){
  "$python_bin" - "$1" <<'PY'
import json,sys
try: x=json.load(open(sys.argv[1]))
except Exception: raise SystemExit(1)
assert x['stage'] in {
 'state8_strong1_widen8_joint_promotion_v1',
 'state8_strong1_rescue_joint_promotion_v1',
 'state8_midpoint_diagnosis_first_joint_promotion_v1',
 'state8_exact_diagnosis_curriculum_joint_promotion_v1',
 'state8_exact_scientific_rescue_joint_promotion_v1',
},x
assert x['pass'] is True and x['selected']['qualified'] is True,x
assert x['selected']['checks']['answer_C4_pass'] is True,x
assert x['selected']['checks']['diagnosis_C4_pass'] is True,x
PY
}

if [[ ! -f "$state/selection.json" ]]; then
  while true; do
    heartbeat
    main_terminal=false; fallback_terminal=false
    [[ -f "$bridge/complete" || -f "$bridge/scientific_gate_failed" \
       || -f "$bridge/superseded_before_run" ]] && main_terminal=true
    [[ -f "$fallback/complete" || -f "$fallback/scientific_gate_failed" \
       || -f "$fallback/superseded_before_run" \
       || -f "$fallback/retired_for_diagnosis_first_priority" ]] && fallback_terminal=true
    rescue_terminal=false
    [[ -f "$rescue/complete" || -f "$rescue/scientific_gate_failed" \
       || -f "$rescue/not_required" ]] && rescue_terminal=true
    midpoint_terminal=false
    [[ -f "$midpoint/complete" || -f "$midpoint/scientific_gate_failed" \
       || -f "$midpoint/superseded_before_run" ]] && midpoint_terminal=true
    exact_terminal=false
    [[ -f "$exact_diagnosis/complete" || -f "$exact_diagnosis/scientific_gate_failed" ]] \
      && exact_terminal=true
    exact_rescue_terminal=false
    [[ -f "$exact_rescue/complete" || -f "$exact_rescue/scientific_gate_failed" \
       || -f "$exact_rescue/not_required" ]] && exact_rescue_terminal=true
    if [[ "$main_terminal" == true && "$fallback_terminal" == true \
          && "$rescue_terminal" == true && "$midpoint_terminal" == true \
          && "$exact_terminal" == true && "$exact_rescue_terminal" == true ]]; then
      selection_args=(
        --selection "$bridge/selection.json"
        --selection "$fallback/selection.json"
        --selection "$rescue/selection.json"
        --selection "$midpoint/selection.json"
        --selection "$exact_diagnosis/selection.json"
      )
      [[ -s "$exact_rescue/selection.json" ]] \
        && selection_args+=(--selection "$exact_rescue/selection.json")
      set +e
      "$python_bin" scripts/select_state8_complete_candidate.py \
        "${selection_args[@]}" \
        --output "$state/selection.json" > "$state/selection.log" 2>&1
      selection_rc=$?
      set -e
      if (( selection_rc == 0 )); then
        log "selected highest-scoring qualified candidate across all terminal branches"
        break
      fi
      if (( selection_rc == 2 )); then
        touch "$state/scientific_gate_failed" "$state/complete"
        fail "all strong1-to-State8 short candidates failed the joint gate"
      fi
      fail "complete candidate selection crashed rc=$selection_rc"
    fi
    sleep_unlocked 60
  done
fi

readarray -t selected < <("$python_bin" - "$state/selection.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); slug=x['selected']['slug']
mapping={
 'lr5e7_seed58':('strong1_widen8_lr5e7_seed58','state8-strong1-widen8-lr5e7-seed58-unified-finalfit-seed42'),
 'lr1e6_seed59':('strong1_widen8_lr1e6_seed59','state8-strong1-widen8-lr1e6-seed59-unified-finalfit-seed42'),
 'single_lr5e7_seed60':('strong1_widen8_single_lr5e7_seed60','state8-strong1-widen8-single-lr5e7-seed60-unified-finalfit-seed42'),
 'taxonomy_joint':('strong1_rescue_taxonomy_joint_seed61','state8-strong1-rescue-taxonomy-joint-seed61-unified-finalfit-seed42'),
 'taskanchor_c4strong':('strong1_rescue_taskanchor_c4strong_seed62','state8-strong1-rescue-taskanchor-c4strong-seed62-unified-finalfit-seed42'),
 'midpoint_diagnosis_first_seed63':('strong1_midpoint_diagnosis_first_seed63','state8-strong1-midpoint-diagnosis-first-seed63-unified-finalfit-seed42'),
 'exact_diagnosis_curriculum_seed65':('strong1_exact_diagnosis_curriculum_seed65','state8-strong1-exact-diagnosis-curriculum-seed65-unified-finalfit-seed42'),
 'exact_rescue_semantic_preserve_seed66':('strong1_exact_rescue_semantic_preserve_seed66','state8-strong1-exact-rescue-semantic-preserve-seed66-unified-finalfit-seed42'),
 'exact_rescue_causal_taskanchor_seed67':('strong1_exact_rescue_causal_taskanchor_seed67','state8-strong1-exact-rescue-causal-taskanchor-seed67-unified-finalfit-seed42'),
 'exact_rescue_retention_preserve_seed68':('strong1_exact_rescue_retention_preserve_seed68','state8-strong1-exact-rescue-retention-preserve-seed68-unified-finalfit-seed42'),
}
variant,job=mapping[slug]
print(slug); print(variant); print(job)
PY
)
[[ ${#selected[@]} -eq 3 ]] || fail "invalid bridge selection mapping"
slug="${selected[0]}"; variant="${selected[1]}"; fit_job="${selected[2]}"

# Once one short candidate has passed both answer and diagnosis causal gates,
# an unstarted sibling cannot affect model selection.  Retire only queued work;
# never kill a healthy running candidate.  The short-candidate controllers also
# honor the central selection after their current remote job reaches a terminal
# state, so a cancellation cannot trigger an automatic recovery submission.
if [[ ! -f "$state/redundant_candidates_retired" ]]; then
  case "$slug" in
    single_lr5e7_seed60)
      retire_queued_experiment "$bridge/train_active_experiment.txt" \
        "$state/redundant_primary_retirement.json"
      ;;
    lr5e7_seed58|lr1e6_seed59)
      retire_queued_experiment "$fallback/active_experiment.txt" \
        "$state/redundant_fallback_retirement.json"
      ;;
    taxonomy_joint|taskanchor_c4strong)
      retire_queued_experiment "$bridge/train_active_experiment.txt" \
        "$state/redundant_primary_retirement.json"
      retire_queued_experiment "$fallback/active_experiment.txt" \
        "$state/redundant_fallback_retirement.json"
      ;;
    midpoint_diagnosis_first_seed63)
      retire_queued_experiment "$bridge/train_active_experiment.txt" \
        "$state/redundant_primary_retirement.json"
      retire_queued_experiment "$fallback/active_experiment.txt" \
        "$state/redundant_fallback_retirement.json"
      ;;
    exact_diagnosis_curriculum_seed65)
      retire_queued_experiment "$bridge/train_active_experiment.txt" \
        "$state/redundant_primary_retirement.json"
      retire_queued_experiment "$fallback/active_experiment.txt" \
        "$state/redundant_fallback_retirement.json"
      ;;
    exact_rescue_semantic_preserve_seed66|exact_rescue_causal_taskanchor_seed67)
      retire_queued_experiment "$bridge/train_active_experiment.txt" \
        "$state/redundant_primary_retirement.json"
      retire_queued_experiment "$fallback/active_experiment.txt" \
        "$state/redundant_fallback_retirement.json"
      ;;
  esac
  touch "$state/redundant_candidates_retired"
fi

topology_receipt="$fit/topology_selection.json"
if [[ ! -f "$topology_receipt" ]]; then
  "$python_bin" scripts/count_active_a100_requests.py --user-limit 32 \
    --output "$fit/active_a100_at_topology_selection.json" \
    > "$fit/active_a100_at_topology_selection.stdout" \
    2> "$fit/active_a100_at_topology_selection.stderr" \
    || fail "could not audit active A100 requests before final-fit topology selection"
  available_slots="$("$python_bin" - "$fit/active_a100_at_topology_selection.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); print(int(x['available_user_slots']))
PY
)"
  amlt --json-tables ti sing -t Feeds \
    > "$fit/quota_at_topology_selection.json" \
    2> "$fit/quota_at_topology_selection.stderr" \
    || fail "could not audit Feeds A100 quota before final-fit topology selection"
  cluster_slots="$("$python_bin" - "$fit/quota_at_topology_selection.json" <<'PY'
import json,sys
rows=json.load(open(sys.argv[1]))
matches=[row for row in rows
 if row.get('series')=='NDAMv4'
 and (row.get('accelerator') or {}).get('type')=='A100'
 and int((row.get('accelerator') or {}).get('memory_gb',-1))==80]
assert len(matches)==1,matches
quota=(matches[0].get('quotas_(used/limit)') or {}).get('standard') or {}
used=int(quota['used']); limit=int(quota['limit'])
assert 0 <= used <= limit,(used,limit)
print(limit-used)
PY
)" || fail "could not parse Feeds NDAMv4 Standard quota"
  "$python_bin" scripts/select_state8_finalfit_topology.py \
    --output "$topology_receipt" --topology "${FINALFIT_TOPOLOGY:-auto}" \
    --available-user-slots "$available_slots" \
    --available-cluster-slots "$cluster_slots" \
    > "$fit/topology_selection.stdout" 2> "$fit/topology_selection.stderr" \
    || fail "final-fit topology selection failed"
fi
readarray -t topology_fields < <("$python_bin" - "$topology_receipt" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); assert x['cross_topology_resume_allowed'] is False,x
print(x['topology']); print(x['config']); print(x['validator']); print(x['gpu_count'])
PY
)
[[ ${#topology_fields[@]} -eq 4 ]] || fail "invalid final-fit topology receipt"
fit_topology="${topology_fields[0]}"
fit_gpu_count="${topology_fields[3]}"
case "$fit_topology" in
  world4) fit_config="$fit_config_world4" ;;
  world2) fit_config="$fit_config_world2" ;;
  *) fail "unsupported pinned final-fit topology=$fit_topology" ;;
esac
test "$fit_config" = "$project/${topology_fields[1]}" \
  || fail "pinned final-fit config path drift"
fit_validator="$project/${topology_fields[2]}"
test -f "$fit_config" && test -f "$fit_validator" \
  || fail "pinned final-fit config/validator missing"

if [[ ! -f "$state/promotion_uploaded" ]]; then
  promotion_name=studentmodeling/gates/state8_strong1_widen8_bridge_v1/selection.json
  if "$python_bin" - "$state/selection.json" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1]))['stage']=='state8_strong1_rescue_joint_promotion_v1' else 1)
PY
  then
    promotion_name=studentmodeling/gates/state8_strong1_rescue_v1/selection.json
  fi
  if "$python_bin" - "$state/selection.json" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1]))['stage']=='state8_midpoint_diagnosis_first_joint_promotion_v1' else 1)
PY
  then
    promotion_name=studentmodeling/gates/state8_midpoint_diagnosis_first_v1/selection.json
  fi
  if "$python_bin" - "$state/selection.json" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1]))['stage']=='state8_exact_diagnosis_curriculum_joint_promotion_v1' else 1)
PY
  then
    promotion_name=studentmodeling/gates/state8_exact_diagnosis_curriculum_v1/selection.json
  fi
  if "$python_bin" - "$state/selection.json" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1]))['stage']=='state8_exact_scientific_rescue_joint_promotion_v1' else 1)
PY
  then
    promotion_name=studentmodeling/gates/state8_exact_scientific_rescue_v1/selection.json
  fi
  az storage blob upload --auth-mode login \
    --account-name csnewsandfeeds4150361735 --container-name amulet \
    --name "$promotion_name" \
    --file "$state/selection.json" --overwrite \
    > "$state/promotion_upload.txt" 2>&1 \
    || fail "bridge promotion-gate upload failed"
  touch "$state/promotion_uploaded"
fi

status_phase(){
  STATUS="$1" "$python_bin" - <<'PY'
import json,os
try: rows=json.loads(os.environ['STATUS'])
except Exception: print('unknown'); raise SystemExit
print(str(rows[0].get('status') or '').casefold() if len(rows)==1 else 'invalid')
PY
}

submit_fit(){
  local experiment="$1"
  local audit available auditable
  exec 8>"$project/results/feeds_submission_global.lock"; flock -x 8
  audit="$fit/$experiment.active_a100.json"
  "$python_bin" scripts/count_active_a100_requests.py \
    --user-limit 32 --recent 500 --output "$audit" \
    > /dev/null 2>> "$fit/controller.log" \
    || { flock -u 8; return 1; }
  read -r available auditable < <("$python_bin" - "$audit" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
print(int(x['available_user_slots']), int(bool(x['auditable'])))
PY
  )
  [[ "$auditable" == 1 && "$available" -ge "$fit_gpu_count" ]] \
    || { flock -u 8; return 1; }
  amlt run "$fit_config" ":$fit_job" --dump -t Feeds \
    > "$fit/$experiment.preflight.yaml" 2> "$fit/$experiment.preflight.stderr" \
    || { flock -u 8; return 1; }
  "$python_bin" "$fit_validator" \
    --config "$fit/$experiment.preflight.yaml" >> "$fit/controller.log" 2>&1 \
    || { flock -u 8; return 1; }
  amlt --json-tables ti sing -t Feeds > "$fit/$experiment.quota.json" \
    2>> "$fit/controller.log" || { flock -u 8; return 1; }
  amlt run "$fit_config" ":$fit_job" "$experiment" -t Feeds -y \
    --description "Selected strong1-State8 bridge=$slug variant=$variant; exact 259845-row strict-state-only epoch" \
    > "$fit/$experiment.submission.txt" 2>&1
  rc=$?; flock -u 8; return "$rc"
}

experiment="state8-strong1-widen8-complete-finalfit-${fit_topology}-20260808"
wave=0
if [[ -s "$fit/active_experiment.txt" ]]; then
  experiment="$(<"$fit/active_experiment.txt")"
  [[ "$experiment" =~ recovery([0-9]+)-${fit_topology}-20260808 ]] \
    && wave="${BASH_REMATCH[1]}"
fi
while [[ ! -f "$fit/remote_passed" ]]; do
  if [[ ! -f "$fit/submitted.$experiment" ]]; then
    submit_fit "$experiment" || { sleep_unlocked 120; continue; }
    printf '%s\n' "$experiment" > "$fit/active_experiment.txt"
    touch "$fit/submitted.$experiment"
    log "submitted exact full fit experiment=$experiment job=$fit_job"
  fi
  heartbeat
  status="$(timeout 90 amlt --json-tables status "$experiment" --hide-urls 2>/dev/null)" \
    || { sleep_unlocked 120; continue; }
  printf '%s\n' "$status" > "$fit/latest_status.json"
  phase="$(status_phase "$status")"
  case "$phase" in
    pass) touch "$fit/remote_passed" ;;
    preparing|pre-queued|queued|running|paused|resuming) sleep_unlocked 120 ;;
    *)
      timeout 90 amlt show "$experiment" > "$fit/$experiment.show.txt" 2>&1 || true
      timeout 180 amlt logs view -n 3000 "$experiment" ":$fit_job" \
        > "$fit/$experiment.failure.log" 2>&1 || true
      (( wave < max_recovery_waves )) || fail "full fit exhausted recovery waves"
      wave=$((wave+1))
      experiment="state8-strong1-widen8-complete-finalfit-recovery${wave}-${fit_topology}-20260808"
      printf '%s\n' "$experiment" > "$fit/active_experiment.txt"
      ;;
  esac
done

if [[ ! -f "$fit/complete" ]]; then
  rm -rf "$fit/download"; mkdir -p "$fit/download"
  amlt results download "$experiment" --status pass \
    --include '*.json' --include '*.safetensors' --include '*.pt' \
    --output "$fit/download" >> "$fit/controller.log" 2>&1 \
    || fail "full-fit result download failed"
  mapfile -t receipts < <(find "$fit/download" -type f -name state8_final_fit_receipt.json)
  mapfile -t published < <(find "$fit/download" -type f -name canonical_publish_receipt.json)
  [[ ${#receipts[@]} -eq 1 && ${#published[@]} -eq 1 ]] \
    || fail "full-fit receipt cardinality drift"
  "$python_bin" - "${receipts[0]}" "${published[0]}" "$variant" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); p=json.load(open(sys.argv[2])); variant=sys.argv[3]
assert x['status']=='final_fit_complete' and x['one_epoch_exact'] is True,x
assert x['real_examples']==259845 and x['selected_replay_variant']==variant,x
assert x['state_shape']==[64,8,5120],x
assert x['history_training_mode']=='strict_state_only',x
assert x['history_dropout_counts']=={'1.0':259845},x
assert x['raw_history_visible_to_decoder'] is False,x
assert x['canonical_label_auxiliary_supervision'] is True,x
assert p['status']=='complete',p
assert p['artifact_sha256']=={
 'adapter_model':x['adapter_sha256'],'recurrent_state':x['recurrent_state_sha256']},(p,x)
PY
  touch "$fit/scientific_gate_pass" "$fit/complete"
fi

"$python_bin" scripts/materialize_state8_strong1_final_eval_config.py \
  --source amlt/feeds_state8_final_eval_20260805.yaml --output "$eval_config" \
  >> "$state/controller.log" 2>&1 || fail "final-eval config materialization failed"
"$python_bin" scripts/validate_state8_final_eval_amlt.py --config "$eval_config" \
  >> "$state/controller.log" 2>&1 || fail "strong1 final-eval preflight failed"

while [[ ! -f "$eval_state/complete" ]]; do
  FIT_DIR="$fit" STATE_DIR="$eval_state" FINAL_EVAL_CONFIG="$eval_config" \
  BASE_EXPERIMENT=state8-strong1-widen8-complete-final-eval-20260808 \
  RECOVERY_EXPERIMENT_PREFIX=state8-strong1-widen8-complete-final-eval-recovery \
  RECOVERY_EXPERIMENT_SUFFIX=-20260808 \
  bash scripts/watch_state8_final_eval.sh
  rc=$?
  [[ -f "$eval_state/complete" ]] && break
  [[ ! -f "$eval_state/scientific_gate_failed" ]] \
    || fail "strong1 final evaluation reached scientific failure rc=$rc"
  log "recoverable final-evaluation controller exit rc=$rc; retrying"
  heartbeat; sleep_unlocked 120
done
[[ -f "$eval_state/complete" ]] || fail "strong1 complete final evaluation incomplete"
if [[ -f "$eval_state/scientific_gate_pass" ]]; then touch "$state/scientific_gate_pass"; fi
if [[ -f "$eval_state/scientific_gate_failed" ]]; then touch "$state/scientific_gate_failed"; fi
touch "$state/complete"; heartbeat
log "strong1-to-State8 complete unified pipeline reached terminal evaluated state"
