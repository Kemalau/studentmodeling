#!/usr/bin/env bash
set -uo pipefail

project_dir=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
candidate_dir="$project_dir/results/state8_semantic_candidates_20260806"
state_dir="$project_dir/results/state8_unified_final_fit_20260805"
config="$project_dir/amlt/feeds_state8_unified_final_fit_20260805.yaml"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
max_recovery_waves="${MAX_RECOVERY_WAVES:-32}"
export AZURE_CONFIG_DIR="${AZURE_CONFIG_DIR:-/home/v-yuejiali/.azure-keyflip-feeds}"
stall_seconds="${STALL_SECONDS:-3600}"
progress_dir="$state_dir/progress"

mkdir -p "$state_dir" "$progress_dir"
cd "$project_dir" || exit 1
exec 9>"$state_dir/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT
rm -f "$state_dir/needs_attention"

heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state_dir/needs_attention"; exit 1; }

observe_progress(){
  local experiment="$1" job="$2" status_value="$3"
  local listing log_file=""
  if [[ "$status_value" != running ]]; then
    "$python_bin" scripts/monitor_layerwise_unified_training_progress.py \
      --awaiting-first-record --experiment "$experiment" --log-file "" \
      --status "$status_value" --output "$progress_dir/progress.json" \
      --total-rounds 16241 >/dev/null 2>&1 || true
    rm -f "$progress_dir/stalled.json" "$progress_dir/health_alert.json"
    return 0
  fi
  listing="$(timeout 90 amlt logs list "$experiment" ":$job" 2>&1)" || listing=""
  log_file="$("$python_bin" - "$listing" <<'PY'
import re,sys
paths=[]
for line in sys.argv[1].splitlines():
    value=line.strip()
    if re.fullmatch(r'user_logs/(?:retry_\d+/)?std_log_process_(?:0|00)\.txt',value):
        retry=re.search(r'retry_(\d+)',value)
        paths.append((int(retry.group(1)) if retry else -1,value))
print(max(paths)[1] if paths else '')
PY
)"
  if [[ -n "$log_file" ]] && timeout 90 amlt logs view -F "$log_file" -n 800 \
      "$experiment" ":$job" > "$progress_dir/latest_rank0.log" \
      2> "$progress_dir/latest_rank0.stderr"; then
    "$python_bin" scripts/monitor_layerwise_unified_training_progress.py \
      --log "$progress_dir/latest_rank0.log" --experiment "$experiment" \
      --log-file "$log_file" --status running \
      --output "$progress_dir/progress.json" --total-rounds 16241 \
      --max-run-seconds 345600 --validation-reserve-seconds 21600 \
      >/dev/null 2>&1 || true
  else
    "$python_bin" scripts/monitor_layerwise_unified_training_progress.py \
      --awaiting-first-record --experiment "$experiment" --log-file "$log_file" \
      --status running --output "$progress_dir/progress.json" \
      --total-rounds 16241 >/dev/null 2>&1 || true
  fi
  if "$python_bin" - "$progress_dir/progress.json" "$stall_seconds" <<'PY'
import json,sys
try: value=json.load(open(sys.argv[1]))
except Exception: raise SystemExit(1)
raise SystemExit(0 if float(value.get('seconds_since_progress') or 0)>=float(sys.argv[2]) else 1)
PY
  then cp "$progress_dir/progress.json" "$progress_dir/stalled.json"
  else rm -f "$progress_dir/stalled.json"; fi
  if "$python_bin" - "$progress_dir/progress.json" <<'PY'
import json,sys
try: value=json.load(open(sys.argv[1]))
except Exception: raise SystemExit(1)
raise SystemExit(0 if (value.get('health') or {}).get('healthy') is False else 1)
PY
  then cp "$progress_dir/progress.json" "$progress_dir/health_alert.json"
  else rm -f "$progress_dir/health_alert.json"; fi
}

status_name(){
  "$python_bin" - "$1" <<'PY'
import re,sys
text=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',sys.argv[1])
values=re.findall(r'\b(preparing|queued|running|paused|resuming|pass|failed|killed|cancelled)\b',text,re.I)
print(values[-1].lower() if values else 'unknown')
PY
}

candidate_slugs=(c4_g005 c4_g010 c4_g025 eedi_g005 eedi_g010 eedi_g025 eedi_g050)
while true; do
  heartbeat
  ready=true
  for slug in "${candidate_slugs[@]}"; do
    root="$candidate_dir/$slug"
    [[ ! -f "$root/needs_attention" ]] || fail "semantic candidate needs attention: $slug"
    [[ -f "$root/complete" ]] || ready=false
  done
  [[ "$ready" == true ]] && break
  sleep 120
done

selection_path="$state_dir/semantic_candidate_selection.json"
selection_args=()
for slug in "${candidate_slugs[@]}"; do
  root="$candidate_dir/$slug"
  selection_args+=(--candidate "$slug" "$root/t3_semantic/metrics.json" "$root/c4_semantic/summary.json")
done
"$python_bin" scripts/select_state8_semantic_candidates.py \
  "${selection_args[@]}" --output "$selection_path" \
  > "$state_dir/semantic_candidate_selection.log" 2>&1
selection_rc=$?
if (( selection_rc != 0 )); then
  touch "$state_dir/scientific_gate_failed"
  fail "no semantic T3+C4 candidate passed; bounded rescue required"
fi

readarray -t selection < <("$python_bin" - "$selection_path" "$candidate_dir" <<'PY'
import json,pathlib,sys
x=json.load(open(sys.argv[1])); selected=x.get('selected')
assert x['stage']=='state8_semantic_candidate_selection_v1' and x['pass'] is True,x
assert selected and selected['qualified'] is True,selected
slug=selected['slug']; root=pathlib.Path(sys.argv[2])/slug
p=json.load(open(root/'publish_receipt.json'))
assert p['stage']=='state8_stateonly_repair_pilot_publish_v1' and p['status']=='complete',p
assert p['repair_arm']=='trajectory01' and p['state_shape']==[64,8,5120],p
assert p['raw_history_visible_to_decoder'] is False and p['history_dropout']==1.0,p
assert p['online_teacher_kd'] is True,p
assert p['train_entrypoint']=='scripts/train_state8_semantic_causal_repair.py',p
a=p['artifact_sha256']; assert set(a)=={'adapter_model','recurrent_state'},a
print(selected['finalfit_variant']); print(selected['finalfit_job'])
print(a['adapter_model']); print(a['recurrent_state']); print(slug)
PY
)
[[ ${#selection[@]} -eq 5 ]] || fail "semantic promotion selection receipt is invalid"
selected_variant="${selection[0]}"
job="${selection[1]}"
selected_adapter_sha="${selection[2]}"
selected_state_sha="${selection[3]}"
selected_slug="${selection[4]}"
base_experiment="${job}-20260806"
recovery_prefix="${job}-recovery"

wait_quota(){
  local quota
  while true; do
    heartbeat
    quota="$(timeout 90 amlt --json-tables ti sing -t Feeds 2>&1)" || { sleep 30; continue; }
    if "$python_bin" - "$quota" "$state_dir/quota.json" <<'PY'
import json,sys
from scripts.feeds_quota_utils import parse_json_table,select_a100_80gb
row=select_a100_80gb(parse_json_table(sys.argv[1])); q=row['quotas_(used/limit)']
standard_limit=int(q['standard']['limit']); available=standard_limit-int(q['standard']['used'])
user_limit=int(q['user_max']['limit']); user_available=user_limit-int(q['user_max'].get('used') or 0)
open(sys.argv[2],'w').write(json.dumps({'selected_row':row,'required_units':16,
 'standard_available':available,'user_available':user_available,
 'scheduler_queue_allowed':True},indent=2,sort_keys=True)+'\n')
assert standard_limit>=16 and user_limit>=16
PY
    then return 0; fi
    sleep 30
  done
}

submit_wave(){
  local wave="$1" experiment output current class rc
  if (( wave == 0 )); then experiment="$base_experiment"
  else experiment="${recovery_prefix}${wave}-seed42-20260805"; fi
  exec 8>"$project_dir/results/feeds_submission_global.lock"; flock 8
  amlt run "$config" --dump -t Feeds > "$state_dir/${experiment}.preflight.dump.yaml" \
    2> "$state_dir/${experiment}.preflight.stderr" || { flock -u 8; return 1; }
  "$python_bin" scripts/validate_state8_unified_final_fit_amlt.py \
    --config "$state_dir/${experiment}.preflight.dump.yaml" \
    >> "$state_dir/controller.log" 2>&1 || { flock -u 8; return 1; }
  current="$(amlt status --no-update-db "$experiment" 2>&1)"; rc=$?
  class="$("$python_bin" - "$rc" "$current" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]),sys.argv[2]))
PY
  )" || { flock -u 8; return 1; }
  if [[ "$class" == missing ]]; then
    wait_quota
    output="$(amlt run "$config" ":$job" "$experiment" -t Feeds -y \
      --description "Semantic T3+C4 selected repair=$selected_variant exact 259845-row unified epoch; decoder always state-only; qualified full-history teacher KD always on; label-weighted natural diagnosis; multiseed C4; 16 A100; atomic resume wave=$wave" 2>&1)"; rc=$?
    printf '[%s] submit rc=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$output" \
      > "$state_dir/${experiment}.submission.txt"
    (( rc == 0 )) || { flock -u 8; return 1; }
  elif [[ "$class" == exists ]]; then
    printf 'status=preexisting_not_resubmitted\n%s\n' "$current" \
      > "$state_dir/${experiment}.submission.txt"
  else flock -u 8; return 1; fi
  flock -u 8
  printf '%s\n' "$experiment" > "$state_dir/active_experiment.txt"
  log "monitoring state8 full fit experiment=$experiment"
}

wave=0
if [[ -s "$state_dir/active_experiment.txt" ]]; then
  experiment="$(<"$state_dir/active_experiment.txt")"
  [[ "$experiment" =~ recovery([0-9]+)-seed42 ]] && wave="${BASH_REMATCH[1]}"
else
  submit_wave 0 || fail "initial state8 full-fit submission failed"
  experiment="$(<"$state_dir/active_experiment.txt")"
fi

while true; do
  heartbeat
  status="$(timeout 90 amlt status "$experiment" 2>&1)" || { sleep 120; continue; }
  printf '[%s]\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" >> "$state_dir/status.log"
  observe_progress "$experiment" "$job" "$(status_name "$status")"
  if "$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import has_forward_progress_job
raise SystemExit(0 if has_forward_progress_job(sys.argv[1]) else 1)
PY
  then
    sleep 120; continue
  fi
  if "$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1],1) else 1)
PY
  then break; fi
  mapfile -t paused < <("$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import paused_job_names
values=paused_job_names(sys.argv[1])
sys.stdout.write(''.join(value+'\n' for value in values))
PY
  )
  if (( ${#paused[@]} > 0 )); then
    log "requesting resume for paused state8 full-fit job"
    timeout 90 amlt resume "$experiment" ":$job" \
      >> "$state_dir/controller.log" 2>&1 || true
    sleep 120; continue
  fi
  printf '%s\n' "$status" > "$state_dir/${experiment}.terminal_status.txt"
  timeout 90 amlt show "$experiment" > "$state_dir/${experiment}.show.txt" 2>&1 || true
  timeout 180 amlt logs view -n 3000 "$experiment" ":$job" \
    > "$state_dir/${experiment}.failure.log" 2>&1 || true
  (( wave < max_recovery_waves )) || fail "state8 full fit exhausted recovery waves"
  wave=$((wave+1))
  submit_wave "$wave" || fail "state8 full-fit recovery submission failed wave=$wave"
  experiment="$(<"$state_dir/active_experiment.txt")"
done

rm -rf "$state_dir/download"; mkdir -p "$state_dir/download"
amlt results download "$experiment" --status pass \
  --include '*.json' --include '*.safetensors' --include '*.pt' \
  --output "$state_dir/download" >> "$state_dir/controller.log" 2>&1 \
  || fail "state8 full-fit result download failed"
readarray -t receipts < <(find "$state_dir/download" -name state8_final_fit_receipt.json -type f)
readarray -t publishes < <(find "$state_dir/download" -name canonical_publish_receipt.json -type f)
[[ ${#receipts[@]} -eq 1 && ${#publishes[@]} -eq 1 ]] \
  || fail "expected one final-fit receipt and one canonical publish receipt"
"$python_bin" - "${receipts[0]}" "${publishes[0]}" "$selected_variant" "$selected_adapter_sha" "$selected_state_sha" <<'PY' \
  || fail "state8 full-fit receipt validation failed"
import json,sys
x=json.load(open(sys.argv[1])); p=json.load(open(sys.argv[2])); variant=sys.argv[3]
selected={'adapter_model':sys.argv[4],'recurrent_state':sys.argv[5]}
assert x['status']=='final_fit_complete',x
assert x['stage']=='state8_qualified_unified_final_fit_v1' and x['one_epoch_exact'] is True,x
assert x['selected_replay_variant']==variant and x['initial_artifact_sha256']==selected,(x,selected)
assert x['real_examples']==259845 and x['state_shape']==[64,8,5120],x
assert len(x['teacher_adapter_sha256'])==64 and x['teacher_gate_required'] is True,x
assert x['teacher_prompt_provenance_stage']=='state8_clean_prompt_teacher_provenance_gate_v2',x
assert x['teacher_evaluation_input_sha256']=='78850c5434f24c3072b7ab89370400347955686e9aac5b99aadb69ae19c7cfa5',x
assert x['teacher_clean_prompt_sha256']=='c35fd46e0cfbccd9141eccb524992bd5452416d32720686f74ad9373f57de314',x
assert x['legacy_answer_adjacent_contracts']==0,x
assert x['raw_history_visible_to_decoder'] is False,x
assert x['persistent_history_kv_cache'] is False,x
assert x['history_training_mode']=='strict_state_only',x
assert x['history_dropout_counts']=={'1.0':259845},x
assert float(x['online_kd_weight'])==1.0,x
assert float(x['label_auxiliary_weight'])==4.0,x
assert float(x['trajectory_weight']) in {0.0,0.1,1.0},x
assert len(x['adapter_sha256'])==64 and len(x['recurrent_state_sha256'])==64,x
assert p['stage']=='state8_canonical_final_checkpoint_publish_v1' and p['status']=='complete',p
assert p['selected_replay_variant']==variant,(p,variant)
assert p['artifact_sha256']=={'adapter_model':x['adapter_sha256'],
                              'recurrent_state':x['recurrent_state_sha256']},(p,x)
PY
touch "$state_dir/scientific_gate_pass" "$state_dir/complete"
heartbeat
log "state8 exact unified final fit completed and artifact receipt verified"
