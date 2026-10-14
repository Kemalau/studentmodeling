#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
source "$project/scripts/use_studentmodeling_feeds_auth.sh"
promotion="${STATE8_FINALFIT_PROMOTION_DIR:-$project/results/state8_eedi_same_question_promotion_20260811}"
state="${STATE8_FINALFIT_STATE_DIR:-$project/results/state8_eedi_same_question_finalfit_world2_20260811}"
template="${STATE8_FINALFIT_TEMPLATE:-$project/amlt/feeds_state8_eedi_same_question_final_fit_world2_20260811.yaml}"
config="$state/bound.yaml"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
base="${STATE8_FINALFIT_BASE_EXPERIMENT:-state8-eedi-same-question-unified-finalfit-world2-20260811}"
experiment="$base"; wave=0; max_waves=12

mkdir -p "$state"; cd "$project" || exit 1
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"; trap 'rm -f "$state/controller.active.pid"' EXIT
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
sleep_unlocked(){ sleep "$1" 9>&-; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }

while [[ ! -f "$promotion/complete" ]]; do heartbeat; sleep_unlocked 60; done
if [[ ! -f "$promotion/scientific_gate_pass" ]]; then
  touch "$state/promotion_failed" "$state/complete"; heartbeat
  log "joint promotion failed; full exact epoch was not submitted"
  exit 0
fi
promotion_json="$promotion/promotion.json"
[[ -f "$promotion_json" ]] || fail "promotion-pass marker has no promotion receipt"

"$python_bin" scripts/materialize_state8_eedi_same_question_finalfit_world2_amlt.py \
  --template "$template" --promotion "$promotion_json" --output "$config" \
  >> "$state/controller.log" 2>&1 || fail "final-fit config materialization failed"
"$python_bin" scripts/validate_state8_eedi_same_question_finalfit_world2_amlt.py \
  --config "$config" >> "$state/controller.log" 2>&1 \
  || fail "final-fit config validation failed"
job="$("$python_bin" - "$config" <<'PY'
import sys,yaml
x=yaml.safe_load(open(sys.argv[1])); assert len(x['jobs'])==1
print(x['jobs'][0]['name'])
PY
)" || fail "final-fit job extraction failed"

submit(){
  local name="$1" current class output rc
  exec 8>"$project/results/feeds_submission_global.lock"; flock -x 8
  amlt run "$config" --dump -t Feeds > "$state/$name.preflight.yaml" \
    2> "$state/$name.preflight.stderr" || { flock -u 8; return 1; }
  "$python_bin" scripts/validate_state8_eedi_same_question_finalfit_world2_amlt.py \
    --config "$state/$name.preflight.yaml" >> "$state/controller.log" 2>&1 \
    || { flock -u 8; return 1; }
  current="$(amlt status --no-update-db "$name" 2>&1)"; rc=$?
  class="$("$python_bin" - "$rc" "$current" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]),sys.argv[2]))
PY
)" || { flock -u 8; return 1; }
  if [[ "$class" == missing ]]; then
    audit="$state/$name.active_a100.json"
    "$python_bin" scripts/count_active_a100_requests.py \
      --user-limit 32 --recent 500 --output "$audit" \
      > /dev/null 2>> "$state/controller.log" \
      || { flock -u 8; return 1; }
    "$python_bin" - "$audit" <<'PY' \
      || { flock -u 8; return 1; }
import json,sys
x=json.load(open(sys.argv[1]))
assert x['auditable'] is True,x
assert int(x['available_user_slots'])>=2,x
PY
    amlt --json-tables ti sing -t Feeds > "$state/$name.quota.json" \
      2>> "$state/controller.log" || { flock -u 8; return 1; }
    "$python_bin" - "$state/$name.quota.json" <<'PY' \
      || { flock -u 8; return 1; }
import sys
from scripts.feeds_quota_utils import parse_json_table,select_a100_80gb
row=select_a100_80gb(parse_json_table(open(sys.argv[1]).read()))
assert row['series']=='NDAMv4',row
assert row['accelerator']['type']=='A100' and row['accelerator']['memory_gb']==80,row
quota=row['quotas_(used/limit)']['standard']
assert quota['limit']-quota['used']>=2,quota
user=row['quotas_(used/limit)']['user_max']
assert int(user['limit'])>=2,user
PY
    output="$(amlt run "$config" "$name" -t Feeds -y \
      --description "Jointly promoted State8 exact 259845-row one epoch; world2 global batch16; strict state-only; same-question causal replay" 2>&1)"; rc=$?
    printf '[%s] rc=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$output" \
      > "$state/$name.submission.txt"
    (( rc == 0 )) || { flock -u 8; return 1; }
  elif [[ "$class" != exists ]]; then flock -u 8; return 1; fi
  flock -u 8
  printf '%s\n' "$name" > "$state/active_experiment.txt"
}

if [[ -s "$state/active_experiment.txt" ]]; then experiment="$(<"$state/active_experiment.txt")"
else until submit "$experiment"; do heartbeat; sleep_unlocked 120; done; fi
touch "$state/submitted"; log "monitoring full exact epoch experiment=$experiment"

while [[ ! -f "$state/remote_passed" ]]; do
  heartbeat
  status="$(timeout 90 amlt --json-tables status "$experiment" --hide-urls \
    --force-status-update 2>/dev/null)" || { sleep_unlocked 60; continue; }
  [[ -n "$status" && "$status" != '[]' ]] || { sleep_unlocked 30; continue; }
  printf '%s\n' "$status" > "$state/latest_status.json"
  phase="$(STATUS="$status" "$python_bin" - <<'PY'
import json,os
s={str(x.get('status') or '').casefold() for x in json.loads(os.environ['STATUS'])}
if s=={'pass'}: print('pass')
elif s&{'preparing','pre-queued','queued','running','resuming'}: print('active')
elif 'paused' in s: print('paused')
elif not s or s<={'','unknown'}: print('unknown')
else: print('terminal')
PY
)"
  case "$phase" in
    pass) touch "$state/remote_passed" ;;
    active|unknown) sleep_unlocked 120 ;;
    paused) timeout 90 amlt resume "$experiment" ":$job" \
      >> "$state/controller.log" 2>&1 || true; sleep_unlocked 120 ;;
    *)
      timeout 180 amlt logs view -n 3000 "$experiment" ":$job" \
        > "$state/$experiment.failure.log" 2>&1 || true
      (( wave < max_waves )) || fail "full exact epoch recovery exhausted"
      wave=$((wave+1)); experiment="${base}-recovery${wave}"
      until submit "$experiment"; do heartbeat; sleep_unlocked 120; done
      log "submitted exact-resume recovery wave=$wave" ;;
  esac
done

rm -rf "$state/download"; mkdir -p "$state/download"
amlt results download "$experiment" ":$job" --include '*.json' \
  --output "$state/download" >> "$state/controller.log" 2>&1 \
  || fail "final-fit receipt download failed"
mapfile -t fits < <(find "$state/download" -name state8_final_fit_receipt.json -type f)
mapfile -t publishes < <(find "$state/download" -name canonical_publish_receipt.json -type f)
[[ ${#fits[@]} -eq 1 && ${#publishes[@]} -eq 1 ]] \
  || fail "final-fit receipt cardinality drift"
"$python_bin" - "${fits[0]}" "${publishes[0]}" "$promotion_json" <<'PY' \
  || fail "final-fit receipt validation failed"
import json,sys
fit,publish,promotion=(json.load(open(path)) for path in sys.argv[1:4])
selected=promotion['selected']; variant=selected['finalfit_variant']
assert fit['status']=='final_fit_complete' and fit['one_epoch_exact'] is True,fit
assert fit['real_examples']==259845 and fit['selected_replay_variant']==variant,fit
assert fit['state_shape']==[64,8,5120],fit
assert fit['raw_history_visible_to_decoder'] is False,fit
assert fit['persistent_history_kv_cache'] is False,fit
assert fit['history_training_mode']=='strict_state_only',fit
assert fit['history_dropout_counts']=={'1.0':259845},fit
assert fit['teacher_gate_required'] is True and float(fit['online_kd_weight'])==1.0,fit
assert fit['control_robust_objective'] is True,fit
assert int(fit['control_random_samples'])==4,fit
assert float(fit['control_random_weight'])==0.25,fit
assert float(fit['control_same_rule_weight'])==0.50,fit
branch=str(selected.get('repair_branch') or '')
expected_teacher=(0.15 if branch in {'causal_teacher_low','early_label_low','early_label_high'} else (0.50 if branch=='causal_teacher_high' else 0.0))
expected_early=(0.25 if branch=='early_label_low' else (0.50 if branch=='early_label_high' else 0.0))
assert float(fit['causal_teacher_trajectory_weight'])==expected_teacher,fit
assert float(fit['early_label_binding_weight'])==expected_early,fit
assert fit['binding_constraints_inherited_from_promotion'] is True,fit
assert publish['stage']=='state8_canonical_final_checkpoint_publish_v1',publish
assert publish['status']=='complete' and publish['selected_replay_variant']==variant,publish
assert publish['artifact_sha256']=={
 'adapter_model':fit['adapter_sha256'],'recurrent_state':fit['recurrent_state_sha256']},publish
PY
touch "$state/scientific_gate_pass" "$state/complete"; heartbeat
log "exact unified one-epoch State8 fit completed and canonical receipt verified"
