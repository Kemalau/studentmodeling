#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
screen="$project/results/state8_highcausal_binding_20260806"
state="$project/results/state8_highcausal_finalfit_20260806"
config=amlt/feeds_state8_highcausal_final_fit_20260806.yaml
validator=scripts/validate_state8_unified_final_fit_amlt.py
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
if [[ -z "${AZURE_CONFIG_DIR:-}" ]]; then
  if AZURE_CONFIG_DIR=/home/v-yuejiali/.azure az account get-access-token \
      --subscription b6dc87f3-c479-49c8-8cb5-7896da3ff895 \
      --query expiresOn -o tsv >/dev/null 2>&1; then
    export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure
  else
    export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure-keyflip-feeds
  fi
fi

mkdir -p "$state"
cd "$project" || exit 1
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"
trap 'rm -f "$state/controller.active.pid"' EXIT
rm -f "$state/needs_attention"
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }

while [[ ! -f "$screen/complete" ]]; do heartbeat; sleep 120; done
selection="$screen/selection.json"
[[ -f "$selection" ]] || fail "screen completed without selection"
selected="$($python_bin - "$selection" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['stage']=='state8_high_causal_candidate_selection_v1',x
if not x['pass'] or x['selected'] not in {'joint','updater','mid','eedi','taxonomy'}: raise SystemExit(2)
print(x['selected'])
PY
)" || fail "no high-causal arm passed the promotion gate"
case "$selected" in
  joint) job=state8-highcausal-joint-unified-finalfit-seed42 ;;
  updater) job=state8-highcausal-updater-unified-finalfit-seed42 ;;
  mid) job=state8-highcausal-mid-unified-finalfit-seed42 ;;
  eedi) job=state8-highcausal-eedi-unified-finalfit-seed42 ;;
  taxonomy) job=state8-highcausal-taxonomy-unified-finalfit-seed42 ;;
  *) fail "unknown selected arm=$selected" ;;
esac
printf '%s\n' "$selected" > "$state/selected_arm.txt"
printf '%s\n' "$job" > "$state/selected_job.txt"

submit_wave(){
  local wave="$1"
  local experiment
  if [[ "$wave" == 0 ]]; then
    experiment=state8-highcausal-promoted-finalfit-20260806
  else
    experiment="state8-highcausal-promoted-finalfit-recovery${wave}-20260806"
  fi
  exec 8>"$project/results/feeds_submission_global.lock"; flock 8
  amlt run "$config" ":$job" --dump -t Feeds \
    > "$state/${experiment}.preflight.yaml" \
    2> "$state/${experiment}.preflight.stderr" \
    || { flock -u 8; return 1; }
  "$python_bin" "$validator" --config "$state/${experiment}.preflight.yaml" \
    >> "$state/controller.log" 2>&1 || { flock -u 8; return 1; }
  amlt --json-tables ti sing -t Feeds > "$state/${experiment}.quota.json" \
    2>> "$state/controller.log" || { flock -u 8; return 1; }
  amlt run "$config" ":$job" "$experiment" -t Feeds -y \
    --description "Promoted exact one-epoch unified State8 final fit arm=$selected wave=$wave" \
    > "$state/${experiment}.submission.txt" 2>&1 \
    || { flock -u 8; return 1; }
  flock -u 8
  printf '%s\n' "$experiment" > "$state/active_experiment.txt"
  log "submitted final fit experiment=$experiment arm=$selected"
}

wave=0
if [[ -s "$state/active_experiment.txt" ]]; then
  experiment="$(<"$state/active_experiment.txt")"
  [[ "$experiment" =~ recovery([0-9]+)-20260806 ]] && wave="${BASH_REMATCH[1]}"
else
  submit_wave 0 || fail "initial final-fit submission failed"
  experiment="$(<"$state/active_experiment.txt")"
fi

while true; do
  heartbeat
  status="$(timeout 90 amlt status "$experiment" 2>&1)" || { sleep 120; continue; }
  printf '[%s]\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" >> "$state/status.log"
  if "$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1],1) else 1)
PY
  then break; fi
  if "$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import has_forward_progress_job
raise SystemExit(0 if has_forward_progress_job(sys.argv[1]) else 1)
PY
  then sleep 120; continue; fi
  printf '%s\n' "$status" > "$state/${experiment}.terminal_status.txt"
  timeout 90 amlt show "$experiment" > "$state/${experiment}.show.txt" 2>&1 || true
  timeout 180 amlt logs view -n 4000 "$experiment" ":$job" \
    > "$state/${experiment}.failure.log" 2>&1 || true
  (( wave < 32 )) || fail "promoted final fit exhausted recovery waves"
  wave=$((wave+1))
  submit_wave "$wave" || fail "final-fit recovery submission failed wave=$wave"
  experiment="$(<"$state/active_experiment.txt")"
done

rm -rf "$state/download"; mkdir -p "$state/download"
amlt results download "$experiment" --status pass --include '*.json' \
  --output "$state/download" >> "$state/controller.log" 2>&1 \
  || fail "final-fit result download failed"
mapfile -t receipts < <(find "$state/download" -type f -path '*/metrics/state8_final_fit_receipt.json')
mapfile -t canonical < <(find "$state/download" -type f -path '*/metrics/canonical_publish_receipt.json')
[[ ${#receipts[@]} -eq 1 && ${#canonical[@]} -eq 1 ]] \
  || fail "final-fit receipt cardinality drift"
"$python_bin" - "${receipts[0]}" "${canonical[0]}" "$selected" <<'PY'
import json,sys
f=json.load(open(sys.argv[1])); c=json.load(open(sys.argv[2])); selected=sys.argv[3]
variant={'joint':'highcausal_joint','updater':'highcausal_updater','mid':'highcausal_mid','eedi':'highcausal_eedi','taxonomy':'highcausal_taxonomy'}[selected]
assert f['stage']=='state8_qualified_unified_final_fit_v1',f
assert f['status']=='final_fit_complete' and f['one_epoch_exact'] is True,f
assert f['selected_replay_variant']==variant,f
assert f['real_examples']==259845 and f['raw_history_visible_to_decoder'] is False,f
assert c['stage']=='state8_canonical_final_checkpoint_publish_v1' and c['status']=='complete',c
assert c['selected_replay_variant']==variant,c
PY
cp "${receipts[0]}" "$state/state8_final_fit_receipt.json"
cp "${canonical[0]}" "$state/canonical_publish_receipt.json"
touch "$state/scientific_gate_pass" "$state/complete"; heartbeat
log "promoted exact unified final fit completed and receipts verified"
