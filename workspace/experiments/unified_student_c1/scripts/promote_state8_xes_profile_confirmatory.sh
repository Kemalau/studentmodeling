#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
auth=${XES_PROFILE_CONFIRMATORY_AUTH:-/home/v-yuejiali/.azure-keyflip-e123-recovery-20260819}
formal=${XES_PROFILE_FORMAL_STATE:-$project/results/state8_profile_ownership_lrfix_v4_formal_20260820}
state=$project/results/state8_xes_profile_confirmatory_promotion_20260821
eval_state=$project/results/state8_xes_profile_confirmatory_lrfix_v4_20260821
config=$project/amlt/feeds_state8_xes_profile_confirmatory_20260820.yaml
experiment=state8-xes-profile-confirmatory-lrfix-v4-20260821
python=/home/v-yuejiali/.venvs/studentmodeling/bin/python

mkdir -p "$state" "$eval_state"
cd "$project" || exit 1
exec 9>"$state/controller.lock"
flock -n 9 || exit 0
echo $$ > "$state/controller.pid"
trap 'rm -f "$state/controller.pid"' EXIT
rm -f "$state/promoted" "$state/suppressed" "$state/needs_attention"

while [[ ! -f "$formal/complete" && ! -f "$formal/needs_attention" ]]; do
  date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"
  sleep 60
done
if [[ ! -f "$formal/complete" ]]; then
  printf 'Development formal run did not complete its audited artifact contract.\n' > "$state/suppressed"
  exit 2
fi

FORMAL="$formal/terminal_summary.json" "$python" - <<'PY' > "$state/development_gate.json"
import json,os
x=json.load(open(os.environ['FORMAL']))
gates=x.get('reusable_interface_gates') or {}
checks={
 'all_passed':x.get('all_passed') is True,
 'independent_audit':x.get('independent_audit_exit_code')==0,
 'scientific_gate_pass':x.get('reusable_interface_scientific_gate_pass') is True,
 'all_preregistered_gates':bool(gates) and all(gates.values()),
}
result={'status':'pass' if all(checks.values()) else 'fail','checks':checks,'gates':gates}
print(json.dumps(result,indent=2,sort_keys=True))
raise SystemExit(0 if result['status']=='pass' else 2)
PY
gate_rc=$?
if (( gate_rc != 0 )); then
  printf 'Frozen DBE development gate failed; XES test remains untouched.\n' > "$state/suppressed"
  exit "$gate_rc"
fi

export PYTHONPATH="$project/scripts:$project"
"$python" scripts/verify_xes_profile_confirmatory_freeze.py > "$state/freeze_verification.json" || {
  touch "$state/needs_attention"; exit 3;
}
"$python" scripts/validate_state8_xes_profile_confirmatory_amlt.py --config "$config" \
  > "$state/config_validation.txt" 2>&1 || { touch "$state/needs_attention"; exit 4; }

export AZURE_CONFIG_DIR="$auth"
amlt --json-tables ti sing -t Feeds -a > "$state/quota.json" || {
  touch "$state/needs_attention"; exit 5;
}
amlt run "$config" --dump -t Feeds > "$state/preflight.dump.yaml" \
  2> "$state/preflight.stderr" || { touch "$state/needs_attention"; exit 6; }
"$python" scripts/validate_state8_xes_profile_confirmatory_amlt.py \
  --config "$state/preflight.dump.yaml" > "$state/preflight_validation.txt" 2>&1 || {
  touch "$state/needs_attention"; exit 7;
}

existing="$(timeout 120 amlt --json-tables status "$experiment" --hide-urls \
  --force-status-update --no-calculate-result-size 2>/dev/null || true)"
if [[ -n "$existing" && "$existing" != '[]' ]]; then
  printf '%s\n' "$existing" > "$state/status.existing.json"
else
  exec 8>"$project/results/feeds_submission_global.lock"
  flock -w 300 8 || { touch "$state/needs_attention"; exit 8; }
  amlt run "$config" "$experiment" -t Feeds -y \
    --description 'One-shot frozen XES confirmation after all DBE development gates; strict State8-only readout; no test-driven selection or retry' \
    > "$state/submission.txt" 2>&1 || { touch "$state/needs_attention"; exit 9; }
  flock -u 8
fi
timeout 120 amlt --json-tables status "$experiment" --hide-urls \
  --force-status-update --no-calculate-result-size > "$state/status.initial.json" \
  2> "$state/status.initial.stderr" || { touch "$state/needs_attention"; exit 10; }

setsid env XES_PROFILE_CONFIRMATORY_EXPERIMENT="$experiment" \
  XES_PROFILE_CONFIRMATORY_STATE="$eval_state" XES_PROFILE_CONFIRMATORY_AUTH="$auth" \
  bash scripts/watch_state8_xes_profile_confirmatory.sh \
  > "$eval_state/watcher.nohup.log" 2>&1 < /dev/null &
echo $! > "$eval_state/watcher.launch.pid"
touch "$state/promoted"
