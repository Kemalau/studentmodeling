#!/usr/bin/env bash
set -uo pipefail
shopt -s nullglob globstar

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
state=${PROFILE_READOUT_STATE:-"$project/results/state8_future_profile_v2_20260818"}
mkdir -p "$state"
cd "$project" || exit 1
exec 9>"$state/independent_audit_watcher.lock"
flock -n 9 || exit 0
echo $$ > "$state/independent_audit_watcher.pid"
trap 'rm -f "$state/independent_audit_watcher.pid"' EXIT

while [[ ! -f "$state/terminal_summary.json" ]]; do
  date -u +%Y-%m-%dT%H:%M:%SZ > "$state/independent_audit_watcher.heartbeat"
  sleep 60
done

rm -f "$state/independent_audit_complete" "$state/independent_audit_needs_attention"

# AMLT downloads may insert both experiment and job directories. Search the
# downloaded tree recursively while retaining the fail-closed cardinality gate.
profile_metrics=("$state"/download/**/profile_readout/metrics.json)
profiles=("$state"/download/**/profile_readout/readable_profiles.jsonl)
transport=("$state"/download/**/profile_readout/fixed_query_transport.jsonl)
interface_metrics=("$state"/download/**/reusable_interface/metrics.json)
interface_features=("$state"/download/**/reusable_interface_features/test.npz)
retention=("$state"/download/**/retention_eval/metrics.json)

if [[ ${#profile_metrics[@]} -ne 1 || ! -f "${profile_metrics[0]}" || \
      ${#profiles[@]} -ne 1 || ! -f "${profiles[0]}" || \
      ${#transport[@]} -ne 1 || ! -f "${transport[0]}" || \
      ${#interface_metrics[@]} -ne 1 || ! -f "${interface_metrics[0]}" || \
      ${#interface_features[@]} -ne 1 || ! -f "${interface_features[0]}" || \
      ${#retention[@]} -ne 1 || ! -f "${retention[0]}" ]]; then
  printf '%s\n' 'required downloaded artifact is missing or duplicated' > "$state/independent_audit.log"
  touch "$state/independent_audit_needs_attention"
  exit 1
fi

feature_dir=$(dirname "${interface_features[0]}")
if /home/v-yuejiali/.venvs/studentmodeling/bin/python \
  scripts/audit_state8_future_profile_v2.py \
  --profile-metrics "${profile_metrics[0]}" \
  --profiles-jsonl "${profiles[0]}" \
  --transport-jsonl "${transport[0]}" \
  --interface-metrics "${interface_metrics[0]}" \
  --interface-feature-dir "$feature_dir" \
  --retention-metrics "${retention[0]}" \
  --output "$state/independent_audit.json" \
  > "$state/independent_audit.log" 2>&1; then
  touch "$state/independent_audit_complete"
else
  touch "$state/independent_audit_needs_attention"
  exit 1
fi
