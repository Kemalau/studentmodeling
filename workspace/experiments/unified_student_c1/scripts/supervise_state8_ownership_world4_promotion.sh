#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
diagnostic=state8-future-profile-ownership-v3-world4-diagnostic-20260820
formal=state8-future-profile-ownership-v3-world4-recovery2-20260820
auth=${PROFILE_OWNERSHIP_WORLD4_AUTH:-/home/v-yuejiali/.azure-keyflip-e123-recovery-20260819}
state="$project/results/state8_future_profile_ownership_v3_world4_promotion_20260820"
diagnostic_state="$project/results/state8_future_profile_ownership_v3_world4_diagnostic_20260820"
formal_state="$project/results/state8_future_profile_ownership_v3_world4_recovery2_20260820"
config="$project/amlt/feeds_state8_future_profile_ownership_v3_world4_20260820.yaml"

mkdir -p "$state"
cd "$project" || exit 1
exec 9>"$state/controller.lock"
flock -n 9 || exit 0
echo $$ > "$state/controller.pid"
trap 'rm -f "$state/controller.pid"' EXIT
export AZURE_CONFIG_DIR="$auth"

while true; do
  date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"
  [[ -f "$diagnostic_state/complete" || -f "$diagnostic_state/needs_attention" ]] && break
  sleep 60
done

if [[ ! -f "$diagnostic_state/complete" ]]; then
  printf 'diagnostic did not pass; formal promotion suppressed\n' > "$state/suppressed.txt"
  exit 1
fi

python - "$diagnostic_state/terminal_summary.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['all_passed'] is True,x
assert x['artifact_cardinality'],x
PY

# Revalidate target headroom immediately before the cost-bearing promotion.
amlt --json-tables ti sing -t Feeds -a > "$state/quota.json" || exit 2
amlt run "$config" --dump -t Feeds > "$state/preflight.dump.yaml" 2> "$state/preflight.stderr" || exit 3
python - "$state/preflight.dump.yaml" <<'PY'
from pathlib import Path
import sys
s=Path(sys.argv[1]).read_text()
assert '80G4-A100' in s and 'rankfun_aml' in s and 'name: Feeds' in s,s
assert 'PROFILE_READOUT_DIAGNOSTIC_STOP_ROUND' not in s,s
assert all(x not in s for x in ('<...>','REQUIRED','msrcommonacr.azurecr.io/ml-common-py3.12:latest')),s
PY

amlt run "$config" "$formal" -t Feeds -y \
  --description 'Formal 80G4-A100 State8 ownership training after bounded collective-order diagnostic passed; strict state-only profile/readout/retention evaluation' \
  > "$state/submission.txt" 2>&1 || exit 4
amlt --json-tables status "$formal" --force-status-update --no-calculate-result-size \
  > "$state/status.initial.json" 2> "$state/status.initial.stderr" || exit 5

rm -f "$formal_state/watcher.lock" "$formal_state/watcher.pid"
setsid env PROFILE_READOUT_EXPERIMENT="$formal" PROFILE_READOUT_STATE="$formal_state" \
  PROFILE_READOUT_AUTH="$auth" bash scripts/watch_state8_student_profile_readout.sh \
  > "$formal_state/watcher.nohup.log" 2>&1 < /dev/null &
echo $! > "$formal_state/watcher.launch.pid"
touch "$state/promoted"
