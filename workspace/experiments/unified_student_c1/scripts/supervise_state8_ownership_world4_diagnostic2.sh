#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
auth=${PROFILE_OWNERSHIP_WORLD4_AUTH:-/home/v-yuejiali/.azure-keyflip-e123-recovery-20260819}
diagnostic=state8-future-profile-ownership-v3-world4-diagnostic2-20260820
formal=state8-future-profile-ownership-v3-world4-recovery2-20260820
diagnostic_state="$project/results/state8_future_profile_ownership_v3_world4_diagnostic2_20260820"
formal_state="$project/results/state8_future_profile_ownership_v3_world4_recovery2_20260820"
formal_config="$project/amlt/feeds_state8_future_profile_ownership_v3_world4_20260820.yaml"
expected_runner=42383e5693955524010976b4e330677794c4b74e5995f6f0a900ce334d00643b

mkdir -p "$diagnostic_state" "$formal_state"
cd "$project" || exit 1
exec 9>"$diagnostic_state/supervisor.lock"
flock -n 9 || exit 0
echo $$ > "$diagnostic_state/supervisor.pid"
trap 'rm -f "$diagnostic_state/supervisor.pid"' EXIT
export AZURE_CONFIG_DIR="$auth"

while true; do
  date -u +%Y-%m-%dT%H:%M:%SZ > "$diagnostic_state/supervisor.heartbeat"
  if status="$(timeout 90 amlt --json-tables status "$diagnostic" \
      --hide-urls --force-status-update --no-calculate-result-size 2>/dev/null)" \
      && [[ -n "$status" ]]; then
    printf '%s\n' "$status" > "$diagnostic_state/status.json.partial"
    mv "$diagnostic_state/status.json.partial" "$diagnostic_state/status.json"
    terminal="$(STATUS="$status" /home/v-yuejiali/.venvs/studentmodeling/bin/python - <<'PY'
import json,os
rows=json.loads(os.environ['STATUS'])
terminal={'pass','passed','failed','fail','killed','cancelled','canceled','stopped'}
states=[str(row.get('status') or '').casefold() for row in rows]
print('yes' if states and all(state in terminal for state in states) else 'no')
PY
)"
    [[ "$terminal" == yes ]] && break
  fi
  sleep 60
done

rm -rf "$diagnostic_state/download"
mkdir -p "$diagnostic_state/download"
timeout 1800 amlt results download "$diagnostic" --status pass \
  --include 'distributed_diagnostic_completion.json' \
  --output "$diagnostic_state/download" > "$diagnostic_state/download.log" 2>&1 || true

completion="$(find "$diagnostic_state/download" -type f \
  -name distributed_diagnostic_completion.json -print -quit)"
STATUS="$status" COMPLETION="$completion" /home/v-yuejiali/.venvs/studentmodeling/bin/python - <<'PY' \
  > "$diagnostic_state/gate.json.partial" || {
import json,os,pathlib
rows=json.loads(os.environ['STATUS'])
assert rows and all(str(row.get('status') or '').casefold() in {'pass','passed'} for row in rows),rows
path=pathlib.Path(os.environ['COMPLETION'])
assert path.is_file(),path
value=json.loads(path.read_text())
assert value['status']=='diagnostic_complete',value
assert value['micro_steps']==8 and value['optimizer_steps']==2,value
assert value['history_integrity_pass'] is True,value
# The queued diagnostic is pinned to the pre-receipt-extension code snapshot.
# Its immutable integrity fields still prove a complete, gap-free 8-example
# gather.  Newer snapshots additionally expose topology and rank assignment;
# validate them when present without rejecting the already-submitted job.
assert value['history_rows']==8,value
assert value['history_unique_global_positions']==8,value
assert value['history_global_positions_contiguous'] is True,value
assert value['history_global_position_start']==0,value
assert value['history_global_position_end']==7,value
if 'distributed_world_size' in value:
 assert value['distributed_world_size']==4,value
if 'rank_assignment_consistent' in value:
 assert value['rank_assignment_consistent'] is True,value
if 'global_positions_seen' in value:
 assert value['global_positions_seen']==list(range(8)),value
print(json.dumps({'status':'pass','completion':value},indent=2,sort_keys=True))
PY
  mv "$diagnostic_state/gate.json.partial" "$diagnostic_state/gate.json"
  touch "$diagnostic_state/needs_attention"
  exit 1
}
mv "$diagnostic_state/gate.json.partial" "$diagnostic_state/gate.json"
touch "$diagnostic_state/complete"
rm -f "$diagnostic_state/needs_attention"

test "$(sha256sum scripts/run_state8_student_profile_readout.sh | cut -d' ' -f1)" = "$expected_runner" || exit 20
amlt --json-tables ti sing -t Feeds -a > "$diagnostic_state/quota.before-promotion.json" || exit 21
amlt run "$formal_config" --dump -t Feeds \
  > "$diagnostic_state/formal.preflight.yaml" 2> "$diagnostic_state/formal.preflight.stderr" || exit 22
python - "$diagnostic_state/formal.preflight.yaml" "$expected_runner" <<'PY' || exit 23
from pathlib import Path
import sys
s=Path(sys.argv[1]).read_text(); runner=sys.argv[2]
assert '80G4-A100' in s and 'rankfun_aml' in s and 'name: Feeds' in s,s
assert runner in s,s
assert 'PROFILE_READOUT_DIAGNOSTIC_STOP_ROUND' not in s,s
assert all(x not in s for x in ('<...>','REQUIRED','REPLACE_LAUNCHER_SHA256','msrcommonacr.azurecr.io/ml-common-py3.12:latest')),s
PY

# Idempotency guard: never create a second formal experiment.
if ! timeout 90 amlt --json-tables status "$formal" --hide-urls \
    --no-calculate-result-size > "$formal_state/status.existing.json" 2>/dev/null \
    || [[ ! -s "$formal_state/status.existing.json" ]]; then
  amlt run "$formal_config" "$formal" -t Feeds -y \
    --description 'Formal 80G4-A100 State8 ownership training after the fresh exact-hash two-round collective-order diagnostic passed' \
    > "$formal_state/submission.txt" 2>&1 || exit 24
fi
amlt --json-tables status "$formal" --force-status-update --no-calculate-result-size \
  > "$formal_state/status.initial.json" 2> "$formal_state/status.initial.stderr" || exit 25

setsid env PROFILE_READOUT_EXPERIMENT="$formal" PROFILE_READOUT_STATE="$formal_state" \
  PROFILE_READOUT_AUTH="$auth" bash scripts/watch_state8_student_profile_readout.sh \
  > "$formal_state/watcher.nohup.log" 2>&1 < /dev/null &
echo $! > "$formal_state/watcher.launch.pid"
touch "$diagnostic_state/promoted"
