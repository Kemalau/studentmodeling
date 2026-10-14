#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
auth=${PROFILE_OWNERSHIP_CP400_AUTH:-/home/v-yuejiali/.azure-keyflip-e123-recovery-20260819}
experiment=state8-profile-ownership-checkpoint400-audit-20260820
state="$project/results/state8_profile_ownership_checkpoint400_audit_20260820"
expected_adapter=e04320134b514946048fa621109a8ae41ebcb4715a985871da1c39c1c3e40d3e
expected_recurrent=6416a45dfd00ba3afc5a6c9cd1546177fea5b3b3aa44cfba6ab2610fbfd79ef5

mkdir -p "$state"
cd "$project" || exit 1
exec 9>"$state/watcher.lock"
flock -n 9 || exit 0
echo $$ > "$state/watcher.pid"
trap 'rm -f "$state/watcher.pid"' EXIT
export AZURE_CONFIG_DIR="$auth"

while true; do
  date -u +%Y-%m-%dT%H:%M:%SZ > "$state/watcher.heartbeat"
  if status="$(timeout 90 amlt --json-tables status "$experiment" \
      --hide-urls --force-status-update --no-calculate-result-size 2>/dev/null)" \
      && [[ -n "$status" ]]; then
    printf '%s\n' "$status" > "$state/status.json.partial"
    mv "$state/status.json.partial" "$state/status.json"
    terminal="$(STATUS="$status" /home/v-yuejiali/.venvs/studentmodeling/bin/python - <<'PY'
import json,os
rows=json.loads(os.environ['STATUS'])
terminal={'pass','passed','failed','fail','killed','cancelled','canceled','stopped'}
states=[str(row.get('status') or '').casefold() for row in rows]
print('yes' if states and all(value in terminal for value in states) else 'no')
PY
)"
    [[ "$terminal" == yes ]] && break
  fi
  sleep 60
done

rm -rf "$state/download"
mkdir -p "$state/download"
timeout 1800 amlt results download "$experiment" --status pass \
  --include '*.json' --output "$state/download" > "$state/download.log" 2>&1 || true

STATUS="$status" STATE="$state" ADAPTER="$expected_adapter" RECURRENT="$expected_recurrent" \
  /home/v-yuejiali/.venvs/studentmodeling/bin/python - <<'PY'
import json,os,pathlib,shutil
state=pathlib.Path(os.environ['STATE'])
rows=json.loads(os.environ['STATUS'])
passed=bool(rows) and all(str(row.get('status') or '').casefold() in {'pass','passed'} for row in rows)
metrics=sorted(state.joinpath('download').rglob('checkpoint_early_audit/metrics.json'))
failures=[]
if not passed: failures.append('remote job did not pass')
if len(metrics)!=1: failures.append(f'expected one metrics artifact, found {len(metrics)}')
payload=None
if len(metrics)==1:
    payload=json.loads(metrics[0].read_text())
    checks={
      'stage':payload.get('stage')=='state8_profile_ownership_checkpoint_early_audit_v1',
      'complete':payload.get('status')=='complete',
      'checkpoint_400':payload.get('checkpoint_step')==400,
      'students_88':payload.get('students')==88,
      'state_only':payload.get('strict_state_only') is True,
      'permutations_2000':payload.get('ownership_permutation_replicates')==2000,
      'adapter_hash':payload.get('artifact_sha256',{}).get('adapter_model')==os.environ['ADAPTER'],
      'recurrent_hash':payload.get('artifact_sha256',{}).get('recurrent_state')==os.environ['RECURRENT'],
    }
    failures += [name for name,value in checks.items() if not value]
    if not failures: shutil.copy2(metrics[0],state/'metrics.json')
summary={'stage':'state8_profile_ownership_checkpoint400_watcher_v1','remote_passed':passed,'rows':rows,'failures':failures,'metrics':payload}
(state/'terminal_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
marker='complete' if not failures else 'needs_attention'
(state/marker).touch()
(state/('needs_attention' if marker=='complete' else 'complete')).unlink(missing_ok=True)
raise SystemExit(0 if not failures else 1)
PY
