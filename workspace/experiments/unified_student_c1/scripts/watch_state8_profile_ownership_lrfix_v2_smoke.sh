#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
auth=${PROFILE_OWNERSHIP_LRFIX_V2_AUTH:-/home/v-yuejiali/.azure-keyflip-e123-recovery-20260819}
experiment=state8-profile-ownership-lrfix-v2-smoke-20260820
state="$project/results/state8_profile_ownership_lrfix_v2_smoke_20260820"

mkdir -p "$state"
cd "$project" || exit 1
test -d "$auth" || {
  printf 'Azure browser-login cache does not exist: %s\n' "$auth" >&2
  exit 2
}
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
success={'pass','passed'}
failure={'failed','fail','killed','cancelled','canceled','stopped'}
states=[str(row.get('status') or '').casefold() for row in rows]
print('yes' if states and all(value in success | failure for value in states) else 'no')
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

STATUS="$status" STATE="$state" /home/v-yuejiali/.venvs/studentmodeling/bin/python - <<'PY'
import json,os,pathlib,shutil

state=pathlib.Path(os.environ['STATE'])
rows=json.loads(os.environ['STATUS'])
passed=bool(rows) and all(
    str(row.get('status') or '').casefold() in {'pass','passed'} for row in rows
)
metrics=sorted(state.joinpath('download').rglob('checkpoint_early_audit/metrics.json'))
completion=sorted(state.joinpath('download').rglob('diagnostic_completion.json'))
repair=sorted(state.joinpath('download').rglob('profile_ownership_repair.json'))
failures=[]
if not passed:
    failures.append('remote job did not pass')
if len(metrics) != 1:
    failures.append(f'expected one metrics artifact, found {len(metrics)}')
if len(completion) != 1:
    failures.append(f'expected one completion artifact, found {len(completion)}')
if len(repair) != 1:
    failures.append(f'expected one repair receipt, found {len(repair)}')

payload=None
completion_payload=None
repair_payload=None
if len(metrics) == 1:
    payload=json.loads(metrics[0].read_text())
    checks={
        'stage': payload.get('stage') == 'state8_profile_ownership_checkpoint_early_audit_v1',
        'complete': payload.get('status') == 'complete',
        'checkpoint_200': payload.get('checkpoint_step') == 200,
        'students_88': payload.get('students') == 88,
        'state_only': payload.get('strict_state_only') is True,
        'permutations_2000': payload.get('ownership_permutation_replicates') == 2000,
        'layernorm': payload.get('profile_ownership_feature_normalization') == 'layernorm',
    }
    failures += [name for name,value in checks.items() if not value]
if len(completion) == 1:
    completion_payload=json.loads(completion[0].read_text())
    if completion_payload.get('status') != 'diagnostic_complete':
        failures.append('diagnostic completion status')
    if completion_payload.get('micro_steps') != 200:
        failures.append('diagnostic micro steps')
if len(repair) == 1:
    repair_payload=json.loads(repair[0].read_text())

summary={
    'stage':'state8_profile_ownership_lrfix_v2_smoke_watcher_v1',
    'remote_passed':passed,
    'rows':rows,
    'failures':failures,
    'metrics':payload,
    'completion':completion_payload,
    'repair':repair_payload,
}
(state/'terminal_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
if not failures:
    shutil.copy2(metrics[0],state/'metrics.json')
marker='complete' if not failures else 'needs_attention'
(state/marker).touch()
(state/('needs_attention' if marker == 'complete' else 'complete')).unlink(missing_ok=True)
raise SystemExit(0 if not failures else 1)
PY
rc=$?

if (( rc != 0 )); then
  timeout 120 amlt logs view -n 300 --no-pager "$experiment" \
    > "$state/terminal_logs.txt" 2>&1 || true
fi
exit "$rc"
