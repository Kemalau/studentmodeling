#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
auth=${PROFILE_OWNERSHIP_AUDIT_AUTH:-/home/v-yuejiali/.azure-studentmodeling-feeds-v2}
config="$project/amlt/feeds_state8_profile_ownership_checkpoint200_audit_20260820.yaml"
experiment=state8-profile-ownership-checkpoint200-audit-20260820
state="$project/results/state8_profile_ownership_checkpoint200_audit_20260820"
expected_launcher=203e9c318169a38c24bd7fa7438f2409d7048d4f412e72009e904df364a88549
expected_adapter=9f7f8e82f6dd5cc80522ebfd99ac972aa45b60234251c5f69af6f94c7d8ede58
expected_recurrent=fc07a859f74ad5addbdf6bd4fde86ae3cfbc5063eb822323eafab097012cf7ba

mkdir -p "$state"
cd "$project" || exit 1
exec 9>"$state/controller.lock"
flock -n 9 || exit 0
echo $$ > "$state/controller.pid"
trap 'rm -f "$state/controller.pid"' EXIT
export AZURE_CONFIG_DIR="$auth"

while true; do
  date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"
  if timeout 45 az account get-access-token \
      --subscription b6dc87f3-c479-49c8-8cb5-7896da3ff895 \
      --scope 'https://storage.azure.com/.default' --query expiresOn -o tsv \
      > "$state/storage_token_expiry.txt.partial" 2> "$state/auth_probe.stderr"; then
    mv "$state/storage_token_expiry.txt.partial" "$state/storage_token_expiry.txt"
    break
  fi
  rm -f "$state/storage_token_expiry.txt.partial"
  sleep 30
done

test "$(sha256sum scripts/run_state8_profile_ownership_checkpoint_audit.sh | cut -d' ' -f1)" = "$expected_launcher" || exit 10
timeout 120 amlt --json-tables ti sing -t Feeds -a > "$state/quota.json" || exit 11
timeout 120 amlt run "$config" --dump -t Feeds \
  > "$state/preflight.dump.yaml" 2> "$state/preflight.stderr" || exit 12
python - "$state/preflight.dump.yaml" <<'PY' || exit 13
from pathlib import Path
import sys
s=Path(sys.argv[1]).read_text()
assert '80G1-A100' in s and 'rankfun_aml' in s and 'name: Feeds' in s,s
assert 'run_state8_profile_ownership_checkpoint_audit.sh' in s,s
assert all(x not in s for x in ('<...>','REQUIRED','REPLACE_LAUNCHER_SHA256','msrcommonacr.azurecr.io/ml-common-py3.12:latest')),s
PY

if ! timeout 90 amlt --json-tables status "$experiment" \
    --hide-urls --no-calculate-result-size > "$state/status.existing.json" 2>/dev/null \
    || [[ ! -s "$state/status.existing.json" ]]; then
  amlt run "$config" "$experiment" -t Feeds -y \
    --description 'Evaluation-only checkpoint-200 held-out ownership audit; 2000-permutation null; strict State8; no training' \
    > "$state/submission.txt" 2>&1 || exit 14
fi

while true; do
  date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"
  if status="$(timeout 90 amlt --json-tables status "$experiment" \
      --hide-urls --force-status-update --no-calculate-result-size 2>/dev/null)" \
      && [[ -n "$status" ]]; then
    printf '%s\n' "$status" > "$state/status.json.partial"
    mv "$state/status.json.partial" "$state/status.json"
    terminal="$(STATUS="$status" /home/v-yuejiali/.venvs/studentmodeling/bin/python - <<'PY'
import json,os
rows=json.loads(os.environ['STATUS'])
states=[str(r.get('status') or '').casefold() for r in rows]
terminal={'pass','passed','failed','fail','killed','cancelled','canceled','stopped'}
print('yes' if states and all(x in terminal for x in states) else 'no')
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
import hashlib,json,os,pathlib,shutil
state=pathlib.Path(os.environ['STATE'])
rows=json.loads(os.environ['STATUS'])
passed=bool(rows) and all(str(r.get('status') or '').casefold() in {'pass','passed'} for r in rows)
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
  'diagnostic_only':payload.get('diagnostic_only') is True,
  'checkpoint_200':payload.get('checkpoint_step')==200,
  'students_88':payload.get('students')==88,
  'state_only':payload.get('strict_state_only') is True,
  'permutations_2000':payload.get('ownership_permutation_replicates')==2000,
  'adapter_hash':payload.get('artifact_sha256',{}).get('adapter_model')==os.environ['ADAPTER'],
  'recurrent_hash':payload.get('artifact_sha256',{}).get('recurrent_state')==os.environ['RECURRENT'],
 }
 failures += [name for name,value in checks.items() if not value]
 if not failures: shutil.copy2(metrics[0],state/'metrics.json')
summary={'stage':'state8_profile_ownership_checkpoint200_supervisor_v1','remote_passed':passed,'rows':rows,'failures':failures,'metrics':payload}
(state/'terminal_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
marker='complete' if not failures else 'needs_attention'
(state/marker).touch()
(state/('needs_attention' if marker=='complete' else 'complete')).unlink(missing_ok=True)
raise SystemExit(0 if not failures else 1)
PY
