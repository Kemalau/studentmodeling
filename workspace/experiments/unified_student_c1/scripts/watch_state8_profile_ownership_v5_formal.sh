#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
experiment=state8-profile-ownership-v5-formal-20260822
state="$project/results/state8_profile_ownership_v5_formal_20260822"
python=/home/v-yuejiali/.venvs/studentmodeling/bin/python

mkdir -p "$state"
cd "$project" || exit 1
exec 9>"$state/watcher.lock"
flock -n 9 || exit 0
echo $$ > "$state/watcher.pid"
trap 'rm -f "$state/watcher.pid"' EXIT
rm -f "$state/complete" "$state/needs_attention"

while true; do
  date -u +%Y-%m-%dT%H:%M:%SZ > "$state/watcher.heartbeat"
  source scripts/use_studentmodeling_feeds_auth.sh
  status="$(timeout 120 amlt --json-tables status "$experiment" \
    --hide-urls --force-status-update --no-calculate-result-size 2>/dev/null)" || {
      sleep 300
      continue
    }
  [[ -n "$status" && "$status" != '[]' ]] || { sleep 300; continue; }
  printf '%s\n' "$status" > "$state/status.json.partial"
  mv "$state/status.json.partial" "$state/status.json"
  terminal="$(STATUS="$status" "$python" - <<'PY'
import json, os
rows=json.loads(os.environ['STATUS'])
success={'pass','passed'}
failure={'failed','fail','killed','cancelled','canceled','stopped'}
states=[str(r.get('status') or '').casefold() for r in rows]
print('yes' if states and all(value in success | failure for value in states) else 'no')
PY
)"
  [[ "$terminal" == yes ]] && break
  sleep 300
done

passed="$(STATUS="$status" "$python" - <<'PY'
import json, os
rows=json.loads(os.environ['STATUS'])
print('yes' if rows and all(str(r.get('status') or '').casefold() in {'pass','passed'} for r in rows) else 'no')
PY
)"
rm -rf "$state/download" "$state/download.partial"
if [[ "$passed" == yes ]]; then
  download_ok=no
  for attempt in $(seq 1 12); do
    date -u +%Y-%m-%dT%H:%M:%SZ > "$state/watcher.heartbeat"
    source scripts/use_studentmodeling_feeds_auth.sh
    rm -rf "$state/download.partial"
    mkdir -p "$state/download.partial"
    if timeout 3600 amlt results download "$experiment" --status pass \
      --include '*.json' --include '*.npz' --include 'readable_profiles.jsonl' \
      --include 'fixed_query_transport.jsonl' \
      --include 'adapter_model.safetensors' --include 'recurrent_state.pt' \
      --output "$state/download.partial" \
      > "$state/download.attempt-${attempt}.log" 2>&1; then
      mv "$state/download.partial" "$state/download"
      download_ok=yes
      break
    fi
    sleep 300
  done
  [[ "$download_ok" == yes ]] || mkdir -p "$state/download"
else
  # Failed jobs can still expose completion/checkpoint receipts that identify
  # an evaluation-only recovery.  Download them without claiming success.
  mkdir -p "$state/download"
  timeout 1800 amlt results download "$experiment" \
    --include '*.json' --include 'adapter_model.safetensors' \
    --include 'recurrent_state.pt' --output "$state/download" \
    > "$state/download.failed-job.log" 2>&1 || true
fi
timeout 300 amlt logs view -n 1000 --no-pager "$experiment" \
  > "$state/terminal.log" 2>&1 || true

STATUS="$status" STATE="$state" EXPERIMENT="$experiment" "$python" - <<'PY'
import hashlib, json, os, pathlib, subprocess
state=pathlib.Path(os.environ['STATE'])
rows=json.loads(os.environ['STATUS'])
passed=bool(rows) and all(str(r.get('status') or '').casefold() in {'pass','passed'} for r in rows)
files=sorted(str(p.relative_to(state)) for p in state.joinpath('download').rglob('*') if p.is_file())
summary={
  'experiment':os.environ['EXPERIMENT'], 'rows':rows, 'all_passed':passed,
  'interpretation':'exploratory_after_nonsignificant_200_step_smoke_not_preregistered_confirmation',
  'downloaded_files':files,
}
metrics=sorted(state.joinpath('download').rglob('reusable_interface/metrics.json'))
profiles=sorted(state.joinpath('download').rglob('profile_readout/metrics.json'))
retention=sorted(state.joinpath('download').rglob('retention_eval/metrics.json'))
for key, paths in [('reusable_interface',metrics),('profile_readout',profiles),('retention',retention)]:
  summary[key+'_count']=len(paths)
  if len(paths)==1:
    summary[key]=json.loads(paths[0].read_text(encoding='utf-8'))
summary['scientific_gate_pass']=bool(
  summary.get('reusable_interface',{}).get('scientific_gate_pass') is True
)
profile_rows=sorted(state.joinpath('download').rglob('profile_readout/readable_profiles.jsonl'))
transport=sorted(state.joinpath('download').rglob('profile_readout/fixed_query_transport.jsonl'))
features=sorted(state.joinpath('download').rglob('reusable_interface_features/test.npz'))
adapter=sorted(state.joinpath('download').rglob('final_adapter/adapter_model.safetensors'))
recurrent=sorted(state.joinpath('download').rglob('recurrent_state.pt'))
required={
  'profile_metrics':profiles, 'profile_rows':profile_rows, 'transport':transport,
  'interface_metrics':metrics, 'test_features':features, 'retention_metrics':retention,
  'adapter_model':adapter, 'recurrent_state':recurrent,
}
summary['artifact_cardinality']={key:len(paths) for key,paths in required.items()}
audit_ready=passed and all(len(paths)==1 for paths in required.values())

def sha256(path):
  digest=hashlib.sha256()
  with path.open('rb') as handle:
    for chunk in iter(lambda:handle.read(1024*1024),b''):
      digest.update(chunk)
  return digest.hexdigest()

if audit_ready:
  summary['downloaded_artifact_sha256']={
    'adapter_model':sha256(adapter[0]), 'recurrent_state':sha256(recurrent[0]),
  }
  command=[
    '/home/v-yuejiali/.venvs/studentmodeling/bin/python',
    'scripts/audit_state8_future_profile_v2.py',
    '--profile-metrics',str(profiles[0]), '--profiles-jsonl',str(profile_rows[0]),
    '--transport-jsonl',str(transport[0]), '--interface-metrics',str(metrics[0]),
    '--interface-feature-dir',str(features[0].parent),
    '--retention-metrics',str(retention[0]),
    '--output',str(state/'independent_audit.json'),
  ]
  with (state/'independent_audit.log').open('w',encoding='utf-8') as handle:
    audit=subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT,check=False)
  summary['independent_audit_exit_code']=audit.returncode
  if (state/'independent_audit.json').is_file():
    summary['independent_audit']=json.loads((state/'independent_audit.json').read_text(encoding='utf-8'))
  expected=summary.get('profile_readout',{}).get('artifact_sha256',{})
  summary['downloaded_artifact_hashes_match_metrics']=(
    summary['downloaded_artifact_sha256']==expected
  )
else:
  summary['independent_audit_exit_code']=None
  summary['independent_audit']={
    'status':'not_run','reason':'remote_failure_or_artifact_cardinality',
  }
summary['completion_ready']=(
  passed and audit_ready
  and summary.get('downloaded_artifact_hashes_match_metrics') is True
  and summary.get('independent_audit',{}).get('status')=='pass'
)
(state/'terminal_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,sort_keys=True)+'\n')
marker='complete' if summary['completion_ready'] else 'needs_attention'
(state/marker).touch()
(state/('needs_attention' if marker=='complete' else 'complete')).unlink(missing_ok=True)
PY

if [[ -f "$state/complete" ]]; then
  "$python" scripts/summarize_state8_future_profile_experiment.py \
    --profile-metrics "$(find "$state/download" -path '*/profile_readout/metrics.json' -print -quit)" \
    --interface-metrics "$(find "$state/download" -path '*/reusable_interface/metrics.json' -print -quit)" \
    --retention-metrics "$(find "$state/download" -path '*/retention_eval/metrics.json' -print -quit)" \
    --independent-audit "$state/independent_audit.json" \
    --experiment "$experiment" \
    --output-json "$state/final_experiment_summary.json" \
    --output-md "$state/final_experiment_summary.md" \
    > "$state/final_experiment_summary.log" 2>&1 || {
      rm -f "$state/complete"
      touch "$state/needs_attention"
    }
fi
