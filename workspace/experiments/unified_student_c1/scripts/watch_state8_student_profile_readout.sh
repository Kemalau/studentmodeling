#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
experiment=${PROFILE_READOUT_EXPERIMENT:-state8-student-profile-readout-seed8601-recovery2-20260817}
state=${PROFILE_READOUT_STATE:-"$project/results/state8_student_profile_readout_20260817"}
auth=${PROFILE_READOUT_AUTH:-/home/v-yuejiali/.azure-studentmodeling-feeds-v2}

mkdir -p "$state"
cd "$project" || exit 1
test -d "$auth" || {
  printf 'PROFILE_READOUT_AUTH does not exist: %s\n' "$auth" >&2
  exit 2
}
exec 9>"$state/watcher.lock"
flock -n 9 || exit 0
echo $$ > "$state/watcher.pid"
trap 'rm -f "$state/watcher.pid"' EXIT
rm -f "$state/complete" "$state/needs_attention"

while true; do
  date -u +%Y-%m-%dT%H:%M:%SZ > "$state/watcher.heartbeat"
  export AZURE_CONFIG_DIR="$auth"
  status="$(timeout 90 amlt --json-tables status "$experiment" \
    --hide-urls --force-status-update --no-calculate-result-size 2>/dev/null)" || {
      sleep 60
      continue
    }
  [[ -n "$status" ]] || {
      sleep 60
      continue
    }
  printf '%s\n' "$status" > "$state/status.json.partial"
  mv "$state/status.json.partial" "$state/status.json"
  terminal="$(STATUS="$status" /home/v-yuejiali/.venvs/studentmodeling/bin/python - <<'PY'
import json,os
rows=json.loads(os.environ['STATUS'])
success={'pass','passed'}
failure={'failed','fail','killed','cancelled','canceled','stopped'}
states=[str(r.get('status') or '').casefold() for r in rows]
# AMLT can transiently return ``unknown`` for a running job.  Only an explicit
# terminal state may trigger result collection; every other value keeps polling.
print('yes' if states and all(state in success | failure for state in states) else 'no')
PY
)"
  [[ "$terminal" == yes ]] && break
  sleep 60
done

rm -rf "$state/download"
mkdir -p "$state/download"
timeout 1800 amlt results download "$experiment" --status pass \
  --include '*.json' --include '*.jsonl' --include '*.npz' --include '*.txt' \
  --output "$state/download" > "$state/download.log" 2>&1 || true

STATUS="$status" STATE="$state" EXPERIMENT="$experiment" \
  /home/v-yuejiali/.venvs/studentmodeling/bin/python - <<'PY'
import json,os,pathlib,shutil,subprocess
state=pathlib.Path(os.environ['STATE'])
rows=json.loads(os.environ['STATUS'])
passed=bool(rows) and all(str(r.get('status') or '').casefold() in {'pass','passed'} for r in rows)
metrics=sorted(state.joinpath('download').rglob('profile_readout/metrics.json'))
profiles=sorted(state.joinpath('download').rglob('profile_readout/readable_profiles.jsonl'))
transport=sorted(state.joinpath('download').rglob('profile_readout/fixed_query_transport.jsonl'))
retention=sorted(state.joinpath('download').rglob('retention_eval/metrics.json'))
interface=sorted(state.joinpath('download').rglob('reusable_interface/metrics.json'))
diagnostic_completion=sorted(
 state.joinpath('download').rglob('distributed_diagnostic_completion.json')
)
complete=passed and all(len(x)==1 for x in (metrics,profiles,transport,retention,interface))
diagnostic_complete=passed and len(diagnostic_completion)==1
summary={
 'experiment':os.environ['EXPERIMENT'],'rows':rows,'all_passed':passed,
 'artifact_cardinality':{
  'profile_metrics':len(metrics),'profiles':len(profiles),
  'transport':len(transport),'retention_metrics':len(retention),
  'interface_metrics':len(interface),
  'distributed_diagnostic_completion':len(diagnostic_completion)},
}
if diagnostic_complete:
 shutil.copy2(diagnostic_completion[0],state/'distributed_diagnostic_completion.json')
 payload=json.loads((state/'distributed_diagnostic_completion.json').read_text())
 if payload.get('status')!='diagnostic_complete' or payload.get('history_integrity_pass') is not True:
  diagnostic_complete=False
 else:
  summary['distributed_diagnostic']=payload
if complete:
 for source,name in ((metrics[0],'metrics.json'),(profiles[0],'readable_profiles.jsonl'),
                     (transport[0],'fixed_query_transport.jsonl'),(retention[0],'retention_metrics.json'),
                     (interface[0],'reusable_interface_metrics.json')):
  shutil.copy2(source,state/name)
 payload=json.loads((state/'metrics.json').read_text())
 summary.update(scientific_gate_pass=payload.get('scientific_gate_pass'),
                profile=payload.get('profile'),transport=payload.get('transport'))
 reusable=json.loads((state/'reusable_interface_metrics.json').read_text())
 summary.update(reusable_interface_scientific_gate_pass=reusable.get('scientific_gate_pass'),
                reusable_interface_gates=reusable.get('gates'),
                reusable_interface_metrics=reusable.get('metrics'),
                reusable_interface_uncertainty=reusable.get('uncertainty'))

 interface_features=sorted(state.joinpath('download').rglob('reusable_interface_features/test.npz'))
 if len(interface_features)==1:
  feature_dir=interface_features[0].parent
  audit_command=[
   '/home/v-yuejiali/.venvs/studentmodeling/bin/python',
   'scripts/audit_state8_future_profile_v2.py',
   '--profile-metrics',str(metrics[0]),
   '--profiles-jsonl',str(profiles[0]),
   '--transport-jsonl',str(transport[0]),
   '--interface-metrics',str(interface[0]),
   '--interface-feature-dir',str(feature_dir),
   '--retention-metrics',str(retention[0]),
   '--output',str(state/'independent_audit.json')]
  with (state/'independent_audit.log').open('w') as handle:
   result=subprocess.run(audit_command,stdout=handle,stderr=subprocess.STDOUT,check=False)
  summary['independent_audit_exit_code']=result.returncode
  if (state/'independent_audit.json').is_file():
   summary['independent_audit']=json.loads((state/'independent_audit.json').read_text())
 else:
  summary['independent_audit_exit_code']=None
  summary['independent_audit_error']='reusable interface test feature artifact missing or duplicated'
else:
 for row in rows:
  if str(row.get('status') or '').casefold()=='pass': continue
  with (state/'failure.log').open('w') as handle:
   subprocess.run(['timeout','180','amlt','logs','view','-n','1000','--no-pager',
                   os.environ['EXPERIMENT'],str(row.get('job_name') or '')],
                  stdout=handle,stderr=subprocess.STDOUT,check=False)
(state/'terminal_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,sort_keys=True)+'\n')
audited=(summary.get('independent_audit_exit_code')==0)
terminal_marker = 'complete' if (complete and audited) or diagnostic_complete else 'needs_attention'
opposite_marker = 'needs_attention' if terminal_marker == 'complete' else 'complete'
(state/terminal_marker).touch()
(state/opposite_marker).unlink(missing_ok=True)
PY
