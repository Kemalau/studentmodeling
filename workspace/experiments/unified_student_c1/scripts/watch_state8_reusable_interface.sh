#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
experiment=${STATE_INTERFACE_EXPERIMENT:-state8-reusable-interface-seed8601-20260818}
state=${STATE_INTERFACE_STATE:-"$project/results/state8_reusable_interface_20260818"}
auth=/home/v-yuejiali/.azure-studentmodeling-feeds-v2
python=/home/v-yuejiali/.venvs/studentmodeling/bin/python

mkdir -p "$state"
cd "$project" || exit 1
exec 9>"$state/watcher.lock"
flock -n 9 || exit 0
echo $$ > "$state/watcher.pid"
trap 'rm -f "$state/watcher.pid"' EXIT

while true; do
  date -u +%Y-%m-%dT%H:%M:%SZ > "$state/watcher.heartbeat"
  export AZURE_CONFIG_DIR="$auth"
  status="$(timeout 90 amlt --json-tables status "$experiment" \
    --hide-urls --force-status-update --no-calculate-result-size 2>/dev/null)" || {
      sleep 60
      continue
    }
  printf '%s\n' "$status" > "$state/status.json.partial"
  mv "$state/status.json.partial" "$state/status.json"
  terminal="$(STATUS="$status" "$python" - <<'PY'
import json,os
rows=json.loads(os.environ['STATUS'])
active={'queued','preparing','running','pausing','resuming','paused'}
print('no' if not rows or any(str(r.get('status') or '').casefold() in active for r in rows) else 'yes')
PY
)"
  [[ "$terminal" == yes ]] && break
  sleep 60
done

rm -rf "$state/download" "$state/features" "$state/analysis"
mkdir -p "$state/download" "$state/features" "$state/analysis"
timeout 1800 amlt results download "$experiment" --status pass \
  --include '*.npz' --include '*.json' --include '*.txt' \
  --output "$state/download" > "$state/download.log" 2>&1 || true

STATUS="$status" STATE="$state" EXPERIMENT="$experiment" PYTHON="$python" \
  "$python" - <<'PY'
import json,os,pathlib,shutil,subprocess

state=pathlib.Path(os.environ['STATE'])
rows=json.loads(os.environ['STATUS'])
passed=bool(rows) and len(rows)==3 and all(
    str(row.get('status') or '').casefold()=='pass' for row in rows
)
artifact_cardinality={}
complete=passed
for split in ('train','valid','test'):
    npz=sorted(state.joinpath('download').rglob(f'{split}.npz'))
    metadata=sorted(state.joinpath('download').rglob(f'{split}.json'))
    artifact_cardinality[split]={'npz':len(npz),'metadata':len(metadata)}
    if len(npz)!=1 or len(metadata)!=1:
        complete=False
        continue
    shutil.copy2(npz[0],state/'features'/f'{split}.npz')
    shutil.copy2(metadata[0],state/'features'/f'{split}.json')

analysis_returncode=None
if complete:
    command=[
        os.environ['PYTHON'],'scripts/evaluate_state8_reusable_interface.py','analyze',
        '--manifest-dir','manifests/dbe_student_profile_v1',
        '--feature-dir',str(state/'features'),
        '--output-dir',str(state/'analysis'),
        '--bootstrap-replicates','5000','--permutation-replicates','2000',
        '--seed','20260818',
    ]
    with (state/'analysis.log').open('w') as handle:
        analysis_returncode=subprocess.run(
            command,stdout=handle,stderr=subprocess.STDOUT,check=False
        ).returncode
    complete=analysis_returncode==0 and (state/'analysis'/'metrics.json').is_file()

summary={
    'experiment':os.environ['EXPERIMENT'],
    'rows':rows,
    'all_jobs_passed':passed,
    'artifact_cardinality':artifact_cardinality,
    'analysis_returncode':analysis_returncode,
}
if complete:
    payload=json.loads((state/'analysis'/'metrics.json').read_text())
    summary.update(
        scientific_gate_pass=payload['scientific_gate_pass'],
        gates=payload['gates'],
        metrics=payload['metrics'],
        uncertainty=payload['uncertainty'],
    )
else:
    failed=[row for row in rows if str(row.get('status') or '').casefold()!='pass']
    for row in failed:
        name=str(row.get('job_name') or '')
        safe=name.replace(':','').replace('/','_') or 'unknown'
        with (state/f'failure.{safe}.log').open('w') as handle:
            subprocess.run(
                ['timeout','180','amlt','logs','view','-n','1000','--no-pager',
                 os.environ['EXPERIMENT'],name],
                stdout=handle,stderr=subprocess.STDOUT,check=False,
            )
(state/'terminal_summary.json').write_text(
    json.dumps(summary,indent=2,ensure_ascii=False,sort_keys=True)+'\n'
)
(state/('complete' if complete else 'needs_attention')).touch()
PY
