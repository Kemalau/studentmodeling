#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
experiment=${XES_PROFILE_CONFIRMATORY_EXPERIMENT:-state8-xes-profile-confirmatory-lrfix-v4-20260821}
state=${XES_PROFILE_CONFIRMATORY_STATE:-$project/results/state8_xes_profile_confirmatory_lrfix_v4_20260821}
auth=${XES_PROFILE_CONFIRMATORY_AUTH:-/home/v-yuejiali/.azure-keyflip-e123-recovery-20260819}
python=/home/v-yuejiali/.venvs/studentmodeling/bin/python

mkdir -p "$state"
cd "$project" || exit 1
exec 9>"$state/watcher.lock"
flock -n 9 || exit 0
echo $$ > "$state/watcher.pid"
trap 'rm -f "$state/watcher.pid"' EXIT
rm -f "$state/complete" "$state/needs_attention"
export AZURE_CONFIG_DIR="$auth"

while true; do
  date -u +%Y-%m-%dT%H:%M:%SZ > "$state/watcher.heartbeat"
  status="$(timeout 90 amlt --json-tables status "$experiment" \
    --hide-urls --force-status-update --no-calculate-result-size 2>/dev/null)" || {
      sleep 60
      continue
    }
  [[ -n "$status" ]] || { sleep 60; continue; }
  printf '%s\n' "$status" > "$state/status.json.partial"
  mv "$state/status.json.partial" "$state/status.json"
  terminal="$(STATUS="$status" "$python" - <<'PY'
import json,os
rows=json.loads(os.environ['STATUS'])
done={'pass','passed','failed','fail','killed','cancelled','canceled','stopped'}
states=[str(row.get('status') or '').casefold() for row in rows]
print('yes' if states and all(value in done for value in states) else 'no')
PY
)"
  [[ "$terminal" == yes ]] && break
  sleep 60
done

rm -rf "$state/download"
mkdir -p "$state/download"
timeout 1800 amlt results download "$experiment" --status pass \
  --include '*.json' --include '*.jsonl' --output "$state/download" \
  > "$state/download.log" 2>&1 || true

STATUS="$status" STATE="$state" EXPERIMENT="$experiment" "$python" - <<'PY'
import json,os,pathlib,shutil,subprocess
state=pathlib.Path(os.environ['STATE']); rows=json.loads(os.environ['STATUS'])
passed=len(rows)==1 and str(rows[0].get('status') or '').casefold() in {'pass','passed'}
metrics=sorted(state.joinpath('download').rglob('analysis/metrics.json'))
profiles=sorted(state.joinpath('download').rglob('analysis/readable_profiles.jsonl'))
freeze=sorted(state.joinpath('download').rglob('freeze_verification.json'))
binding=sorted(state.joinpath('download').rglob('checkpoint_binding.json'))
failures=[]
if not passed: failures.append('remote confirmatory job did not pass')
for name,paths in [('metrics',metrics),('profiles',profiles),('freeze',freeze),('binding',binding)]:
    if len(paths)!=1: failures.append(f'expected one {name} artifact, found {len(paths)}')
payload=json.loads(metrics[0].read_text()) if len(metrics)==1 else None
freeze_payload=json.loads(freeze[0].read_text()) if len(freeze)==1 else None
binding_payload=json.loads(binding[0].read_text()) if len(binding)==1 else None
if payload:
    checks={
      'complete':payload.get('status')=='complete',
      'student_counts':payload.get('students')=={'train':800,'valid':200,'test':466},
      'student_disjoint':payload.get('student_disjoint') is True,
      'chronological':payload.get('chronological_future_targets') is True,
      'state_only':payload.get('raw_history_visible_to_readout') is False,
      'anonymous':payload.get('student_id_visible_to_readout') is False,
      'no_future_support_input':payload.get('future_support_visible_to_readout') is False,
      'validation_selection':(payload.get('readout_selection') or {}).get('selection_split')=='validation_students_only',
      'no_test_selection':(payload.get('readout_selection') or {}).get('test_labels_used_for_selection') is False,
      'bootstrap_5000':all((payload.get('uncertainty') or {}).get(k,{}).get('replicates')==5000 for k in ('state8_vs_population','state8_vs_same_window_kc_outcomes')),
      'permutation_2000':(payload.get('uncertainty') or {}).get('state_ownership_permutation_null',{}).get('replicates')==2000,
    }
    failures += [key for key,value in checks.items() if not value]
if freeze_payload and not (
    freeze_payload.get('status')=='pass'
    and freeze_payload.get('test_metrics_inspected_by_this_verifier') is False
): failures.append('frozen holdout verification failed')
if binding_payload and not (
    binding_payload.get('status')=='frozen_checkpoint_bound'
    and binding_payload.get('development_scientific_gate_pass') is True
): failures.append('checkpoint binding failed')
if not failures:
    shutil.copy2(metrics[0],state/'metrics.json')
    shutil.copy2(profiles[0],state/'readable_profiles.jsonl')
summary={
  'stage':'state8_xes_profile_confirmatory_watcher_v1',
  'experiment':os.environ['EXPERIMENT'],'rows':rows,'remote_passed':passed,
  'failures':failures,'metrics':payload,'freeze':freeze_payload,'binding':binding_payload,
  'scientific_gate_pass':payload.get('scientific_gate_pass') if payload else None,
  'gates':payload.get('gates') if payload else None,
}
(state/'terminal_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,sort_keys=True)+'\n')
(state/('complete' if not failures else 'needs_attention')).touch()
if failures and rows:
    with (state/'terminal_logs.txt').open('w') as handle:
        subprocess.run(['timeout','180','amlt','logs','view','-n','500','--no-pager',os.environ['EXPERIMENT']],stdout=handle,stderr=subprocess.STDOUT,check=False)
raise SystemExit(0 if not failures else 1)
PY
