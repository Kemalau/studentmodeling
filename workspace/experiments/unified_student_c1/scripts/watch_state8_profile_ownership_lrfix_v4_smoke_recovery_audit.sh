#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
config="$project/amlt/feeds_state8_profile_ownership_lrfix_v4_smoke_recovery_audit_20260821.yaml"
experiment=state8-profile-ownership-lrfix-v4-smoke-recovery-audit-20260821
job=state8-lrfix-v4-smoke-r2-metadatafix-evalonly-seed8801
state="$project/results/state8_profile_ownership_lrfix_v4_smoke_recovery_audit_20260821"
original="$project/results/state8_profile_ownership_lrfix_v4_sparsekc_smoke_r2_20260820"
promotion="$project/results/state8_profile_ownership_lrfix_v4_formal_promotion_20260820"
auth=${PROFILE_OWNERSHIP_LRFIX_V4_AUTH:-/home/v-yuejiali/.azure-studentmodeling-feeds-v2}
python=/home/v-yuejiali/.venvs/studentmodeling/bin/python

mkdir -p "$state"
cd "$project" || exit 1
exec 9>"$state/controller.lock"
flock -n 9 || exit 0
echo $$ > "$state/controller.pid"
trap 'rm -f "$state/controller.pid"' EXIT
rm -f "$state/complete" "$state/needs_attention" "$state/promoted_original_smoke"
export AZURE_CONFIG_DIR="$auth"

while true; do
  date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"
  existing="$(timeout 120 amlt --json-tables status "$experiment" \
    --hide-urls --force-status-update --no-calculate-result-size 2>/dev/null || true)"
  if [[ -n "$existing" && "$existing" != '[]' ]]; then
    printf '%s\n' "$existing" > "$state/status.json.partial"
    mv "$state/status.json.partial" "$state/status.json"
    break
  fi
  exec 8>"$project/results/feeds_submission_global.lock"
  if flock -w 60 8; then
    existing="$(timeout 120 amlt --json-tables status "$experiment" \
      --hide-urls --force-status-update --no-calculate-result-size 2>/dev/null || true)"
    if [[ -z "$existing" || "$existing" == '[]' ]]; then
      timeout 900 amlt run "$config" "$experiment" -t Feeds -y \
        --description 'Evaluation-only recovery of completed State8 lrfix-v4 DBE 200-step smoke after checkpoint-step metadata reader fix; immutable checkpoint and held-out DBE rows; no retraining; XES official test untouched' \
        > "$state/submission.txt" 2>&1
      submit_rc=$?
      if (( submit_rc != 0 )); then
        printf '%s\n' "$submit_rc" > "$state/submission.rc"
        flock -u 8
        sleep 60
        continue
      fi
    fi
    flock -u 8
  fi
  sleep 30
done

while true; do
  date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"
  status="$(timeout 120 amlt --json-tables status "$experiment" \
    --hide-urls --force-status-update --no-calculate-result-size 2>/dev/null)" || {
      sleep 60
      continue
    }
  [[ -n "$status" ]] || { sleep 60; continue; }
  printf '%s\n' "$status" > "$state/status.json.partial"
  mv "$state/status.json.partial" "$state/status.json"
  terminal="$(STATUS="$status" "$python" - <<'PY'
import json, os
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
timeout 1800 amlt results download "$experiment" --status pass --include '*.json' \
  --output "$state/download" > "$state/download.log" 2>&1 || true

STATUS="$status" STATE="$state" JOB="$job" "$python" - <<'PY'
import json, os, pathlib, shutil
state=pathlib.Path(os.environ['STATE']); rows=json.loads(os.environ['STATUS'])
passed=bool(rows) and all(str(row.get('status') or '').casefold() in {'pass','passed'} for row in rows)
metrics=sorted(state.joinpath('download').rglob('checkpoint_early_audit/metrics.json'))
completion=sorted(state.joinpath('download').rglob('diagnostic_completion.json'))
repair=sorted(state.joinpath('download').rglob('profile_ownership_repair.json'))
receipt=sorted(state.joinpath('download').rglob('recovery_receipt.json'))
failures=[]
if not passed: failures.append('remote recovery job did not pass')
for name, values in [('metrics',metrics),('completion',completion),('repair',repair),('receipt',receipt)]:
    if len(values)!=1: failures.append(f'expected one {name} artifact, found {len(values)}')
m=json.loads(metrics[0].read_text()) if len(metrics)==1 else None
c=json.loads(completion[0].read_text()) if len(completion)==1 else None
r=json.loads(repair[0].read_text()) if len(repair)==1 else None
q=json.loads(receipt[0].read_text()) if len(receipt)==1 else None
if m:
    checks={
      'stage':m.get('stage')=='state8_profile_ownership_checkpoint_early_audit_v1',
      'complete':m.get('status')=='complete','checkpoint_200':m.get('checkpoint_step')==200,
      'students_88':m.get('students')==88,'cells_634':m.get('supported_student_kc_cells')==634,
      'state_only':m.get('strict_state_only') is True,'layernorm':m.get('profile_ownership_feature_normalization')=='layernorm',
      'permutations_2000':m.get('ownership_permutation_replicates')==2000,
    }
    failures += [name for name, ok in checks.items() if not ok]
if c and not (c.get('status')=='diagnostic_complete' and c.get('micro_steps')==200 and c.get('history_integrity_pass') is True):
    failures.append('completion contract')
if r and not (r.get('status')=='diagnostic_complete' and r.get('minimum_kc_train_support')==4 and r.get('bias_prior_strength')==8.0):
    failures.append('repair contract')
if q and not (q.get('status')=='complete' and q.get('training_repeated') is False and q.get('official_xes_test_inspected') is False):
    failures.append('recovery receipt contract')
summary={'stage':'state8_profile_ownership_lrfix_v4_smoke_recovery_watcher_v1','remote_passed':passed,
         'rows':rows,'failures':failures,'metrics':m,'completion':c,'repair':r,'recovery_receipt':q}
(state/'terminal_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
for source, name in ((metrics,'metrics.json'),(completion,'diagnostic_completion.json'),(repair,'profile_ownership_repair.json'),(receipt,'recovery_receipt.json')):
    if len(source)==1: shutil.copy2(source[0],state/name)
(state/('complete' if not failures else 'needs_attention')).touch()
raise SystemExit(0 if not failures else 1)
PY
audit_rc=$?
if (( audit_rc != 0 )); then
  timeout 180 amlt logs view -n 500 --no-pager "$experiment" > "$state/terminal_logs.txt" 2>&1 || true
  exit "$audit_rc"
fi

set +e
"$python" scripts/validate_state8_profile_ownership_lrfix_v4_promotion.py \
  --smoke-summary "$state/terminal_summary.json" --output "$state/promotion_gate.json" \
  > "$state/promotion_gate.log" 2>&1
gate_rc=$?
set -e
if (( gate_rc != 0 )); then
  touch "$state/scientific_gate_failed"
  exit "$gate_rc"
fi

# The old job failed only after publishing immutable checkpoint artifacts.
# Promote its smoke state atomically from the independently audited recovery;
# retain the old terminal summary as failure provenance.
cp "$original/terminal_summary.json" "$original/terminal_summary.metadata_bug_failure.json" 2>/dev/null || true
cp "$state/terminal_summary.json" "$original/terminal_summary.json.partial"
mv "$original/terminal_summary.json.partial" "$original/terminal_summary.json"
cp "$state/metrics.json" "$original/metrics.json.partial"
mv "$original/metrics.json.partial" "$original/metrics.json"
rm -f "$original/needs_attention"
touch "$original/complete" "$state/promoted_original_smoke"

# Restart the fail-closed promotion controller only after the scientific gate
# passed.  Its own lock prevents duplicate submissions.
if [[ ! -f "$promotion/promoted" && ! -f "$promotion/controller.pid" ]]; then
  setsid env PROFILE_OWNERSHIP_LRFIX_V4_AUTH="$auth" \
    bash scripts/promote_state8_profile_ownership_lrfix_v4_formal.sh \
    >> "$promotion/controller.nohup.log" 2>&1 < /dev/null &
  echo $! > "$promotion/controller.launch.pid"
fi
