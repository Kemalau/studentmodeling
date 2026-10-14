#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
experiment=state8-label-first-natural-c4-recovery3-20260816
state="$project/results/state8_label_first_evalonly_recovery1_20260816"
auth=/home/v-yuejiali/.azure-studentmodeling-feeds-v2

mkdir -p "$state"
cd "$project" || exit 1
exec 9>"$state/watcher.lock"
flock -n 9 || exit 0
echo $$ > "$state/watcher.pid"
trap 'rm -f "$state/watcher.pid"' EXIT

while true; do
  date -u +%Y-%m-%dT%H:%M:%SZ > "$state/heartbeat"
  export AZURE_CONFIG_DIR="$auth"
  status="$(timeout 90 amlt --json-tables status "$experiment" \
    --hide-urls --force-status-update --no-calculate-result-size 2>/dev/null)" || {
      sleep 60
      continue
    }
  printf '%s\n' "$status" > "$state/status.json.partial"
  mv "$state/status.json.partial" "$state/status.json"
  terminal="$(STATUS="$status" /home/v-yuejiali/.venvs/studentmodeling/bin/python - <<'PY'
import json, os
rows = json.loads(os.environ['STATUS'])
active = {'queued', 'preparing', 'running', 'pausing', 'resuming', 'paused'}
print('no' if any(str(row.get('status') or '').casefold() in active for row in rows) else 'yes')
PY
)"
  [[ "$terminal" == yes ]] && break
  sleep 60
done

rm -rf "$state/download"
mkdir -p "$state/download"
timeout 1800 amlt results download "$experiment" --status pass \
  --include '*.json' --include '*.jsonl' --include '*.txt' \
  --output "$state/download" > "$state/download.log" 2>&1 || true

mapfile -t natural_predictions < <(find "$state/download" -type f \
  -path '*/eval/eedi_natural_c4_25/judge_predictions.jsonl' | sort)
mapfile -t natural_eval < <(find "$state/download" -type f \
  -path '*/eval/eedi_natural_c4_25/judge_eval.jsonl' | sort)
mapfile -t natural_metrics < <(find "$state/download" -type f \
  -path '*/eval/eedi_natural_c4_25/metrics.json' | sort)
if [[ ${#natural_predictions[@]} -eq 1 && ${#natural_eval[@]} -eq 1 \
      && ${#natural_metrics[@]} -eq 1 ]]; then
  semantic="$state/natural_c4_semantic"
  judge_python=/home/v-yuejiali/fkft/.venv/bin/python
  judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
  mkdir -p "$semantic"
  bash scripts/wait_for_trapi_auth.sh
  PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
    --predictions-jsonl "${natural_predictions[0]}" \
    --eval-jsonl "${natural_eval[0]}" --output-dir "$semantic" \
    --judge-backend trapi --trapi-deployment gpt-4o_2024-11-20 \
    --judge-workers 12 --judge-max-retries 12 --judge-timeout-seconds 60 \
    --judge-cache "$semantic/cache.jsonl" --judge-audit-samples 25 \
    --seed 20260811 > "$semantic/judge.log" 2>&1 && \
  PYTHONPATH="$project/scripts" /home/v-yuejiali/.venvs/studentmodeling/bin/python \
    scripts/summarize_state8_diagnosis_c4.py \
    --judge-redacted "$semantic/predictions_redacted.jsonl" \
    --generation-metrics "${natural_metrics[0]}" \
    --output "$semantic/summary.json" > "$semantic/summarize.log" 2>&1 || true
fi

STATUS="$status" STATE="$state" EXPERIMENT="$experiment" \
  /home/v-yuejiali/.venvs/studentmodeling/bin/python - <<'PY'
import json, os, pathlib, subprocess
state = pathlib.Path(os.environ['STATE'])
rows = json.loads(os.environ['STATUS'])
summary = {
    'experiment': os.environ['EXPERIMENT'],
    'rows': rows,
    'all_passed': bool(rows) and all(str(r.get('status') or '').casefold() == 'pass' for r in rows),
}
(state / 'terminal_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
for row in rows:
    if str(row.get('status') or '').casefold() != 'pass':
        job = str(row['job_name'])
        safe = job.lstrip(':').replace('/', '_')
        with (state / f'{safe}.failure.log').open('w') as handle:
            subprocess.run(
                ['timeout', '180', 'amlt', 'logs', 'view', '-n', '1000', '--no-pager',
                 os.environ['EXPERIMENT'], job], stdout=handle, stderr=subprocess.STDOUT,
                check=False,
            )
(state / ('complete' if summary['all_passed'] else 'needs_attention')).touch()
PY
