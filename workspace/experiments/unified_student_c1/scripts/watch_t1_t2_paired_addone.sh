#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
state="$project/results/t1_t2_paired_addone_seed42_20260830"
experiment=t1-t2-paired-addone-seed42-20260830
job=qwen3-32b-t1-t2-paired-addone-seed42
mkdir -p "$state"
cd "$project" || exit 1
exec 9>"$state/watcher.lock"
flock -n 9 || exit 0
echo $$ > "$state/watcher.pid"
trap 'rm -f "$state/watcher.pid"' EXIT

while true; do
  date -u +%Y-%m-%dT%H:%M:%SZ > "$state/heartbeat"
  source scripts/use_studentmodeling_feeds_auth.sh
  if [[ "${STUDENTMODELING_FEEDS_AUTH_HEALTH:-unhealthy}" == unhealthy ]]; then
    printf '%s\tauth_unhealthy\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$state/latest_status.tsv"
    sleep 60
    continue
  fi
  status="$(timeout 90 amlt --json-tables status "$experiment" ":$job" \
    --no-calculate-result-size --force-status-update 2>/dev/null)" || { sleep 60; continue; }
  printf '%s\n' "$status" > "$state/latest_status.json"
  job_status="$(STATUS="$status" python - <<'PY'
import json, os
rows=json.loads(os.environ['STATUS'])
print(str(rows[0].get('status') or '').casefold() if len(rows)==1 else 'unknown')
PY
)"
  printf '%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$job_status" > "$state/latest_status.tsv"
  case "$job_status" in
    pass) break ;;
    preparing|pre-queued|queued|running|resuming|paused) sleep 60 ;;
    failed|killed|cancelled|canceled|stopped)
      timeout 180 amlt logs view -n 4000 "$experiment" ":$job" > "$state/failure.log" 2>&1 || true
      touch "$state/needs_attention"
      exit 1 ;;
    *) sleep 60 ;;
  esac
done

rm -rf "$state/download"
mkdir -p "$state/download"
timeout --foreground 900 amlt results download "$experiment" ":$job" \
  --include '*.json' --include '*.jsonl' --output "$state/download" </dev/null \
  > "$state/download.log" 2>&1 || { touch "$state/needs_attention"; exit 1; }

python - "$state" <<'PY'
import json, pathlib, sys

state=pathlib.Path(sys.argv[1])
metrics=list((state/'download').rglob('metrics.json'))
metrics=[p for p in metrics if 'eval_t1' in str(p)]
if len(metrics) != 1:
    raise RuntimeError(f'expected one eval_t1 metrics file, found {metrics}')
x=json.load(open(metrics[0], encoding='utf-8'))
groups=x['metrics_by_group']
n=sum(int(v['n']) for v in groups.values())
if n != 10913:
    raise RuntimeError(f'unexpected T1 row count: {n}')
accuracy=sum(int(v['n'])*float(v['accuracy']) for v in groups.values())/n
summary={
    'status':'complete',
    'protocol':'T1 + interaction-matched T2 add-one; T1 exposure matched, total updates not matched',
    'experiment':'t1-t2-paired-addone-seed42-20260830',
    'job_id':'funny_animal_r4qzc5pzt3',
    'n':n,
    't1_t2_accuracy':accuracy,
    't1_only_accuracy':0.691652157976725,
    'unified_accuracy':0.6875286355722533,
    't1_t2_minus_t1_only':accuracy-0.691652157976725,
    't1_t2_minus_unified':accuracy-0.6875286355722533,
    'metrics_path':str(metrics[0]),
}
(state/'comparison.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n', encoding='utf-8')
(state/'comparison.md').write_text(
    '# T1 + paired T2 add-one result\n\n'
    f'- T1+T2 accuracy: {accuracy:.6f}\n'
    f'- T1-only accuracy: {summary["t1_only_accuracy"]:.6f}\n'
    f'- Unified accuracy: {summary["unified_accuracy"]:.6f}\n'
    f'- T1+T2 minus T1-only: {summary["t1_t2_minus_t1_only"]:+.6f}\n'
    f'- T1+T2 minus unified: {summary["t1_t2_minus_unified"]:+.6f}\n',
    encoding='utf-8')
PY
touch "$state/complete"
