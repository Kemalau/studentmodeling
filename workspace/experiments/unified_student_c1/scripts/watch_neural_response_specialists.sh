#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
state="$project/results/neural_response_specialists_20260814"
experiment=neural-response-specialists-formal-20260814
jobs=(neural-response-specialist-t2-seed42 neural-response-specialist-t4-seed42)
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
[[ -x "$python_bin" ]] || python_bin=python
mkdir -p "$state"; cd "$project" || exit 1
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"
trap 'rm -f "$state/controller.active.pid"' EXIT
rm -f "$state/needs_attention"

heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }

while true; do
  heartbeat
  source scripts/use_studentmodeling_feeds_auth.sh
  if [[ "${STUDENTMODELING_FEEDS_AUTH_HEALTH:-unhealthy}" == unhealthy ]]; then
    printf '%s\tAUTH_OR_STATUS_RETRY\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$state/latest_status.tsv"
    sleep 60; continue
  fi
  status="$(timeout 90 amlt --json-tables status "$experiment" --hide-urls --force-status-update 2>/dev/null)" \
    || { sleep 60; continue; }
  [[ -n "$status" && "$status" != '[]' ]] || { sleep 60; continue; }
  printf '%s\n' "$status" > "$state/latest_status.json"
  summary="$(STATUS="$status" "$python_bin" - <<'PY'
import json,os
rows=json.loads(os.environ['STATUS'])
expected={'neural-response-specialist-t2-seed42','neural-response-specialist-t4-seed42'}
got={str(row.get('job_name') or '').lstrip(':'):str(row.get('status') or '').casefold() for row in rows}
if set(got)!=expected: print('CARDINALITY_DRIFT')
elif all(value=='pass' for value in got.values()): print('ALL_PASS')
elif any(value in {'failed','killed','cancelled','canceled','stopped'} for value in got.values()): print('TERMINAL_FAILURE')
else: print(','.join(f'{key}={got[key]}' for key in sorted(got)))
PY
)"
  printf '%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$summary" > "$state/latest_status.tsv"
  case "$summary" in
    ALL_PASS) break ;;
    TERMINAL_FAILURE)
      timeout 180 amlt logs view -n 1000 --no-pager "$experiment" > "$state/terminal.log" 2>&1 || true
      fail "one or more neural response specialists ended unsuccessfully" ;;
    CARDINALITY_DRIFT) fail "remote job cardinality drift" ;;
  esac
  sleep 60
done

download_failures=0
while true; do
  heartbeat
  source scripts/use_studentmodeling_feeds_auth.sh
  [[ "${STUDENTMODELING_FEEDS_AUTH_HEALTH:-unhealthy}" != unhealthy ]] || { sleep 60; continue; }
  rm -rf "$state/download"; mkdir -p "$state/download"
  if timeout --foreground 900 amlt results download "$experiment" --status pass \
      --include '*.json' --include '*.jsonl' --include '*.txt' \
      --output "$state/download" </dev/null >> "$state/controller.log" 2>&1; then
    break
  fi
  download_failures=$((download_failures+1))
  (( download_failures < 8 )) || fail "result download exhausted retries"
  sleep 60
done

rm -rf "$state/final"
mkdir -p "$state/final/T2" "$state/final/T4"
for task in T2 T4; do
  mapfile -t metrics < <(find "$state/download" -type f -path "*/eval/$task/metrics.json")
  mapfile -t predictions < <(find "$state/download" -type f -path "*/eval/$task/predictions.jsonl")
  mapfile -t completions < <(find "$state/download" -type f -path "*/eval/$task/completion.json")
  [[ ${#metrics[@]} -eq 1 && ${#predictions[@]} -eq 1 && ${#completions[@]} -eq 1 ]] \
    || fail "$task artifact cardinality drift metrics=${#metrics[@]} predictions=${#predictions[@]} completions=${#completions[@]}"
  "$python_bin" - "$task" "${metrics[0]}" "${predictions[0]}" "${completions[0]}" "$state/final/$task" <<'PY' \
    || fail "$task formal artifact validation failed"
import hashlib,json,pathlib,shutil,sys
task=sys.argv[1]; metrics_path,pred_path,completion_path,out=map(pathlib.Path,sys.argv[2:])
metrics=json.load(metrics_path.open(encoding='utf-8')); receipt=json.load(completion_path.open(encoding='utf-8'))
expected={'T2':('predict_response',55569,3693,'math_equivalence','e44efac7f183fdb7b5fa4dfe2075e5b10efaa5abf689dc33e0fd718256dfc62e'),
          'T4':('predict_prospective_error',78600,4600,'joint_math_equivalence','240dc2f1a02bbb446e4d73b0301ee4da7f6716c088746f7ec17bf82934c2aa1e')}[task]
internal,train_n,test_n,metric,train_sha=expected
assert receipt['status']=='complete' and receipt['public_task']==task and receipt['internal_task']==internal,receipt
assert receipt['train_rows']==train_n and receipt['test_rows']==test_n,receipt
assert receipt['train_data_sha256']==train_sha,receipt
assert metrics['support']['train']+metrics['support']['valid']==train_n and metrics['support']['test']==test_n,metrics
assert metrics['metrics']['overall']['n']==test_n and metric in metrics['metrics']['overall'],metrics
records=[json.loads(line) for line in pred_path.open(encoding='utf-8') if line.strip()]
assert len(records)==test_n and len({row['example_id'] for row in records})==test_n
assert {row['task'] for row in records}=={internal}
assert receipt['predictions_sha256']==hashlib.sha256(pred_path.read_bytes()).hexdigest(),receipt
out.mkdir(parents=True,exist_ok=True)
shutil.copy2(metrics_path,out/'metrics.json'); shutil.copy2(pred_path,out/'predictions.jsonl'); shutil.copy2(completion_path,out/'completion.json')
PY
done

"$python_bin" scripts/summarize_neural_response_specialists.py \
  --t2 "$state/final/T2/metrics.json" --t4 "$state/final/T4/metrics.json" \
  --traditional results/formal_specialist_v1/task_specialists.json \
  --matrix results/formal_specialist_v2/formal_evaluation_matrix.json \
  --output-json "$state/comparison.json" --output-md "$state/comparison.md" \
  || fail "comparison summary failed"
touch "$state/complete"; rm -f "$state/needs_attention"; heartbeat
log "formal neural response specialists validated and summarized"
